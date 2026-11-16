"""v5 bid-competition diagnostics: the v4 check, run against data/synthetic/v5

  PYTHONPATH=. python -m src.data_synthesis.v5.diagnostics
  PYTHONPATH=. python -m src.data_synthesis.v5.diagnostics --fuel-counterfactual

the same class of report as src/visualization/bid_competition_diagnostics.py, which
is read but not imported or modified: eligibility, bidder counts, feasible chains,
the three infeasibility cases, top-2 price gaps, win shares, and feasibility by
scenario and material. every order is simulated from the csv files alone

what differs from v4, because v5 differs:
  material offers use the agent's price ratio times the reference price on the
  order's own date; polyester's reference price is fixed and flagged as such
  manufacturer offers use the USD unit costs
  a logistics offer depends on the manufacturer the shipment leaves from, so it is
  priced per manufacturer candidate with the v5 route cost, which applies the shared
  fuel series, damage risk and priority but not initial_markup or min_profit_margin
  a logistics agent is eligible only inside the weight range its reading is evidenced for
  the chain checks cover lowest-cost and fastest selection. v4's balanced 0.4/0.35/0.25
  rule is not reproduced, it is not the current selection rule

the added checks: whether the shared fuel series moves with the order date, whether
dated material prices move across the window, whether budgets track the prices, which
calibration corner and which agent wins at which weight, and whether any agent ever
wins outside its weight range. anything that looks structurally wrong is printed as a
WARNING and left for a decision; nothing here adjusts a calibration

--fuel-counterfactual replays the same orders with every agent on another shared
series, and on the per-agent mapping the previous build used, and reports how the
logistics win shares move. it writes a report and changes no dataset file
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd

from src.data_synthesis.v5 import fuel
from src.data_synthesis.v5 import logistics as logistics_module
from src.data_synthesis.v5.settings import (
    ALL_MATERIALS, FIXED_PRICE_MATERIALS, PRIORITY_MARKUP, REPORTS_V5, SHARED_FUEL_SERIES_ID, V5_DIR, rel,
    write_json,
)

TOP_N = 5
# half-open [low, high) bands on the parcel limits and the Lavarahti weight segment edges
WEIGHT_BANDS = ((0, 25, "under 25 kg"), (25, 35, "25-35 kg"), (35, 500, "35-500 kg"), (500, 1000, "500-1000 kg"),
                (1000, 1500, "1000-1500 kg"), (1500, 2000, "1500-2000 kg"), (2000, 2500, "2000-2500 kg"),
                (2500, float("inf"), "2500 kg and over"))
PRODUCTION_TITLE = ("BID COMPETITION DIAGNOSTICS - version v5 (corrected: damage and priority applied, "
                    "shared fuel, parcel weight caps, polyester fixed)")
REPORT_STEM = "bid_competition_v5"

# the per-agent mapping the previous v5 build used, as recorded in
# prompt-reports/v5_synthesis_and_diagnostics_20260913_131847.md. replayed only by the
# counterfactual, to measure what a mapping choice does; nothing prices with it
PRIOR_PER_AGENT_MAPPING = {
    "L01": "matkahuolto_lavarahti", "L02": "matkahuolto_lavarahti", "L03": "postnord_parcels",
    "L04": "matkahuolto_lavarahti", "L05": "postnord_pallets_and_groupage", "L06": "pakettipiste_domestic",
    "L07": "matkahuolto_parcel_and_transport", "L08": "postnord_pallets_and_groupage",
    "L09": "postnord_pallets_and_groupage", "L10": "postnord_parcels", "L11": "matkahuolto_lavarahti",
    "L12": "pakettipiste_domestic",
}


def workload_markup(ratio):
    if ratio < 0.50:
        return 0.00
    if ratio < 0.75:
        return 0.05
    if ratio < 0.90:
        return 0.10
    if ratio <= 1.00:
        return 0.15
    return None


def weight_band(kg):
    return next(label for lo, hi, label in WEIGHT_BANDS if lo <= kg < hi)


def describe_domain(domain):
    low, cap, inclusive = domain
    if low is None and cap is None:
        return "any weight"
    upper = "inf)" if cap is None else f"{cap:g}{']' if inclusive else ')'}"
    return f"[{low or 0:g}, {upper}"


def widen(bids):
    """the cheapest TOP_N plus the fastest, as the v4 diagnostic does"""
    candidates = bids[:TOP_N]
    if len(bids) > TOP_N:
        fastest = min(bids, key=lambda b: b[1])
        if fastest not in candidates:
            candidates = candidates + [fastest]
    return candidates


class Dataset:
    """material, manufacturer and product files from directory; the logistics profile and
    orders from there too, unless a calibration proposal supplies its own"""

    def __init__(self, directory=V5_DIR, logistics_profile_path=None, orders_path=None):
        logistics_profile_path = logistics_profile_path or directory / "logistics_agents.csv"
        self.materials = pd.read_csv(directory / "material_agents.csv").to_dict("records")
        self.reference = {(r.material, r.date): r.usd_per_kg for r in
                          pd.read_csv(directory / "material_reference_prices.csv").itertuples()}
        self.manufacturers = pd.read_csv(directory / "manufacturer_agents.csv").to_dict("records")
        self.logistics = pd.read_csv(logistics_profile_path).to_dict("records")
        self.products = {(r.product_id, r.material_type): (r.material_kg_per_unit, r.finished_weight_kg_per_unit)
                         for r in pd.read_csv(directory / "products.csv").itertuples()}
        self.orders = pd.read_csv(orders_path or directory / "orders.csv").to_dict("records")
        self.route = logistics_module.RouteCost(profile_path=logistics_profile_path)
        self.by_material = {m: [a for a in self.materials if m in str(a["material_types"]).split("|")]
                            for m in ALL_MATERIALS}
        for agent in self.logistics:
            agent["regions"] = set(str(agent["service_regions"]).split("|"))


def material_bids(order, data, day, days_until, kg):
    bids, material = [], order["material_type"]
    markup = PRIORITY_MARKUP[order["priority"]]
    reference = data.reference[(material, day)]
    for a in data.by_material[material]:
        if a["defect_probability"] > order["max_defect_rate"] or a["reputation_score"] < order["min_reputation_score"]:
            continue
        available, restock = a[f"{material}_available_kg"], a[f"{material}_restock_amount_kg"]
        if available >= kg:
            completion = a["default_completion_days"]
        elif not pd.isna(restock) and available + restock >= kg:
            completion = a["default_completion_days"] + a[f"{material}_restock_time_days"]
        else:
            continue
        if completion > days_until:
            continue
        internal = kg * a[f"{material}_price_ratio_to_reference"] * reference * (1 + a["defect_probability"])
        offer = max(internal * (1 + a["min_profit_margin"]), internal * (1 + a["initial_markup"]) * (1 + markup))
        bids.append((round(offer, 2), completion, a["agent_id"]))
    return sorted(bids)


def manufacturer_bids(order, data, days_until):
    bids, product = [], order["product_id"]
    markup = PRIORITY_MARKUP[order["priority"]]
    for a in data.manufacturers:
        if product not in str(a["product_types"]).split("|"):
            continue
        if a["defect_probability"] > order["max_defect_rate"] or a["reputation_score"] < order["min_reputation_score"]:
            continue
        capacity = a["daily_capacity_units"]
        if capacity <= 0:
            continue
        completion = (a["current_workload_units"] + order["quantity"]) / capacity
        if completion > days_until:
            continue
        wl = workload_markup((a["current_workload_units"] + order["quantity"]) / (capacity * days_until))
        cost = a[f"unit_cost_usd_{product}"]
        if wl is None or pd.isna(cost):
            continue
        internal = order["quantity"] * cost * (1 + a["defect_probability"])
        offer = max(internal * (1 + a["min_profit_margin"]),
                    internal * (1 + a["initial_markup"]) * (1 + wl) * (1 + markup))
        bids.append((round(offer, 2), round(completion, 2), a["agent_id"], a["location"]))
    return sorted(bids)


def logistics_eligible(order, data, days_until, weight):
    out = []
    for a in data.logistics:
        if order["deliver_to"] not in a["regions"]:
            continue
        if a["reputation_score"] < order["min_reputation_score"] or a["damage_probability"] > order["max_defect_rate"]:
            continue
        if not data.route.eligible_weight(a["agent_id"], weight):
            continue
        days = (a["default_completion_same_city_days"] if a["location"] == order["deliver_to"]
                else a["default_completion_intercity_days"])
        if days <= days_until:
            out.append((a["agent_id"], days))
    return out


def logistics_bids(order, data, eligible, origin, day_date, weight):
    distance = data.route.distance_km[(origin, order["deliver_to"])]
    return sorted((round(data.route.price_for_distance(agent, weight, distance, day_date, order["priority"]), 2),
                   days, agent) for agent, days in eligible)


def run(data):
    corner = {a["agent_id"]: a["calibration_corner"] for a in data.logistics}
    domain = {a["agent_id"]: data.route.weight_domain(a["agent_id"]) for a in data.logistics}
    eligible_count = {s: defaultdict(int) for s in ("material", "manufacturer", "logistics")}
    wins_low = {s: defaultdict(int) for s in ("material", "manufacturer", "logistics")}
    wins_fast = {s: defaultdict(int) for s in ("material", "manufacturer", "logistics")}
    bidders = {s: [] for s in ("material", "manufacturer", "logistics")}
    gaps = {s: [] for s in ("material", "manufacturer", "logistics")}
    corner_wins = defaultdict(lambda: defaultdict(int))
    agent_band_wins = defaultdict(lambda: defaultdict(int))
    max_weight_won, min_weight_won = {}, {}
    domain_violations = []
    records = []

    for order in data.orders:
        created = datetime.strptime(order["order_created"], "%Y-%m-%dT%H:%M:%S")
        deadline = datetime.strptime(order["deadline"], "%Y-%m-%d")
        days_raw = (deadline - created).days
        days_until = max(days_raw, 1)
        day = created.date().isoformat()
        mat_kg, fin_kg = data.products[(order["product_id"], order["material_type"])]
        kg, weight = order["quantity"] * mat_kg, order["quantity"] * fin_kg

        mat = material_bids(order, data, day, days_raw, kg)
        mfr = manufacturer_bids(order, data, days_until)
        log_eligible = logistics_eligible(order, data, days_until, weight)

        for stage, bids in (("material", mat), ("manufacturer", mfr)):
            bidders[stage].append(len(bids))
            for b in bids:
                eligible_count[stage][b[2]] += 1
            if len(bids) >= 2:
                gaps[stage].append((bids[1][0] - bids[0][0]) / bids[0][0] * 100)
        bidders["logistics"].append(len(log_eligible))
        for agent, _ in log_eligible:
            eligible_count["logistics"][agent] += 1
        if mfr and len(log_eligible) >= 2:
            first = logistics_bids(order, data, log_eligible, mfr[0][3], created.date(), weight)
            gaps["logistics"].append((first[1][0] - first[0][0]) / first[0][0] * 100)

        budget, feasible, cheapest_any, lowest, fastest = order["budget_usd"], 0, None, None, None
        over_budget_all, over_time_all = True, True
        mat_c = widen(mat)
        for m_bid in widen(mfr):
            log_c = widen(logistics_bids(order, data, log_eligible, m_bid[3], created.date(), weight))
            for l_bid in log_c:
                for mt in mat_c:
                    total = mt[0] + m_bid[0] + l_bid[0]
                    days = mt[1] + m_bid[1] + l_bid[1]
                    if cheapest_any is None or total < cheapest_any:
                        cheapest_any = total
                    over_budget_all &= total > budget
                    over_time_all &= days > days_until
                    if total <= budget and days <= days_until:
                        feasible += 1
                        chain = (mt[2], m_bid[2], l_bid[2])
                        if lowest is None or total < lowest[0]:
                            lowest = (total, chain)
                        if fastest is None or (days, total) < fastest[0]:
                            fastest = ((days, total), chain)

        empty = sum(1 for n in (len(mat), len(mfr), len(log_eligible)) if n == 0)
        if feasible:
            case = "feasible"
            for stage, agent in zip(("material", "manufacturer", "logistics"), lowest[1]):
                wins_low[stage][agent] += 1
            for stage, agent in zip(("material", "manufacturer", "logistics"), fastest[1]):
                wins_fast[stage][agent] += 1
            winner = lowest[1][2]
            corner_wins[weight_band(weight)][corner[winner]] += 1
            agent_band_wins[winner][weight_band(weight)] += 1
            max_weight_won[winner] = max(max_weight_won.get(winner, weight), weight)
            min_weight_won[winner] = min(min_weight_won.get(winner, weight), weight)
            for chosen in (lowest[1][2], fastest[1][2]):
                if not data.route.eligible_weight(chosen, weight):
                    domain_violations.append((order["order_id"], chosen, weight, domain[chosen]))
        elif empty == 3:
            case = "case1_no_bidder_at_any_stage"
        elif empty:
            case = "case2_a_stage_with_no_bidder"
        elif over_budget_all and not over_time_all:
            case = "case3_over_budget_only"
        elif over_time_all and not over_budget_all:
            case = "case3_over_deadline_only"
        else:
            case = "case3_mixed"
        records.append({"scenario_type": order["scenario_type"], "material_type": order["material_type"],
                        "priority": order["priority"], "month": day[:7], "feasible": bool(feasible),
                        "case": case, "chains": feasible, "weight_band": weight_band(weight),
                        "logistics_winner": lowest[1][2] if feasible else None,
                        "logistics_eligible": tuple(agent for agent, _ in log_eligible),
                        "budget_to_cheapest_chain": (budget / cheapest_any) if cheapest_any else None,
                        "budget_per_unit": budget / order["quantity"],
                        "pair": f"{order['product_id']}/{order['material_type']}",
                        "reference_usd_per_kg": data.reference[(order["material_type"], day)],
                        "order_day": day})
    return {"eligible": eligible_count, "wins_low": wins_low, "wins_fast": wins_fast, "bidders": bidders,
            "gaps": gaps, "corner_wins": corner_wins, "agent_band_wins": agent_band_wins,
            "max_weight_won": max_weight_won, "min_weight_won": min_weight_won, "domain": domain,
            "domain_violations": domain_violations, "records": pd.DataFrame(records)}


def fuel_date_pattern(data):
    """how each series in use resolves across the actual order dates"""
    used = sorted({a["fuel_surcharge_series_id"] for a in data.logistics})
    days = sorted({o["order_created"][:10] for o in data.orders})
    counts = pd.Series([o["order_created"][:10] for o in data.orders]).value_counts()
    out = {}
    for series_id in used:
        lookups = {d: data.route.fuel_lookup(series_id, datetime.strptime(d, "%Y-%m-%d").date()) for d in days}
        per_order = [(lookups[d], int(counts[d])) for d in days]
        total = sum(n for _, n in per_order)
        by_month = defaultdict(set)
        for d in days:
            by_month[d[:7]].add(lookups[d].value)
        out[series_id] = {
            "agents": sorted(a["agent_id"] for a in data.logistics if a["fuel_surcharge_series_id"] == series_id),
            "distinct_percentages_over_orders": len({l.value for l, _ in per_order}),
            "distinct_observation_dates_used": len({l.observation_date for l, _ in per_order}),
            "percent_min": min(l.value for l, _ in per_order),
            "percent_max": max(l.value for l, _ in per_order),
            "share_of_orders_after_last_observation": round(
                sum(n for l, n in per_order if l.after_last_observation) / total, 4),
            "distinct_percentages_by_month": {m: len(v) for m, v in sorted(by_month.items())},
            "window_coverage": fuel.window_coverage(data.route.series[series_id]),
        }
    return out


def tied_price_groups(data, result):
    """agents whose base price is identical for every shipment, and how the orders two or more of
    them could carry are split; with identical prices only damage, speed and eligibility decide"""
    keyed = defaultdict(list)
    for agent_id, a in data.route.agents.items():
        keyed[(a["fixed_charge_eur"], a["alpha_eur_per_kg"], a["beta_eur_per_tonne_km"], a["fuel_surcharge_series_id"],
               a["oversized_above_kg"], a["oversized_fee_eur"], a["min_shipment_weight_kg"],
               a["max_shipment_weight_kg"], a["max_weight_inclusive"])].append(agent_id)
    records, groups = result["records"], []
    for members in sorted(sorted(m) for m in keyed.values() if len(m) > 1):
        contested = records[records["logistics_winner"].isin(members)
                            & records["logistics_eligible"].apply(lambda e: sum(m in e for m in members) >= 2)]
        wins = contested["logistics_winner"].value_counts().to_dict()
        groups.append({"agents": members, "contested_orders": int(len(contested)),
                       "wins": {m: int(wins.get(m, 0)) for m in members},
                       "one_agent_wins_every_contested_order": bool(len(contested))
                                                              and max(wins.values()) == len(contested)})
    return groups


def material_movement():
    reference = pd.read_csv(V5_DIR / "material_reference_prices.csv")
    out = {}
    for m in ALL_MATERIALS:
        rows = reference[reference["material"] == m]
        out[m] = {"price_basis": rows["price_basis"].iloc[0],
                  "distinct_prices": int(rows["usd_per_kg"].nunique()),
                  "min_usd_per_kg": round(float(rows["usd_per_kg"].min()), 4),
                  "max_usd_per_kg": round(float(rows["usd_per_kg"].max()), 4),
                  "observation_dates_used": sorted(rows["series_observation_date"].dropna().unique().tolist()),
                  "days_after_last_observation": int((rows["series_days_carried"] > 31).sum())}
    return out


def share(counter):
    total = sum(counter.values())
    return {k: round(v / total * 100, 1) for k, v in sorted(counter.items(), key=lambda kv: -kv[1])} if total else {}


def report(data, result, title=PRODUCTION_TITLE):
    lines, warnings = [], []
    total = len(data.orders)
    records = result["records"]

    def section(heading):
        lines.append(f"\n{'=' * 70}\n{heading}\n{'=' * 70}")

    section(title)
    lines.append(f"orders simulated: {total}")
    lines.append(f"orders by material: {records['material_type'].value_counts().to_dict()}")

    section("ELIGIBILITY")
    for stage in ("material", "manufacturer", "logistics"):
        counts = result["bidders"][stage]
        zero = sum(1 for c in counts if c == 0)
        two = sum(1 for c in counts if c >= 2)
        lines.append(f"{stage:12s} zero bidders {zero} ({zero / total * 100:.1f}%)  2+ bidders {two} "
                     f"({two / total * 100:.1f}%)  mean {sum(counts) / total:.1f}  max {max(counts)}")
        if two / total < 0.85:
            warnings.append(f"{stage}: only {two / total * 100:.1f}% of orders have 2+ eligible bidders (v4 target 85%)")

    all_agents = {"material": [a["agent_id"] for a in data.materials],
                  "manufacturer": [a["agent_id"] for a in data.manufacturers],
                  "logistics": [a["agent_id"] for a in data.logistics]}
    section("ELIGIBILITY VS WINS PER AGENT (lowest cost)")
    for stage, agents in all_agents.items():
        lines.append(f"{stage}:")
        for agent in agents:
            elig, wins = result["eligible"][stage].get(agent, 0), result["wins_low"][stage].get(agent, 0)
            lines.append(f"  {agent}: eligible {elig}, wins {wins}, win rate {wins / elig * 100 if elig else 0:.1f}%")
            if elig == 0:
                warnings.append(f"{stage} agent {agent} is never eligible")
            elif wins == 0:
                warnings.append(f"{stage} agent {agent} is eligible {elig} times but never wins lowest cost")
            elif wins / elig < 0.01:
                warnings.append(f"{stage} agent {agent} wins lowest cost in only {wins} of {elig} eligible orders")

    section("FEASIBLE CHAINS AND INFEASIBILITY CASES")
    feasible = int(records["feasible"].sum())
    lines.append(f"feasible orders: {feasible} / {total} ({feasible / total * 100:.1f}%)")
    lines.append(records["case"].value_counts().to_string())
    lines.append(f"mean feasible chains per order: {records['chains'].mean():.1f}")
    if not 0.85 <= feasible / total <= 0.995:
        warnings.append(f"overall feasibility {feasible / total * 100:.1f}% is outside 85-99.5%")

    section("BID PRICE CLOSENESS (top-2 gap %)")
    for stage, values in result["gaps"].items():
        s = pd.Series(values)
        lines.append(f"{stage:12s} median {s.median():.1f}%  p75 {s.quantile(0.75):.1f}%  "
                     f"p90 {s.quantile(0.9):.1f}%  max {s.max():.1f}%")
    lines.append("logistics gaps are priced from the cheapest manufacturer's location")

    section("WIN SHARE (lowest-cost chain, then fastest chain)")
    for label, wins, limit in (("lowest cost", result["wins_low"], 30), ("fastest", result["wins_fast"], 55)):
        for stage in ("material", "manufacturer", "logistics"):
            shares = share(wins[stage])
            lines.append(f"{label} {stage}: {shares}")
            if shares and max(shares.values()) > limit:
                top = max(shares, key=shares.get)
                warnings.append(f"{label} {stage}: {top} takes {shares[top]}% of wins (v4 threshold {limit}%)")

    section("LOGISTICS WINS BY CALIBRATION CORNER AND SHIPMENT WEIGHT (lowest cost)")
    for _, _, band in WEIGHT_BANDS:
        wins = result["corner_wins"].get(band, {})
        n = sum(wins.values())
        lines.append(f"  {band:14s} parcel {wins.get('parcel', 0):6d}  freight {wins.get('freight', 0):6d}"
                     f"  ({wins.get('parcel', 0) / n * 100 if n else 0:.1f}% parcel)")
    labels = [label for _, _, label in WEIGHT_BANDS]
    lines.append("\nlowest-cost wins per logistics agent and shipment weight, against the agent's weight domain:")
    lines.append(f"  {'agent':5s}  {'domain kg':16s}" + "".join(f"{label:>17s}" for label in labels)
                 + f"{'lightest':>10s}{'heaviest':>10s}")
    for agent in all_agents["logistics"]:
        bands = result["agent_band_wins"].get(agent, {})
        light, heavy = result["min_weight_won"].get(agent), result["max_weight_won"].get(agent)
        lines.append(f"  {agent:5s}  {describe_domain(result['domain'][agent]):16s}"
                     + "".join(f"{bands.get(label, 0):17d}" for label in labels)
                     + (f"{light:10.1f}{heavy:10.1f}" if light is not None else f"{'-':>10s}{'-':>10s}"))
    lines.append(f"wins outside an agent's own weight domain (lowest cost or fastest): "
                 f"{len(result['domain_violations'])}")
    if result["domain_violations"]:
        warnings.append(f"{len(result['domain_violations'])} chains pick a logistics agent outside its weight domain")

    section("FEASIBILITY BY SCENARIO, MATERIAL, PRIORITY AND MONTH")
    for column in ("scenario_type", "material_type", "priority", "month"):
        grouped = records.groupby(column)["feasible"].agg(["sum", "count"])
        grouped["pct"] = (grouped["sum"] / grouped["count"] * 100).round(1)
        lines.append(f"\nby {column}:\n{grouped.to_string()}")
        if column == "material_type":
            for m, row in grouped.iterrows():
                if row["pct"] < 50:
                    warnings.append(f"material {m} is feasible in only {row['pct']}% of its orders")
    lines.append("\nscenario vs case:\n" + pd.crosstab(records["scenario_type"], records["case"]).to_string())
    lines.append("\nmaterial vs case:\n" + pd.crosstab(records["material_type"], records["case"]).to_string())

    section("BUDGETS AGAINST THE CHEAPEST CHAIN (budget / cheapest chain, any deadline)")
    ratio = records.dropna(subset=["budget_to_cheapest_chain"])
    for column in ("priority", "month", "material_type"):
        lines.append(f"\nby {column}:\n" + ratio.groupby(column)["budget_to_cheapest_chain"]
                     .describe()[["count", "25%", "50%", "75%"]].round(3).to_string())
    median = ratio["budget_to_cheapest_chain"].median()
    if not 1.0 <= median <= 1.6:
        warnings.append(f"median budget to cheapest chain is {median:.3f}, outside 1.0-1.6")
    correlations = {}
    for pair, rows in records.groupby("pair"):
        if rows["reference_usd_per_kg"].nunique() > 1:
            correlations[pair] = round(float(rows["budget_per_unit"].corr(rows["reference_usd_per_kg"])), 3)
    lines.append(f"\ncorrelation of budget per unit with the material reference price, per dated pair: {correlations}")

    section("MATERIAL PRICE MOVEMENT ACROSS THE WINDOW")
    movement = material_movement()
    for m, v in movement.items():
        lines.append(f"  {m:9s} {v}")
        if m in FIXED_PRICE_MATERIALS:
            lines.append(f"            fixed by design: {FIXED_PRICE_MATERIALS[m]}")
        elif v["distinct_prices"] <= 1:
            warnings.append(f"dated material {m} has one price across the whole window")

    section("FUEL SURCHARGE RESOLUTION BY ORDER DATE")
    pattern = fuel_date_pattern(data)
    for series_id, v in pattern.items():
        lines.append(f"  {series_id}: agents {v['agents']}, {v['distinct_percentages_over_orders']} distinct "
                     f"percentages ({v['percent_min']} to {v['percent_max']}), "
                     f"{v['share_of_orders_after_last_observation'] * 100:.1f}% of orders carried past the "
                     f"last observation, by month {v['distinct_percentages_by_month']}")
        if v["distinct_percentages_over_orders"] <= 1:
            warnings.append(f"fuel series {series_id} resolves to one percentage for every order")
        if v["share_of_orders_after_last_observation"] > 0.25:
            warnings.append(f"fuel series {series_id}: {v['share_of_orders_after_last_observation'] * 100:.1f}% "
                            "of orders use a percentage carried past its last observation")
    if len(pattern) != 1 or SHARED_FUEL_SERIES_ID not in pattern:
        warnings.append(f"agents do not all share {SHARED_FUEL_SERIES_ID}: {list(pattern)}")

    section("TIED PRICE GROUPS (identical F, alpha, beta, fuel, oversized and weight terms)")
    groups = tied_price_groups(data, result)
    for group in groups:
        lines.append(f"  {group['agents']}: {group['contested_orders']} orders with two or more members eligible "
                     f"and a member winning lowest cost, wins {group['wins']}")
        if group["one_agent_wins_every_contested_order"]:
            warnings.append(f"tied price group {group['agents']}: one agent wins every contested order {group['wins']}")
    if not groups:
        lines.append("  none")

    section("WARNINGS (reported for a decision, nothing adjusted)")
    lines.extend(f"  WARNING {w}" for w in warnings) if warnings else lines.append("  none")
    return lines, warnings, movement, pattern


def fuel_counterfactual(data):
    """logistics lowest-cost win shares with the same orders on different fuel mappings"""
    stored = {a: v["fuel_surcharge_series_id"] for a, v in data.route.agents.items()}
    shared_options = [sid for sid in fuel.FUEL_SERIES
                      if fuel.window_coverage(data.route.series[sid])["usable"]]
    variants = {f"shared {sid}": {a: sid for a in stored} for sid in shared_options}
    variants["prior per-agent carrier mapping"] = PRIOR_PER_AGENT_MAPPING
    out = {}
    try:
        for label, mapping in variants.items():
            for agent, sid in mapping.items():
                data.route.agents[agent]["fuel_surcharge_series_id"] = sid
            data.route._fuel.clear()
            result = run(data)
            out[label] = {"lowest_cost_logistics_win_share": share(result["wins_low"]["logistics"]),
                          "feasible_orders": int(result["records"]["feasible"].sum())}
    finally:
        for agent, sid in stored.items():
            data.route.agents[agent]["fuel_surcharge_series_id"] = sid
        data.route._fuel.clear()

    agents = sorted(stored)
    shared_labels = [f"shared {sid}" for sid in shared_options]

    def swing(labels):
        return {a: round(max(out[l]["lowest_cost_logistics_win_share"].get(a, 0.0) for l in labels)
                         - min(out[l]["lowest_cost_logistics_win_share"].get(a, 0.0) for l in labels), 1)
                for a in agents}

    stored_label = f"shared {SHARED_FUEL_SERIES_ID}"
    return {
        "variants": out,
        "stored_mapping": stored_label,
        "largest_swing_across_shared_series_points": max(swing(shared_labels).values()),
        "swing_across_shared_series_points_by_agent": swing(shared_labels),
        "swing_stored_vs_prior_mapping_points_by_agent": swing([stored_label, "prior per-agent carrier mapping"]),
        "largest_swing_stored_vs_prior_mapping_points": max(
            swing([stored_label, "prior per-agent carrier mapping"]).values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuel-counterfactual", action="store_true",
                        help="replay the orders on other fuel mappings, report only")
    args = parser.parse_args()

    data = Dataset()
    REPORTS_V5.mkdir(parents=True, exist_ok=True)
    if args.fuel_counterfactual:
        result = fuel_counterfactual(data)
        for label, v in result["variants"].items():
            print(f"  {label:48s} feasible {v['feasible_orders']}  {v['lowest_cost_logistics_win_share']}")
        print(f"\nlargest swing in any agent's win share across shared series: "
              f"{result['largest_swing_across_shared_series_points']} points")
        print(f"largest swing between the stored shared series and the prior per-agent mapping: "
              f"{result['largest_swing_stored_vs_prior_mapping_points']} points")
        write_json(REPORTS_V5 / "fuel_counterfactual_v5.json", result)
        print(f"wrote {rel(REPORTS_V5 / 'fuel_counterfactual_v5.json')}")
        return 0

    write_reports(data, run(data), REPORTS_V5, echo=True)
    return 0


def write_reports(data, result, out_dir, title=PRODUCTION_TITLE, echo=False):
    """the text and json diagnostics for one dataset, written into out_dir; returns the json"""
    lines, warnings, movement, pattern = report(data, result, title)
    text = "\n".join(lines)
    if echo:
        print(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{REPORT_STEM}.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    records = result["records"]
    summary = {
        "orders": len(records),
        "orders_by_material": {k: int(v) for k, v in records["material_type"].value_counts().items()},
        "feasible_orders": int(records["feasible"].sum()),
        "cases": records["case"].value_counts().to_dict(),
        "feasibility_by_material_pct": (records.groupby("material_type")["feasible"].mean() * 100).round(1).to_dict(),
        "feasibility_by_scenario_pct": (records.groupby("scenario_type")["feasible"].mean() * 100).round(1).to_dict(),
        "logistics_zero_bidders": sum(1 for count in result["bidders"]["logistics"] if count == 0),
        "eligible_logistics_material_manufacturer": {s: dict(v) for s, v in result["eligible"].items()},
        "win_share_lowest_cost": {s: share(w) for s, w in result["wins_low"].items()},
        "win_share_fastest": {s: share(w) for s, w in result["wins_fast"].items()},
        "wins_lowest_cost": {s: dict(w) for s, w in result["wins_low"].items()},
        "logistics_corner_wins_by_weight": {b: dict(v) for b, v in result["corner_wins"].items()},
        "logistics_wins_by_agent_and_weight": {a: dict(v) for a, v in result["agent_band_wins"].items()},
        "logistics_lightest_win_kg": result["min_weight_won"],
        "logistics_heaviest_win_kg": result["max_weight_won"],
        "logistics_weight_domains_kg": {a: {"min": low, "max": cap, "max_inclusive": inclusive}
                                        for a, (low, cap, inclusive) in result["domain"].items()},
        "weight_domain_violations": len(result["domain_violations"]),
        "logistics_tied_price_groups": tied_price_groups(data, result),
        "median_budget_to_cheapest_chain": float(records["budget_to_cheapest_chain"].median()),
        "material_movement": movement,
        "fuel_date_pattern": pattern,
        "warnings": warnings,
    }
    write_json(out_dir / f"{REPORT_STEM}.json", summary)
    if echo:
        print(f"\nwrote {rel(out_dir / f'{REPORT_STEM}.txt')} and {rel(out_dir / f'{REPORT_STEM}.json')}")
    return summary


if __name__ == "__main__":
    sys.exit(main())
