"""phase 0: the L09 counterfactual, which closes logistics calibration one way or the other

  PYTHONPATH=. python -m src.data_synthesis.v5.l09_counterfactual

every sourced freight candidate L09 does not already hold is tried on L09 alone. nothing else
moves: the other eleven agents keep their production calibration rows. each counterfactual
rebuilds its own logistics profile and order budgets in a temporary directory, exactly as a
promotion would, and is diagnosed on the production order design. no production file is written,
and production hashes are compared before and after

the convention was fixed before any result was seen and is production's own: a freight reading
is its F and alpha with no weight domain, which is how the production calibration applies every
freight reading. two sensitivities are reported and can never promote:
  budget held   the counterfactual profile against the production budgets, unrebuilt
  weight domain the candidate's evidenced weight domain applied to L09 as well

a control rebuild with the production rows unchanged must reproduce the production profile and
orders byte for byte, otherwise the machinery is not faithful and the run stops

the guards, every one strict against production, decided mechanically:
  feasible orders                        not fewer
  high_quality feasible orders           not fewer
  orders with no eligible logistics bidder  not more
  coverage   no (city, weight band) cell gains an order with no structurally eligible agent
             (region and weight only), and no cell gains an order with no eligible bidder
  niche      L09's lowest-cost win share in Jyvaskyla, Oulu and Tampere, each not lower
  groups     no tied-price group and no structural-domination pair production lacks, in any city

a structural-domination pair (city, A, B) means B serves the city no slower than A, and on every
order in that city where A is eligible, B is eligible too and prices no higher than A from every
manufacturer location that makes the product, so A can never beat B. identical prices dominate
both ways, which is how a tie shows up here

a candidate that clears every guard improves on production only if it also removes the L05/L09
tied-price group. among those, the smallest total absolute lowest-cost win-share change summed
over all twelve agents wins, the lower candidate number breaking an exact tie. if none qualifies,
production stays as it is
"""

import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import pandas as pd

from src.data_synthesis.v5 import diagnostics, freight_candidates, logistics, orders, proposal_validation
from src.data_synthesis.v5.settings import (
    AGENT_CALIBRATION_PATH, PROCESSED_V5, REPORTS_V5, V5_DIR, rel, sha256_file, write_csv, write_json,
)

AGENT = "L09"
NICHE_CITIES = ("Jyvaskyla", "Oulu", "Tampere")
TIE_PARTNER = "L05"
TIE_CITY = "Helsinki"
DATA_DIR = PROCESSED_V5 / "l09_counterfactual"
REPORT_DIR = REPORTS_V5 / "l09_counterfactual"
PRICE_TOLERANCE = 1e-9

CONVENTION = ("production's: a freight reading is its F and alpha with no weight domain, as the production "
              "calibration applies every freight reading; fixed before any counterfactual was run")


# ── calibration rows ────────────────────────────────────────────────────────

def production_calibration():
    # round_trip keeps every float exactly as written: pandas' default parser reads
    # 0.054540000000000005 back as 0.05454, which would neither match the candidate
    # exactly nor rewrite the production bytes
    frame = pd.read_csv(AGENT_CALIBRATION_PATH, float_precision="round_trip")
    return frame.to_dict("records"), list(frame.columns)


def held_candidate(rows, candidates):
    """the candidate number L09 holds in production, matched on reading id, F and alpha exactly"""
    row = next(r for r in rows if r["agent_id"] == AGENT)
    held = [n for n, (reading_id, _) in freight_candidates.CANDIDATES.items()
            if reading_id == row["calibration_reading_id"]
            and candidates[n]["fixed_charge_eur"] == row["fixed_charge_eur"]
            and candidates[n]["alpha_eur_per_kg"] == row["alpha_eur_per_kg"]]
    if len(held) != 1:
        raise ValueError(f"{AGENT}'s production row matches {len(held)} candidates exactly, expected one")
    return held[0]


def counterfactual_rows(rows, candidate_number, candidate, with_domain):
    out = []
    for row in rows:
        row = dict(row)
        if row["agent_id"] == AGENT:
            row["calibration_reading_id"] = freight_candidates.CANDIDATES[candidate_number][0]
            row["fixed_charge_eur"] = candidate["fixed_charge_eur"]
            row["alpha_eur_per_kg"] = candidate["alpha_eur_per_kg"]
            if with_domain:
                row["min_shipment_weight_kg"] = candidate["min_shipment_weight_kg"]
                row["max_shipment_weight_kg"] = candidate["max_shipment_weight_kg"]
                row["max_weight_inclusive"] = candidate["max_weight_inclusive"]
        out.append(row)
    return out


def rebuild(directory, rows, columns):
    """calibration, logistics profile and orders with budgets, in directory only"""
    for protected in (V5_DIR, AGENT_CALIBRATION_PATH.parent):
        if Path(directory).resolve() == protected.resolve():
            raise ValueError(f"a counterfactual must never be written into {rel(protected)}")
    directory.mkdir(parents=True, exist_ok=True)
    calibration_path = directory / "agent_calibration.csv"
    profile_path = directory / "logistics_agents.csv"
    orders_path = directory / "orders.csv"
    frame = pd.DataFrame(rows)
    write_csv(frame[[c for c in columns if c in frame.columns] + [c for c in frame.columns if c not in columns]],
              calibration_path)
    logistics.build(calibration_path, profile_path)
    order_frame, _, _ = orders.generate_frame(profile_path)
    write_csv(order_frame, orders_path)
    return calibration_path, profile_path, orders_path


# ── measuring one variant ───────────────────────────────────────────────────

def order_facts(data):
    makers = {}
    for m in data.manufacturers:
        for product in str(m["product_types"]).split("|"):
            makers.setdefault(product, set()).add(m["location"])
    facts = []
    for order in data.orders:
        weight = order["quantity"] * data.products[(order["product_id"], order["material_type"])][1]
        city = order["deliver_to"]
        km = [data.route.distance_km[(origin, city)] for origin in makers[order["product_id"]]]
        facts.append({"order_id": order["order_id"], "city": city, "weight": weight,
                      "band": diagnostics.weight_band(weight), "scenario": order["scenario_type"],
                      "day": datetime.strptime(order["order_created"], "%Y-%m-%dT%H:%M:%S").date(),
                      "priority": order["priority"], "km_range": (min(km), max(km))})
    return facts


def route_days(agent, city):
    return (agent["default_completion_same_city_days"] if agent["location"] == city
            else agent["default_completion_intercity_days"])


def domination_pairs(data, eligible_sets, facts):
    """(city, dominated, dominating) pairs, see the module docstring

    a price is linear in distance, so being no higher at the nearest and the farthest maker
    location is being no higher at every maker location"""
    agents = {a["agent_id"]: a for a in data.logistics}
    serving = {}
    support = {}
    for city in sorted({f["city"] for f in facts}):
        serving[city] = sorted(a for a, v in agents.items() if city in v["regions"])
        for low in serving[city]:
            for high in serving[city]:
                if low != high and route_days(agents[high], city) <= route_days(agents[low], city):
                    support[(city, low, high)] = 0
    broken = set()
    for fact, eligible in zip(facts, eligible_sets):
        if not eligible:
            continue
        city = fact["city"]
        prices = {a: tuple(data.route.price_for_distance(a, fact["weight"], km, fact["day"], fact["priority"])
                           for km in fact["km_range"]) for a in eligible}
        for low in eligible:
            for high in serving[city]:
                key = (city, low, high)
                if key not in support or key in broken:
                    continue
                if high not in prices or any(ph > pl + PRICE_TOLERANCE for ph, pl in zip(prices[high], prices[low])):
                    broken.add(key)
                else:
                    support[key] += 1
    return {key for key, n in support.items() if n and key not in broken}


def measure(data, result, facts):
    records = result["records"]
    # pandas stores an infeasible order's None winner as NaN, which is truthy, so every winner
    # is normalized to an agent id or None before anything counts it
    winners = [w if isinstance(w, str) else None for w in records["logistics_winner"]]
    eligible_sets = [set(e) for e in records["logistics_eligible"]]
    feasible = list(records["feasible"])
    wins = Counter(w for w in winners if w)
    total_wins = sum(wins.values())
    agent_ids = sorted(a["agent_id"] for a in data.logistics)

    city_wins, city_total = Counter(), Counter()
    no_bidder_cells, structural_gap_cells = Counter(), Counter()
    for fact, winner, eligible in zip(facts, winners, eligible_sets):
        if winner:
            city_total[fact["city"]] += 1
            city_wins[(fact["city"], winner)] += 1
        cell = f"{fact['city']} | {fact['band']}"
        if not eligible:
            no_bidder_cells[cell] += 1
        if not any(fact["city"] in a["regions"] and data.route.eligible_weight(a["agent_id"], fact["weight"])
                   for a in data.logistics):
            structural_gap_cells[cell] += 1

    def city_share(city, agent):
        return Fraction(city_wins[(city, agent)], city_total[city]) if city_total[city] else Fraction(0)

    tie_orders = [i for i, (fact, winner, eligible) in enumerate(zip(facts, winners, eligible_sets))
                  if fact["city"] == TIE_CITY and {AGENT, TIE_PARTNER} <= eligible and winner in (AGENT, TIE_PARTNER)]
    groups = diagnostics.tied_price_groups(data, result)
    return {
        "orders": len(facts),
        "feasible_orders": int(sum(feasible)),
        "high_quality_feasible_orders": int(sum(1 for fact, ok in zip(facts, feasible)
                                                if ok and fact["scenario"] == "high_quality")),
        "high_quality_orders": int(sum(1 for fact in facts if fact["scenario"] == "high_quality")),
        "orders_with_no_eligible_logistics_bidder": int(sum(1 for e in eligible_sets if not e)),
        "lowest_cost_logistics_wins": {a: int(wins.get(a, 0)) for a in agent_ids},
        "lowest_cost_logistics_win_share": {a: Fraction(wins.get(a, 0), total_wins) for a in agent_ids},
        "l09_win_share_by_city": {city: city_share(city, AGENT) for city in sorted(city_total)},
        "l09_wins_by_city": {city: int(city_wins[(city, AGENT)]) for city in sorted(city_total)},
        "lowest_cost_wins_by_city": {city: int(n) for city, n in sorted(city_total.items())},
        "no_bidder_orders_by_city_and_band": dict(sorted(no_bidder_cells.items())),
        "structural_gap_orders_by_city_and_band": dict(sorted(structural_gap_cells.items())),
        "tied_price_groups": groups,
        "tied_price_group_sets": {tuple(g["agents"]) for g in groups},
        "domination_pairs": domination_pairs(data, eligible_sets, facts),
        "l05_l09_helsinki_contested": {"orders": len(tie_orders),
                                       "wins": dict(Counter(winners[i] for i in tie_orders))},
        "winners": winners,
        "tie_order_positions": tie_orders,
    }


def analyse(label, profile_path, orders_path, report_dir=None):
    data = diagnostics.Dataset(V5_DIR, profile_path, orders_path)
    result = diagnostics.run(data)
    if report_dir is not None:
        diagnostics.write_reports(data, result, report_dir / label,
                                  f"BID COMPETITION DIAGNOSTICS - v5 L09 counterfactual: {label}")
    return measure(data, result, order_facts(data))


# ── the guards and the decision ─────────────────────────────────────────────

def guards(production, variant):
    checks = {
        "feasible_orders_not_fewer": variant["feasible_orders"] >= production["feasible_orders"],
        "high_quality_feasible_orders_not_fewer":
            variant["high_quality_feasible_orders"] >= production["high_quality_feasible_orders"],
        "no_bidder_orders_not_more": (variant["orders_with_no_eligible_logistics_bidder"]
                                      <= production["orders_with_no_eligible_logistics_bidder"]),
        "no_cell_gains_a_structural_coverage_gap": all(
            n <= production["structural_gap_orders_by_city_and_band"].get(cell, 0)
            for cell, n in variant["structural_gap_orders_by_city_and_band"].items()),
        "no_cell_gains_a_no_bidder_order": all(
            n <= production["no_bidder_orders_by_city_and_band"].get(cell, 0)
            for cell, n in variant["no_bidder_orders_by_city_and_band"].items()),
    }
    for city in NICHE_CITIES:
        checks[f"l09_share_not_lower_in_{city}"] = (variant["l09_win_share_by_city"].get(city, Fraction(0))
                                                    >= production["l09_win_share_by_city"].get(city, Fraction(0)))
    new_groups = sorted(variant["tied_price_group_sets"] - production["tied_price_group_sets"])
    new_pairs = sorted(variant["domination_pairs"] - production["domination_pairs"])
    checks["no_new_tied_price_group"] = not new_groups
    checks["no_new_structural_domination_pair"] = not new_pairs
    return checks, new_groups, new_pairs


def tie_removed(production, variant):
    pair = {AGENT, TIE_PARTNER}
    held = any(pair <= set(g) for g in production["tied_price_group_sets"])
    return held and not any(pair <= set(g) for g in variant["tied_price_group_sets"])


def total_share_change(production, variant):
    return sum(abs(variant["lowest_cost_logistics_win_share"][a] - production["lowest_cost_logistics_win_share"][a])
               for a in production["lowest_cost_logistics_win_share"])


def decide(production, primary):
    rows = []
    for number, variant in sorted(primary.items()):
        checks, new_groups, new_pairs = guards(production, variant)
        rows.append({"candidate": number, "checks": checks, "clears_every_guard": all(checks.values()),
                     "failed_guards": [k for k, ok in checks.items() if not ok],
                     "new_tied_price_groups": new_groups, "new_domination_pairs": new_pairs,
                     "removes_l05_l09_tie": tie_removed(production, variant),
                     "total_abs_win_share_change": total_share_change(production, variant)})
    qualifying = [r for r in rows if r["clears_every_guard"] and r["removes_l05_l09_tie"]]
    winner = min(qualifying, key=lambda r: (r["total_abs_win_share_change"], r["candidate"])) if qualifying else None
    return rows, winner


# ── serializing ─────────────────────────────────────────────────────────────

def plain(value):
    if isinstance(value, Fraction):
        return {"value": round(float(value), 6), "exact": f"{value.numerator}/{value.denominator}"}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def public(metrics):
    return plain({k: v for k, v in metrics.items() if k not in ("winners", "tie_order_positions")})


def production_contested_now(production, variant):
    """who wins, under the variant, the Helsinki orders production's L05/L09 group contested

    orders are the same in every variant apart from budgets, so positions line up"""
    return dict(Counter(str(variant["winners"][i]) for i in production["tie_order_positions"]))


def render(document):
    lines = ["L09 COUNTERFACTUAL (phase 0)", f"convention: {document['convention']}", ""]
    production = document["production"]
    head = f"  {'':44s}{'production':>12s}" + "".join(f"{'cand ' + str(r['candidate']):>12s}"
                                                       for r in document["candidates"])
    lines.append(head)

    def row(label, fn):
        lines.append(f"  {label:44s}{fn(production):>12}" + "".join(
            f"{fn(document['primary'][str(r['candidate'])]):>12}" for r in document["candidates"]))

    row("feasible orders", lambda m: m["feasible_orders"])
    row("high_quality feasible orders", lambda m: m["high_quality_feasible_orders"])
    row("orders with no eligible logistics bidder", lambda m: m["orders_with_no_eligible_logistics_bidder"])
    row("L09 lowest-cost win share overall %",
        lambda m: f"{m['lowest_cost_logistics_win_share'][AGENT]['value'] * 100:.2f}")
    for city in sorted(production["l09_win_share_by_city"]):
        row(f"L09 win share {city} %", lambda m, c=city: f"{m['l09_win_share_by_city'][c]['value'] * 100:.2f}")
    row("L05/L09 Helsinki contested orders", lambda m: m["l05_l09_helsinki_contested"]["orders"])
    row("tied-price groups", lambda m: len(m["tied_price_groups"]))
    row("structural-domination pairs", lambda m: len(m["domination_pairs"]))
    lines.append("")
    for r in document["candidates"]:
        lines.append(f"candidate {r['candidate']} ({r['label']}): clears every guard {r['clears_every_guard']}, "
                     f"removes the L05/L09 tie {r['removes_l05_l09_tie']}, total abs share change "
                     f"{r['total_abs_win_share_change']['value'] * 100:.2f} points")
        if r["failed_guards"]:
            lines.append(f"    failed: {', '.join(r['failed_guards'])}")
        if r["new_tied_price_groups"]:
            lines.append(f"    new tied-price groups: {r['new_tied_price_groups']}")
        if r["new_domination_pairs"]:
            lines.append(f"    new domination pairs (city, dominated, dominating): {r['new_domination_pairs']}")
    lines += ["", f"decision: {document['decision']['outcome']}", document["decision"]["reason"]]
    return lines


def main():
    before = proposal_validation.production_hashes()
    candidates = freight_candidates.load_candidates()
    verification = freight_candidates.verify_candidates(candidates)
    rows, columns = production_calibration()
    held = held_candidate(rows, candidates)
    others = [n for n in sorted(candidates) if n != held]
    print(f"L09 holds candidate {held}; trying {others}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    production = analyse("production", None, None, REPORT_DIR)
    print(f"production: feasible {production['feasible_orders']}, no bidder "
          f"{production['orders_with_no_eligible_logistics_bidder']}")

    primary, budget_held, weight_domain, builds = {}, {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="l09_counterfactual_") as tmp:
        tmp = Path(tmp)
        control_calibration, control_profile, control_orders = rebuild(tmp / "control", rows, columns)
        control = {"agent_calibration_csv_identical": sha256_file(control_calibration) == sha256_file(AGENT_CALIBRATION_PATH),
                   "logistics_agents_csv_identical": sha256_file(control_profile) == sha256_file(V5_DIR / "logistics_agents.csv"),
                   "orders_csv_identical": sha256_file(control_orders) == sha256_file(V5_DIR / "orders.csv")}
        print(f"control rebuild: {control}")
        if not all(control.values()):
            raise SystemExit("the control rebuild does not reproduce production, the counterfactual machinery is not faithful")

        for number in others:
            label = f"candidate_{number}"
            variant_rows = counterfactual_rows(rows, number, candidates[number], with_domain=False)
            calibration_path, profile_path, orders_path = rebuild(tmp / label, variant_rows, columns)
            primary[number] = analyse(label, profile_path, orders_path, REPORT_DIR)
            budget_held[number] = analyse(label, profile_path, V5_DIR / "orders.csv")
            keep = DATA_DIR / label
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(calibration_path, keep / "agent_calibration.csv")
            shutil.copyfile(profile_path, keep / "logistics_agents.csv")
            builds[number] = {"calibration": rel(keep / "agent_calibration.csv"),
                              "logistics_profile": rel(keep / "logistics_agents.csv"),
                              "orders_csv_sha256": sha256_file(orders_path),
                              "orders_csv_rows": int(len(pd.read_csv(orders_path, usecols=["order_id"]))),
                              "orders_csv_persisted": False}
            print(f"{label}: feasible {primary[number]['feasible_orders']}, "
                  f"L09 share {float(primary[number]['lowest_cost_logistics_win_share'][AGENT]) * 100:.2f}%")

        for number in others:
            label = f"candidate_{number}_weight_domain"
            variant_rows = counterfactual_rows(rows, number, candidates[number], with_domain=True)
            _, profile_path, orders_path = rebuild(tmp / label, variant_rows, columns)
            weight_domain[number] = analyse(label, profile_path, orders_path)
            print(f"{label}: feasible {weight_domain[number]['feasible_orders']}")

    decisions, winner = decide(production, primary)
    sensitivity = {}
    for name, variants in (("budget_held", budget_held), ("weight_domain", weight_domain)):
        rows_s, winner_s = decide(production, variants)
        sensitivity[name] = {"candidates": plain(rows_s), "would_qualify": winner_s["candidate"] if winner_s else None,
                             "metrics": {str(n): public(m) for n, m in variants.items()}}

    contested_now = {str(n): production_contested_now(production, m) for n, m in primary.items()}
    if winner is None:
        outcome = "keep production unchanged"
        reason = ("no candidate clears every guard and removes the L05/L09 tie, so production stays as it is. "
                  "the L05/L09 tie and L04's lowest-cost share are disclosed dataset characteristics, and logistics "
                  "calibration is closed without a further freight redesign")
    else:
        outcome = f"promote candidate {winner['candidate']} on L09"
        reason = (f"candidate {winner['candidate']} clears every guard, removes the L05/L09 tie and has the smallest "
                  f"total absolute win-share change among qualifying candidates")
    after = proposal_validation.production_hashes()
    document = {
        "phase": "0, the L09 counterfactual",
        "convention": CONVENTION,
        "candidate_verification": verification,
        "l09_production_candidate": held,
        "candidates_tried": others,
        "control_rebuild": control,
        "production": public(production),
        "primary": {str(n): public(m) for n, m in primary.items()},
        "candidates": [{**plain(r), "label": candidates[r["candidate"]]["label"],
                        "reading_id": freight_candidates.CANDIDATES[r["candidate"]][0],
                        "build": builds[r["candidate"]]} for r in decisions],
        "production_l05_l09_helsinki_contested_orders_won_now_by": contested_now,
        "sensitivities_reported_never_promote": sensitivity,
        "decision": {"outcome": outcome, "reason": reason,
                     "winner": winner["candidate"] if winner else None},
        "production_files_unchanged": before == after,
    }
    write_json(REPORT_DIR / "decision.json", document)
    text = render(document)
    (REPORT_DIR / "decision.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text))
    print(f"\nproduction files unchanged: {before == after}")
    print(f"wrote {rel(REPORT_DIR / 'decision.json')} and {rel(REPORT_DIR / 'decision.txt')}")
    return 0 if before == after else 1


if __name__ == "__main__":
    sys.exit(main())
