"""
bid competition diagnostics for chain-of-bid synthetic data.

simulates rule-based bidding on all orders using the formulas from
the agent_logic_design document
pure python calculation against the csv files.
"""

import pandas as pd
from collections import defaultdict

from config import DATA_VERSION
from src.visualization.data_utils import load_all_data, ALL_MATERIALS, ALL_PRODUCTS
from src.visualization import version_guard


USE_CAPACITY_CHECK = False

PRIORITY_MARKUP = {
    "urgent":   0.30,
    "normal":   0.00,
    "flexible": -0.15,
}


def get_workload_markup(workload_ratio):
    if workload_ratio < 0.50:
        return 0.00
    elif workload_ratio < 0.75:
        return 0.05
    elif workload_ratio < 0.90:
        return 0.10
    elif workload_ratio <= 1.00:
        return 0.15
    return None  # above 1.0 means no bid


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def pct_gap(price_a, price_b):
    if price_a <= 0:
        return None
    return abs(price_b - price_a) / price_a * 100


# ── material stage ────────────────────────────────────────────────────────────

def simulate_material_bids(order, materials, products):
    product_id    = order["product_id"]
    material_type = order["material_type"]
    priority      = order["priority"]
    quantity      = order["quantity"]

    prod_row = products[
        (products["product_id"] == product_id) &
        (products["material_type"] == material_type)
    ]
    if prod_row.empty:
        return []

    required_kg    = quantity * prod_row.iloc[0]["material_kg_per_unit"]
    max_defect     = order["max_defect_rate"]
    min_reputation = order["min_reputation_score"]
    priority_mkup  = PRIORITY_MARKUP.get(priority, 0.0)

    try:
        created  = pd.to_datetime(order["order_created"])
        deadline = pd.to_datetime(order["deadline"])
        days_until_deadline = (deadline - created).days
    except Exception:
        days_until_deadline = 14

    bids = []
    for _, agent in materials.iterrows():
        mat_types = str(agent.get("material_types", "")).split("|")
        if material_type not in mat_types:
            continue
        if agent["defect_probability"] > max_defect:
            continue
        if agent["reputation_score"] < min_reputation:
            continue

        available = agent.get(f"{material_type}_available_kg", None)
        restock   = agent.get(f"{material_type}_restock_amount_kg", None)
        restock_days = agent.get(f"{material_type}_restock_time_days", 0)
        base_cost = agent.get(f"{material_type}_base_cost_eur_per_kg", None)
        default_days = agent.get("default_completion_days", 2)

        if pd.isna(available) or pd.isna(base_cost):
            continue

        if available >= required_kg:
            completion_days = default_days
        elif not pd.isna(restock) and (available + restock) >= required_kg:
            completion_days = default_days + restock_days
        else:
            continue

        if completion_days > days_until_deadline:
            continue

        material_cost = required_kg * base_cost
        internal_cost = material_cost * (1 + agent["defect_probability"])
        minimum_bid   = internal_cost * (1 + agent["min_profit_margin"])
        initial_bid   = internal_cost * (1 + agent["initial_markup"]) * (1 + priority_mkup)
        offer_amount  = max(minimum_bid, initial_bid)

        bids.append({
            "agent_id":    agent["agent_id"],
            "offer":       round(offer_amount, 2),
            "defect":      agent["defect_probability"],
            "completion":  completion_days,
            "reputation":  agent["reputation_score"],
        })

    return sorted(bids, key=lambda x: x["offer"])


# read-only diagnostic, mirrors the checks above only to explain a zero
# bidder outcome, never used to decide a bid
def diagnose_material_reasons(order, materials, products):
    product_id    = order["product_id"]
    material_type = order["material_type"]
    quantity      = order["quantity"]

    prod_row = products[
        (products["product_id"] == product_id) &
        (products["material_type"] == material_type)
    ]
    if prod_row.empty:
        return ["no_product_material_pair"]

    required_kg    = quantity * prod_row.iloc[0]["material_kg_per_unit"]
    max_defect     = order["max_defect_rate"]
    min_reputation = order["min_reputation_score"]

    try:
        created  = pd.to_datetime(order["order_created"])
        deadline = pd.to_datetime(order["deadline"])
        days_until_deadline = (deadline - created).days
    except Exception:
        days_until_deadline = 14

    reasons = []
    for _, agent in materials.iterrows():
        mat_types = str(agent.get("material_types", "")).split("|")
        if material_type not in mat_types:
            reasons.append("does_not_supply_material")
            continue
        if agent["defect_probability"] > max_defect:
            reasons.append("defect_too_strict")
            continue
        if agent["reputation_score"] < min_reputation:
            reasons.append("reputation_too_strict")
            continue

        available = agent.get(f"{material_type}_available_kg", None)
        restock   = agent.get(f"{material_type}_restock_amount_kg", None)
        restock_days = agent.get(f"{material_type}_restock_time_days", 0)
        default_days = agent.get("default_completion_days", 2)

        if pd.isna(available):
            reasons.append("missing_stock_data")
            continue

        if available >= required_kg:
            completion_days = default_days
        elif not pd.isna(restock) and (available + restock) >= required_kg:
            completion_days = default_days + restock_days
        else:
            reasons.append("stock_and_restock_insufficient")
            continue

        if completion_days > days_until_deadline:
            reasons.append("completion_too_slow_for_deadline")
            continue

        reasons.append("eligible_but_not_returned")

    return reasons


# ── manufacturer stage ────────────────────────────────────────────────────────

def simulate_manufacturer_bids(order, manufacturers):
    product_id    = order["product_id"]
    quantity      = order["quantity"]
    priority      = order["priority"]
    max_defect    = order["max_defect_rate"]
    min_reputation = order["min_reputation_score"]
    priority_mkup = PRIORITY_MARKUP.get(priority, 0.0)

    try:
        created  = pd.to_datetime(order["order_created"])
        deadline = pd.to_datetime(order["deadline"])
        days_until_deadline = max((deadline - created).days, 1)
    except Exception:
        days_until_deadline = 14

    bids = []
    for _, agent in manufacturers.iterrows():
        prod_types = str(agent.get("product_types", "")).split("|")
        if product_id not in prod_types:
            continue
        if agent["defect_probability"] > max_defect:
            continue
        if agent["reputation_score"] < min_reputation:
            continue

        daily_cap = agent["daily_capacity_units"]
        if daily_cap <= 0:
            continue

        backlog_days     = agent["current_workload_units"] / daily_cap
        production_days  = quantity / daily_cap
        est_completion   = backlog_days + production_days

        if est_completion > days_until_deadline:
            continue

        workload_ratio = (agent["current_workload_units"] + quantity) / (daily_cap * days_until_deadline)
        workload_mkup  = get_workload_markup(workload_ratio)
        if workload_mkup is None:
            continue

        unit_cost_col = f"unit_cost_{product_id}"
        unit_cost = agent.get(unit_cost_col, None)
        if pd.isna(unit_cost) or unit_cost is None:
            continue

        base_cost    = quantity * unit_cost
        internal_cost = base_cost * (1 + agent["defect_probability"])
        minimum_bid  = internal_cost * (1 + agent["min_profit_margin"])
        initial_bid  = internal_cost * (1 + agent["initial_markup"]) * (1 + workload_mkup) * (1 + priority_mkup)
        offer_amount = max(minimum_bid, initial_bid)

        bids.append({
            "agent_id":    agent["agent_id"],
            "offer":       round(offer_amount, 2),
            "defect":      agent["defect_probability"],
            "completion":  round(est_completion, 2),
            "reputation":  agent["reputation_score"],
            "workload_ratio": round(workload_ratio, 3),
        })

    return sorted(bids, key=lambda x: x["offer"])


# ── logistics stage ───────────────────────────────────────────────────────────

def simulate_logistics_bids(order, logistics, products):
    product_id    = order["product_id"]
    material_type = order["material_type"]
    quantity      = order["quantity"]
    deliver_to    = order["deliver_to"]
    priority      = order["priority"]
    max_defect    = order["max_defect_rate"]
    min_reputation = order["min_reputation_score"]
    priority_mkup = PRIORITY_MARKUP.get(priority, 0.0)

    prod_row = products[
        (products["product_id"] == product_id) &
        (products["material_type"] == material_type)
    ]
    if prod_row.empty:
        return []

    shipment_weight = quantity * prod_row.iloc[0]["finished_weight_kg_per_unit"]

    try:
        created  = pd.to_datetime(order["order_created"])
        deadline = pd.to_datetime(order["deadline"])
        days_until_deadline = max((deadline - created).days, 1)
    except Exception:
        days_until_deadline = 14

    bids = []
    for _, agent in logistics.iterrows():
        regions = str(agent.get("service_regions", "")).split("|")
        if deliver_to not in regions:
            continue
        if agent["reputation_score"] < min_reputation:
            continue
        if agent["damage_probability"] > max_defect:
            continue

        if USE_CAPACITY_CHECK:
            avail_cap = agent.get("available_capacity_kg", None)
            if not pd.isna(avail_cap) and avail_cap is not None:
                if shipment_weight > avail_cap:
                    continue

        if deliver_to == agent.get("location", ""):
            route_days = agent["default_completion_same_city_days"]
        else:
            route_days = agent["default_completion_intercity_days"]

        if route_days > days_until_deadline:
            continue

        base_cost    = agent["base_delivery_cost"] + (agent["cost_per_kg"] * shipment_weight)
        internal_cost = base_cost * (1 + agent["damage_probability"])
        minimum_bid  = internal_cost * (1 + agent["min_profit_margin"])
        initial_bid  = internal_cost * (1 + agent["initial_markup"]) * (1 + priority_mkup)
        offer_amount = max(minimum_bid, initial_bid)

        bids.append({
            "agent_id":    agent["agent_id"],
            "offer":       round(offer_amount, 2),
            "damage":      agent["damage_probability"],
            "delivery":    route_days,
            "reputation":  agent["reputation_score"],
        })

    return sorted(bids, key=lambda x: x["offer"])


# ── chain builder ─────────────────────────────────────────────────────────────

def build_chains(mat_bids, mfr_bids, log_bids, top_n=3):
    chains = []
    for m in mat_bids[:top_n]:
        for t in mfr_bids[:top_n]:
            for l in log_bids[:top_n]:
                total = m["offer"] + t["offer"] + l["offer"]
                chains.append({
                    "material":    m["agent_id"],
                    "manufacturer": t["agent_id"],
                    "logistics":   l["agent_id"],
                    "total_cost":  round(total, 2),
                    "max_defect":  max(m["defect"], t["defect"], l.get("damage", 0)),
                    "max_days":    m["completion"] + t["completion"] + l["delivery"],
                    "min_rep":     min(m["reputation"], t["reputation"], l["reputation"]),
                })
    return chains


def select_chains(chains):
    if not chains:
        return None, None, None

    lowest = min(chains, key=lambda x: x["total_cost"])
    fastest = min(chains, key=lambda x: x["max_days"])

    costs = [c["total_cost"] for c in chains]
    days  = [c["max_days"]   for c in chains]
    defs  = [c["max_defect"] for c in chains]

    cost_range = max(costs) - min(costs) if max(costs) != min(costs) else 1
    days_range = max(days)  - min(days)  if max(days)  != min(days)  else 1
    def_range  = max(defs)  - min(defs)  if max(defs)  != min(defs)  else 1

    def balanced_score(c):
        cost_norm = (c["total_cost"] - min(costs)) / cost_range
        days_norm = (c["max_days"]   - min(days))  / days_range
        def_norm  = (c["max_defect"] - min(defs))  / def_range
        return 0.4 * cost_norm + 0.35 * days_norm + 0.25 * def_norm

    balanced = min(chains, key=balanced_score)
    return lowest, fastest, balanced

# keep the fastest bid outside the price shortlist so time filtering stays honest
def widen_candidates(bids, time_key, top_n=5):
    cands = list(bids[:top_n])
    if len(bids) > top_n:
        fastest = min(bids, key=lambda b: b[time_key])
        if fastest not in cands:
            cands.append(fastest)
    return cands
# ── main simulation ───────────────────────────────────────────────────────────

def main():
    # this script reads config.DATA_VERSION, which the environment cannot move, so a v5 request has
    # to be routed to the v5 diagnostic rather than answered with a v4 report
    pair = version_guard.mismatch()
    if pair:
        return version_guard.route_or_refuse(
            "make bid_check", pair, route="src.data_synthesis.v5.diagnostics",
            equivalent="PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.data_synthesis.v5.diagnostics")

    materials, manufacturers, logistics, products, orders = load_all_data()

    section(f"BID COMPETITION DIAGNOSTICS - version {DATA_VERSION}")
    print(f"simulating rule-based bidding on {len(orders)} orders")
    print("no agents, no nats, no docker, pure formula calculation")

    mat_eligible_counts  = []
    mfr_eligible_counts  = []
    log_eligible_counts  = []
    chain_counts         = []
    top2_mat_gaps        = []
    top2_mfr_gaps        = []
    top2_log_gaps        = []
    feasibility_records  = []
    win_records          = []

    max_log_gap_info = None
    l07_eligible     = 0
    l07_lost_to_l12  = 0

    # track eligibility separately from wins
    # distinguish rare access from poor pricing
    mat_eligible_agents = defaultdict(int)
    mfr_eligible_agents = defaultdict(int)
    log_eligible_agents = defaultdict(int)

    material_reason_counts = defaultdict(lambda: defaultdict(int))

    mat_wins  = defaultdict(int)
    mfr_wins  = defaultdict(int)
    log_wins  = defaultdict(int)

    mat_wins_fast = defaultdict(int)
    mfr_wins_fast = defaultdict(int)
    log_wins_fast = defaultdict(int)

    mat_wins_bal  = defaultdict(int)
    mfr_wins_bal  = defaultdict(int)
    log_wins_bal  = defaultdict(int)

    criteria_differ_count = 0
    urgent_differs_count  = 0
    urgent_order_count    = 0
    urgent_feasible_count = 0
    feasible_count        = 0

    zero_mat = 0
    zero_mfr = 0
    zero_log = 0
    zero_chain = 0

    case1_all_stages_empty = 0
    case2_some_stage_empty = 0
    case3_chain_constraint = 0
    case3_over_budget      = 0
    case3_over_time        = 0
    case3_both             = 0

    exactly_one_chain = 0
    two_plus_chains   = 0

    for _, order in orders.iterrows():
        mat_bids = simulate_material_bids(order, materials, products)
        mfr_bids = simulate_manufacturer_bids(order, manufacturers)
        log_bids = simulate_logistics_bids(order, logistics, products)

        log_ids = [b["agent_id"] for b in log_bids]
        if "L07" in log_ids:
            l07_eligible += 1

        for b in mat_bids:
            mat_eligible_agents[b["agent_id"]] += 1
        for b in mfr_bids:
            mfr_eligible_agents[b["agent_id"]] += 1
        for b in log_bids:
            log_eligible_agents[b["agent_id"]] += 1

        mat_eligible_counts.append(len(mat_bids))
        mfr_eligible_counts.append(len(mfr_bids))
        log_eligible_counts.append(len(log_bids))

        if len(mat_bids) == 0:
            zero_mat += 1
            for reason in diagnose_material_reasons(order, materials, products):
                material_reason_counts[order["material_type"]][reason] += 1
        if len(mfr_bids) == 0: zero_mfr += 1
        if len(log_bids) == 0: zero_log += 1

        if len(mat_bids) >= 2:
            top2_mat_gaps.append(pct_gap(mat_bids[0]["offer"], mat_bids[1]["offer"]))
        if len(mfr_bids) >= 2:
            top2_mfr_gaps.append(pct_gap(mfr_bids[0]["offer"], mfr_bids[1]["offer"]))
        if len(log_bids) >= 2:
            gap = pct_gap(log_bids[0]["offer"], log_bids[1]["offer"])
            top2_log_gaps.append(gap)
            if max_log_gap_info is None or gap > max_log_gap_info[0]:
                max_log_gap_info = (gap, order["order_id"], log_bids[0]["agent_id"], log_bids[1]["agent_id"])

        mat_cands = widen_candidates(mat_bids, "completion")
        mfr_cands = widen_candidates(mfr_bids, "completion")
        log_cands = widen_candidates(log_bids, "delivery")

        chains = build_chains(mat_cands, mfr_cands, log_cands, top_n=6)

        # stage checks are not enough
        # the full chain must fit its budget
        # summed stage days must fit the deadline
        try:
            created  = pd.to_datetime(order["order_created"])
            deadline = pd.to_datetime(order["deadline"])
            days_until_deadline = max((deadline - created).days, 1)
        except Exception:
            days_until_deadline = 14

        budget = order["budget"]
        valid_chains = [
            c for c in chains
            if c["total_cost"] <= budget and c["max_days"] <= days_until_deadline
        ]
        chain_counts.append(len(valid_chains))

        if len(valid_chains) == 1:
            exactly_one_chain += 1
        elif len(valid_chains) >= 2:
            two_plus_chains += 1

        case_label = "feasible"

        if len(valid_chains) == 0:
            stages_empty = sum(1 for n in (len(mat_bids), len(mfr_bids), len(log_bids)) if n == 0)
            if stages_empty == 3:
                case1_all_stages_empty += 1
                case_label = "case1"
            elif stages_empty >= 1:
                case2_some_stage_empty += 1
                case_label = "case2"
            else:
                case3_chain_constraint += 1
                case_label = "case3"
                over_b = all(c["total_cost"] > budget for c in chains)
                over_t = all(c["max_days"] > days_until_deadline for c in chains)
                if over_b and not over_t:
                    case3_over_budget += 1
                elif over_t and not over_b:
                    case3_over_time += 1
                else:
                    case3_both += 1

        feasibility_records.append({
            "scenario_type": order["scenario_type"],
            "material_type": order["material_type"],
            "feasible": len(valid_chains) > 0,
            "case_type": case_label,
        })

        if order["priority"] == "urgent":
            urgent_order_count += 1

        if len(valid_chains) == 0:
            zero_chain += 1
            continue

        feasible_count += 1
        lowest, fastest, balanced = select_chains(valid_chains)

        if lowest:
            mat_wins[lowest["material"]] += 1
            mfr_wins[lowest["manufacturer"]] += 1
            log_wins[lowest["logistics"]] += 1

            if "L07" in log_ids and lowest["logistics"] == "L12":
                l07_lost_to_l12 += 1

        if fastest:
            mat_wins_fast[fastest["material"]] += 1
            mfr_wins_fast[fastest["manufacturer"]] += 1
            log_wins_fast[fastest["logistics"]] += 1

        if balanced:
            mat_wins_bal[balanced["material"]] += 1
            mfr_wins_bal[balanced["manufacturer"]] += 1
            log_wins_bal[balanced["logistics"]] += 1

            win_records.append({
                "scenario": order["scenario_type"],
                "low_mat": lowest["material"], "low_log": lowest["logistics"],
                "fast_mat": fastest["material"], "fast_log": fastest["logistics"],
                "bal_mat": balanced["material"], "bal_log": balanced["logistics"],
            })

        if lowest and fastest and balanced:
            unique_chains = len({
                (lowest["material"], lowest["manufacturer"], lowest["logistics"]),
                (fastest["material"], fastest["manufacturer"], fastest["logistics"]),
                (balanced["material"], balanced["manufacturer"], balanced["logistics"]),
            })
            if unique_chains > 1:
                criteria_differ_count += 1

        if order["priority"] == "urgent":
            if lowest and fastest:
                urgent_feasible_count += 1
                if (lowest["material"], lowest["manufacturer"], lowest["logistics"]) != \
                   (fastest["material"], fastest["manufacturer"], fastest["logistics"]):
                    urgent_differs_count += 1

    total = len(orders)

    section("ELIGIBILITY")
    print(f"orders with zero eligible material bidders:      {zero_mat} / {total} ({zero_mat/total*100:.1f}%)")
    print(f"orders with zero eligible manufacturer bidders:  {zero_mfr} / {total} ({zero_mfr/total*100:.1f}%)")
    print(f"orders with zero eligible logistics bidders:     {zero_log} / {total} ({zero_log/total*100:.1f}%)")
    print(f"orders with zero feasible chains:                {zero_chain} / {total} ({zero_chain/total*100:.1f}%)")
    print()
    mat_2plus = sum(1 for x in mat_eligible_counts if x >= 2)
    mfr_2plus = sum(1 for x in mfr_eligible_counts if x >= 2)
    log_2plus = sum(1 for x in log_eligible_counts if x >= 2)
    print(f"orders with 2+ eligible material bidders:    {mat_2plus} / {total} ({mat_2plus/total*100:.1f}%)")
    print(f"orders with 2+ eligible manufacturer bidders: {mfr_2plus} / {total} ({mfr_2plus/total*100:.1f}%)")
    print(f"orders with 2+ eligible logistics bidders:   {log_2plus} / {total} ({log_2plus/total*100:.1f}%)")

    section("ELIGIBLE BIDDER COUNTS")
    print(f"material   - mean: {sum(mat_eligible_counts)/len(mat_eligible_counts):.1f}  min: {min(mat_eligible_counts)}  max: {max(mat_eligible_counts)}")
    print(f"manufacturer - mean: {sum(mfr_eligible_counts)/len(mfr_eligible_counts):.1f}  min: {min(mfr_eligible_counts)}  max: {max(mfr_eligible_counts)}")
    print(f"logistics  - mean: {sum(log_eligible_counts)/len(log_eligible_counts):.1f}  min: {min(log_eligible_counts)}  max: {max(log_eligible_counts)}")

    section("ELIGIBILITY VS WINS PER AGENT (lowest cost)")
    print("material, eligible count -> wins -> win rate when eligible:")
    for agent in sorted(mat_eligible_agents, key=lambda a: -mat_eligible_agents[a]):
        elig = mat_eligible_agents[agent]
        wins = mat_wins.get(agent, 0)
        rate = wins / elig * 100 if elig else 0
        print(f"  {agent}: eligible {elig}, wins {wins}, win rate {rate:.1f}%")

    print("\nmanufacturer, eligible count -> wins -> win rate when eligible:")
    for agent in sorted(mfr_eligible_agents, key=lambda a: -mfr_eligible_agents[a]):
        elig = mfr_eligible_agents[agent]
        wins = mfr_wins.get(agent, 0)
        rate = wins / elig * 100 if elig else 0
        print(f"  {agent}: eligible {elig}, wins {wins}, win rate {rate:.1f}%")

    print("\nlogistics, eligible count -> wins -> win rate when eligible:")
    for agent in sorted(log_eligible_agents, key=lambda a: -log_eligible_agents[a]):
        elig = log_eligible_agents[agent]
        wins = log_wins.get(agent, 0)
        rate = wins / elig * 100 if elig else 0
        print(f"  {agent}: eligible {elig}, wins {wins}, win rate {rate:.1f}%")

    section("FEASIBLE CHAINS (budget and deadline constrained)")
    print(f"mean chains per order:   {sum(chain_counts)/len(chain_counts):.1f}")
    print(f"min chains:              {min(chain_counts)}")
    print(f"max chains:              {max(chain_counts)}")
    chain_3plus = sum(1 for x in chain_counts if x >= 3)
    print(f"orders with exactly 1 feasible chain:  {exactly_one_chain} / {total} ({exactly_one_chain/total*100:.1f}%)")
    print(f"orders with 2+ feasible chains:        {two_plus_chains} / {total} ({two_plus_chains/total*100:.1f}%)")
    print(f"orders with 3+ feasible chains:        {chain_3plus} / {total} ({chain_3plus/total*100:.1f}%)")

    section("INFEASIBILITY CASE BREAKDOWN (three cases)")
    total_infeasible = case1_all_stages_empty + case2_some_stage_empty + case3_chain_constraint
    print(f"total infeasible orders: {total_infeasible} / {total} ({total_infeasible/total*100:.1f}%)")
    print(f"  case 1, no bidder at any stage:            {case1_all_stages_empty}  (target ~300-500)")
    print(f"  case 2, at least one stage with no bidder: {case2_some_stage_empty}  (target ~1000-1500)")
    print(f"  case 3, all stages bid, chain fails:       {case3_chain_constraint}  (target: remainder)")
    if case3_chain_constraint > 0:
        print(f"    case 3 over budget only:   {case3_over_budget}")
        print(f"    case 3 over deadline only: {case3_over_time}")
        print(f"    case 3 mixed or both:      {case3_both}")

    section("BID PRICE CLOSENESS (top-2 gap %)")
    for label, gaps in [("material", top2_mat_gaps), ("manufacturer", top2_mfr_gaps), ("logistics", top2_log_gaps)]:
        if not gaps:
            continue
        s = pd.Series(gaps)
        q = s.quantile([0.5, 0.75, 0.9, 0.95])
        print(f"{label:12s} - mean: {s.mean():.1f}%  median: {q[0.5]:.1f}%  "
              f"p75: {q[0.75]:.1f}%  p90: {q[0.9]:.1f}%  p95: {q[0.95]:.1f}%  max: {s.max():.1f}%")

    if max_log_gap_info:
        gap, oid, a1, a2 = max_log_gap_info
        print(f"  max gap order: {oid}, agents: {a1} vs {a2}")

    section("MATERIAL INFEASIBILITY REASONS (why zero bidders)")
    for material, reasons in material_reason_counts.items():
        print(f"\n{material}:")
        total_reasons = sum(reasons.values())
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} ({count/total_reasons*100:.1f}%)")

    section("WIN SHARE - MATERIAL AGENTS (lowest cost chain)")
    total_mat_wins = sum(mat_wins.values())
    if total_mat_wins > 0:
        for agent, wins in sorted(mat_wins.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_mat_wins*100:.1f}%)")

    section("WIN SHARE - MANUFACTURER AGENTS (lowest cost chain)")
    total_mfr_wins = sum(mfr_wins.values())
    if total_mfr_wins > 0:
        for agent, wins in sorted(mfr_wins.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_mfr_wins*100:.1f}%)")

    section("WIN SHARE - LOGISTICS AGENTS (lowest cost chain)")
    total_log_wins = sum(log_wins.values())
    if total_log_wins > 0:
        for agent, wins in sorted(log_wins.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_log_wins*100:.1f}%)")

    section("WIN SHARE - MATERIAL AGENTS (fastest chain)")
    total_mat_fast = sum(mat_wins_fast.values())
    if total_mat_fast > 0:
        for agent, wins in sorted(mat_wins_fast.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_mat_fast*100:.1f}%)")

    section("WIN SHARE - MANUFACTURER AGENTS (fastest chain)")
    total_mfr_fast = sum(mfr_wins_fast.values())
    if total_mfr_fast > 0:
        for agent, wins in sorted(mfr_wins_fast.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_mfr_fast*100:.1f}%)")

    section("WIN SHARE - LOGISTICS AGENTS (fastest chain)")
    total_log_fast = sum(log_wins_fast.values())
    if total_log_fast > 0:
        for agent, wins in sorted(log_wins_fast.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_log_fast*100:.1f}%)")

    section("WIN SHARE - MATERIAL AGENTS (balanced chain)")
    total_mat_bal = sum(mat_wins_bal.values())
    if total_mat_bal > 0:
        for agent, wins in sorted(mat_wins_bal.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_mat_bal*100:.1f}%)")

    section("WIN SHARE - MANUFACTURER AGENTS (balanced chain)")
    total_mfr_bal = sum(mfr_wins_bal.values())
    if total_mfr_bal > 0:
        for agent, wins in sorted(mfr_wins_bal.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_mfr_bal*100:.1f}%)")

    section("WIN SHARE - LOGISTICS AGENTS (balanced chain)")
    total_log_bal = sum(log_wins_bal.values())
    if total_log_bal > 0:
        for agent, wins in sorted(log_wins_bal.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {wins} wins ({wins/total_log_bal*100:.1f}%)")

    section("CHAIN SELECTION DIVERSITY")
    if feasible_count > 0:
        print(f"orders where criteria differ (of feasible orders): {criteria_differ_count} / {feasible_count} ({criteria_differ_count/feasible_count*100:.1f}%)")
    print(f"orders where criteria differ (of all orders):      {criteria_differ_count} / {total} ({criteria_differ_count/total*100:.1f}%)")
    if urgent_order_count > 0:
        print(f"urgent orders total: {urgent_order_count}  |  urgent with feasible chains: {urgent_feasible_count}")
        if urgent_feasible_count > 0:
            print(f"urgent feasible orders where fastest differs from lowest-cost: {urgent_differs_count} / {urgent_feasible_count} ({urgent_differs_count/urgent_feasible_count*100:.1f}%)")

    section("FEASIBILITY BY SCENARIO AND MATERIAL")
    feas_df = pd.DataFrame(feasibility_records)
    grouped = feas_df.groupby("scenario_type")["feasible"].agg(["sum", "count"])
    grouped["pct"] = (grouped["sum"] / grouped["count"] * 100).round(1)
    print("\nby scenario_type:")
    print(grouped.to_string())

    grouped_mat = feas_df.groupby("material_type")["feasible"].agg(["sum", "count"])
    grouped_mat["pct"] = (grouped_mat["sum"] / grouped_mat["count"] * 100).round(1)
    print("\nby material_type:")
    print(grouped_mat.to_string())

    section("SCENARIO VS CASE TYPE CROSS TAB")
    cross = pd.crosstab(feas_df["scenario_type"], feas_df["case_type"])
    print(cross.to_string())

    section("WIN SHARE BY SCENARIO (top 3)")
    wdf = pd.DataFrame(win_records)
    for col in ["low_mat", "low_log", "fast_mat", "fast_log", "bal_mat", "bal_log"]:
        print(f"\n{col}:")
        print(wdf.groupby("scenario")[col].value_counts().groupby(level=0).head(3).to_string())

    section("SUMMARY")
    max_mat_share = max(v/total_mat_wins*100 for v in mat_wins.values()) if mat_wins else 0
    max_mfr_share = max(v/total_mfr_wins*100 for v in mfr_wins.values()) if mfr_wins else 0
    max_log_share = max(v/total_log_wins*100 for v in log_wins.values()) if log_wins else 0

    max_mat_fast = max(v/total_mat_fast*100 for v in mat_wins_fast.values()) if mat_wins_fast else 0
    max_mfr_fast = max(v/total_mfr_fast*100 for v in mfr_wins_fast.values()) if mfr_wins_fast else 0
    max_log_fast = max(v/total_log_fast*100 for v in log_wins_fast.values()) if log_wins_fast else 0

    max_mat_bal = max(v/total_mat_bal*100 for v in mat_wins_bal.values()) if mat_wins_bal else 0
    max_mfr_bal = max(v/total_mfr_bal*100 for v in mfr_wins_bal.values()) if mfr_wins_bal else 0
    max_log_bal = max(v/total_log_bal*100 for v in log_wins_bal.values()) if log_wins_bal else 0

    print("lowest-cost chain:")
    print(f"  material max:      {max_mat_share:.1f}%  {'good' if max_mat_share < 30 else 'WARNING'}")
    print(f"  manufacturer max:  {max_mfr_share:.1f}%  {'good' if max_mfr_share < 30 else 'WARNING'}")
    print(f"  logistics max:     {max_log_share:.1f}%  {'good' if max_log_share < 30 else 'WARNING'}")

    print("fastest chain (threshold 55%, speed specialists naturally concentrate here):")
    print(f"  material max:      {max_mat_fast:.1f}%  {'good' if max_mat_fast < 55 else 'WARNING'}")
    print(f"  manufacturer max:  {max_mfr_fast:.1f}%  {'good' if max_mfr_fast < 55 else 'WARNING'}")
    print(f"  logistics max:     {max_log_fast:.1f}%  {'good' if max_log_fast < 55 else 'WARNING'}")

    print("balanced chain (threshold 35%):")
    print(f"  material max:      {max_mat_bal:.1f}%  {'good' if max_mat_bal < 35 else 'WARNING'}")
    print(f"  manufacturer max:  {max_mfr_bal:.1f}%  {'good' if max_mfr_bal < 35 else 'WARNING'}")
    print(f"  logistics max:     {max_log_bal:.1f}%  {'good' if max_log_bal < 35 else 'WARNING'}")
    print()

    print(f"orders with 2+ material bidders:      {mat_2plus/total*100:.1f}%  {'good' if mat_2plus/total > 0.85 else 'WARNING'}")
    print(f"orders with 2+ manufacturer bidders:  {mfr_2plus/total*100:.1f}%  {'good' if mfr_2plus/total > 0.85 else 'WARNING'}")
    print(f"orders with 2+ logistics bidders:     {log_2plus/total*100:.1f}%  {'good' if log_2plus/total > 0.85 else 'WARNING'}")
    diversity_pct = criteria_differ_count / feasible_count if feasible_count > 0 else 0
    print(f"chain diversity (feasible orders):    {diversity_pct*100:.1f}%  {'good' if diversity_pct > 0.30 else 'WARNING'}")

    print(f"\nL07 eligible: {l07_eligible}, lost to L12 on lowest cost: {l07_lost_to_l12}")


if __name__ == "__main__":
    main()
