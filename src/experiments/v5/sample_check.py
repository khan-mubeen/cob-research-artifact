"""phase 3: a sample-scale run of the v5 generators and its sanity gate

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.sample_check

the sample is every scenario_type x priority stratum of the v5 orders, each stratum's share of
133 orders with at least 8 from each, which lifts the small strata and gives 143 orders, taken in
the generators' own walk order. that yields low
thousands of rows across the three stages, both modes and the supervisor, reaches every agent
and every stratum, and runs in well under a minute, which is the point: a gate cheap enough to
run before 50,000 orders are generated

the rows come from the very build_rows functions the full generators write from, run twice in
memory and never written as corpus files. the checks:
  usd          every v5 prompt carries USD and none carries v4-only wording; every auction field is USD
  formula      each stage's TASK block holds the v5 price formula, and v4 prompts built from the
               v4 corpus for the same agents carry no v5 wording
  prices       every BID offer is re-derived independently from the stored v5 csv files, without
               calling src/common/v5 or bidding_formulas: material and manufacturer exactly, logistics
               against the full-precision v5 RouteCost, reporting the largest difference
  determinism  a second run in the same process gives byte-identical rows
  clock        the first run executes under sys.setprofile, which records every call to a system
               clock function; there must be none
  edge, floor  the edge cases and the floor-boundary rows are built for the sample and pass their
               own assertions

writes reports/v5/phase3_sample_check.json and exits non-zero if any check fails
"""

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime

import pandas as pd

from src.common import bidding_formulas, slm
from src.common.reverse_auction import agent_minimum_bid
from src.data_synthesis.v5.logistics import RouteCost
from src.experiments.v5 import generate_edge_examples as edge
from src.experiments.v5 import generate_finetune_data as sealed
from src.experiments.v5 import generate_floor_boundary as floor
from src.experiments.v5 import generate_reverse_auction_data as auction
from src.experiments.v5 import generate_supervisor_data as supervisor
from src.experiments.v5.common import STAGES, Market, load_corpus, rel, require_v5, stage_rfqs, walk_orders, write_json
from src.experiments.v5.split import assign
from config import PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "v5" / "phase3_sample_check.json"
SAMPLE_ORDERS = 133
MIN_PER_STRATUM = 8

V4_ONLY_WORDING = ("base cost per kg (EUR)", "unit cost for requested product (EUR)", "base delivery cost",
                   "cost per kg (EUR)", "current_best_bid (EUR)", "platform_minimum_decrement (EUR)",
                   "your minimum_bid (EUR)", "budget (EUR)", "cost_eur=", "total_cost_eur")
V5_ONLY_WORDING = ("USD", "pricing date", "reference price", "fixed charge", "fuel surcharge", "EUR to USD",
                   "pickup from", "route distance")
FORMULA_LINES = {
    "material": "material_cost  = required_kg x price_ratio_to_reference x reference_price_usd_per_kg",
    "tailoring": "base_cost      = quantity x unit_cost_for_product",
    "logistics": "freight_eur    = fixed_charge + alpha x shipment_weight_kg + beta x shipment_weight_kg / 1000 x route_distance_km",
}
CLOCK_OWNERS = (datetime, date, time)
CLOCK_NAMES = {"now", "utcnow", "today", "time", "time_ns", "localtime", "gmtime", "monotonic"}


def sample(corpus):
    walk = walk_orders(corpus["orders"])
    frame = pd.DataFrame(corpus["orders"])
    shares = frame.groupby(["scenario_type", "priority"]).size() / len(frame)
    quota = {key: max(MIN_PER_STRATUM, round(SAMPLE_ORDERS * share)) for key, share in shares.items()}
    taken, picked = Counter(), []
    for order in walk:
        key = (order["scenario_type"], order["priority"])
        if taken[key] < quota[key]:
            taken[key] += 1
            picked.append(order)
    return picked, {f"{k[0]}|{k[1]}": v for k, v in sorted(taken.items())}


def run_generators(context, corpus, market, split_of, walk):
    rows = {"sealed": [], "reverse_auction": [], "supervisor": []}
    rows["sealed"] = [row for _, row in sealed.build_rows(context, corpus, market, split_of, walk, Counter())]
    tally = auction.new_tally()
    rows["reverse_auction"] = [row for _, row in auction.build_rows(
        context, corpus, market, split_of, walk, tally, auction.floor_pairs_per_stage(len(walk)))]
    rows["supervisor"] = [row for _, row in supervisor.build_rows(context, corpus, market, split_of, walk, Counter())]
    return rows, tally


def digest(rows):
    return {mode: hashlib.sha256("\n".join(json.dumps(r, ensure_ascii=False) for r in items).encode()).hexdigest()
            for mode, items in rows.items()}


class ClockTrap:
    def __init__(self):
        self.calls = Counter()

    def __call__(self, frame, event, arg):
        # matched by name, then by identity of the owner, never by hashing the callable: a builtin
        # bound to a dict or list is unhashable
        if event == "c_call" and getattr(arg, "__name__", "") in CLOCK_NAMES:
            owner = getattr(arg, "__self__", None)
            if any(owner is clock for clock in CLOCK_OWNERS):
                self.calls[f"{getattr(owner, '__name__', '')}.{arg.__name__}"] += 1


def independent_offer(stage, agent, order, rfq, csv_frames, route):
    """a BID offer from the stored v5 csv files, written without src/common/v5 or bidding_formulas"""
    markup = {"urgent": 0.30, "normal": 0.00, "flexible": -0.15}[order["priority"]]
    if stage == "material":
        material = order["material_type"]
        reference = round(csv_frames["reference"][(material, order["pricing_date"])], 4)
        kg = round(order["quantity"] * csv_frames["products"][(order["product_id"], material)][0], 4)
        internal = kg * agent[f"{material}_price_ratio_to_reference"] * reference * (1 + agent["defect_probability"])
        return round(max(internal * (1 + agent["min_profit_margin"]),
                         internal * (1 + agent["initial_markup"]) * (1 + markup)), 2), None
    if stage == "tailoring":
        quantity, capacity = order["quantity"], agent["daily_capacity_units"]
        days = rfq["days_until_deadline"]
        ratio = (agent["current_workload_units"] + quantity) / (capacity * days)
        workload = 0.0 if ratio < 0.5 else 0.05 if ratio < 0.75 else 0.10 if ratio < 0.9 else 0.15
        internal = quantity * agent[f"unit_cost_usd_{order['product_id']}"] * (1 + agent["defect_probability"])
        return round(max(internal * (1 + agent["min_profit_margin"]),
                         internal * (1 + agent["initial_markup"]) * (1 + workload) * (1 + markup)), 2), None
    weight = round(order["quantity"] * csv_frames["products"][(order["product_id"], order["material_type"])][1], 4)
    day = datetime.strptime(order["pricing_date"], "%Y-%m-%d").date()
    exact = route.price_for_distance(agent["agent_id"], weight, route.distance_km[(rfq["pickup_from"], order["deliver_to"])],
                                     day, order["priority"])
    return None, exact


def main():
    context = require_v5()
    corpus = load_corpus()
    market = Market()
    walk, strata = sample(corpus)
    split_of, _, _ = assign(pd.DataFrame(corpus["orders"])[["order_id", "scenario_type", "priority"]])
    checks, details = {}, {"sample_orders": len(walk), "orders_per_stratum": strata}

    trap = ClockTrap()
    sys.setprofile(trap)
    try:
        first, tally = run_generators(context, corpus, market, split_of, walk)
    finally:
        sys.setprofile(None)
    second, _ = run_generators(context, corpus, market, split_of, walk)
    details["rows"] = {mode: len(items) for mode, items in first.items()}
    details["rows_by_stage"] = {mode: dict(Counter(r["stage"] for r in items)) for mode, items in first.items()}
    details["class_balance"] = {mode: dict(Counter(r.get("rule_decision", "SELECT") for r in items))
                                for mode, items in first.items()}
    details["digest_first_run"], details["digest_second_run"] = digest(first), digest(second)
    checks["deterministic_across_two_runs"] = details["digest_first_run"] == details["digest_second_run"]
    details["clock_calls_during_generation"] = dict(trap.calls)
    checks["no_system_clock_read_during_generation"] = not trap.calls

    all_rows = [r for items in first.values() for r in items]
    checks["every_prompt_carries_usd"] = all("USD" in r["user"] for r in all_rows)
    leaks = Counter(w for r in all_rows for w in V4_ONLY_WORDING if w in r["user"])
    details["v4_only_wording_found"] = dict(leaks)
    checks["no_v4_only_wording_in_v5_prompts"] = not leaks
    auction_lines = [line for r in first["reverse_auction"] for line in r["user"].splitlines()
                     if line.startswith(("current_best_bid", "platform_minimum_decrement", "your minimum_bid"))]
    checks["every_auction_field_is_usd"] = bool(auction_lines) and all("(USD)" in line for line in auction_lines)
    checks["every_stage_prompt_holds_the_v5_formula"] = all(
        FORMULA_LINES[r["stage"]] in r["user"].replace("multiplied by (x)", "x").replace("plus (+)", "+")
        .replace("divided by (/)", "/") for r in first["sealed"] + first["reverse_auction"])
    checks["every_row_is_labeled_v5"] = all(r["dataset_version"] == "v5" for r in all_rows)

    from tests.golden.v4_baseline import load_v4
    from tests.golden.v4_baseline import stage_rfqs as v4_stage_rfqs
    v4 = load_v4()
    v5_leaks = Counter()
    for order in v4["orders"][:40]:
        rfqs = v4_stage_rfqs(order, v4["products"])
        for stage, build in (("material", slm.build_material_prompt), ("tailoring", slm.build_manufacturer_prompt),
                             ("logistics", slm.build_logistics_prompt)):
            for agent in v4[stage]:
                if bidding_formulas.eligibility_reason(stage, agent, rfqs[stage]) is None:
                    v5_leaks.update(w for w in V5_ONLY_WORDING if w in build(agent, rfqs[stage])[1])
    details["v5_wording_found_in_v4_prompts"] = dict(v5_leaks)
    checks["no_v5_wording_in_v4_prompts"] = not v5_leaks

    csv_frames = {"reference": market.reference,
                  "products": {k: (v["material_kg_per_unit"], v["finished_weight_kg_per_unit"])
                               for k, v in corpus["products"].items()}}
    route = RouteCost()
    agents = {stage: {a["agent_id"]: a for a in corpus[stage]} for stage in STAGES}
    orders = {o["order_id"]: o for o in corpus["orders"]}
    rfq_cache = {}
    mismatches, logistics_worst, compared = [], 0.0, Counter()
    for r in first["sealed"] + [x for x in first["reverse_auction"] if x["round_number"] == 1]:
        label = json.loads(r["completion"])
        if label["decision"] != "BID":
            continue
        order = orders[r["order_id"]]
        if r["order_id"] not in rfq_cache:
            rfq_cache[r["order_id"]] = stage_rfqs(order, corpus, market, context)
        rfq = rfq_cache[r["order_id"]][r["stage"]]
        offer, exact = independent_offer(r["stage"], agents[r["stage"]][r["agent_id"]], order, rfq, csv_frames, route)
        compared[r["stage"]] += 1
        if offer is not None and offer != label["offer_amount"]:
            mismatches.append((r["order_id"], r["agent_id"], r["stage"], offer, label["offer_amount"]))
        if exact is not None:
            logistics_worst = max(logistics_worst, abs(round(exact, 2) - label["offer_amount"]))
    details["independent_price_checks"] = {"bid_offers_compared": dict(compared), "exact_mismatches": mismatches[:10],
                                           "logistics_largest_difference_vs_full_precision_route_cost_usd":
                                               round(logistics_worst, 4)}
    checks["material_and_manufacturer_offers_re_derive_exactly"] = not mismatches
    checks["logistics_offers_within_one_cent_of_full_precision_route_cost"] = logistics_worst <= 0.01 + 1e-9

    decided = [r for r in first["reverse_auction"] if r["stage"] == "logistics" and r["round_number"] == 2
               and r["category"] == "auction_round"]
    details["logistics_round_2_rows"] = dict(Counter(r["rule_decision"] for r in decided))
    details["replay_lengths_by_stage"] = {k: dict(v) for k, v in tally["replay_lengths"].items()}

    edge_cases = (edge.material_cases(corpus["material"])
                  + edge.tailoring_cases(pd.DataFrame(corpus["tailoring"]))
                  + edge.logistics_cases(pd.DataFrame(corpus["logistics"]), set()))
    maker_locations = sorted(set(corpus["manufacturer_location"].values()))
    edge_labels = Counter()
    for index, case in enumerate(edge_cases):
        order = edge.v5_fields(case, index, market, context, maker_locations)
        agent = case["agent"]
        assert bidding_formulas.eligibility_reason(case["stage"], agent, order) is None, order["order_id"]
        rule = auction.SEALED_DECIDERS[case["stage"]](agent, order)
        expected, reason = edge.EXPECTED[case["category"]]
        assert rule["decision"] == expected and (reason is None or rule.get("debug_reason") == reason), (order, rule)
        edge_labels[(case["stage"], case["category"], rule["decision"])] += 1
    details["edge_cases_built"] = {f"{s}:{c}:{d}": n for (s, c, d), n in sorted(edge_labels.items())}
    checks["edge_cases_pass_their_label_assertions"] = bool(edge_labels)

    floor_rows = 0
    for stage in STAGES:
        pairs = 0
        for order in walk:
            if pairs == 5:
                break
            rfq = (stage_rfqs(order, corpus, market, context) or {}).get(stage)
            if rfq is None:
                continue
            opener = next((a for a in corpus[stage] if bidding_formulas.eligibility_reason(stage, a, rfq) is None
                           and auction.SEALED_DECIDERS[stage](a, rfq)["decision"] == "BID"), None)
            if opener is None:
                continue
            pairs += 1
            numbered = {**rfq, "_round_number": 2 + (pairs % 2)}
            for case, best in auction.floor_cases(stage, opener, rfq, 5.0):
                row = floor.checked_row(stage, opener, numbered, case, best, context)
                assert row["displayed_minimum_bid"] == agent_minimum_bid(stage, opener, rfq)
                floor_rows += 1
    details["floor_boundary_rows_built"] = floor_rows
    checks["floor_boundary_rows_pass_the_shared_floor_checks"] = floor_rows == 45

    details["checks"] = checks
    details["passed"] = all(checks.values())
    details["examples"] = {mode: {s: next((r["user"] for r in items if r["stage"] == s), None)
                                  for s in {r["stage"] for r in items}} for mode, items in first.items()}
    write_json(REPORT_PATH, details)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    print(f"rows {details['rows']}, logistics round-2 rows {details['logistics_round_2_rows']}, "
          f"largest logistics difference {details['independent_price_checks']['logistics_largest_difference_vs_full_precision_route_cost_usd']}")
    print(f"wrote {rel(REPORT_PATH)}")
    return 0 if details["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
