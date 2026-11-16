"""v5 edge examples, role robustness_eval, the v4 edge methodology rebuilt on the v5 profiles

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_edge_examples

these rows are an evaluation set for robustness at the dynamic-feasibility boundary. they are
not training data and are not part of any train, validation or test count

src/experiments/generate_edge_examples.py is the source of truth and is not edited: every case
is built from a real agent profile, verified against the rule before it is labeled, labeled with
the category's expected decision asserted, checked for no parameter overlap with the fixed
baseline cases, within itself, and with every real train, validation and test row, and
serialized through completion_json. its tailoring_cases and logistics_cases are imported and run
unchanged. what differs, because v5 differs:
  rfqs       each case carries the v5 context and dated market values for a pricing date cycled
             deterministically across the v5 window, and a logistics case picks up at a
             manufacturer location cycled the same way
  weights    a v5 logistics agent is eligible only inside its evidenced weight range, so v4's
             200-1500 kg draw is moved to 80% of a capped agent's limit when it falls outside;
             the categories depend only on route days, so no category changes
  material   v4 builds no material edge set. the same construction is applied to the material
             dynamic check, so the three stages share one method: per agent and supplied
             material, a restock just too slow for the deadline (NO_BID), the same restock
             exactly at the deadline (BID), stock and restock together short of the order
             (NO_BID) and an in-stock order due exactly at default completion (BID)
  holdout    one file per stage, no train/holdout split, because the whole set is evaluation

output: data/finetune/v5/edge/{material,tailoring,logistics}_edge.jsonl and metadata.json
"""

import math
import re
import sys
from collections import Counter

import pandas as pd

from src.common.bidding_formulas import eligibility_reason
from src.common.reverse_auction import SEALED_DECIDERS
from src.common.v5 import pricing
from src.data_synthesis.v5.settings import WINDOW_DAYS
from src.experiments.generate_edge_examples import ORDER_BASE, _param_key, logistics_cases, tailoring_cases
from src.experiments.generate_finetune_data import completion_json
from src.experiments.generate_reverse_auction_data import STAGE_PROMPTS
from src.experiments.nobid_edge_baseline import CASES as BASELINE_CASES
from src.experiments.v5.common import (
    FINETUNE_V5, SPLITS, STAGE_DEFECT_KEY, JsonlFiles, Market, load_corpus, read_jsonl, rel, require_v5, write_json,
)

OUT_DIR = FINETUNE_V5 / "edge"
ROLE = "robustness_eval"
STAGES = ("material", "tailoring", "logistics")
CAPPED_WEIGHT_SHARE = 0.8
EXPECTED = {
    "infeasible": ("NO_BID", "completion_exceeds_deadline"),
    "boundary_infeasible": ("NO_BID", "completion_exceeds_deadline"),
    "at_boundary": ("BID", None),
    "near_boundary": ("BID", None),
    "restock_too_slow": ("NO_BID", "completion_exceeds_deadline"),
    "restock_at_boundary": ("BID", None),
    "stock_and_restock_short": ("NO_BID", "insufficient_stock_and_restock"),
    "in_stock_at_boundary": ("BID", None),
}


def material_cases(materials):
    cases = []
    for agent in materials:
        for material in sorted(str(agent["material_types"]).split("|")):
            available = agent[f"{material}_available_kg"]
            restock = agent[f"{material}_restock_amount_kg"]
            restock_days = agent[f"{material}_restock_time_days"]
            default = agent["default_completion_days"]
            wanted = []
            if not pricing._missing(restock) and restock > 1:
                required = available + max(1.0, math.floor(restock / 2))
                completion = default + restock_days
                if math.ceil(completion) - 1 >= 1:
                    wanted.append(("restock_too_slow", required, math.ceil(completion) - 1))
                wanted.append(("restock_at_boundary", required, math.ceil(completion)))
                wanted.append(("stock_and_restock_short", available + restock + 1.0, math.ceil(completion) + 5))
            if available >= 2:
                wanted.append(("in_stock_at_boundary", float(math.floor(available / 2)), math.ceil(default)))
            for category, required, days in wanted:
                cases.append({"stage": "material", "category": category, "agent": agent, "order": {
                    **ORDER_BASE,
                    "order_id": f"SYNEDGE5-M-{category.upper()[:7]}-{agent['agent_id']}-{material}",
                    "material_type": material, "required_kg": round(required, 4), "days_until_deadline": days}})
    return cases


def material_key(agent_id, order):
    return ("material", agent_id, order["material_type"], order["required_kg"], order["days_until_deadline"])


def key_of(case):
    if case["stage"] == "material":
        return material_key(case["agent"]["agent_id"], case["order"])
    return _param_key(case["stage"], case["agent"]["agent_id"], case["order"])


def real_param_keys():
    """parameter tuples of every real v5 train, validation and test row, read from the prompts"""
    keys = set()
    patterns = {
        "material": (re.compile(r"^material needed: (\S+)$", re.M), re.compile(r"^required quantity \(kg\): (\S+)$", re.M)),
        "tailoring": (re.compile(r"^product: (\S+)$", re.M), re.compile(r"^quantity: (\d+)$", re.M)),
        "logistics": (re.compile(r"^deliver to: (\S+)$", re.M), None),
    }
    days_re = re.compile(r"^days until deadline: (\d+)$", re.M)
    for stage in STAGES:
        first, second = patterns[stage]
        for split in SPLITS:
            for row in read_jsonl(FINETUNE_V5 / f"{stage}_{split}.jsonl"):
                user = row["user"]
                days = int(days_re.search(user).group(1))
                if stage == "material":
                    keys.add((stage, row["agent_id"], first.search(user).group(1), float(second.search(user).group(1)), days))
                elif stage == "tailoring":
                    keys.add((stage, row["agent_id"], first.search(user).group(1), int(second.search(user).group(1)), days))
                else:
                    keys.add((stage, row["agent_id"], first.search(user).group(1), days))
    return keys


def v5_fields(case, index, market, context, maker_locations):
    """the v5 context and dated market values, for a pricing date and pickup cycled by case index"""
    day = WINDOW_DAYS[(index * 37) % len(WINDOW_DAYS)]
    order = {**case["order"], **context.as_dict(), "pricing_date": day.isoformat()}
    if case["stage"] == "material":
        order["reference_price_usd_per_kg"] = market.reference_usd_per_kg(order["material_type"], order["pricing_date"])
    elif case["stage"] == "logistics":
        agent = case["agent"]
        if not pricing.weight_inside_domain(agent, order["shipment_weight_kg"]):
            _, cap, _ = pricing.weight_domain(agent)
            order["shipment_weight_kg"] = round(cap * CAPPED_WEIGHT_SHARE, 1)
        origin = maker_locations[index % len(maker_locations)]
        order.update({"pickup_from": origin, "route_distance_km": market.distance_km(origin, order["deliver_to"]),
                      "fuel_surcharge_percent": market.fuel_surcharge_percent(day),
                      "eur_to_usd": market.eur_to_usd(day)})
    return order


def main():
    context = require_v5()
    corpus = load_corpus()
    market = Market()
    real_keys = real_param_keys()
    baseline_keys = {_param_key(c["stage"], c["agent_id"], c["order"]) for c in BASELINE_CASES}
    maker_locations = sorted(set(corpus["manufacturer_location"].values()))

    cases = (material_cases(corpus["material"])
             + tailoring_cases(pd.DataFrame(corpus["tailoring"]))
             + logistics_cases(pd.DataFrame(corpus["logistics"]), real_keys))
    seen, counts, labels, skipped = set(), Counter(), Counter(), Counter()
    paths = {stage: OUT_DIR / f"{stage}_edge.jsonl" for stage in STAGES}
    with JsonlFiles(paths) as files:
        for index, case in enumerate(cases):
            stage, agent, category = case["stage"], case["agent"], case["category"]
            order = v5_fields(case, index, market, context, maker_locations)
            case = {**case, "order": order}
            if eligibility_reason(stage, agent, order) is not None:
                raise AssertionError(f"{order['order_id']}: statically ineligible under v5")
            rule = SEALED_DECIDERS[stage](agent, order)
            decision, reason = EXPECTED[category]
            if rule["decision"] != decision or (reason is not None and rule.get("debug_reason") != reason):
                raise AssertionError(f"{order['order_id']}: expected {decision} {reason or ''}, got {rule}")
            key = key_of(case)
            if key in baseline_keys:
                raise AssertionError(f"{order['order_id']}: overlaps a fixed baseline case")
            if key in seen:
                raise AssertionError(f"{order['order_id']}: duplicate within the edge set")
            if key in real_keys:
                raise AssertionError(f"{order['order_id']}: coincides with a real train, validation or test row")
            seen.add(key)
            system_prompt, user_prompt = STAGE_PROMPTS[stage](agent, order)
            files.write(stage, {
                "order_id": order["order_id"], "agent_id": agent["agent_id"], "stage": stage,
                "dataset_version": context.version, "role": ROLE, "category": category,
                "rule_decision": rule["decision"], "system": system_prompt, "user": user_prompt,
                "completion": completion_json(stage, order["order_id"], agent["agent_id"], rule,
                                              agent.get(STAGE_DEFECT_KEY[stage])),
            })
            counts[(stage, category)] += 1
            labels[(stage, rule["decision"])] += 1

    for (stage, category), n in sorted(counts.items()):
        print(f"  {stage:10s} {category:24s} {n:4d}")
    write_json(OUT_DIR / "metadata.json", {
        "role": ROLE,
        "role_statement": ("robustness evaluation only: not training data, and not counted in any train, validation "
                           "or test total"),
        "data_version": context.version,
        "currency": context.currency,
        "method": "src/experiments/generate_edge_examples.py, rebuilt for v5 as this module's docstring records",
        "worded_formulas": True,
        "prompt_wording": ("rendered through the shared v5 stage prompt builders, so in the validated worded style "
                           "(reports/v5/wording/validated_mapping.json)"),
        "counts": {stage: files.counts[stage] for stage in STAGES},
        "by_category": {f"{stage}:{category}": n for (stage, category), n in sorted(counts.items())},
        "label_balance": {f"{stage}:{label}": n for (stage, label), n in sorted(labels.items())},
        "expected_decision_by_category": {k: {"decision": d, "reason": r} for k, (d, r) in EXPECTED.items()},
        "verification": ("every label asserted through the dispatched decide_*_bid with its category's expected "
                         "decision and reason; no parameter tuple overlaps the fixed baseline cases, another edge "
                         "case, or any real v5 train, validation or test row"),
        "real_rows_checked_for_overlap": len(real_keys),
        "deferred": ("tailoring_boundary_stress, tailoring_final_boundary_holdout and material or logistics "
                     "equivalents are intentionally not generated tonight; they are supplementary evaluation "
                     "artifacts for the tailoring workload-markup boundary investigation, not required for the "
                     "main corpus"),
    })
    print(f"written to {rel(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
