"""
boundary stress test set, evaluation only, never used in training.

per manufacturer, three cases straddling the completion/deadline boundary:
ratio just below 1.00 (BID), exactly 1.00 (BID), just above 1.00 (NO_BID).
generalization evidence around the exact threshold, reported separately from
the four official criteria, never folded into the pass/fail bar.

deadlines deliberately differ from the training edge cases for each agent so
parameter tuples cannot collide, and disjointness against both the training
edge set and the six fixed baseline cases is asserted, not assumed.

output: data/finetune/{version}/boundary_stress.jsonl
"""

import json

from config import DATA_VERSION, PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_manufacturer_bid,
    manufacturer_eligibility_reason,
)
from src.common.slm import build_manufacturer_prompt
from src.experiments.generate_edge_examples import (
    ORDER_BASE,
    _param_key,
    completion_json,
    load_real_param_keys,
    logistics_cases,
    solve_quantity,
    tailoring_cases,
)
from src.experiments.nobid_edge_baseline import CASES as BASELINE_CASES
from src.visualization.data_utils import load_all_data

STRESS_RATIOS = (0.98, 1.00, 1.02)

OUT_PATH = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION / "boundary_stress.jsonl"


def main():
    materials, manufacturers, logistics, products, orders = load_all_data()

    # parameter keys of everything this set must stay disjoint from,
    # real examples included so stress results are pure generalization
    real_keys = load_real_param_keys()
    forbidden = set(real_keys)
    forbidden.update(_param_key(c["stage"], c["agent_id"], c["order"]) for c in BASELINE_CASES)
    for case in tailoring_cases(manufacturers) + logistics_cases(logistics, real_keys):
        forbidden.add(_param_key(case["stage"], case["agent"]["agent_id"], case["order"]))

    records = []
    for idx, agent in enumerate(manufacturers.to_dict("records")):
        cap = agent["daily_capacity_units"]
        workload = agent["current_workload_units"]
        products_list = str(agent["product_types"]).split("|")
        # training edge cases use days = 3+(idx%4), shift by two so no
        # (agent, days) pair repeats and quantities differ with it
        days = 3 + ((idx + 2) % 4)
        product = products_list[idx % len(products_list)]

        for ratio in STRESS_RATIOS:
            qty = solve_quantity(cap, workload, days, ratio)
            if ratio < 1.0:
                while qty > 1 and (workload + qty) / cap >= days:
                    qty -= 1
            elif ratio > 1.0:
                while (workload + qty) / cap <= days:
                    qty += 1

            order = {
                **ORDER_BASE,
                "order_id": f"STRESS-T-{str(ratio).replace('.', 'P')}-{agent['agent_id']}",
                "product_id": product,
                "quantity": qty,
                "days_until_deadline": days,
            }

            assert manufacturer_eligibility_reason(agent, order) is None, \
                f"{order['order_id']}: statically ineligible"
            rule = decide_manufacturer_bid(agent, order)
            expected = "BID" if ratio <= 1.0 else "NO_BID"
            assert rule["decision"] == expected, \
                f"{order['order_id']}: rule said {rule['decision']}, expected {expected}"

            key = _param_key("tailoring", agent["agent_id"], order)
            assert key not in forbidden, f"{order['order_id']}: collides with training or baseline"
            forbidden.add(key)  # also guards against internal duplicates

            realized = (workload + qty) / cap / days
            system_prompt, user_prompt = build_manufacturer_prompt(agent, order)
            records.append({
                "order_id": order["order_id"],
                "agent_id": agent["agent_id"],
                "stage": "tailoring",
                "category": f"stress_ratio_{ratio}",
                "realized_ratio": round(realized, 6),
                "rule_decision": rule["decision"],
                "system": system_prompt,
                "user": user_prompt,
                "completion": completion_json("tailoring", order["order_id"],
                                              agent["agent_id"], rule,
                                              agent.get("defect_probability")),
            })
            print(f"{order['order_id']:24s} days={days} qty={qty:6d} "
                  f"realized_ratio={realized:.6f} -> {rule['decision']}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(records)} stress cases, all rule-verified, disjoint from "
          f"training edge set and baseline cases, written to {OUT_PATH}")


if __name__ == "__main__":
    main()
