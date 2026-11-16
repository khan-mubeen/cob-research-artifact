"""
final boundary holdout, untouched independent evidence for round three.

the original boundary stress set influenced round-three design (its results
motivated the reweighting), so it is demoted to a development set. this set
is generated before round-three training starts, evaluated once per model,
and must stay disjoint from training, edge holdout, the six baseline cases,
the existing stress set, and every real train/holdout example.

per manufacturer, three cases: ratio ~0.97 (feasible, BID), exactly 1.00
(feasible, BID), ~1.03 (infeasible, NO_BID). deadlines use a day offset that
neither the training edge cases (3+(idx%4)) nor the stress set (3+((idx+2)%4))
use, so parameter tuples cannot repeat either.

PRE-REGISTERED QUALITY BAR, declared before any round-three result exists:
round three passes the boundary condition only if each class independently
scores >= 9/12 (75%) correct decisions: below_feasible, exact_feasible,
above_infeasible. no aggregate score, one-sided failure must stay visible.

output: data/finetune/{version}/final_boundary_holdout.jsonl
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

FINAL_RATIOS = ((0.97, "below_feasible"), (1.00, "exact_feasible"), (1.03, "above_infeasible"))

OUT_PATH = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION / "final_boundary_holdout.jsonl"
STRESS_PATH = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION / "boundary_stress.jsonl"


def main():
    materials, manufacturers, logistics, products, orders = load_all_data()

    real_keys = load_real_param_keys()
    forbidden = set(real_keys)
    forbidden.update(_param_key(c["stage"], c["agent_id"], c["order"]) for c in BASELINE_CASES)
    for case in tailoring_cases(manufacturers) + logistics_cases(logistics, real_keys):
        forbidden.add(_param_key(case["stage"], case["agent"]["agent_id"], case["order"]))
    with open(STRESS_PATH, encoding="utf-8") as f:
        import re
        for line in f:
            r = json.loads(line)
            forbidden.add(("tailoring", r["agent_id"],
                           re.search(r"^product: (\S+)", r["user"], re.M).group(1),
                           int(re.search(r"^quantity: (\d+)", r["user"], re.M).group(1)),
                           int(re.search(r"^days until deadline: (\d+)", r["user"], re.M).group(1))))

    records = []
    for idx, agent in enumerate(manufacturers.to_dict("records")):
        cap = agent["daily_capacity_units"]
        workload = agent["current_workload_units"]
        products_list = str(agent["product_types"]).split("|")
        days = 3 + ((idx + 1) % 4)  # distinct from training and stress day cycles
        product = products_list[(idx + 1) % len(products_list)]

        for ratio, klass in FINAL_RATIOS:
            qty = solve_quantity(cap, workload, days, ratio)
            if ratio < 1.0:
                while qty > 1 and (workload + qty) / cap >= days:
                    qty -= 1
            elif ratio > 1.0:
                while (workload + qty) / cap <= days:
                    qty += 1

            order = {
                **ORDER_BASE,
                "order_id": f"FINALB-T-{klass.upper()[:5]}-{agent['agent_id']}",
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
            assert key not in forbidden, \
                f"{order['order_id']}: collides with training, stress, baseline, or real data"
            forbidden.add(key)

            realized = (workload + qty) / cap / days
            system_prompt, user_prompt = build_manufacturer_prompt(agent, order)
            records.append({
                "order_id": order["order_id"],
                "agent_id": agent["agent_id"],
                "stage": "tailoring",
                "category": klass,
                "realized_ratio": round(realized, 6),
                "rule_decision": rule["decision"],
                "system": system_prompt,
                "user": user_prompt,
                "completion": completion_json("tailoring", order["order_id"],
                                              agent["agent_id"], rule,
                                              agent.get("defect_probability")),
            })
            print(f"{order['order_id']:26s} days={days} qty={qty:6d} "
                  f"realized_ratio={realized:.6f} -> {rule['decision']}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(records)} final-holdout cases, all rule-verified, disjoint from "
          f"training, stress, baseline, and real data, written to {OUT_PATH}")


if __name__ == "__main__":
    main()
