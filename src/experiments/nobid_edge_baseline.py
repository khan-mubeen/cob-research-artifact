"""
baseline: does the un-fine-tuned slm return NO_BID on dynamic edge cases?

the sampled v4 training data never triggers the dynamic NO_BID conditions for
tailoring and logistics, deadline-infeasible completion and workload ratio
above 1.00. these six hand-constructed rfqs hit exactly those branches using
real agent profiles, so this script records the reference behavior of the
current base model that every later comparison, fine-tuned or not, is checked
against.

each case is checked against the rule-based decision before model inference.
the rule path must return NO_BID for the intended dynamic reason.

workload_ratio = (workload + quantity) / (capacity x days) is equivalent to
estimated_completion / days, so a ratio above 1.00 always coincides with a
missed deadline and the rule reason is always completion_exceeds_deadline. the
workload cases remain well above 1.00 regardless of which constraint the model
evaluates.

results: printed table plus reports/epoch-1/production/evaluation/guard/nobid_edge_baseline.json
"""

import asyncio
import json
import os
import time
from datetime import datetime

import pandas as pd

from config import DATA_VERSION, PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_logistics_bid,
    decide_manufacturer_bid,
    logistics_eligibility_reason,
    manufacturer_eligibility_reason,
)
# importing the diagnostics module also sets USE_SLM / gpu env before load_slm
from src.visualization.slm_competition_diagnostics import GenerationRecorder
from src.common.slm import (
    load_slm,
    slm_decide_logistics_bid,
    slm_decide_manufacturer_bid,
    warmup_slm,
)


REPORT_PATH = (PROJECT_ROOT / "reports" / "epoch-1" / "production" / "evaluation" /
               "guard" / "nobid_edge_baseline.json")

# eligibility fields are deliberately permissive so the only NO_BID trigger
# left is the dynamic condition each case is built around
_ORDER_BASE = {
    "max_defect_rate": 0.5,
    "min_reputation_score": 0.1,
    "priority": "normal",
}


def _mfr_case(case_id, description, agent_id, product_id, quantity, days):
    return {
        "case_id": case_id,
        "description": description,
        "stage": "tailoring",
        "agent_id": agent_id,
        "order": {
            **_ORDER_BASE,
            "order_id": case_id,
            "product_id": product_id,
            "quantity": quantity,
            "days_until_deadline": days,
        },
    }


def _log_case(case_id, description, agent_id, deliver_to, weight_kg, days):
    return {
        "case_id": case_id,
        "description": description,
        "stage": "logistics",
        "agent_id": agent_id,
        "order": {
            **_ORDER_BASE,
            "order_id": case_id,
            "deliver_to": deliver_to,
            "shipment_weight_kg": weight_kg,
            "days_until_deadline": days,
        },
    }


# T9: cap 803, workload 64  -> est (64+2500)/803 = 3.19 days vs deadline 3, ratio 1.06
# T12: cap 804, workload 603 -> est (603+3000)/804 = 4.48 days vs deadline 4, ratio 1.12
# T3: cap 830, workload 706 -> est (706+8000)/830 = 10.49 days vs deadline 5, ratio 2.10
# T1: cap 883, workload 177 -> est (177+5000)/883 = 5.86 days vs deadline 5, ratio 1.17
# L06: Lahti, intercity 4.0 days vs deadline 2
# L07: Helsinki, intercity 1.5 days vs deadline 1
CASES = [
    _mfr_case("EDGE-T-DEADLINE-01",
              "completion barely exceeds deadline (3.19 vs 3 days)",
              "T9", "shirt", 2500, 3),
    _mfr_case("EDGE-T-DEADLINE-02",
              "large backlog pushes completion past deadline (4.48 vs 4 days)",
              "T12", "jeans", 3000, 4),
    _mfr_case("EDGE-T-WORKLOAD-01",
              "workload ratio 2.10, far above the 1.00 no-bid line",
              "T3", "jacket", 8000, 5),
    _mfr_case("EDGE-T-WORKLOAD-02",
              "workload ratio 1.17, just above the 1.00 no-bid line",
              "T1", "hoodie", 5000, 5),
    _log_case("EDGE-L-ROUTE-01",
              "intercity route 4.0 days vs 2 day deadline",
              "L06", "Tampere", 900.0, 2),
    _log_case("EDGE-L-ROUTE-02",
              "intercity route 1.5 days vs 1 day deadline",
              "L07", "Turku", 450.0, 1),
]


def load_agent(stage, agent_id):
    filename = "manufacturer_agents.csv" if stage == "tailoring" else "logistics_agents.csv"
    df = pd.read_csv(PROJECT_ROOT / "data" / "synthetic" / DATA_VERSION / filename)
    row = df[df["agent_id"] == agent_id]
    if row.empty:
        raise SystemExit(f"agent {agent_id} not found in {filename}")
    return row.iloc[0].to_dict()


async def run():
    print(f"NO_BID edge case baseline, un-fine-tuned model, version {DATA_VERSION}")
    print(f"model: {os.getenv('SLM_MODEL_PATH', '') or 'default from src/common/slm.py'}\n")

    slm = load_slm("nobid_baseline")
    await warmup_slm(slm, "nobid_baseline")
    recorder = GenerationRecorder(slm._llama)

    results = []
    for case in CASES:
        agent = load_agent(case["stage"], case["agent_id"])
        order = case["order"]

        if case["stage"] == "tailoring":
            eligibility_fn, rule_fn, slm_fn = (
                manufacturer_eligibility_reason, decide_manufacturer_bid, slm_decide_manufacturer_bid)
        else:
            eligibility_fn, rule_fn, slm_fn = (
                logistics_eligibility_reason, decide_logistics_bid, slm_decide_logistics_bid)

        # the case must be eligible and must fail only on the dynamic check
        assert eligibility_fn(agent, order) is None, \
            f"{case['case_id']}: agent unexpectedly ineligible, case is built wrong"
        rule = rule_fn(agent, order)
        assert rule["decision"] == "NO_BID", \
            f"{case['case_id']}: rule path returned {rule['decision']}, case is built wrong"

        started = time.perf_counter()
        slm_dec = await slm_fn(slm, agent, order)
        elapsed = time.perf_counter() - started

        correct = slm_dec["decision"] == "NO_BID"
        results.append({
            "case_id": case["case_id"],
            "description": case["description"],
            "stage": case["stage"],
            "agent_id": case["agent_id"],
            "rule_reason": rule["debug_reason"],
            "slm_decision": slm_dec["decision"],
            "slm_offer_amount": slm_dec.get("offer_amount"),
            "correct_no_bid": correct,
            "seconds": round(elapsed, 2),
            "raw_response": recorder.calls[-1]["raw"],
        })
        print(f"{case['case_id']:22s} rule={rule['debug_reason']:28s} "
              f"slm={slm_dec['decision']:6s} {'OK' if correct else 'WRONG - model bid on an infeasible rfq'}")

    correct_count = sum(r["correct_no_bid"] for r in results)
    print(f"\nbase model NO_BID accuracy on dynamic edge cases: "
          f"{correct_count} / {len(results)} ({correct_count / len(results) * 100:.0f}%)")

    print("\nfull raw response for the first case:")
    print(results[0]["raw_response"])

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": os.getenv("SLM_MODEL_PATH", "") or "models/Qwen3.5-4B-Q8_0.gguf",
        "model_state": "base, un-fine-tuned",
        "correct": correct_count,
        "total": len(results),
        "cases": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nbaseline written to {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
