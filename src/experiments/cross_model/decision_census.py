"""frozen snapshot of rule-based decide_* output, the regression baseline for
the Guard B refactor of bidding_formulas.py.

run this ONCE before the refactor (writes the baseline) and again after (must
produce an identical content hash). proves that extracting the shared dynamic
feasibility helpers changed no rule decision.

coverage:
  real_tailoring: every manufacturer agent x every real order whose product the
    agent makes -> exhaustive over the manufacturer stage on real v4 data
    (needs only fields present in orders.csv)
  synthetic_logistics / synthetic_material: real agents x constructed orders
    sweeping deadlines and quantities/weights around each agent's feasibility
    boundary, so every dynamic reason branch is exercised (weight/kg are
    derived per-RFQ at runtime and are not columns in orders.csv)

output: reports/epoch-1/production/diagnostics/decision_census_baseline.json (+ _after.json on
rerun with --after), each with a sha256 over the sorted decision tuples.
"""

import argparse
import sys
import hashlib
import json
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, SYNTHETIC_DATA_DIR
from src.common.bidding_formulas import (
    decide_logistics_bid,
    decide_manufacturer_bid,
    decide_material_bid,
)
from src.visualization.data_utils import load_all_data

sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
from report_paths import report_dir  # noqa: E402
OUT_DIR = report_dir("production", "diagnostics")
MATERIALS = ["cotton", "polyester", "linen", "denim", "leather"]


def order_days(orders):
    created = pd.to_datetime(orders["order_created"])
    deadline = pd.to_datetime(orders["deadline"])
    return ((deadline - created).dt.days).clip(lower=1)


def rec(stage, agent_id, oid, d):
    return {"stage": stage, "agent_id": agent_id, "order_id": oid,
            "decision": d["decision"], "reason": d.get("debug_reason", ""),
            "offer": d.get("offer_amount"), "days": d.get("estimated_completion_days")}


def census():
    materials, manufacturers, logistics, products, orders = load_all_data()
    orders = orders.copy()
    orders["days_until_deadline"] = order_days(orders)
    out = []

    # exhaustive real-data manufacturer stage
    for agent in manufacturers.to_dict("records"):
        made = str(agent["product_types"]).split("|")
        for o in orders.to_dict("records"):
            order = {"order_id": o["order_id"], "product_id": o["product_id"],
                     "quantity": int(o["quantity"]), "max_defect_rate": o["max_defect_rate"],
                     "min_reputation_score": o["min_reputation_score"],
                     "priority": o["priority"], "days_until_deadline": int(o["days_until_deadline"])}
            # only score pairs whose product the agent could plausibly make, but
            # still include a slice of non-made products to exercise eligibility
            if order["product_id"] in made or (o["order_id"][-1] in "0"):
                out.append(rec("tailoring", agent["agent_id"], o["order_id"],
                               decide_manufacturer_bid(agent, order)))

    # synthetic logistics sweep: real agents, constructed delivery orders
    for agent in logistics.to_dict("records"):
        regions = str(agent["service_regions"]).split("|")
        route = float(agent["default_completion_intercity_days"])
        for i, region in enumerate(regions[:3]):
            for days in range(1, int(route) + 4):
                order = {"order_id": f"SYN-L-{agent['agent_id']}-{i}-{days}",
                         "deliver_to": region, "shipment_weight_kg": 500.0 + 50 * days,
                         "max_defect_rate": 0.5, "min_reputation_score": 0.1,
                         "priority": "normal", "days_until_deadline": days}
                out.append(rec("logistics", agent["agent_id"], order["order_id"],
                               decide_logistics_bid(agent, order)))
        # eligibility: an unserviced region
        bad = {"order_id": f"SYN-L-{agent['agent_id']}-BAD", "deliver_to": "__nowhere__",
               "shipment_weight_kg": 500.0, "max_defect_rate": 0.5,
               "min_reputation_score": 0.1, "priority": "normal", "days_until_deadline": 5}
        out.append(rec("logistics", agent["agent_id"], bad["order_id"],
                       decide_logistics_bid(agent, bad)))

    # synthetic material sweep: real agents, constructed material orders
    for agent in materials.to_dict("records"):
        supplied = str(agent["material_types"]).split("|")
        for m in MATERIALS:
            avail = agent.get(f"{m}_available_kg")
            restock = agent.get(f"{m}_restock_amount_kg")
            base_qtys = [10.0]
            if pd.notna(avail):
                base_qtys += [float(avail) * 0.5, float(avail) + 1,
                              float(avail) + (float(restock) if pd.notna(restock) else 0) + 1]
            for q in base_qtys:
                for days in (1, 5, 60):
                    order = {"order_id": f"SYN-M-{agent['agent_id']}-{m}-{int(q)}-{days}",
                             "material_type": m, "required_kg": q, "max_defect_rate": 0.5,
                             "min_reputation_score": 0.0, "priority": "normal",
                             "days_until_deadline": days}
                    out.append(rec("material", agent["agent_id"], order["order_id"],
                                   decide_material_bid(agent, order)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", action="store_true", help="write _after.json and compare to baseline")
    args = ap.parse_args()

    out = census()
    key = lambda r: (r["stage"], r["agent_id"], r["order_id"])
    out.sort(key=key)
    blob = json.dumps(out, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()

    name = "decision_census_after.json" if args.after else "decision_census_baseline.json"
    summary = {"n_decisions": len(out), "sha256": digest,
               "by_stage": {s: sum(1 for r in out if r["stage"] == s)
                            for s in ("tailoring", "logistics", "material")},
               "bids": sum(1 for r in out if r["decision"] == "BID"),
               "no_bids": sum(1 for r in out if r["decision"] == "NO_BID")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "decisions": out}, f, sort_keys=True, default=str)
    print(json.dumps(summary, indent=2))

    if args.after:
        base = json.load(open(OUT_DIR / "decision_census_baseline.json", encoding="utf-8"))
        if base["summary"]["sha256"] == digest:
            print("\nREGRESSION PASS: post-refactor decisions byte-identical to baseline")
        else:
            print("\nREGRESSION FAIL: hash mismatch")
            b = {tuple(key(r)): r for r in base["decisions"]}
            for r in out:
                k = key(r)
                if k in b and b[k] != r:
                    print("  DIFF", k, "\n    was", b[k], "\n    now", r)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
