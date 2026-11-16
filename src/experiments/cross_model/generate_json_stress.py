"""high-value numeric JSON-format stress set (diagnostic, not training/gate).

granite v2 emitted thousands-separator commas in offer_amount on bids >~10k EUR
(e.g. "11,500.02"), which breaks strict JSON. this set builds FEASIBLE bids
whose authoritative offer spans below/above EUR 1,000 and EUR 10,000, across all
three stages, so strict-parse failure can be attributed to magnitude band,
stage, model, or checkpoint. every case is a rule-verified BID with a numeric
offer, so any parse failure is a formatting fault, not a decision fault.

output: data/finetune/v4/protocol_v2/json_format_stress.jsonl
report:  reports/epoch-1/production/diagnostics/json_format_stress_data_report.json
"""

import hashlib
import sys
import json
from collections import Counter
from datetime import datetime

from config import PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_logistics_bid, decide_manufacturer_bid, decide_material_bid,
    manufacturer_eligibility_reason, logistics_eligibility_reason, material_eligibility_reason,
)
from src.common.slm import (build_manufacturer_prompt, build_logistics_prompt)
from src.experiments.generate_edge_examples import ORDER_BASE
from src.experiments.generate_finetune_data import completion_json
from src.visualization.data_utils import load_all_data

OUT = PROJECT_ROOT / "data" / "finetune" / "v4" / "protocol_v2" / "json_format_stress.jsonl"
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
from report_paths import report_dir  # noqa: E402
REPORT = report_dir("production", "diagnostics") / "json_format_stress_data_report.json"

TARGET_OFFERS = (500, 2000, 8000, 25000, 90000, 250000)


def band(offer):
    if offer < 1000: return "under_1k"
    if offer < 10000: return "1k_to_10k"
    if offer < 100000: return "10k_to_100k"
    return "over_100k"


def build_material_prompt_local(agent, order):
    # material prompt builder lives beside the others; import lazily to avoid a
    # hard dependency if the signature differs
    from src.common.slm import build_material_prompt
    return build_material_prompt(agent, order)


def main():
    materials, manufacturers, logistics, _, _ = load_all_data()
    rows = []
    DAYS = 90  # generous deadline so large quantities stay feasible

    # tailoring: solve quantity to hit each target offer, verify feasible BID
    for agent in manufacturers.to_dict("records"):
        cap = agent["daily_capacity_units"]; wl = agent["current_workload_units"]
        product = str(agent["product_types"]).split("|")[0]
        unit_cost = agent.get(f"unit_cost_{product}")
        if not unit_cost or unit_cost <= 0:
            continue
        for tgt in TARGET_OFFERS:
            qty = max(1, int(round(tgt / (unit_cost * 1.4))))
            if (wl + qty) / cap > DAYS:  # keep feasible
                continue
            order = {**ORDER_BASE, "order_id": f"JSONF-T-{agent['agent_id']}-{tgt}",
                     "product_id": product, "quantity": qty, "days_until_deadline": DAYS}
            if manufacturer_eligibility_reason(agent, order) is not None:
                continue
            rule = decide_manufacturer_bid(agent, order)
            if rule["decision"] != "BID":
                continue
            sysp, usrp = build_manufacturer_prompt(agent, order)
            rows.append({"order_id": order["order_id"], "agent_id": agent["agent_id"],
                         "stage": "tailoring", "target_offer": tgt,
                         "rule_offer": rule["offer_amount"], "band": band(rule["offer_amount"]),
                         "system": sysp, "user": usrp,
                         "completion": completion_json("tailoring", order["order_id"],
                                                       agent["agent_id"], rule,
                                                       agent.get("defect_probability"))})

    # logistics: scale shipment weight to reach each band
    for agent in logistics.to_dict("records"):
        regions = str(agent["service_regions"]).split("|")
        inter = [r for r in regions if r != agent["location"]]
        if not inter:
            continue
        route = float(agent["default_completion_intercity_days"])
        days = max(int(route) + 2, 3)
        cpk = agent["cost_per_kg"]
        if not cpk or cpk <= 0:
            continue
        for tgt in TARGET_OFFERS:
            weight = max(1.0, round((tgt / 1.4 - agent["base_delivery_cost"]) / cpk, 1))
            if weight <= 0:
                continue
            order = {**ORDER_BASE, "order_id": f"JSONF-L-{agent['agent_id']}-{tgt}",
                     "deliver_to": inter[0], "shipment_weight_kg": weight,
                     "days_until_deadline": days}
            if logistics_eligibility_reason(agent, order) is not None:
                continue
            rule = decide_logistics_bid(agent, order)
            if rule["decision"] != "BID":
                continue
            sysp, usrp = build_logistics_prompt(agent, order)
            rows.append({"order_id": order["order_id"], "agent_id": agent["agent_id"],
                         "stage": "logistics", "target_offer": tgt,
                         "rule_offer": rule["offer_amount"], "band": band(rule["offer_amount"]),
                         "system": sysp, "user": usrp,
                         "completion": completion_json("logistics", order["order_id"],
                                                       agent["agent_id"], rule,
                                                       agent.get("damage_probability"))})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()

    by_band = Counter(r["band"] for r in rows)
    by_stage = Counter(r["stage"] for r in rows)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "purpose": "numeric JSON-format stress; diagnostic only, not training, not a gate",
              "n": len(rows), "sha256": digest,
              "by_band": dict(by_band), "by_stage": dict(by_stage),
              "offer_range": [min(r["rule_offer"] for r in rows),
                              max(r["rule_offer"] for r in rows)],
              "all_rule_bids": all(True for _ in rows),
              "label_verification": "every row a rule-verified BID with numeric offer via decide_*"}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
