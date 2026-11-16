"""scale-vs-magnitude diagnostic set (EXPLANATORY, not training, not a gate).

tests the Protocol v2 hypothesis that models pattern-match absolute magnitudes
(deadline, quantity, capacity) rather than applying completion/deadline. built
from matched counterfactual families so that, as far as the formula permits,
one axis moves while others are pinned. every label is verified through
decide_manufacturer_bid; the set is frozen and hashed before any model runs.

the manufacturer rule: completion = (workload + qty) / capacity;
NO_BID iff completion > deadline (equivalently ratio = completion/deadline > 1).

families (what is HELD vs what NECESSARILY MOVES):
  F1 ratio_fixed_deadline_varies   HELD target ratio + agent; MOVES deadline,
       quantity, completion (all scale together). isolates "same ratio".
  F2 quantity_fixed_deadline_varies HELD agent + quantity -> completion is
       CONSTANT; MOVES deadline, ratio. the rule flips exactly when deadline
       crosses the fixed completion. cleanest deadline-vs-rule probe: if the
       model tracks the rule, decisions flip at completion; if it keys off
       quantity magnitude, they will not (quantity is pinned).
  F3 capacity_varies_dl_qty_fixed  HELD deadline + quantity (the RFQ numbers);
       MOVES capacity (synthetic profile copies), completion, ratio. isolates
       whether the model reads the PROFILE to judge feasibility with RFQ numbers
       pinned; if decisions ignore capacity, the model ignores the profile.
  F4 crossed_ratio_deadline         ratio x deadline grid, quantity solved. the
       decision surface for attribution.

perfect isolation is impossible (completion is a function of qty, cap, workload);
each row records target_ratio, realized_ratio, deadline, quantity, capacity,
workload, completion so any residual confound is inspectable.

output: data/finetune/v4/protocol_v2/scale_diagnostic.jsonl
report:  reports/epoch-1/production/diagnostics/scale_diagnostic_data_report.json
"""

import hashlib
import sys
import json
from collections import Counter
from datetime import datetime

from config import PROJECT_ROOT
from src.common.bidding_formulas import decide_manufacturer_bid, manufacturer_eligibility_reason
from src.common.slm import build_manufacturer_prompt
from src.experiments.generate_edge_examples import ORDER_BASE
from src.experiments.generate_finetune_data import completion_json
from src.visualization.data_utils import load_all_data

OUT = PROJECT_ROOT / "data" / "finetune" / "v4" / "protocol_v2" / "scale_diagnostic.jsonl"
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
from report_paths import report_dir  # noqa: E402
REPORT = report_dir("production", "diagnostics") / "scale_diagnostic_data_report.json"

DEADLINES = (2, 4, 7, 10, 14)
F1_RATIOS = (0.95, 1.05)
F4_RATIOS = (0.90, 0.97, 1.03, 1.10)
F4_DEADLINES = (2, 5, 9, 13)


def solve_qty(cap, workload, days, ratio):
    return max(1, int(round(cap * days * ratio - workload)))


def build(agent, oid, family, product, cap, workload, qty, days, target_ratio):
    a = dict(agent)
    a["daily_capacity_units"] = cap
    a["current_workload_units"] = workload
    order = {**ORDER_BASE, "order_id": oid, "product_id": product,
             "quantity": qty, "days_until_deadline": days}
    if manufacturer_eligibility_reason(a, order) is not None:
        return None
    rule = decide_manufacturer_bid(a, order)
    completion = (workload + qty) / cap
    realized = completion / days
    system_prompt, user_prompt = build_manufacturer_prompt(a, order)
    return {"order_id": oid, "agent_id": agent["agent_id"], "stage": "tailoring",
            "family": family, "target_ratio": round(target_ratio, 4),
            "realized_ratio": round(realized, 6), "deadline": days, "quantity": qty,
            "capacity": cap, "workload": workload, "completion": round(completion, 4),
            "rule_decision": rule["decision"], "system": system_prompt, "user": user_prompt,
            "completion_json": completion_json("tailoring", oid, agent["agent_id"], rule,
                                               agent.get("defect_probability"))}


def main():
    _, manufacturers, _, _, _ = load_all_data()
    rows = []
    for idx, agent in enumerate(manufacturers.to_dict("records")):
        cap0 = int(agent["daily_capacity_units"])
        wl0 = int(agent["current_workload_units"])
        product = str(agent["product_types"]).split("|")[0]

        # F1: hold target ratio, vary deadline (quantity solved)
        for r in F1_RATIOS:
            for d in DEADLINES:
                q = solve_qty(cap0, wl0, d, r)
                oid = f"SCALE-F1-{'I' if r>1 else 'F'}-{agent['agent_id']}-d{d}"
                rec = build(agent, oid, "ratio_fixed_deadline_varies", product,
                            cap0, wl0, q, d, r)
                if rec:
                    rows.append(rec)

        # F2: hold quantity so completion is constant ~ mid-range, vary deadline
        comp_target = 6.0
        q_fixed = max(1, int(round(comp_target * cap0 - wl0)))
        for d in DEADLINES:
            oid = f"SCALE-F2-{agent['agent_id']}-d{d}"
            rec = build(agent, oid, "quantity_fixed_deadline_varies", product,
                        cap0, wl0, q_fixed, d, (wl0 + q_fixed) / cap0 / d)
            if rec:
                rows.append(rec)

        # F3: hold deadline+quantity, vary capacity (synthetic copies) so ratio crosses 1
        d_fixed = 7
        q3 = max(1, int(round(1.0 * cap0 * d_fixed - wl0)))  # ~ratio 1 at base capacity
        for cap_mult, tag in ((0.8, "r125"), (0.9, "r111"), (1.0, "r100"),
                              (1.1, "r091"), (1.3, "r077")):
            cap = max(1, int(round(cap0 * cap_mult)))
            completion = (wl0 + q3) / cap
            oid = f"SCALE-F3-{agent['agent_id']}-{tag}"
            rec = build(agent, oid, "capacity_varies_dl_qty_fixed", product,
                        cap, wl0, q3, d_fixed, completion / d_fixed)
            if rec:
                rows.append(rec)

        # F4: crossed ratio x deadline grid
        for r in F4_RATIOS:
            for d in F4_DEADLINES:
                q = solve_qty(cap0, wl0, d, r)
                oid = f"SCALE-F4-{agent['agent_id']}-r{int(r*100)}-d{d}"
                rec = build(agent, oid, "crossed_ratio_deadline", product,
                            cap0, wl0, q, d, r)
                if rec:
                    rows.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()

    fam = Counter(r["family"] for r in rows)
    dec = Counter((r["family"], r["rule_decision"]) for r in rows)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "purpose": "explanatory diagnostic only; NOT a training set, NOT an acceptance gate",
              "n": len(rows), "sha256": digest,
              "families": dict(fam),
              "label_balance_by_family": {f"{k[0]}:{k[1]}": v for k, v in sorted(dec.items())},
              "held_vs_moved": {
                  "F1_ratio_fixed_deadline_varies": "HELD target_ratio, agent; MOVES deadline, quantity, completion",
                  "F2_quantity_fixed_deadline_varies": "HELD agent, quantity (=> completion constant); MOVES deadline, ratio",
                  "F3_capacity_varies_dl_qty_fixed": "HELD deadline, quantity; MOVES capacity, completion, ratio",
                  "F4_crossed_ratio_deadline": "ratio x deadline grid, quantity solved"},
              "label_verification": "every row via decide_manufacturer_bid; eligibility via manufacturer_eligibility_reason"}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
