"""magnitude / decomposition diagnostic (EXPLANATORY, not training, not a gate).

directly tests whether a model keys off absolute magnitudes or the Q-vs-W split
rather than the completion/deadline rule. two matched families in which the RULE
DECISION IS INVARIANT by construction (completion and workload-ratio held fixed):

Family A — absolute-scale: deadline, target ratio, completion, product fixed;
  quantity, workload, capacity all scaled by the SAME factor s (the only way to
  hold both ratio and completion fixed when the deadline is fixed). so ONLY the
  absolute magnitudes move. if a model's BID flips across s, it is magnitude-
  sensitive. scale factors chosen so quantity stays within the real v4 range
  [100,2000] and capacity within a plausible [500,1600] (<=1.8x the real max
  883, under one order of magnitude); realized magnitude span reported.

Family B — quantity/workload decomposition: deadline, capacity, T=workload+
  quantity, ratio, completion all fixed; only the split between quantity and
  current workload moves (Q up, W down by the same amount). if a model's BID
  flips across the split, it reacts to quantity or backlog specifically rather
  than their rule-relevant sum. fully within real ranges.

every label verified through decide_manufacturer_bid. within each (family,
group) the expected decision is constant; the diagnostic measures whether the
model's decision is likewise constant.

output: data/finetune/v4/protocol_v2/magnitude_diagnostic.jsonl
report:  reports/epoch-1/production/diagnostics/magnitude_diagnostic_data_report.json
"""

import hashlib
import sys
import json
from collections import Counter, defaultdict
from datetime import datetime

from config import PROJECT_ROOT
from src.common.bidding_formulas import decide_manufacturer_bid, manufacturer_eligibility_reason
from src.common.slm import build_manufacturer_prompt
from src.experiments.generate_edge_examples import ORDER_BASE
from src.experiments.generate_finetune_data import completion_json
from src.visualization.data_utils import load_all_data

OUT = PROJECT_ROOT / "data" / "finetune" / "v4" / "protocol_v2" / "magnitude_diagnostic.jsonl"
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
from report_paths import report_dir  # noqa: E402
REPORT = report_dir("production", "diagnostics") / "magnitude_diagnostic_data_report.json"

# real v4 bounds (checked): capacity 645-883, quantity 100-2000, workload 43-706
Q_MIN, Q_MAX = 100, 2000
C_MIN, C_MAX = 500, 1600
W_MAX = 900
A_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)
B_FRACS = (0.2, 0.35, 0.5, 0.65, 0.8)
# Family A (deadline, ratio, side): ratios off 1.00 so rounding cannot flip side
CONDS = [(4, 0.8, "feasible"), (5, 0.9, "feasible"),
         (3, 1.10, "infeasible"), (2, 1.15, "infeasible")]
# Family B (deadline, ratio, capacity): totals kept moderate so BOTH quantity and
# workload can span ~[100,800] within real bounds as the split varies
B_CONDS = [(2, 0.60, 800, "feasible"), (3, 0.50, 700, "feasible"),
           (1, 1.30, 700, "infeasible"), (2, 1.10, 500, "infeasible")]


def build(agent_tmpl, cap, workload, qty, days, group, family, cond_tag):
    a = dict(agent_tmpl)
    a["daily_capacity_units"] = cap
    a["current_workload_units"] = workload
    product = str(agent_tmpl["product_types"]).split("|")[0]
    order = {**ORDER_BASE, "order_id": f"MAG-{family}-{group}-{cond_tag}",
             "product_id": product, "quantity": qty, "days_until_deadline": days}
    if manufacturer_eligibility_reason(a, order) is not None:
        return None
    rule = decide_manufacturer_bid(a, order)
    completion = (workload + qty) / cap
    sysp, usrp = build_manufacturer_prompt(a, order)
    return {"order_id": order["order_id"], "agent_id": agent_tmpl["agent_id"],
            "stage": "tailoring", "family": family, "group": group, "cond": cond_tag,
            "deadline": days, "quantity": qty, "workload": workload, "capacity": cap,
            "completion": round(completion, 4), "realized_ratio": round(completion / days, 6),
            "rule_decision": rule["decision"],
            "system": sysp, "user": usrp,
            "completion_json": completion_json("tailoring", order["order_id"],
                                              agent_tmpl["agent_id"], rule,
                                              agent_tmpl.get("defect_probability"))}


def main():
    _, manufacturers, _, _, _ = load_all_data()
    templates = manufacturers.to_dict("records")[:4]  # a few real profiles for defect/cost fields
    rows = []

    # ---- Family A: absolute scale (uniform s on Q,W,C; ratio+completion fixed) ----
    for ti, tmpl in enumerate(templates):
        for ci, (days, ratio, side) in enumerate(CONDS):
            # base capacity mid-real; base totals from ratio*days*cap
            C0 = 800
            group = f"T{ti}c{ci}"
            for s in A_SCALES:
                cap = int(round(C0 * s))
                total = ratio * days * cap          # W+Q to hit exact ratio at this deadline
                qty = int(round(0.75 * total)); wl = int(round(total)) - qty
                if not (C_MIN <= cap <= C_MAX and Q_MIN <= qty <= Q_MAX and 0 <= wl <= W_MAX):
                    continue
                rec = build(tmpl, cap, wl, qty, days, group, "A_absolute_scale", f"s{int(s*100)}")
                if rec:
                    rows.append(rec)

    # ---- Family B: Q vs W decomposition (T, C, D, ratio, completion fixed) ----
    for ti, tmpl in enumerate(templates):
        for ci, (days, ratio, cap, side) in enumerate(B_CONDS):
            total = int(round(ratio * days * cap))
            group = f"T{ti}c{ci}"
            for f in B_FRACS:
                qty = int(round(f * total)); wl = total - qty
                if not (Q_MIN <= qty <= Q_MAX and 0 <= wl <= W_MAX):
                    continue
                rec = build(tmpl, cap, wl, qty, days, group, "B_qty_workload_split", f"q{int(f*100)}")
                if rec:
                    rows.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()

    # per-group rule-decision must be constant (invariance is well-formed)
    grp = defaultdict(set)
    for r in rows:
        grp[(r["family"], r["group"])].add(r["rule_decision"])
    nonconst = {k: sorted(v) for k, v in grp.items() if len(v) > 1}

    qA = [r["quantity"] for r in rows if r["family"] == "A_absolute_scale"]
    cA = [r["capacity"] for r in rows if r["family"] == "A_absolute_scale"]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "explanatory magnitude/decomposition diagnostic; NOT training, NOT a gate",
        "n": len(rows), "sha256": digest,
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "familyA_quantity_span": [min(qA), max(qA)] if qA else None,
        "familyA_capacity_span": [min(cA), max(cA)] if cA else None,
        "familyA_capacity_note": "capacity 500-1600 spans <=1.8x the real max 883 (under one order "
                                 "of magnitude); quantity kept within real [100,2000]",
        "invariant_by_construction": "within each (family,group) completion and ratio are fixed so "
                                     "the rule decision is constant; the model's decision constancy "
                                     "is what the diagnostic measures",
        "groups_with_nonconstant_rule_label": nonconst,
        "label_verification": "every row via decide_manufacturer_bid",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
