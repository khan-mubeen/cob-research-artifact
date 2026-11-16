"""
consolidate the qwen3.5-4b experiment into machine-readable summary tables.

reads only saved evaluation artifacts, never reruns inference:
- eval_base-audit_*.json          base model, all six sets
- eval_finetuned-r1-audit_*.json  round one, all six sets
- eval_finetuned-r2-finalb_*.json round two, all six sets
- eval_finetuned-r3_*.json        round three, all six sets
- guard_audit.json                per-class confusions incl. guard b arms

arms emitted:
- rule_based:      reference arm, labels on every set ARE the rule outputs,
                   so agreement is 1.0 and deviation 0.0 by construction
- base:            un-fine-tuned qwen3.5-4b
- ft_r1:           round one, selected pure fine-tuned checkpoint
- ft_r1_guard_b:   hybrid engineering arm, round one + authoritative guard,
                   never a substitute for the pure result
- ft_r2 / ft_r3:   boundary-focused experimental variants, kept for the record

outputs (never overwriting raw reports):
- reports/epoch-1/ablation/reports/qwen_comparison_summary.json
- reports/epoch-1/ablation/reports/qwen_comparison_summary.csv
"""

import csv
import glob
import json
from datetime import datetime

from config import DATA_VERSION, PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "reports" / "epoch-1" / "ablation" / "reports"

EVAL_FILES = {
    "base": "eval_base-audit_*.json",
    "ft_r1": "eval_finetuned-r1-audit_*.json",
    "ft_r2": "eval_finetuned-r2-finalb_*.json",
    "ft_r3": "eval_finetuned-r3_*.json",
}
GUARD_MODEL_KEY = {"ft_r1": "r1", "ft_r2": "r2", "ft_r3": "r3", "base": "base"}
SETS = ("baseline_six", "edge_holdout", "main_holdout", "boundary_stress", "final_boundary_holdout")

# documented latency measurements with their sources, no new benchmarks here
LATENCY_NOTES = {
    "rule_based": "sub-millisecond per decision, pure python formulas (bid_check sweeps 50k orders in minutes)",
    "gguf_q8_base": "1.05-1.5 s/decision warm, 85.2 gen tok/s over 829 calls (slm_check 2026-07-15, RTX 5090, full offload)",
    "hf_bf16_eval": "~1.9-2.0 examples/s at generation batch 16 (audit evals 2026-07-18/19, same gpu)",
    "note": "fine-tuned adapters share the base architecture, per-call latency is equivalent; gguf latency for the fine-tuned checkpoint pends re-quantization",
}


def newest(pattern):
    matches = sorted(glob.glob(str(REPORT_DIR / pattern)))
    if not matches:
        raise SystemExit(f"missing artifact: {pattern}")
    return matches[-1]


def confusion_from_rows(rows):
    c = {"BID/BID": 0, "BID/NO_BID": 0, "NO_BID/BID": 0, "NO_BID/NO_BID": 0}
    for r in rows:
        pred = r["model_decision"] if r["model_decision"] in ("BID", "NO_BID") else "NO_BID"
        c[f"{r['label_decision']}/{pred}"] += 1
    return c


def arm_from_eval(path):
    report = json.load(open(path, encoding="utf-8"))
    arm = {"source": path.split("\\")[-1]}
    for set_name in SETS:
        if set_name not in report:
            continue
        arm[set_name] = {}
        rows = report[set_name]["rows"]
        by_class = {}
        for r in rows:
            by_class.setdefault(r["stage"], []).append(r)
        for cls, cls_rows in sorted(by_class.items()):
            summary = report[set_name]["summary"].get(cls, {})
            c = confusion_from_rows(cls_rows)
            arm[set_name][cls] = {
                "n": len(cls_rows),
                "decision_agreement_pct": summary.get("decision_agreement_pct"),
                "json_valid_pct": summary.get("json_valid_pct"),
                "false_bid": c["NO_BID/BID"],
                "false_no_bid": c["BID/NO_BID"],
                "mean_abs_offer_dev_pct": summary.get("mean_abs_offer_dev_pct"),
                "median_abs_offer_dev_pct": summary.get("median_abs_offer_dev_pct"),
                "days_exact_match_pct": summary.get("days_exact_match_pct"),
            }
    return arm


def guard_b_arm(base_arm, guard_audit, model_key):
    """same offers and json validity as the pure arm, decisions replaced by
    the guard_b confusions from the audit"""
    g = guard_audit[model_key]
    arm = {"source": f"guard_audit.json[{model_key}].guard_b + {base_arm['source']} offers",
           "hybrid": True}
    for set_name in SETS:
        if set_name not in base_arm or set_name not in g:
            continue
        arm[set_name] = {}
        for cls, pure in base_arm[set_name].items():
            gb = g[set_name][cls]["guard_b"]
            n = pure["n"]
            correct = gb["BID/BID"] + gb["NO_BID/NO_BID"]
            arm[set_name][cls] = {
                **pure,
                "decision_agreement_pct": round(correct / n * 100, 2),
                "false_bid": gb["NO_BID/BID"],
                "false_no_bid": gb["BID/NO_BID"],
                "guard_b_conversions": g[set_name][cls]["b_converted"],
            }
    return arm


def rule_based_arm(reference_arm):
    """labels are the rule outputs, so the rule arm is the identity reference"""
    arm = {"source": "labels are rule-based outputs by construction", "reference": True}
    for set_name in SETS:
        if set_name not in reference_arm:
            continue
        arm[set_name] = {}
        for cls, pure in reference_arm[set_name].items():
            arm[set_name][cls] = {
                "n": pure["n"], "decision_agreement_pct": 100.0, "json_valid_pct": 100.0,
                "false_bid": 0, "false_no_bid": 0,
                "mean_abs_offer_dev_pct": 0.0, "median_abs_offer_dev_pct": 0.0,
                "days_exact_match_pct": 100.0,
            }
    return arm


def main():
    guard_audit = json.load(open(REPORT_DIR / "guard_audit.json", encoding="utf-8"))

    arms = {}
    for name, pattern in EVAL_FILES.items():
        arms[name] = arm_from_eval(newest(pattern))
    arms["rule_based"] = rule_based_arm(arms["ft_r1"])
    arms["ft_r1_guard_b"] = guard_b_arm(arms["ft_r1"], guard_audit, "r1")
    arms["ft_r3_guard_b_context"] = guard_b_arm(arms["ft_r3"], guard_audit, "r3")

    # checkpoint-selection verification, printed from artifacts
    print("checkpoint selection verification (main_holdout, from saved artifacts):")
    print(f"{'model':6s} {'stage':10s} {'decisions%':>10s} {'mean dev%':>9s} {'median dev%':>11s} {'days%':>7s}")
    for model in ("ft_r1", "ft_r2", "ft_r3"):
        for stage, m in arms[model]["main_holdout"].items():
            print(f"{model:6s} {stage:10s} {m['decision_agreement_pct']:10.2f} "
                  f"{m['mean_abs_offer_dev_pct']:9.2f} {m['median_abs_offer_dev_pct']:11.2f} "
                  f"{m['days_exact_match_pct']:7.2f}")
    print("\nbaseline six correct (NO_BID) per model:")
    for model in ("base", "ft_r1", "ft_r2", "ft_r3"):
        six = arms[model]["baseline_six"]
        correct = sum(v["n"] - v["false_bid"] for v in six.values())
        print(f"  {model:6s} {correct}/6")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_version": DATA_VERSION,
        "selected_pure_checkpoint": "ft_r1 (models/epoch-1/production/qwen3.5-4b-bidding-lora-r1/final)",
        "hybrid_engineering_config": "ft_r1 + authoritative dynamic-feasibility guard (Guard B), offline audit only, not implemented in the live path",
        "latency_notes": LATENCY_NOTES,
        "arms": arms,
    }
    json_path = REPORT_DIR / "qwen_comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = REPORT_DIR / "qwen_comparison_summary.csv"
    fields = ["arm", "set", "class", "n", "decision_agreement_pct", "false_bid", "false_no_bid",
              "json_valid_pct", "mean_abs_offer_dev_pct", "median_abs_offer_dev_pct",
              "days_exact_match_pct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for arm_name, arm in arms.items():
            for set_name in SETS:
                for cls, m in arm.get(set_name, {}).items():
                    writer.writerow({"arm": arm_name, "set": set_name, "class": cls,
                                     **{k: m.get(k) for k in fields[3:]}})

    print(f"\nwritten: {json_path}")
    print(f"written: {csv_path}")


if __name__ == "__main__":
    main()
