"""protocol v2 required comparison: base / v1-FT / v2-FT for qwen and granite,
plus qwen-v2 vs granite-v2 under the identical frozen protocol.

reads only saved eval artifacts; robustness-set rows for base/v1 checkpoints
come from the --only-set runs. writes protocol_v2_results.json and prints the
comparison tables + pre-registered criteria verdicts.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "experiments"))
from report_paths import find_evals, report_dir  # noqa: E402

def _resolve(filename):
    """exact-filename lookup across every post-reorganization raw-eval root"""
    hits = find_evals(filename)
    if not hits:
        raise FileNotFoundError(f"{filename} not found under any eval root")
    return hits[-1]

CONTAMINATED = {"SYNEDGE-T-INF-T2/T2", "SYNEDGE-T-INF-T11/T11"}

INPUTS = {
    "qwen_base":   {"full": "eval_base-audit_20260718_235632.json",
                    "robust": "eval_qwen-base-robust_20260720_031455.json"},
    "qwen_r1_v1":  {"full": "eval_finetuned-r1-audit_20260719_001029.json",
                    "robust": "eval_qwen-r1-robust_20260720_031545.json"},
    "qwen_v2":     {"full": "eval_qwen-v2_20260720_025436.json"},
    "granite_base": {"full": "eval_granite-base_20260719_170737.json",
                     "robust": "eval_granite-base-robust_20260720_031624.json"},
    "granite_v1":  {"full": "eval_granite-ft_20260719_172254.json",
                    "robust": "eval_granite-v1-robust_20260720_031710.json"},
    "granite_v2":  {"full": "eval_granite-v2_20260720_031152.json"},
}


def rs(rows):
    n = len(rows)
    devs = [r["offer_pct_dev"] for r in rows if r.get("offer_pct_dev") is not None]
    days = [r for r in rows if "days_matched" in r]
    return {"n": n,
            "decisions_pct": round(sum(r["decision_matched"] for r in rows) / n * 100, 2),
            "json_pct": round(sum(r["json_valid"] for r in rows) / n * 100, 2),
            "false_bid": sum(1 for r in rows if r["label_decision"] == "NO_BID"
                             and r["model_decision"] == "BID"),
            "false_nobid": sum(1 for r in rows if r["label_decision"] == "BID"
                               and r["model_decision"] != "BID"),
            "mean_offer_dev_pct": round(sum(devs) / len(devs), 3) if devs else None,
            "days_exact_pct": round(sum(r["days_matched"] for r in days) / len(days) * 100, 2)
                              if days else None}


def per_class(rows):
    return {c: {"correct": sum(r["decision_matched"] for r in rows if r["stage"] == c),
                "of": sum(1 for r in rows if r["stage"] == c),
                **{k: v for k, v in rs([r for r in rows if r["stage"] == c]).items()
                   if k in ("false_bid", "false_nobid", "json_pct")}}
            for c in sorted({r["stage"] for r in rows})}


def main():
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "inputs": {}, "models": {}}
    for tag, files in INPUTS.items():
        full = json.load(open(_resolve(files["full"]), encoding="utf-8"))
        robust_src = full
        if "robust" in files:
            robust_src = json.load(open(_resolve(files["robust"]), encoding="utf-8"))
        out["inputs"][tag] = files
        m = {
            "baseline_six": f"{sum(r['model_decision'] == 'NO_BID' for r in full['baseline_six']['rows'])}/6",
            "main_holdout": {st: rs([r for r in full["main_holdout"]["rows"] if r["stage"] == st])
                             for st in ("material", "tailoring", "logistics")},
            "edge_holdout_all12": rs(full["edge_holdout"]["rows"]),
            "edge_holdout_clean10": rs([r for r in full["edge_holdout"]["rows"]
                                        if r["id"] not in CONTAMINATED]),
            "final_boundary_per_class": per_class(full["final_boundary_holdout"]["rows"]),
            "boundary_stress_per_class": per_class(full["boundary_stress"]["rows"]),
        }
        if "robustness_holdout_v2" in robust_src:
            m["robustness_v2_per_class"] = per_class(robust_src["robustness_holdout_v2"]["rows"])
        malformed = [(s, r["id"]) for s in ("baseline_six", "edge_holdout", "main_holdout",
                                            "boundary_stress", "final_boundary_holdout",
                                            "robustness_holdout_v2")
                     if s in full for r in full[s]["rows"] if not r["json_valid"]]
        m["malformed_total_full_eval"] = len(malformed)
        out["models"][tag] = m

    # pre-registered v2 criteria for the two v2 runs
    for model, base in (("qwen_v2", "qwen_base"), ("granite_v2", "granite_base")):
        mm, bb = out["models"][model], out["models"][base]
        c = {}
        c["1_json_100_all_sets"] = {"malformed": mm["malformed_total_full_eval"],
                                    "verdict": "PASS" if mm["malformed_total_full_eval"] == 0 else "FAIL"}
        reg = {st: {"base": bb["main_holdout"][st]["decisions_pct"],
                    "v2": mm["main_holdout"][st]["decisions_pct"]}
               for st in ("material", "tailoring", "logistics")}
        c["2_no_decision_regression"] = {"value": reg,
            "verdict": "PASS" if all(v["v2"] >= v["base"] for v in reg.values()) else "FAIL"}
        imp = {st: {"base": bb["main_holdout"][st]["mean_offer_dev_pct"],
                    "v2": mm["main_holdout"][st]["mean_offer_dev_pct"]}
               for st in ("material", "tailoring", "logistics")}
        c["3_offer_improvement"] = {"value": imp,
            "verdict": "PASS" if all(v["v2"] is not None and v["base"] is not None
                                     and v["v2"] < v["base"] for v in imp.values()) else "FAIL"}
        rc = mm["robustness_v2_per_class"]
        c["4_robustness_9of12_each_class"] = {
            "value": {k: f"{v['correct']}/{v['of']}" for k, v in rc.items()},
            "verdict": "PASS" if all(v["correct"] >= 9 for v in rc.values()) else "FAIL"}
        out[f"{model}_criteria"] = c
        out[f"{model}_overall"] = ("PASS" if all(x["verdict"] == "PASS" for x in c.values())
                                   else "FAIL on: " + ", ".join(k for k, x in c.items()
                                                                if x["verdict"] != "PASS"))

    dst = report_dir("ablation", "reports") / "protocol_v2_results.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("robustness_holdout_v2 per class (correct/12):")
    for tag, m in out["models"].items():
        rc = m.get("robustness_v2_per_class", {})
        print(f"  {tag:13s}", {k: f"{v['correct']}/{v['of']}" for k, v in rc.items()},
              "| baseline_six", m["baseline_six"],
              "| malformed", m["malformed_total_full_eval"])
    print("\nfinal_boundary per class (days 3-6):")
    for tag, m in out["models"].items():
        print(f"  {tag:13s}", {k: f"{v['correct']}/{v['of']}"
                               for k, v in m["final_boundary_per_class"].items()})
    for model in ("qwen_v2", "granite_v2"):
        print(f"\n{model}: {out[f'{model}_overall']}")
        for k, v in out[f"{model}_criteria"].items():
            print(f"  {k}: {v['verdict']}")
    print(f"\nwritten {dst}")


if __name__ == "__main__":
    main()
