"""consolidate granite round-one results against the pre-registered protocol
criteria and the qwen reference numbers, from saved eval artifacts only.

outputs reports/epoch-1/ablation/reports/granite_round1_summary.json with:
- per-set/per-stage metrics for granite-base, granite-ft, qwen-base, qwen-r1
- edge_holdout reported all-12 AND clean-10 (contamination disclosure)
- final boundary per class (never aggregated)
- the five pre-registered criteria evaluated for granite-ft with PASS/FAIL
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


INPUTS = {
    "granite_base": _resolve("eval_granite-base_20260719_170737.json"),
    "granite_ft": _resolve("eval_granite-ft_20260719_172254.json"),
    "qwen_base": _resolve("eval_base-audit_20260718_235632.json"),
    "qwen_r1": _resolve("eval_finetuned-r1-audit_20260719_001029.json"),
}
CONTAMINATED = {"SYNEDGE-T-INF-T2/T2", "SYNEDGE-T-INF-T11/T11"}


def rows_summary(rows):
    n = len(rows)
    match = sum(r["decision_matched"] for r in rows)
    jv = sum(r["json_valid"] for r in rows)
    devs = [r["offer_pct_dev"] for r in rows if r.get("offer_pct_dev") is not None]
    return {"n": n, "decisions_pct": round(match / n * 100, 2),
            "json_pct": round(jv / n * 100, 2),
            "false_bid": sum(1 for r in rows if r["label_decision"] == "NO_BID"
                             and r["model_decision"] == "BID"),
            "false_nobid": sum(1 for r in rows if r["label_decision"] == "BID"
                               and r["model_decision"] != "BID"),
            "mean_offer_dev_pct": round(sum(devs) / len(devs), 3) if devs else None}


def main():
    data = {}
    for tag, path in INPUTS.items():
        with open(path, encoding="utf-8") as f:
            data[tag] = json.load(f)

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "inputs": {k: str(v) for k, v in INPUTS.items()}, "models": {}}

    for tag, d in data.items():
        m = {"baseline_six_no_bid": sum(
                r["model_decision"] == "NO_BID" for r in d["baseline_six"]["rows"]),
             "baseline_six_rows": {r["id"]: r["model_decision"]
                                   for r in d["baseline_six"]["rows"]},
             "main_holdout_per_stage": d["main_holdout"]["summary"],
             "edge_holdout_all12": rows_summary(d["edge_holdout"]["rows"]),
             "edge_holdout_clean10": rows_summary(
                 [r for r in d["edge_holdout"]["rows"] if r["id"] not in CONTAMINATED]),
             "boundary_stress_context_only": d["boundary_stress"]["summary"],
             "final_boundary_per_class": {}}
        fb = d["final_boundary_holdout"]["rows"]
        for cls in sorted({r["stage"] for r in fb}):
            cr = [r for r in fb if r["stage"] == cls]
            m["final_boundary_per_class"][cls] = {
                "correct": sum(r["decision_matched"] for r in cr), "of": len(cr)}
        json_all = all(
            all(r["json_valid"] for r in d[s]["rows"])
            for s in ("baseline_six", "edge_holdout", "main_holdout",
                      "boundary_stress", "final_boundary_holdout"))
        m["json_100_all_sets"] = json_all
        out["models"][tag] = m

    gb = out["models"]["granite_base"]
    gf = out["models"]["granite_ft"]
    crit = {}
    crit["1_baseline_six_6of6"] = {
        "value": f"{gf['baseline_six_no_bid']}/6",
        "verdict": "PASS" if gf["baseline_six_no_bid"] == 6 else "FAIL"}
    reg = {}
    for stage in ("material", "tailoring", "logistics"):
        base_pct = gb["main_holdout_per_stage"][stage]["decision_agreement_pct"]
        ft_pct = gf["main_holdout_per_stage"][stage]["decision_agreement_pct"]
        reg[stage] = {"base": base_pct, "ft": ft_pct, "ok": ft_pct >= base_pct}
    crit["2_no_decision_regression_vs_own_base"] = {
        "value": reg, "verdict": "PASS" if all(v["ok"] for v in reg.values()) else "FAIL"}
    imp = {}
    for stage in ("material", "tailoring", "logistics"):
        b = gb["main_holdout_per_stage"][stage]["mean_abs_offer_dev_pct"]
        f = gf["main_holdout_per_stage"][stage]["mean_abs_offer_dev_pct"]
        imp[stage] = {"base": b, "ft": f, "ok": (f is not None and b is not None and f < b)}
    crit["3_offer_dev_improvement_vs_own_base"] = {
        "value": imp, "verdict": "PASS" if all(v["ok"] for v in imp.values()) else "FAIL"}
    crit["4_json_validity_100_all_sets"] = {
        "value": gf["json_100_all_sets"],
        "verdict": "PASS" if gf["json_100_all_sets"] else "FAIL"}
    fb_ok = all(v["correct"] >= 9 for v in gf["final_boundary_per_class"].values())
    crit["5_final_boundary_9of12_each_class"] = {
        "value": {k: f"{v['correct']}/{v['of']}"
                  for k, v in gf["final_boundary_per_class"].items()},
        "verdict": "PASS" if fb_ok else "FAIL"}
    out["granite_ft_pre_registered_criteria"] = crit
    out["overall"] = ("PASS" if all(c["verdict"] == "PASS" for c in crit.values())
                      else "FAIL on: " + ", ".join(
                          k for k, c in crit.items() if c["verdict"] != "PASS"))

    dst = report_dir("ablation", "reports") / "granite_round1_summary.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["granite_ft_pre_registered_criteria"], indent=2))
    print("\noverall:", out["overall"])
    print("\nfinal boundary per class, all models:")
    for tag in INPUTS:
        print(f"  {tag:14s}",
              {k: f"{v['correct']}/{v['of']}"
               for k, v in out["models"][tag]["final_boundary_per_class"].items()})
    print("\nbaseline six:", {t: f"{out['models'][t]['baseline_six_no_bid']}/6" for t in INPUTS})
    print(f"written {dst}")


if __name__ == "__main__":
    main()
