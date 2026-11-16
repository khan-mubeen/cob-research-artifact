"""consolidate the scale-vs-magnitude and JSON-format diagnostics across every
checkpoint that has been evaluated on them. reads eval_<tag>-diag_*.json.

outputs reports/epoch-1/production/diagnostics/diagnostics_results.json and prints:
- scale: decision accuracy per counterfactual family per checkpoint
- F2 focus (completion constant, deadline varies): BID rate by deadline, the
  clean test of deadline sensitivity vs magnitude-matching
- F3 focus (deadline+quantity fixed, capacity varies): BID rate by ratio, the
  test of whether the model reads the profile
- json: strict-parse validity by offer band and by stage
"""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "experiments"))
from report_paths import find_evals, report_dir  # noqa: E402

TAGS = sys.argv[1:] or [
    "qwen-base-diag", "qwen-r1-diag", "qwen-v2-diag",
    "granite-base-diag", "granite-v1-diag", "granite-v2-diag",
]


def load(tag):
    fs = [str(p) for p in find_evals(f"eval_{tag}_*.json")]
    return json.load(open(fs[-1], encoding="utf-8")) if fs else None


def main():
    out = {"checkpoints": {}}
    data = {t: load(t) for t in TAGS}
    data = {t: d for t, d in data.items() if d}

    fams = ["ratio_fixed_deadline_varies", "quantity_fixed_deadline_varies",
            "capacity_varies_dl_qty_fixed", "crossed_ratio_deadline"]

    for tag, d in data.items():
        c = {"scale_family_decision_acc": {}, "scale_family_json": {}}
        for fam in fams:
            rows = [r for r in d["scale_diagnostic"]["rows"] if r["stage"] == fam]
            if rows:
                c["scale_family_decision_acc"][fam] = round(
                    sum(r["decision_matched"] for r in rows) / len(rows) * 100, 1)
        # F2: BID rate by deadline (completion constant); rule flips near deadline 7
        f2 = defaultdict(lambda: [0, 0, 0])
        for r in [r for r in d["scale_diagnostic"]["rows"]
                  if r["stage"] == "quantity_fixed_deadline_varies"]:
            dl = int(r["id"].split("-d")[-1])
            f2[dl][0] += 1
            f2[dl][1] += (r["model_decision"] == "BID")
            f2[dl][2] += (r["label_decision"] == "BID")
        c["F2_bid_rate_by_deadline"] = {dl: {"model_bid": v[1], "rule_bid": v[2], "n": v[0]}
                                        for dl, v in sorted(f2.items())}
        f2_range = [v[1] for v in (f2[dl] for dl in sorted(f2))]
        c["F2_deadline_sensitive"] = (max(f2_range) - min(f2_range)) if f2_range else 0

        # F3: BID rate by capacity-driven ratio (deadline+quantity fixed)
        f3 = defaultdict(lambda: [0, 0, 0])
        for r in [r for r in d["scale_diagnostic"]["rows"]
                  if r["stage"] == "capacity_varies_dl_qty_fixed"]:
            tag3 = r["id"].split("-")[-1]  # r125,r111,r100,r091,r077
            f3[tag3][0] += 1
            f3[tag3][1] += (r["model_decision"] == "BID")
            f3[tag3][2] += (r["label_decision"] == "BID")
        c["F3_bid_rate_by_capacity_ratio"] = {k: {"model_bid": v[1], "rule_bid": v[2], "n": v[0]}
                                              for k, v in sorted(f3.items())}
        f3_range = [v[1] for v in (f3[k] for k in sorted(f3))]
        c["F3_profile_sensitive"] = (max(f3_range) - min(f3_range)) if f3_range else 0

        # JSON format: validity by band and by stage
        if "json_format_stress" in d:
            byband = defaultdict(lambda: [0, 0])
            bystage = defaultdict(lambda: [0, 0])
            for r in d["json_format_stress"]["rows"]:
                band = r["stage"]  # loader set stage=band
                byband[band][0] += 1
                byband[band][1] += r["json_valid"]
                st = "tailoring" if "-T-" in r["id"] else "logistics"
                bystage[st][0] += 1
                bystage[st][1] += r["json_valid"]
            c["json_valid_by_band"] = {b: round(v[1] / v[0] * 100, 1) for b, v in
                                       sorted(byband.items())}
            c["json_valid_by_stage"] = {s: round(v[1] / v[0] * 100, 1) for s, v in
                                        sorted(bystage.items())}
            c["json_malformed_examples"] = [
                (r["id"], r["raw"][:90].replace("\n", " "))
                for r in d["json_format_stress"]["rows"] if not r["json_valid"]][:5]
        out["checkpoints"][tag] = c

    with open(report_dir("production", "diagnostics") / "diagnostics_results.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== scale decision accuracy by family ===")
    print(f"  {'checkpoint':20s} " + "".join(f"{f[:12]:>14s}" for f in fams))
    for tag, c in out["checkpoints"].items():
        a = c["scale_family_decision_acc"]
        print(f"  {tag:20s} " + "".join(f"{a.get(f, 0):13.0f}%" for f in fams))

    print("\n=== F2 deadline-sensitivity (spread of BID count across deadlines 2..14; "
          "high=tracks deadline, ~0=ignores it) and F3 profile-sensitivity ===")
    for tag, c in out["checkpoints"].items():
        print(f"  {tag:20s} F2_spread={c['F2_deadline_sensitive']:2d}/12  "
              f"F3_spread={c['F3_profile_sensitive']:2d}/12")

    print("\n=== JSON validity by offer band ===")
    bands = ["under_1k", "1k_to_10k", "10k_to_100k", "over_100k"]
    print(f"  {'checkpoint':20s} " + "".join(f"{b:>13s}" for b in bands) + "   by_stage")
    for tag, c in out["checkpoints"].items():
        bb = c.get("json_valid_by_band", {})
        line = f"  {tag:20s} " + "".join(f"{bb.get(b, float('nan')):12.0f}%" for b in bands)
        print(line + f"   {c.get('json_valid_by_stage', {})}")


if __name__ == "__main__":
    main()
