"""three-way comparison of the round-one fine-tuned qwen across serving forms:
native (base+adapter, hf runtime), merged (fused checkpoint, hf runtime),
quantized (Q8_0 gguf, llama-cpp-python 0.3.34).

inputs are eval json files produced by eval_finetune.py / eval_gguf.py on the
same frozen sets. reports per set and stage: decision agreement, json
validity, offer deviation, days match, false BID / false NO_BID counts, and
the edge_holdout metric both on all 12 rows and on the clean 10 (two rows are
verbatim in the round-one edge training set, found 2026-07-19, ids below).

usage:
  python src/experiments/gguf_fix/compare_configs.py \
      --native <eval json> --merged <eval json> --quantized <eval json>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "reports" / "epoch-1" / "deployment" / "qwen-r1-gguf-fix"

# verbatim overlap with round1_edge training prompts (train == holdout rows)
CONTAMINATED_EDGE_IDS = {"SYNEDGE-T-INF-T2/T2", "SYNEDGE-T-INF-T11/T11"}

SETS = ("baseline_six", "edge_holdout", "main_holdout",
        "boundary_stress", "final_boundary_holdout")


def summarize_rows(rows):
    n = len(rows)
    if not n:
        return None
    match = sum(r["decision_matched"] for r in rows)
    jv = sum(r["json_valid"] for r in rows)
    false_bid = sum(1 for r in rows
                    if r["label_decision"] == "NO_BID" and r["model_decision"] == "BID")
    false_nobid = sum(1 for r in rows
                      if r["label_decision"] == "BID" and r["model_decision"] != "BID")
    devs = [r["offer_pct_dev"] for r in rows if r.get("offer_pct_dev") is not None]
    days = [r for r in rows if "days_matched" in r]
    return {
        "n": n,
        "decision_agreement_pct": round(match / n * 100, 2),
        "json_valid_pct": round(jv / n * 100, 2),
        "false_bid": false_bid,
        "false_nobid": false_nobid,
        "mean_abs_offer_dev_pct": round(sum(devs) / len(devs), 3) if devs else None,
        "days_exact_match_pct": round(
            sum(r["days_matched"] for r in days) / len(days) * 100, 2) if days else None,
    }


def per_class(rows):
    classes = sorted({r["stage"] for r in rows})
    return {c: summarize_rows([r for r in rows if r["stage"] == c]) for c in classes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--quantized", required=True)
    args = parser.parse_args()

    configs = {}
    for name, path in (("native", args.native), ("merged", args.merged),
                       ("quantized", args.quantized)):
        with open(path, encoding="utf-8") as f:
            configs[name] = {"path": path, "data": json.load(f)}

    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "inputs": {k: v["path"] for k, v in configs.items()},
              "contaminated_edge_ids_excluded_variant": sorted(CONTAMINATED_EDGE_IDS),
              "sets": {}}

    for set_name in SETS:
        entry = {}
        for cfg, blob in configs.items():
            data = blob["data"]
            if set_name not in data:
                entry[cfg] = None
                continue
            rows = data[set_name]["rows"]
            e = {"overall": summarize_rows(rows)}
            if set_name == "edge_holdout":
                clean = [r for r in rows if r["id"] not in CONTAMINATED_EDGE_IDS]
                e["clean_10"] = summarize_rows(clean)
            if set_name in ("boundary_stress", "final_boundary_holdout"):
                e["per_class"] = per_class(rows)
                e["per_class_correct"] = {
                    c: f"{sum(r['decision_matched'] for r in rows if r['stage'] == c)}"
                       f"/{sum(1 for r in rows if r['stage'] == c)}"
                    for c in sorted({r["stage"] for r in rows})}
            if set_name == "baseline_six":
                e["no_bid_correct"] = sum(
                    r["model_decision"] == "NO_BID" for r in rows)
            entry[cfg] = e
        report["sets"][set_name] = entry

    # decision flips between configs, row-aligned by id, per set
    flips = {}
    for set_name in SETS:
        rows_by = {}
        for cfg, blob in configs.items():
            data = blob["data"]
            if set_name in data:
                rows_by[cfg] = {r["id"]: r["model_decision"]
                                for r in data[set_name]["rows"]}
        if len(rows_by) == 3:
            ids = set(rows_by["native"]) & set(rows_by["merged"]) & set(rows_by["quantized"])
            flips[set_name] = {
                "native_vs_merged": sorted(i for i in ids
                                           if rows_by["native"][i] != rows_by["merged"][i]),
                "merged_vs_quantized": sorted(i for i in ids
                                              if rows_by["merged"][i] != rows_by["quantized"][i]),
                "native_vs_quantized": sorted(i for i in ids
                                              if rows_by["native"][i] != rows_by["quantized"][i]),
            }
    report["decision_flips"] = flips

    for cfg, blob in configs.items():
        data = blob["data"]
        lat = data.get("overall_latency")
        if lat:
            report.setdefault("latency", {})[cfg] = lat

    out = OUT_DIR / "native_merged_quantized_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "sets"}, indent=2))
    for set_name, entry in report["sets"].items():
        print(f"\n== {set_name} ==")
        for cfg, e in entry.items():
            if e:
                print(f"  {cfg:10s} {json.dumps(e['overall'])}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
