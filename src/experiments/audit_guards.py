"""
offline audit of deterministic post-decision guards, no live-path changes.

two guards, evaluated by post-processing saved eval outputs, inference is
never rerun here and model weights, prompts, decoding are untouched:

- guard A, self-consistency: when the model answered BID, parse its own
  estimated_completion_days from the saved raw output and convert to NO_BID
  when that reported value strictly exceeds the rfq deadline. uses the
  unrounded parsed value, so a model that rounds 3.09 down to 3 against a
  3-day deadline slips through, that miss is part of what is measured.
- guard B, authoritative recheck: when the model answered BID, recompute the
  dynamic constraint with the production decide_* formulas and convert to
  NO_BID when the rule path says NO_BID. on every audit set the stored label
  IS the rule decision (asserted at generation time), so the label is used
  as the rule verdict. this arm is a hybrid symbolic-SLM configuration and
  must be reported as such, never as pure SLM performance.

neither guard can recover a false NO_BID, there is no offer to publish.

usage:
    python src/experiments/audit_guards.py --scan-band   # real-data exposure
    python src/experiments/audit_guards.py --guards      # guard analysis
"""

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict

from config import DATA_VERSION, PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "reports" / "epoch-1" / "ablation" / "evaluation" / "raw"

# newest full-coverage eval json per model, all six sets each
MODEL_REPORTS = {
    "base": "eval_base-audit_*.json",
    "r1": "eval_finetuned-r1-audit_*.json",
    "r2": "eval_finetuned-r2-finalb_*.json",
    "r3": "eval_finetuned-r3_*.json",
}

SETS = ("baseline_six", "edge_holdout", "main_holdout", "boundary_stress", "final_boundary_holdout")


# ── item 2: exhaustive real-data boundary exposure ───────────────────────────

def scan_band():
    import pandas as pd
    from src.common.bidding_formulas import manufacturer_eligibility_reason
    from src.visualization.data_utils import load_all_data
    from src.visualization.slm_competition_diagnostics import build_stage_orders

    materials, manufacturers, logistics, products, orders = load_all_data()
    agents = manufacturers.to_dict("records")

    buckets = {"below_0.97": 0, "0.97_to_1.00": 0, "exactly_1.00": 0, "1.00_to_1.03": 0, "above_1.03": 0}
    closest_below, closest_above = 0.0, None
    total_pairs = 0
    tol = 1e-9

    for _, order in orders.iterrows():
        stage_orders = build_stage_orders(order, products)
        if stage_orders is None:
            continue
        so = stage_orders["tailoring"]
        days = so["days_until_deadline"]
        qty = so["quantity"]
        for agent in agents:
            if manufacturer_eligibility_reason(agent, so) is not None:
                continue
            total_pairs += 1
            ratio = (agent["current_workload_units"] + qty) / agent["daily_capacity_units"] / days
            if abs(ratio - 1.0) <= tol:
                buckets["exactly_1.00"] += 1
            elif ratio < 0.97:
                buckets["below_0.97"] += 1
            elif ratio < 1.0:
                buckets["0.97_to_1.00"] += 1
            elif ratio <= 1.03:
                buckets["1.00_to_1.03"] += 1
            else:
                buckets["above_1.03"] += 1
            if ratio < 1.0 - tol and ratio > closest_below:
                closest_below = ratio
            if ratio > 1.0 + tol and (closest_above is None or ratio < closest_above):
                closest_above = ratio

    print(f"statically eligible manufacturer-order pairs over all {len(orders)} orders: {total_pairs}")
    print(f"tolerance for 'exactly 1.00': {tol}")
    for k, v in buckets.items():
        print(f"  {k:14s} {v:8d}  ({v / total_pairs * 100:.4f}%)")
    print(f"closest feasible ratio below 1.00:   {closest_below:.6f}")
    print(f"closest infeasible ratio above 1.00: "
          f"{closest_above if closest_above is not None else 'none exists in the data'}")


# ── guard analysis ────────────────────────────────────────────────────────────

def load_examples():
    """same set construction as eval_finetune, deadline parsed per example"""
    from src.experiments.eval_finetune import load_eval_sets
    sets = load_eval_sets()
    for examples in sets.values():
        for e in examples:
            e["deadline"] = float(re.search(r"^days until deadline: (\S+)", e["user"], re.M).group(1))
    return sets


def parse_reported_days(raw):
    m = re.search(r'"estimated_completion_days":\s*([0-9.eE+-]+)', raw)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def apply_guards(example, row):
    """returns dict of decisions under no_guard / guard_a / guard_b plus
    guard bookkeeping for one evaluated example"""
    model_decision = row["model_decision"] if row["model_decision"] in ("BID", "NO_BID") else "NO_BID"
    label = row["label_decision"]

    out = {"no_guard": model_decision, "guard_a": model_decision, "guard_b": model_decision,
           "a_converted": False, "b_converted": False, "a_unparseable": False}
    if model_decision != "BID":
        return out

    reported = parse_reported_days(row["raw"])
    if reported is None:
        out["a_unparseable"] = True
    elif reported > example["deadline"]:
        out["guard_a"] = "NO_BID"
        out["a_converted"] = True

    # label is the rule verdict on every audit set, asserted at generation
    if label == "NO_BID":
        out["guard_b"] = "NO_BID"
        out["b_converted"] = True
    return out


def confusion(rows_out, labels):
    c = {"BID/BID": 0, "BID/NO_BID": 0, "NO_BID/BID": 0, "NO_BID/NO_BID": 0}
    for decision, label in zip(rows_out, labels):
        c[f"{label}/{decision}"] += 1
    return c


def analyze():
    sets = load_examples()
    audit = {}

    for model, pattern in MODEL_REPORTS.items():
        matches = sorted(glob.glob(str(REPORT_DIR / pattern)))
        if not matches:
            print(f"!! no eval report found for {model} ({pattern}), skipping")
            continue
        report = json.load(open(matches[-1], encoding="utf-8"))
        audit[model] = {"source": matches[-1].split("\\")[-1]}

        for set_name in SETS:
            if set_name not in report or set_name not in sets:
                continue
            examples = {e["id"]: e for e in sets[set_name]}
            rows = report[set_name]["rows"]

            per_class = defaultdict(lambda: {"labels": [], "decisions": {"no_guard": [], "guard_a": [], "guard_b": []},
                                             "a_conv": 0, "b_conv": 0, "a_new_false_nobid": 0,
                                             "ab_disagree": 0, "a_unparseable": 0, "devs": [],
                                             "json_valid": 0, "n": 0})
            for row in rows:
                e = examples.get(row["id"])
                if e is None:
                    continue
                g = apply_guards(e, row)
                cls = row["stage"]
                bucket = per_class[cls]
                bucket["n"] += 1
                bucket["json_valid"] += bool(row["json_valid"])
                bucket["labels"].append(row["label_decision"])
                for guard in ("no_guard", "guard_a", "guard_b"):
                    bucket["decisions"][guard].append(g[guard])
                bucket["a_conv"] += g["a_converted"]
                bucket["b_conv"] += g["b_converted"]
                bucket["a_unparseable"] += g["a_unparseable"]
                if g["a_converted"] != g["b_converted"]:
                    bucket["ab_disagree"] += 1
                if g["a_converted"] and not g["b_converted"]:
                    bucket["a_new_false_nobid"] += 1
                if row.get("offer_pct_dev") is not None and row["label_decision"] == "BID" \
                        and g["no_guard"] == "BID":
                    bucket["devs"].append(row["offer_pct_dev"])

            audit[model][set_name] = {}
            for cls, bucket in sorted(per_class.items()):
                entry = {"n": bucket["n"], "json_valid": bucket["json_valid"],
                         "a_converted": bucket["a_conv"], "b_converted": bucket["b_conv"],
                         "a_vs_b_disagree": bucket["ab_disagree"],
                         "a_new_false_nobid": bucket["a_new_false_nobid"],
                         "a_unparseable": bucket["a_unparseable"],
                         "mean_offer_dev": round(statistics.mean(bucket["devs"]), 2) if bucket["devs"] else None}
                for guard in ("no_guard", "guard_a", "guard_b"):
                    entry[guard] = confusion(bucket["decisions"][guard], bucket["labels"])
                audit[model][set_name][cls] = entry

    out_path = REPORT_DIR / "guard_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # printed summary: per model per set, aggregated over classes, plus the
    # final-boundary per-class bar check
    for model, data in audit.items():
        print(f"\n================ {model}  (source: {data['source']}) ================")
        for set_name in SETS:
            if set_name not in data:
                continue
            agg = defaultdict(lambda: defaultdict(int))
            for cls, entry in data[set_name].items():
                for guard in ("no_guard", "guard_a", "guard_b"):
                    for k, v in entry[guard].items():
                        agg[guard][k] += v
            line = " | ".join(
                f"{g}: tb->b {agg[g]['BID/BID']}, tb->n {agg[g]['BID/NO_BID']}, "
                f"tn->b {agg[g]['NO_BID/BID']}, tn->n {agg[g]['NO_BID/NO_BID']}"
                for g in ("no_guard", "guard_a", "guard_b"))
            print(f"{set_name:24s} {line}")

        if "final_boundary_holdout" in data:
            print("  final boundary bar (>=9/12 per class):")
            for guard in ("no_guard", "guard_a", "guard_b"):
                verdicts = []
                for cls, entry in sorted(data["final_boundary_holdout"].items()):
                    c = entry[guard]
                    correct = c["BID/BID"] + c["NO_BID/NO_BID"]
                    verdicts.append(f"{cls} {correct}/{entry['n']}")
                    ok = all(
                        (e[guard]["BID/BID"] + e[guard]["NO_BID/NO_BID"]) >= 9
                        for e in data["final_boundary_holdout"].values()
                    )
                print(f"    {guard:9s} {'PASS' if ok else 'FAIL'}  ({', '.join(verdicts)})")
    print(f"\nfull per-class confusion matrices written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-band", action="store_true")
    parser.add_argument("--guards", action="store_true")
    args = parser.parse_args()
    if args.scan_band:
        scan_band()
    if args.guards:
        analyze()
