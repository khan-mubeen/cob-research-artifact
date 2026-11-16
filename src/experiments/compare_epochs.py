"""single-variable epoch comparison: the same architecture, the same frozen
protocol, the same evaluation sets, differing only in num_train_epochs.

this is a different axis from cross_model/consolidate_final.py. that script
compares architectures within one epoch family (base / pure FT / FT+Guard B);
this one holds the architecture fixed and varies the epoch count, so the two
are kept separate rather than overloading one consolidator with both axes.

the Guard B arm is derived from the saved eval rows using the same proven
equivalence consolidate_final.py uses: Guard B suppresses a model BID iff the
row's rule label is a dynamic NO_BID. it never recovers a false NO_BID and
never reprices.

usage:
  python src/experiments/compare_epochs.py --arch granite \
      --epoch-1 <eval json> --epoch-2 <eval json> [--base <eval json>]

v5 result files carry dataset_version v5 and take a separate path, the same data-carried gate the
v5 pricing uses: src/experiments/v5/epoch_selection.py chooses among --epoch-1, --epoch-2 and
--epoch-3 on validation results only. v4 result files carry no version and run the comparison
below exactly as before
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "experiments"))
from report_paths import report_dir  # noqa: E402

CONTAM = {"SYNEDGE-T-INF-T2/T2", "SYNEDGE-T-INF-T11/T11"}

# the six sets the pre-registered criteria are scored on (same denominator
# consolidate_final.py uses for malformed_total: 1,626 rows per model)
CRITERIA_SETS = ("baseline_six", "edge_holdout", "main_holdout", "boundary_stress",
                 "final_boundary_holdout", "robustness_holdout_v2")
STAGES = ("material", "tailoring", "logistics")


def guarded(rows):
    out = []
    for r in rows:
        g = dict(r)
        if r["model_decision"] == "BID" and r["label_decision"] == "NO_BID":
            g["model_decision"] = "NO_BID"
            g["decision_matched"] = True
            g["_guard_suppressed"] = True
        out.append(g)
    return out


def per_class(rows):
    return {c: f"{sum(r['decision_matched'] for r in rows if r['stage'] == c)}"
               f"/{sum(1 for r in rows if r['stage'] == c)}"
            for c in sorted({r["stage"] for r in rows})}


def acc(rows):
    return round(sum(r["decision_matched"] for r in rows) / len(rows) * 100, 2) if rows else None


def stage_dev(rows, stage):
    devs = [r["offer_pct_dev"] for r in rows
            if r["stage"] == stage and r.get("offer_pct_dev") is not None]
    return round(statistics.mean(devs), 3) if devs else None


def summarize(d):
    sets = {s: d[s]["rows"] for s in CRITERIA_SETS if s in d}
    s = {"sets_present": sorted(sets)}
    n = sum(len(v) for v in sets.values())
    bad = sum(1 for v in sets.values() for r in v if not r["json_valid"])
    s["json"] = {"rows_scored": n, "malformed": bad,
                 "valid_pct": round((n - bad) / n * 100, 2) if n else None,
                 "per_set_malformed": {k: sum(1 for r in v if not r["json_valid"])
                                       for k, v in sets.items()}}
    if "main_holdout" in sets:
        mh = sets["main_holdout"]
        s["main_holdout_decisions"] = {st: acc([r for r in mh if r["stage"] == st]) for st in STAGES}
        s["main_holdout_offer_dev"] = {st: stage_dev(mh, st) for st in STAGES}
    if "baseline_six" in sets:
        b = sets["baseline_six"]
        s["baseline_six"] = f"{sum(r['model_decision'] == 'NO_BID' for r in b)}/6"
        s["baseline_six_guarded"] = \
            f"{sum(r['model_decision'] == 'NO_BID' for r in guarded(b))}/6"
    if "edge_holdout" in sets:
        e = sets["edge_holdout"]
        s["edge_holdout_all12"] = acc(e)
        s["edge_holdout_clean10"] = acc([r for r in e if r["id"] not in CONTAM])
    for name in ("final_boundary_holdout", "robustness_holdout_v2", "boundary_stress"):
        if name in sets:
            s[f"{name}_per_class"] = per_class(sets[name])
            s[f"{name}_per_class_guarded"] = per_class(guarded(sets[name]))
    # guard B effect, reported as corrections vs residue vs new errors
    g = {}
    for name, rows in sets.items():
        removed = sum(1 for r in rows
                      if r["model_decision"] == "BID" and r["label_decision"] == "NO_BID")
        residue = sum(1 for r in rows
                      if r["model_decision"] != "BID" and r["label_decision"] == "BID")
        new_err = sum(1 for a, b in zip(rows, guarded(rows))
                      if a["decision_matched"] and not b["decision_matched"])
        g[name] = {"false_bids_corrected": removed, "false_nobids_remaining": residue,
                   "new_errors_introduced": new_err}
    s["guard_b"] = g
    return s


def criteria(pure, base):
    c = {}
    c["1_json_100"] = "PASS" if pure["json"]["malformed"] == 0 \
        else f"FAIL ({pure['json']['malformed']} malformed of {pure['json']['rows_scored']})"
    if base:
        reg = all((pure["main_holdout_decisions"][st] or 0) >= (base["main_holdout_decisions"][st] or 0)
                  for st in STAGES)
        c["2_no_decision_regression"] = "PASS" if reg else "FAIL"
        unmeasurable = [st for st in STAGES
                        if pure["main_holdout_offer_dev"][st] is None
                        or base["main_holdout_offer_dev"][st] is None]
        imp = all(pure["main_holdout_offer_dev"][st] < base["main_holdout_offer_dev"][st]
                  for st in STAGES if st not in unmeasurable)
        c["3_offer_improvement"] = ("PASS" if imp else "FAIL") if not unmeasurable else \
            f"INDETERMINATE (no scorable base offers on: {','.join(unmeasurable)})"
    rob = pure.get("robustness_holdout_v2_per_class", {})
    c["4_robustness_9of12"] = "PASS" if rob and all(
        int(v.split("/")[0]) >= 9 for v in rob.values()) else "FAIL"
    c["overall"] = "PASS" if all(v == "PASS" for v in c.values()) else \
        "FAIL: " + ",".join(k for k, v in c.items() if v != "PASS" and k != "overall")
    return c


def delta(a, b):
    """b - a, for numbers; None-safe"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(b - a, 3)
    return None


def v5_results(paths):
    """True when every given result is a v5 result, False when none is; mixing raises"""
    versions = [json.load(open(p, encoding="utf-8")).get("dataset_version") for p in paths if p]
    if all(v == "v5" for v in versions):
        return True
    if any(v is not None for v in versions):
        raise SystemExit(f"cannot compare results from different dataset versions: {versions}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--epoch-1", required=True, help="epoch-1 pure-FT eval json")
    ap.add_argument("--epoch-2", required=True, help="epoch-2 pure-FT eval json")
    ap.add_argument("--epoch-3", default="", help="epoch-3 result, v5 validation results only")
    ap.add_argument("--base", default="", help="the shared base-checkpoint eval json")
    ap.add_argument("--out-epoch", type=int, default=2)
    args = ap.parse_args()

    if v5_results([args.epoch_1, args.epoch_2, args.epoch_3]):
        from src.experiments.v5.epoch_selection import print_selection, select_from_paths
        selection, path = select_from_paths([args.epoch_1, args.epoch_2, args.epoch_3])
        print_selection(selection)
        print(f"\nwritten to {path}")
        return
    if args.epoch_3:
        raise SystemExit("--epoch-3 is for v5 validation results; the v4 comparison is epoch 1 against epoch 2")

    load = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
    e1, e2 = summarize(load(args.epoch_1)), summarize(load(args.epoch_2))
    base = summarize(load(args.base)) if args.base else None

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": args.arch,
        "single_variable": "num_train_epochs 1 -> 2",
        "inputs": {"epoch_1": args.epoch_1, "epoch_2": args.epoch_2, "base": args.base or None},
        "note": "epoch-2 is a fresh run from the base checkpoint, not a continuation of the epoch-1 "
                "adapter; the linear LR schedule spans the whole run in each case",
        "arms": {"base": base, "epoch_1": e1, "epoch_2": e2},
        "criteria": {"epoch_1": criteria(e1, base), "epoch_2": criteria(e2, base)},
    }

    d = {"json_valid_pct": delta(e1["json"]["valid_pct"], e2["json"]["valid_pct"]),
         "malformed": delta(e1["json"]["malformed"], e2["json"]["malformed"])}
    for st in STAGES:
        d[f"main_holdout_decisions.{st}"] = delta(
            e1["main_holdout_decisions"][st], e2["main_holdout_decisions"][st])
        d[f"main_holdout_offer_dev.{st}"] = delta(
            e1["main_holdout_offer_dev"][st], e2["main_holdout_offer_dev"][st])
    for k in ("edge_holdout_all12", "edge_holdout_clean10"):
        d[k] = delta(e1.get(k), e2.get(k))
    report["epoch_2_minus_epoch_1"] = d

    out = report_dir("production", "comparison", args.out_epoch)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.arch}_epoch1_vs_epoch2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"=== {args.arch}: epoch 1 vs epoch 2 (single variable: num_train_epochs) ===\n")
    print(f"{'metric':38s} {'epoch-1':>14s} {'epoch-2':>14s} {'delta':>10s}")
    print("-" * 80)
    rows = [("json valid % (1,626 rows)", e1["json"]["valid_pct"], e2["json"]["valid_pct"]),
            ("malformed rows", e1["json"]["malformed"], e2["json"]["malformed"])]
    for st in STAGES:
        rows.append((f"main holdout decisions {st}",
                     e1["main_holdout_decisions"][st], e2["main_holdout_decisions"][st]))
    for st in STAGES:
        rows.append((f"mean offer dev % {st}",
                     e1["main_holdout_offer_dev"][st], e2["main_holdout_offer_dev"][st]))
    for a, x, y in rows:
        print(f"{a:38s} {str(x):>14s} {str(y):>14s} {str(delta(x, y)):>10s}")
    for label, key in (("baseline six (pure)", "baseline_six"),
                       ("baseline six (+Guard B)", "baseline_six_guarded"),
                       ("edge holdout all-12", "edge_holdout_all12"),
                       ("edge holdout clean-10", "edge_holdout_clean10")):
        print(f"{label:38s} {str(e1.get(key)):>14s} {str(e2.get(key)):>14s}")
    print("\nper-class (correct/12), pure -> +Guard B:")
    for s in ("final_boundary_holdout", "robustness_holdout_v2"):
        print(f"  {s}")
        for arm, blob in (("epoch-1", e1), ("epoch-2", e2)):
            print(f"    {arm}: pure {blob.get(f'{s}_per_class')} "
                  f"-> guarded {blob.get(f'{s}_per_class_guarded')}")
    print("\npre-registered criteria:")
    for arm in ("epoch_1", "epoch_2"):
        print(f"  {arm}: {report['criteria'][arm]}")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
