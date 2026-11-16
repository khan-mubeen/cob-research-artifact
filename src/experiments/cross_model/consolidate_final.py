"""final cross-model consolidation: base / pure Protocol-v2 FT / FT+Guard-B for
each architecture, plus the pre-registered criteria verdicts.

the guarded arm is derived from the pure-FT saved eval rows using the proven
equivalence (experiment_framing.md, decision_census.py): Guard B suppresses a
model BID iff the row's rule label is a dynamic NO_BID. it never recovers a
false NO_BID and never reprices. false BIDs removed and false NO_BIDs remaining
are reported separately.

usage: python consolidate_final.py <tag=path.json> ...  OR rely on DEFAULTS.
"""

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "experiments"))
from report_paths import find_evals, report_dir  # noqa: E402
CONTAM = {"SYNEDGE-T-INF-T2/T2", "SYNEDGE-T-INF-T11/T11"}

# architecture -> tag of its base-checkpoint eval. qwen's base full eval is
# tagged "base-audit" (the later run that also carries the boundary sets);
# every other architecture uses "<arch>-base".
ARCHS = {
    "qwen": "base-audit",
    "granite": "granite-base",
    "phi": "phi-base",
    "gemma": "gemma-base",
    "ministral": "ministral-base",
    "smollm3": "smollm3-base",
}

GATED_SETS = ("baseline_six", "main_holdout", "final_boundary_holdout",
              "robustness_holdout_v2")
ALL_SETS = ("baseline_six", "edge_holdout", "main_holdout", "boundary_stress",
            "final_boundary_holdout", "robustness_holdout_v2")


def latest(tag):
    fs = [str(p) for p in find_evals(f"eval_{tag}_*.json")]
    return json.load(open(fs[-1], encoding="utf-8")) if fs else None


def guarded_rows(rows):
    """apply Guard B via guard==rule==label: a model BID on a dynamic-NO_BID
    label is suppressed to NO_BID (correct); everything else unchanged."""
    g = []
    for r in rows:
        gr = dict(r)
        if r["model_decision"] == "BID" and r["label_decision"] == "NO_BID":
            gr["model_decision"] = "NO_BID"
            gr["decision_matched"] = (r["label_decision"] == "NO_BID")
            gr["_guard_suppressed"] = True
        g.append(gr)
    return g


def acc(rows):
    n = len(rows)
    return round(sum(r["decision_matched"] for r in rows) / n * 100, 1) if n else None


def per_class(rows):
    return {c: f"{sum(r['decision_matched'] for r in rows if r['stage'] == c)}"
               f"/{sum(1 for r in rows if r['stage'] == c)}"
            for c in sorted({r["stage"] for r in rows})}


def stage_metric(rows, stage, field):
    sub = [r for r in rows if r["stage"] == stage]
    if field == "decisions":
        return acc(sub)
    devs = [r["offer_pct_dev"] for r in sub if r.get("offer_pct_dev") is not None]
    return round(sum(devs) / len(devs), 3) if devs else None


def summarize(tag, d, base_d=None, robust_d=None):
    rows_by = {s: d[s]["rows"] for s in ALL_SETS if s in d}
    # base checkpoints predate robustness_holdout_v2 in their full eval; pull it
    # from a dedicated --only-set robustness run if provided
    if "robustness_holdout_v2" not in rows_by and robust_d and "robustness_holdout_v2" in robust_d:
        rows_by["robustness_holdout_v2"] = robust_d["robustness_holdout_v2"]["rows"]
    s = {"baseline_six": f"{sum(r['model_decision']=='NO_BID' for r in rows_by['baseline_six'])}/6"
         if "baseline_six" in rows_by else None}
    if "main_holdout" in rows_by:
        s["main_holdout_decisions"] = {st: stage_metric(rows_by["main_holdout"], st, "decisions")
                                       for st in ("material", "tailoring", "logistics")}
        s["main_holdout_offer_dev"] = {st: stage_metric(rows_by["main_holdout"], st, "offer")
                                       for st in ("material", "tailoring", "logistics")}
        s["main_holdout_json"] = round(sum(r["json_valid"] for r in rows_by["main_holdout"])
                                       / len(rows_by["main_holdout"]) * 100, 1)
    if "edge_holdout" in rows_by:
        s["edge_holdout_all12"] = acc(rows_by["edge_holdout"])
        s["edge_holdout_clean10"] = acc([r for r in rows_by["edge_holdout"] if r["id"] not in CONTAM])
    if "final_boundary_holdout" in rows_by:
        s["final_boundary_per_class"] = per_class(rows_by["final_boundary_holdout"])
    s["robustness_v2_per_class"] = per_class(rows_by["robustness_holdout_v2"]) \
        if "robustness_holdout_v2" in rows_by else {}
    s["malformed_total"] = sum(1 for st in ALL_SETS if st in rows_by
                               for r in rows_by[st] if not r["json_valid"])

    # guarded arm
    guard = {"false_bids_removed": {}, "false_nobids_remaining": {},
             "final_boundary_per_class": {}, "robustness_v2_per_class": {}}
    for st in ("final_boundary_holdout", "robustness_holdout_v2", "baseline_six", "edge_holdout"):
        if st not in rows_by:
            continue
        g = guarded_rows(rows_by[st])
        removed = sum(1 for r in rows_by[st]
                      if r["model_decision"] == "BID" and r["label_decision"] == "NO_BID")
        remaining = sum(1 for r in rows_by[st]
                        if r["model_decision"] != "BID" and r["label_decision"] == "BID")
        guard["false_bids_removed"][st] = removed
        guard["false_nobids_remaining"][st] = remaining
        if st in ("final_boundary_holdout", "robustness_holdout_v2"):
            guard[f"{'final_boundary' if 'final' in st else 'robustness_v2'}_per_class"] = per_class(g)
    if "baseline_six" in rows_by:
        gb = guarded_rows(rows_by["baseline_six"])
        guard["baseline_six_guarded"] = f"{sum(r['model_decision']=='NO_BID' for r in gb)}/6"
    s["guarded_arm"] = guard
    return s


def criteria(pure, base):
    """the four pre-registered criteria, verdict per architecture.

    SEMANTIC CHANGE 2026-07-26 — criterion 3 can now return INDETERMINATE.

    Before this date criterion 3 was `PASS if <every stage strictly improved>
    else FAIL`, with a `x is not None` guard folded into the same `all(...)`.
    That silently collapsed *no measurement* into *measured regression*: if a
    base checkpoint never emitted a scorable offer on some stage, that stage's
    deviation is None, the conjunction went False, and the model was recorded
    as having FAILED to improve pricing. It had not; the comparison simply did
    not exist. A missing measurement is not evidence of a bad result.

    Phi-4-mini is the live case. Phi-base produced no scorable tailoring
    offers, so "improved on every stage" is undefined there, while Phi-v2
    improved pricing roughly 30x on the two measurable stages. The narrative
    report (protocol_v2_six_model_report.md, footnote 2) always called this
    "indeterminate"; only this script disagreed, and this script is the
    generator, so the JSON contradicted the prose it was supposed to support.

    Reading old vs new output:
      - `3_offer_improvement` is now one of PASS / FAIL / "INDETERMINATE (...)".
      - A cached pre-2026-07-26 final_comparison.json shows phi_v2 criterion 3
        as a bare "FAIL". That is the old logic, NOT a changed measurement --
        the underlying eval rows are untouched and re-running reproduces the
        same numbers. Do not read it as a regression that was later fixed.
      - `overall` is unchanged in both meaning and value: it is still PASS only
        when every criterion is exactly "PASS", so INDETERMINATE keeps a model
        out of PASS. No architecture's overall verdict moved because of this.
        Phi was already FAIL on criterion 4 independently.

    Deliberately NOT auto-passed: an unmeasurable stage means the pre-registered
    bar was not met on the evidence available, so it must not read as PASS.
    """
    c = {}
    c["1_json_100"] = "PASS" if pure["malformed_total"] == 0 else f"FAIL ({pure['malformed_total']} malformed)"
    reg = all((pure["main_holdout_decisions"][st] or 0) >= (base["main_holdout_decisions"][st] or 0)
              for st in ("material", "tailoring", "logistics"))
    c["2_no_decision_regression"] = "PASS" if reg else "FAIL"
    # a base checkpoint that never emits a scorable offer on some stage makes
    # "strictly improved on every stage" undefined there, which is not the same
    # finding as a measured regression. report it as such rather than as FAIL.
    stages = ("material", "tailoring", "logistics")
    unmeasurable = [st for st in stages
                    if pure["main_holdout_offer_dev"][st] is None
                    or base["main_holdout_offer_dev"][st] is None]
    imp = all(pure["main_holdout_offer_dev"][st] < base["main_holdout_offer_dev"][st]
              for st in stages if st not in unmeasurable)
    if unmeasurable:
        c["3_offer_improvement"] = (
            f"INDETERMINATE (no scorable base offers on: {','.join(unmeasurable)}; "
            f"improved on all measurable stages)" if imp else
            f"FAIL (regression on a measurable stage; also unmeasurable on: {','.join(unmeasurable)})")
    else:
        c["3_offer_improvement"] = "PASS" if imp else "FAIL"
    rob = pure["robustness_v2_per_class"]
    ok = all(int(v.split("/")[0]) >= 9 for v in rob.values())
    c["4_robustness_9of12"] = "PASS" if ok else "FAIL"
    c["overall"] = "PASS" if all(v == "PASS" for v in c.values()) else \
        "FAIL: " + ",".join(k for k, v in c.items() if v != "PASS" and k != "overall")
    return c


def main():
    pairs = {}
    for arg in sys.argv[1:]:
        tag, path = arg.split("=", 1)
        pairs[tag] = json.load(open(path, encoding="utf-8"))
    # defaults from saved artifacts. ARCHS is the single source of truth for
    # scope: the criteria loop below is derived from it, so adding an
    # architecture here cannot leave the verdict table behind.
    defaults = {}
    for arch, base_tag in ARCHS.items():
        defaults[f"{arch}_base"] = base_tag
        defaults[f"{arch}_v2"] = f"{arch}-v2"
    for k, t in defaults.items():
        if k not in pairs:
            d = latest(t)
            if d:
                pairs[k] = d

    # qwen/granite base checkpoints were evaluated before robustness_holdout_v2
    # existed, so their tier-3 rows come from dedicated --only-set reruns. later
    # architectures carry the set in their full eval and have no *-base-robust
    # artifact; latest() returns None for those and summarize() falls through.
    robust = {f"{arch}_base": latest(f"{arch}-base-robust") for arch in ARCHS}

    out = {
        "scope": {
            "architectures": sorted(ARCHS),
            "arms": ["base", "pure Protocol-v2 fine-tune", "fine-tune + Guard B (derived)"],
            "sets_scored": list(ALL_SETS),
            "malformed_total_denominator": "the six sets listed in sets_scored (1,626 rows); "
                                           "the scale/json-stress/magnitude diagnostic sets are "
                                           "section-3/4 explanatory evidence and are excluded",
            "generated_by": "src/experiments/cross_model/consolidate_final.py",
            "criteria_semantics_version": "2026-07-26",
            "criteria_semantics_note":
                "criterion 3 (offer-deviation improvement) may return "
                "INDETERMINATE when a base checkpoint produced no scorable "
                "offers on a stage, making the comparison undefined rather "
                "than failed. Before 2026-07-26 that case collapsed to a bare "
                "FAIL. If this file lacks criteria_semantics_version it was "
                "produced by the older logic: a phi_v2 criterion-3 'FAIL' "
                "there is the old collapse, not a different measurement. "
                "'overall' semantics are unchanged (PASS only if every "
                "criterion is exactly PASS) and no overall verdict moved. "
                "See the criteria() docstring for the full rationale.",
        },
        "models": {}, "criteria": {},
    }
    for k, d in pairs.items():
        base_key = k.replace("_v2", "_base")
        out["models"][k] = summarize(k, d, pairs.get(base_key), robust.get(k))
    for arch in ARCHS:
        if f"{arch}_v2" in out["models"] and f"{arch}_base" in out["models"]:
            out["criteria"][f"{arch}_v2"] = criteria(out["models"][f"{arch}_v2"],
                                                     out["models"][f"{arch}_base"])

    with open(report_dir("production") / "final_comparison.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== pure FT: baseline six | robustness v2 per class | malformed | final-boundary per class ===")
    for k, s in out["models"].items():
        print(f"  {k:14s} b6={s['baseline_six']}  robust={s['robustness_v2_per_class']}  "
              f"malformed={s['malformed_total']}")
    print("\n=== pre-registered criteria (pure FT arm) ===")
    for k, c in out["criteria"].items():
        print(f"  {k}: {c['overall']}   {c}")
    print("\n=== Guard B impact (false BIDs removed / false NO_BIDs remaining) ===")
    for k, s in out["models"].items():
        if "_v2" in k:
            g = s["guarded_arm"]
            print(f"  {k}: removed={g['false_bids_removed']} remaining={g['false_nobids_remaining']}")
            print(f"      guarded robustness per class: {g['robustness_v2_per_class']}")


if __name__ == "__main__":
    main()
