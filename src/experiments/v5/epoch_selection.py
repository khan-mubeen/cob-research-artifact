"""v5 epoch selection: the validation split chooses among epochs 1, 2 and 3, the test split is read once

compare_epochs.py dispatches here when the result files it is given carry dataset_version v5, the
same data-carried gate the v5 pricing uses; v4 result files carry none and take v4's two-epoch
comparison, untouched. eval_finetune.py dispatches to src/experiments/v5/evaluate.py the same way

the discipline this module enforces:
  selection  exactly three results, epochs 1, 2 and 3 of one run, every one scored on the full
             validation split of the same mode, over the same rows (their row-identity digests
             must agree), none of them a smoke run. a test result is refused outright
  signal     validation metrics only, one pre-registered ordered key per mode, higher is better:
               sealed           macro decision agreement over the three stages, then macro json
                                validity, then lower macro mean offer deviation
               reverse_auction  macro decision accuracy over the three stages, then mean floor-case
                                accuracy, then offer exact-match rate
               supervisor       selection accuracy, then json validity, then lower mean regret
             an exact tie goes to the lower epoch, the cheaper and less trained checkpoint
  test       authorize_test allows a test evaluation only for the adapter the selection chose, and
             only while the run has no test result yet, so the test split is used once
  layout     a canonical run <model>-<mode>-v5-r01 lives in reports/v5/selection/<mode>/<model>/:
               selection.json
               validation_base.json, validation_epoch<N>.json
               test_base.json          the base arm's one-shot test read
               test_selected.json      the selected epoch's one-shot test read
               test_descriptive_epoch<N>.json   the other epochs on test, read after selection,
                                                descriptive only and never an input to selection
             any other run name (smoke, pilot, equivalence runs) goes to selection/_scratch/<run>/

nothing here imports unsloth or reads a model; the scoring of generated outputs lives in
evaluate.py and feeds selection_metrics
"""

import hashlib
import json
import re
import statistics
from pathlib import Path

from config import PROJECT_ROOT

RESULT_ROOT = PROJECT_ROOT / "reports" / "v5" / "selection"
CANONICAL_RUN = re.compile(r"^(?P<model>[a-z0-9]+)-(?P<mode>sealed|reverse_auction|supervisor)-v5-r01$")
SELECTION_FILE = "selection.json"
TEST_SELECTED_FILE = "test_selected.json"
TEST_BASE_FILE = "test_base.json"
MODES = ("sealed", "reverse_auction", "supervisor")
EPOCHS = (1, 2, 3)
STAGES = ("material", "tailoring", "logistics")
KEY_DESCRIPTION = {
    "sealed": ["macro decision agreement %", "macro json validity %", "negated macro mean abs offer deviation %"],
    "reverse_auction": ["macro decision accuracy %", "mean floor-case accuracy %", "offer exact-match %"],
    "supervisor": ["selection accuracy %", "json validity %", "negated mean regret %"],
}


class SelectionError(ValueError):
    """results that cannot legitimately be compared, or a test evaluation that is not allowed"""


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 6) if values else None


def selection_metrics(mode, summary):
    """the ordered selection key and its named parts, from one result's summary"""
    if mode == "sealed":
        parts = {"macro_decision_agreement_pct": _mean(summary[s]["decision_agreement_pct"] for s in STAGES),
                 "macro_json_valid_pct": _mean(summary[s]["json_valid_pct"] for s in STAGES),
                 "macro_mean_abs_offer_dev_pct": _mean(summary[s]["mean_abs_offer_dev_pct"] for s in STAGES)}
        key = [parts["macro_decision_agreement_pct"], parts["macro_json_valid_pct"],
               -parts["macro_mean_abs_offer_dev_pct"] if parts["macro_mean_abs_offer_dev_pct"] is not None else None]
    elif mode == "reverse_auction":
        floor = summary.get("by_floor_case", {})
        parts = {"macro_accuracy_pct": _mean(summary["by_stage"][s]["accuracy_pct"] for s in STAGES),
                 "mean_floor_case_accuracy_pct": _mean(v["accuracy_pct"] for v in floor.values()),
                 "offer_exact_match_pct": summary["overall"]["offer_exact_match_pct"]}
        key = [parts["macro_accuracy_pct"], parts["mean_floor_case_accuracy_pct"], parts["offer_exact_match_pct"]]
    elif mode == "supervisor":
        parts = {"selection_accuracy_pct": summary["selection_accuracy_pct"], "json_valid_pct": summary["json_valid_pct"],
                 "mean_regret_pct": summary["mean_regret_pct"]}
        key = [parts["selection_accuracy_pct"], parts["json_valid_pct"],
               -parts["mean_regret_pct"] if parts["mean_regret_pct"] is not None else None]
    else:
        raise SelectionError(f"unknown mode {mode!r}")
    # a missing part sorts below any real value
    return {"parts": parts, "key": [float("-inf") if k is None else k for k in key], "key_order": KEY_DESCRIPTION[mode]}


def row_ids_digest(identities):
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("dataset_version") != "v5":
        raise SelectionError(f"{path} is not a v5 result")
    return result


def select(results):
    """the chosen epoch among three validation results, with the full ranking"""
    if len(results) != 3:
        raise SelectionError(f"selection needs epochs 1, 2 and 3, got {len(results)} result(s)")
    for path, r in results:
        if r.get("split") != "validation":
            raise SelectionError(f"{path} is a {r.get('split')!r} result; only validation results may choose an epoch")
        if r.get("smoke"):
            raise SelectionError(f"{path} is a smoke evaluation on a partial split and cannot choose an epoch")
        if r.get("descriptive"):
            raise SelectionError(f"{path} is a descriptive read, recorded after selection; it never chooses an epoch")
        # a base-model arm is evaluated and reported beside the epochs, and is structurally not a
        # candidate: there is no checkpoint to select, so it is refused here exactly as a smoke or
        # test result is, rather than being allowed to win a ranking by accident
        if r.get("arm") == "base" or r.get("epoch") == 0:
            raise SelectionError(f"{path} is a base-model result; the base arm is reported alongside "
                                 f"epochs 1, 2 and 3 but is never a selectable checkpoint")
    epochs = sorted(r["epoch"] for _, r in results)
    if epochs != list(EPOCHS):
        raise SelectionError(f"selection needs exactly epochs 1, 2 and 3, got {epochs}")
    for field in ("mode", "run", "row_ids_sha256", "rows"):
        values = {r[field] for _, r in results}
        if len(values) != 1:
            raise SelectionError(f"results disagree on {field}: {sorted(map(str, values))}")
    mode = results[0][1]["mode"]
    ranked = []
    for path, r in results:
        metrics = selection_metrics(mode, r["summary"])
        ranked.append({"epoch": r["epoch"], "adapter": r["adapter"], "result": str(path), "result_sha256": sha256_file(path),
                       "metrics": metrics["parts"], "key": metrics["key"]})
    ranked.sort(key=lambda e: ([-k for k in e["key"]], e["epoch"]))
    chosen = ranked[0]
    return {
        "dataset_version": "v5",
        "run": results[0][1]["run"],
        "mode": mode,
        "split_used": "validation",
        "validation_rows": results[0][1]["rows"],
        "validation_row_ids_sha256": results[0][1]["row_ids_sha256"],
        "rule": {"key_order": KEY_DESCRIPTION[mode], "higher_is_better": True, "tie": "the lower epoch"},
        "ranking": ranked,
        "chosen": {"epoch": chosen["epoch"], "adapter": chosen["adapter"]},
        "test_split": "not read during selection; evaluate it once, on the chosen adapter only",
    }


def run_dir(run, root=RESULT_ROOT):
    """<root>/<mode>/<model> for a canonical run, <root>/_scratch/<run> for anything else"""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run):
        raise SelectionError(f"run name {run!r} must be letters, digits, dot, dash or underscore")
    m = CANONICAL_RUN.match(run)
    if m:
        return Path(root) / m["mode"] / m["model"]
    return Path(root) / "_scratch" / run


def validation_path(run, epoch, smoke=False, root=RESULT_ROOT):
    """validation_base.json for epoch 0 (the base arm), validation_epoch<N>.json otherwise"""
    name = "validation_base" if epoch == 0 else f"validation_epoch{epoch}"
    return run_dir(run, root) / f"{name}{'_smoke' if smoke else ''}.json"


def descriptive_test_path(run, epoch, root=RESULT_ROOT):
    return run_dir(run, root) / f"test_descriptive_epoch{epoch}.json"


def select_from_paths(paths, root=RESULT_ROOT):
    """load three validation results, choose, write selection.json once; returns (selection, path)"""
    results = [(p, load_result(p)) for p in paths if p]
    selection = select(results)
    out = run_dir(selection["run"], root) / SELECTION_FILE
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing["chosen"] != selection["chosen"] or existing["validation_row_ids_sha256"] != selection["validation_row_ids_sha256"]:
            raise SelectionError(f"{out} already records a different selection; a selection is made once")
        return existing, out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    return selection, out


def test_result_path(run, root=RESULT_ROOT):
    return run_dir(run, root) / TEST_SELECTED_FILE


def authorize_test(run, adapter, epoch, root=RESULT_ROOT):
    """the selection record a test evaluation must match; raises when test is not allowed"""
    selection_path = run_dir(run, root) / SELECTION_FILE
    if not selection_path.exists():
        raise SelectionError(f"no selection for run {run!r}: choose an epoch on validation before touching test")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection["chosen"]["adapter"], selection["chosen"]["epoch"]) != (adapter, epoch):
        raise SelectionError(f"run {run!r} chose epoch {selection['chosen']['epoch']} ({selection['chosen']['adapter']}); "
                             f"test may only be evaluated on that checkpoint")
    if test_result_path(run, root).exists():
        raise SelectionError(f"run {run!r} already has a test result; the test split is evaluated exactly once")
    return {"selection": str(selection_path), "selection_sha256": sha256_file(selection_path)}


def base_test_result_path(run, root=RESULT_ROOT):
    return run_dir(run, root) / TEST_BASE_FILE


def authorize_base_test(run, root=RESULT_ROOT):
    """the base arm's own one-shot test authorization, separate from the tuned checkpoint's

    two independent things each get exactly one test read, so this mirrors authorize_test's
    check-then-write shape rather than sharing its file: a base test result and a tuned test result
    never compete for the same slot.

    the selection precondition is deliberately the same one authorize_test requires, even though the
    base arm was never a candidate. reading test before the epochs have been compared would let a
    base number exist for a run whose tuned arm was never chosen, which reverses the order this
    whole gate exists to impose: validation first, then one test read each"""
    selection_path = run_dir(run, root) / SELECTION_FILE
    if not selection_path.exists():
        raise SelectionError(
            f"no selection for run {run!r}: the base arm's test read is allowed only after validation "
            f"has chosen an epoch, even though the base arm is never itself a candidate")
    if base_test_result_path(run, root).exists():
        raise SelectionError(
            f"run {run!r} already has a base test result; the base arm's test split is evaluated exactly once")
    return {"arm": "base", "selection": str(selection_path), "selection_sha256": sha256_file(selection_path)}


def print_selection(selection):
    print(f"=== {selection['run']} ({selection['mode']}): epoch selection on validation "
          f"({selection['validation_rows']} rows) ===")
    print(f"key order: {', '.join(selection['rule']['key_order'])}; tie goes to the lower epoch")
    for entry in selection["ranking"]:
        mark = "  <- chosen" if entry["epoch"] == selection["chosen"]["epoch"] else ""
        print(f"  epoch {entry['epoch']}: {entry['metrics']}{mark}")
    print(f"test split: {selection['test_split']}")
