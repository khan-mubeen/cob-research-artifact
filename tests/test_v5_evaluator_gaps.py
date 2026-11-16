"""unit tests for the four tooling and evaluator gaps.
runnable directly (python ...) or under pytest.

gap 1  the v4-only diagnostics refuse or route under a v5 environment instead of silently
       reporting v4, and behave exactly as before when nothing is set
gap 2  a base-model arm can be evaluated, and is structurally refused everywhere an epoch is chosen
gap 3  one macro decision-agreement figure, the same quantity the selection key ranks by
gap 4  completion-time deviation magnitude, absolute as primary and percentage as secondary

nothing here loads a model or needs an accelerator
"""

import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import config
from src.experiments.eval_finetune import score, summarize
from src.experiments.eval_reverse_auction import group_stats
from src.experiments.v5 import epoch_selection, evaluate, results_store
from src.visualization import version_guard

ROOT = Path(config.PROJECT_ROOT)


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


# ── gap 1: version guard ────────────────────────────────────────────────────

def test_the_guard_is_silent_unless_the_environment_disagrees_with_the_pinned_version():
    assert version_guard.mismatch({}) is None
    assert version_guard.mismatch({"COB_DATASET_VERSION": ""}) is None
    assert version_guard.mismatch({"COB_DATASET_VERSION": config.DATA_VERSION}) is None
    assert version_guard.mismatch({"COB_DATASET_VERSION": "v5"}) == ("v5", config.DATA_VERSION)


def test_a_v5_request_never_yields_a_v4_report():
    # slm_check has no v5 equivalent, so it must refuse and name the command to run instead
    raises(lambda: version_guard.route_or_refuse("make slm_check", ("v5", "v4"),
                                                 equivalent="python -m src.experiments.v5.sample_check"),
           SystemExit, "cannot run under COB_DATASET_VERSION=v5")
    raises(lambda: version_guard.route_or_refuse("make slm_check", ("v5", "v4"),
                                                 equivalent="python -m src.experiments.v5.sample_check"),
           SystemExit, "src.experiments.v5.sample_check")
    # an unknown version is refused even where a v5 route exists
    raises(lambda: version_guard.route_or_refuse("make bid_check", ("v9", "v4"),
                                                 route="src.data_synthesis.v5.diagnostics",
                                                 equivalent="n/a"), SystemExit, "v9")


def test_both_diagnostics_check_the_version_before_reading_any_data():
    for relative, entry in (("src/visualization/bid_competition_diagnostics.py", "main"),
                            ("src/visualization/slm_competition_diagnostics.py", "run")):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry)
        body = ast.unparse(fn)
        guard_at, load_at = body.index("version_guard.mismatch"), body.index("load_all_data")
        assert guard_at < load_at, f"{relative}: the guard must precede load_all_data"


def test_config_data_version_default_is_untouched():
    assert config.DATA_VERSION == "v4"


# ── gap 2: the base arm, and its refusal at selection ───────────────────────

def base_args(split="validation", adapter="", epoch=0, smoke=0, run="qwen-sealed"):
    return SimpleNamespace(run=run, split=split, adapter=adapter, epoch=epoch, smoke_rows=smoke)


def test_a_base_arm_is_evaluated_on_validation_and_written_to_its_own_file():
    with tempfile.TemporaryDirectory() as tmp:
        out, authorization = evaluate.guard(base_args(), root=tmp)
        assert out.name == "validation_base.json" and authorization is None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        raises(lambda: evaluate.guard(base_args(), root=tmp), epoch_selection.SelectionError, "written once")


def test_the_base_arm_test_read_needs_its_own_authorization():
    """superseded rule, kept as a pointer rather than left asserting the opposite

    this test used to require that the base arm was refused on test outright. that rule was replaced
    deliberately: the base arm now gets its OWN one-shot test read, gated by
    epoch_selection.authorize_base_test and writing base_test_result.json. the full discipline,
    including the ordering precondition and the repeat refusal, lives in
    tests/test_v5_base_test_arm.py; what is checked here is only that the guard still refuses a base
    test read that has not been authorized"""
    with tempfile.TemporaryDirectory() as tmp:
        raises(lambda: evaluate.guard(base_args(split="test"), root=tmp),
               epoch_selection.SelectionError, "only after validation has chosen an epoch")


def test_an_adapter_and_epoch_zero_cannot_be_mixed():
    with tempfile.TemporaryDirectory() as tmp:
        raises(lambda: evaluate.guard(base_args(adapter="models/x", epoch=0), root=tmp),
               epoch_selection.SelectionError, "base-model arm is --adapter")


def test_selection_refuses_a_base_result_exactly_as_it_refuses_smoke_and_test():
    """the non-negotiable rule: a base arm is context, never a selectable checkpoint"""
    def result(epoch, arm=None, **extra):
        r = {"dataset_version": "v5", "mode": "sealed", "split": "validation", "run": "qwen-sealed",
             "epoch": epoch, "adapter": f"models/epoch{epoch}", "smoke": False, "rows": 10,
             "row_ids_sha256": "abc",
             "summary": {s: {"decision_agreement_pct": 90.0, "json_valid_pct": 100.0,
                             "mean_abs_offer_dev_pct": 1.0} for s in ("material", "tailoring", "logistics")}}
        if arm:
            r["arm"] = arm
        r.update(extra)
        return r

    with tempfile.TemporaryDirectory() as tmp:
        # select() hashes each result file for the ranking record, so these have to be real files
        def written(name, payload):
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return str(path)

        trio = [(written(f"e{e}", result(e)), result(e)) for e in (1, 2, 3)]
        assert epoch_selection.select(trio)["chosen"]["epoch"] in (1, 2, 3)

        # by epoch 0
        with_base = [(written("base", result(0)), result(0))] + trio[1:]
        raises(lambda: epoch_selection.select(with_base), epoch_selection.SelectionError,
               "never a selectable checkpoint")
        # and by the arm marker, even if an epoch number were somehow present
        tagged = [(written("tagged", result(1, arm="base")), result(1, arm="base"))] + trio[1:]
        raises(lambda: epoch_selection.select(tagged), epoch_selection.SelectionError, "base-model result")


def test_the_base_checkpoint_resolves_to_the_registry_pin():
    checkpoint, revision = evaluate.resolve_checkpoint(
        SimpleNamespace(adapter="", epoch=0, base_model="", model_name="granite"))
    assert checkpoint == "ibm-granite/granite-4.1-3b" and revision == "c0650403e44e78ec0262dab1c90914c65b196c4e"
    # a trained arm pins nothing, it loads the adapter directory it was given
    assert evaluate.resolve_checkpoint(
        SimpleNamespace(adapter="models/x/final", epoch=2, base_model="", model_name="")) == ("models/x/final", None)


# ── gap 3: macro decision agreement ─────────────────────────────────────────

def sealed_rows(agreements):
    """rows whose per-stage decision agreement is exactly the requested percentage"""
    rows = []
    for stage, pct in agreements.items():
        for i in range(10):
            matched = i < pct / 10
            rows.append({"id": f"{stage}{i}", "stage": stage, "json_valid": True,
                         "label_decision": "BID", "model_decision": "BID" if matched else "NO_BID",
                         "decision_matched": matched, "raw": ""})
    return rows


def test_the_macro_figure_is_the_equal_mean_of_the_stage_agreements():
    summary = summarize("probe", sealed_rows({"material": 90, "tailoring": 80, "logistics": 100}))
    assert summary["macro"]["macro_decision_agreement_pct"] == 90.0
    assert summary["macro"]["stages"] == ["logistics", "material", "tailoring"]
    assert "equal mean" in summary["macro"]["definition"]


def test_the_macro_figure_equals_the_selection_key_the_checkpoints_are_ranked_by():
    """the reported headline and the selection metric must be one quantity, not two"""
    summary = summarize("probe", sealed_rows({"material": 90, "tailoring": 80, "logistics": 100}))
    for stage in ("material", "tailoring", "logistics"):
        summary[stage].setdefault("mean_abs_offer_dev_pct", 1.0)
    metrics = epoch_selection.selection_metrics("sealed", summary)
    assert metrics["parts"]["macro_decision_agreement_pct"] == summary["macro"]["macro_decision_agreement_pct"]


def test_adding_the_macro_key_does_not_disturb_the_per_stage_entries():
    summary = summarize("probe", sealed_rows({"material": 100, "tailoring": 100, "logistics": 100}))
    assert set(summary) == {"material", "tailoring", "logistics", "macro"}
    assert summary["material"]["n"] == 10


# ── gap 4: completion-time deviation ────────────────────────────────────────

def days_case(stage, expected, predicted):
    return ({"id": f"{stage}-{expected}-{predicted}", "stage": stage,
             "label": {"decision": "BID", "offer_amount": 100.0, "estimated_completion_days": expected}},
            json.dumps({"decision": "BID", "offer_amount": 100.0, "estimated_completion_days": predicted}))


def test_sealed_reports_absolute_days_deviation_and_a_percentage_that_excludes_zero_expectations():
    cases = [days_case("material", 4.0, 7.0), days_case("material", 2.0, 2.0),
             days_case("logistics", 0.0, 1.0)]
    rows = score([c[0] for c in cases], [c[1] for c in cases])
    summary = summarize("probe", rows)
    material = summary["material"]
    assert material["mean_abs_days_dev"] == 1.5                       # (3.0 + 0.0) / 2
    assert material["mean_days_dev_pct"] == 37.5                      # (75.0 + 0.0) / 2
    assert material["days_dev_pct_rows_included"] == 2
    logistics = summary["logistics"]
    # the primary metric survives an expected duration of zero; only the percentage drops the row
    assert logistics["mean_abs_days_dev"] == 1.0
    assert logistics["mean_days_dev_pct"] is None
    assert logistics["days_dev_pct_rows_included"] == 0
    assert logistics["days_dev_pct_rows_excluded_expected_days_zero"] == 1


def test_a_one_day_miss_and_a_thirty_day_miss_no_longer_score_the_same():
    small = [days_case("material", 5.0, 6.0)]
    large = [days_case("material", 5.0, 35.0)]
    a = summarize("a", score([c[0] for c in small], [c[1] for c in small]))["material"]
    b = summarize("b", score([c[0] for c in large], [c[1] for c in large]))["material"]
    assert a["days_exact_match_pct"] == b["days_exact_match_pct"] == 0.0     # the old metric ties them
    assert a["mean_abs_days_dev"] == 1.0 and b["mean_abs_days_dev"] == 30.0  # the new one separates them


def test_reverse_auction_reports_the_same_two_metrics():
    rows = [{"decision_matched": True, "json_valid": True, "label_decision": "BID",
             "model_decision": "BID", "offer_exact": True, "offer_pct_dev": 0.0,
             "days_matched": False, "days_abs_dev": 2.0, "days_pct_dev": 50.0},
            {"decision_matched": True, "json_valid": True, "label_decision": "BID",
             "model_decision": "BID", "offer_exact": True, "offer_pct_dev": 0.0,
             "days_matched": True, "days_abs_dev": 0.0, "days_pct_dev": None}]
    stats = group_stats(rows)
    assert stats["mean_abs_days_dev"] == 1.0 and stats["days_dev_rows"] == 2
    assert stats["mean_days_dev_pct"] == 50.0
    assert stats["days_dev_pct_rows_included"] == 1
    assert stats["days_dev_pct_rows_excluded_expected_days_zero"] == 1


# ── the recorded training caveat ────────────────────────────────────────────

def test_every_result_record_carries_the_epoch_comparability_caveat():
    record = results_store.provenance("qwen", "sealed", 1, "models/x/epoch-1", hardware={})
    assert "epoch_comparability" in results_store.PROVENANCE_FIELDS
    note = record["epoch_comparability"]
    assert "NOT directly comparable" in note and "LR schedule" in note and "--save-epoch-adapters" in note


def test_the_generated_job_scripts_carry_the_same_caveat():
    from src.experiments.v5 import slurm
    with tempfile.TemporaryDirectory() as tmp:
        slurm.main(["--backend", "rocm", "--profile", "lumi", "--profile-root", "/scratch/test",
                    "--account", "test", "--out-dir", tmp])
        text = (Path(tmp) / "train_qwen_sealed.sh").read_text(encoding="utf-8")
        assert "NOT directly comparable" in text and "EPOCH_COMPARABILITY_NOTE" in text


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 evaluator gap unit tests passed")


if __name__ == "__main__":
    main()
