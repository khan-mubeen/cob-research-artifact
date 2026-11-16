"""unit tests for v5 epoch selection and the one-time test split.
runnable directly (python ...) or under pytest.

every result here is synthetic and lives in a temporary directory: the tests prove the selection
discipline itself, validation chooses, ties go low, test is refused until a selection exists and
then allowed once for the chosen checkpoint only, and that compare_epochs.py and eval_finetune.py
dispatch v5 results to it while leaving v4 alone
"""

import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import config
from src.experiments import compare_epochs
from src.experiments.v5 import epoch_selection as sel
from src.experiments.v5 import evaluate

ROOT = Path(config.PROJECT_ROOT)


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def sealed_summary(agreement, json_valid=100.0, dev=0.5):
    return {s: {"decision_agreement_pct": agreement, "json_valid_pct": json_valid, "mean_abs_offer_dev_pct": dev}
            for s in ("material", "tailoring", "logistics")}


def write_result(folder, epoch, summary, mode="sealed", split="validation", run="qwen-sealed", rows=100,
                 digest="abc", smoke=False):
    path = Path(folder) / f"epoch_{epoch}_{split}.json"
    path.write_text(json.dumps({"dataset_version": "v5", "mode": mode, "split": split, "run": run, "epoch": epoch,
                                "adapter": f"models/v5/{run}/epoch{epoch}", "smoke": smoke, "rows": rows,
                                "row_ids_sha256": digest, "summary": summary}), encoding="utf-8")
    return str(path)


def test_validation_chooses_the_best_epoch_by_the_pre_registered_key():
    with tempfile.TemporaryDirectory() as tmp:
        paths = [write_result(tmp, 1, sealed_summary(97.0)), write_result(tmp, 2, sealed_summary(99.0)),
                 write_result(tmp, 3, sealed_summary(98.0))]
        selection, out = sel.select_from_paths(paths, root=tmp)
        assert selection["chosen"]["epoch"] == 2 and selection["split_used"] == "validation"
        assert [e["epoch"] for e in selection["ranking"]] == [2, 3, 1]
        assert json.loads(Path(out).read_text(encoding="utf-8"))["chosen"]["epoch"] == 2


def test_secondary_keys_decide_before_the_tie_rule_and_an_exact_tie_goes_to_the_lower_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        paths = [write_result(tmp, 1, sealed_summary(99.0, dev=0.9)), write_result(tmp, 2, sealed_summary(99.0, dev=0.4)),
                 write_result(tmp, 3, sealed_summary(99.0, dev=0.4))]
        assert sel.select([(p, sel.load_result(p)) for p in paths])["chosen"]["epoch"] == 2


def test_a_test_or_smoke_result_can_never_choose_an_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        base = [write_result(tmp, 1, sealed_summary(97.0)), write_result(tmp, 2, sealed_summary(99.0))]
        test = write_result(tmp, 3, sealed_summary(100.0), split="test")
        raises(lambda: sel.select_from_paths(base + [test], root=tmp), sel.SelectionError, "only validation")
        smoke = write_result(tmp, 3, sealed_summary(100.0), smoke=True)
        raises(lambda: sel.select_from_paths(base + [smoke], root=tmp), sel.SelectionError, "smoke")


def test_results_must_be_three_epochs_of_one_run_on_the_same_rows():
    with tempfile.TemporaryDirectory() as tmp:
        one, two = write_result(tmp, 1, sealed_summary(97.0)), write_result(tmp, 2, sealed_summary(99.0))
        raises(lambda: sel.select_from_paths([one, two], root=tmp), sel.SelectionError, "epochs 1, 2 and 3")
        other_rows = write_result(tmp, 3, sealed_summary(98.0), digest="different")
        raises(lambda: sel.select_from_paths([one, two, other_rows], root=tmp), sel.SelectionError, "row_ids_sha256")
        other_run = write_result(tmp, 3, sealed_summary(98.0), run="granite-sealed")
        raises(lambda: sel.select_from_paths([one, two, other_run], root=tmp), sel.SelectionError, "run")
        duplicate = write_result(Path(tmp) / ".." / Path(tmp).name, 2, sealed_summary(98.0))
        raises(lambda: sel.select_from_paths([one, two, duplicate], root=tmp), sel.SelectionError, "epochs 1, 2 and 3")


def test_the_test_split_is_allowed_once_and_only_for_the_chosen_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        raises(lambda: sel.authorize_test("qwen-sealed", "models/v5/qwen-sealed/epoch2", 2, root=tmp),
               sel.SelectionError, "choose an epoch on validation")
        paths = [write_result(tmp, e, sealed_summary(a)) for e, a in ((1, 97.0), (2, 99.0), (3, 98.0))]
        sel.select_from_paths(paths, root=tmp)
        raises(lambda: sel.authorize_test("qwen-sealed", "models/v5/qwen-sealed/epoch3", 3, root=tmp),
               sel.SelectionError, "chose epoch 2")
        assert sel.authorize_test("qwen-sealed", "models/v5/qwen-sealed/epoch2", 2, root=tmp)["selection_sha256"]
        sel.test_result_path("qwen-sealed", root=tmp).write_text("{}", encoding="utf-8")
        raises(lambda: sel.authorize_test("qwen-sealed", "models/v5/qwen-sealed/epoch2", 2, root=tmp),
               sel.SelectionError, "exactly once")


def test_a_selection_is_made_once():
    with tempfile.TemporaryDirectory() as tmp:
        paths = [write_result(tmp, e, sealed_summary(a)) for e, a in ((1, 97.0), (2, 99.0), (3, 98.0))]
        sel.select_from_paths(paths, root=tmp)
        changed = [write_result(tmp, 1, sealed_summary(99.9))] + paths[1:]
        raises(lambda: sel.select_from_paths(changed, root=tmp), sel.SelectionError, "made once")


def test_the_evaluation_guard_refuses_test_without_selection_and_never_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        args = SimpleNamespace(run="qwen-sealed", split="test", adapter="a", epoch=1, smoke_rows=0)
        raises(lambda: evaluate.guard(args, root=tmp), sel.SelectionError, "choose an epoch")
        raises(lambda: evaluate.guard(SimpleNamespace(**{**vars(args), "smoke_rows": 5}), root=tmp),
               sel.SelectionError, "never smoke")
        validation = SimpleNamespace(**{**vars(args), "split": "validation"})
        out, _ = evaluate.guard(validation, root=tmp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        raises(lambda: evaluate.guard(validation, root=tmp), sel.SelectionError, "written once")
        raises(lambda: evaluate.guard(SimpleNamespace(**{**vars(validation), "epoch": 4}), root=tmp),
               sel.SelectionError, "epoch must be")


def test_selection_keys_for_reverse_auction_and_supervisor():
    ra = {"by_stage": {s: {"accuracy_pct": 90.0} for s in ("material", "tailoring", "logistics")},
          "by_floor_case": {"a": {"accuracy_pct": 80.0}, "b": {"accuracy_pct": 100.0}},
          "overall": {"offer_exact_match_pct": 75.0}}
    assert sel.selection_metrics("reverse_auction", ra)["key"] == [90.0, 90.0, 75.0]
    sup = {"selection_accuracy_pct": 96.0, "json_valid_pct": 100.0, "mean_regret_pct": 0.03}
    assert sel.selection_metrics("supervisor", sup)["key"] == [96.0, 100.0, -0.03]


def test_the_usd_supervisor_prompt_parses():
    user = ("budget (USD): 7788.74\ndays remaining until deadline: 13\nmax acceptable defect or damage probability: 0.046\n"
            "candidate 0: material(cost_usd=4297.04, days=2, defect_p=0.04), manufacturing(cost_usd=2727.15, days=0.59, "
            "defect_p=0.023), logistics(cost_usd=87.75, days=2.0, damage_p=0.032), total_cost_usd=7111.94, "
            "total_completion_days=4.59")
    candidates, order = evaluate.parse_supervisor(user)
    assert candidates[0]["total_cost"] == 7111.94 and order["budget"] == 7788.74


def test_compare_epochs_dispatches_on_the_result_version_and_leaves_v4_alone():
    published = json.loads((ROOT / "reports" / "epoch-2" / "production" / "comparison" / "qwen_epoch1_vs_epoch2.json")
                           .read_text(encoding="utf-8"))["inputs"]
    assert compare_epochs.v5_results([published["epoch_1"], published["epoch_2"]]) is False
    with tempfile.TemporaryDirectory() as tmp:
        v5 = [write_result(tmp, e, sealed_summary(98.0)) for e in (1, 2, 3)]
        assert compare_epochs.v5_results(v5) is True
        raises(lambda: compare_epochs.v5_results([published["epoch_1"], v5[0]]), SystemExit, "different dataset versions")


def test_eval_finetune_takes_the_v5_path_only_under_a_v5_context():
    tree = ast.parse((ROOT / "src" / "experiments" / "eval_finetune.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    first = main.body[0]
    assert isinstance(first, ast.If) and "COB_DATASET_VERSION" in ast.unparse(first.test)
    assert "evaluate.main()" in ast.unparse(first)


def test_the_v4_pipeline_regression_is_identical_apart_from_the_three_edited_scripts():
    result = json.loads((ROOT / "reports" / "v5" / "pipeline" / "pipeline_regression_before_vs_after.json")
                        .read_text(encoding="utf-8"))
    assert result["identical_apart_from_allowed"] is True
    import re
    snapshots = {label: json.loads((ROOT / "reports" / "v5" / "pipeline" / f"pipeline_regression_{label}.json")
                                   .read_text(encoding="utf-8"))["cli"] for label in ("before", "after")}
    for script, blob in snapshots["before"].items():
        flags = lambda text: set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", text))  # noqa: E731
        assert flags(blob["help"]) <= flags(snapshots["after"][script]["help"]), f"{script} lost an option"
        assert blob["exit"] == snapshots["after"][script]["exit"] == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 epoch selection unit tests passed")


if __name__ == "__main__":
    main()
