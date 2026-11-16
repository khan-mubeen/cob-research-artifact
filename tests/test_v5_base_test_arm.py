"""unit tests for the base arm's own one-shot test read.
runnable directly (python ...) or under pytest.

the rule being enforced: base and the selected tuned checkpoint are two independent things, each
getting exactly one test evaluation, each with its own authorization, and both reading the same
canonical test rows. the tuned arm's existing behaviour is covered here too, as a regression against
what already worked rather than only as new coverage.

base cannot enter select() under any code path: that is already proven by
tests/test_v5_evaluator_gaps.py::test_selection_refuses_a_base_result_exactly_as_it_refuses_smoke_and_test,
which refuses both by epoch 0 and by the arm marker while asserting a clean trio still selects. it is
cited rather than duplicated here
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import config
from src.experiments.v5 import epoch_selection, evaluate

ROOT = Path(config.PROJECT_ROOT)
CHOSEN_ADAPTER = "models/v5/sealed/qwen-3epoch/epoch-2"


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def args(split="test", adapter="", epoch=0, smoke=0, run="qwen-sealed", mode="sealed"):
    return SimpleNamespace(run=run, split=split, adapter=adapter, epoch=epoch, smoke_rows=smoke, mode=mode)


def write_selection(root, run="qwen-sealed"):
    """the validation selection that must exist before either arm may read test"""
    path = epoch_selection.run_dir(run, root) / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset_version": "v5", "run": run, "mode": "sealed", "split_used": "validation",
        "validation_rows": 100, "validation_row_ids_sha256": "abc",
        "chosen": {"epoch": 2, "adapter": CHOSEN_ADAPTER},
    }), encoding="utf-8")
    return path


# ── the ordering precondition ───────────────────────────────────────────────

def test_the_base_test_read_is_refused_before_validation_has_selected():
    with tempfile.TemporaryDirectory() as tmp:
        raises(lambda: epoch_selection.authorize_base_test("qwen-sealed", root=tmp),
               epoch_selection.SelectionError, "no selection for run")
        # and through the guard the cli actually goes through, not only the helper
        raises(lambda: evaluate.guard(args(), root=tmp),
               epoch_selection.SelectionError, "only after validation has chosen an epoch")
        assert not epoch_selection.base_test_result_path("qwen-sealed", root=tmp).exists()


def test_the_base_test_read_is_allowed_once_the_selection_exists():
    with tempfile.TemporaryDirectory() as tmp:
        selection = write_selection(tmp)
        out, authorization = evaluate.guard(args(), root=tmp)
        assert out.name == "test_base.json"
        assert authorization["arm"] == "base"
        assert authorization["selection"] == str(selection)
        assert authorization["selection_sha256"]


def test_a_second_base_test_evaluation_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        write_selection(tmp)
        out, _ = evaluate.guard(args(), root=tmp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        raises(lambda: evaluate.guard(args(), root=tmp),
               epoch_selection.SelectionError, "evaluated exactly once")


def test_the_base_test_read_is_never_smoke_evaluated():
    with tempfile.TemporaryDirectory() as tmp:
        write_selection(tmp)
        raises(lambda: evaluate.guard(args(smoke=5), root=tmp),
               epoch_selection.SelectionError, "never smoke-evaluated")


# ── the tuned arm, unchanged: a regression against what already worked ──────

def test_the_tuned_one_shot_test_is_completely_unaffected():
    with tempfile.TemporaryDirectory() as tmp:
        write_selection(tmp)
        tuned = args(adapter=CHOSEN_ADAPTER, epoch=2)
        out, authorization = evaluate.guard(tuned, root=tmp)
        assert out.name == "test_selected.json"                # still its own file, not the base one
        assert authorization["selection_sha256"]
        out.write_text("{}", encoding="utf-8")
        raises(lambda: evaluate.guard(tuned, root=tmp), epoch_selection.SelectionError, "exactly once")
        # a checkpoint the selection did not choose is still refused
        raises(lambda: evaluate.guard(args(adapter="models/v5/sealed/qwen-3epoch/epoch-3", epoch=3), root=tmp),
               epoch_selection.SelectionError, "chose epoch 2")


def test_the_two_arms_do_not_share_or_block_each_other_s_slot():
    """each arm spends only its own one-shot read, in either order"""
    with tempfile.TemporaryDirectory() as tmp:
        write_selection(tmp)
        base_out, _ = evaluate.guard(args(), root=tmp)
        base_out.parent.mkdir(parents=True, exist_ok=True)
        base_out.write_text("{}", encoding="utf-8")
        # the tuned arm is still allowed after the base arm has been written
        tuned_out, _ = evaluate.guard(args(adapter=CHOSEN_ADAPTER, epoch=2), root=tmp)
        assert tuned_out.name == "test_selected.json" and not tuned_out.exists()
        tuned_out.write_text("{}", encoding="utf-8")
        # and now both are spent, independently
        raises(lambda: evaluate.guard(args(), root=tmp), epoch_selection.SelectionError, "exactly once")
        raises(lambda: evaluate.guard(args(adapter=CHOSEN_ADAPTER, epoch=2), root=tmp),
               epoch_selection.SelectionError, "exactly once")


# ── the canonical test rows ─────────────────────────────────────────────────

def test_a_canonical_test_digest_exists_independently_of_either_arm():
    canonical = evaluate.canonical_test_digest("supervisor")
    assert len(canonical["row_ids_sha256"]) == 64
    assert canonical["rows"] > 0
    # it is tied to the manifest's own record of those files, not to any evaluation
    assert canonical["files"] and all(f["sha256"] and f["rows"] for f in canonical["files"])
    manifest = json.loads((evaluate.FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))
    listed = {f["path"]: f["sha256"] for f in manifest["files"]}
    for entry in canonical["files"]:
        assert listed[entry["path"]] == entry["sha256"]
    assert canonical == evaluate.canonical_test_digest("supervisor")      # stable


def test_both_arms_verify_against_the_canonical_value_and_therefore_match_each_other():
    canonical = evaluate.canonical_test_digest("supervisor")
    identities = [evaluate.identity(r) for r in evaluate.load_rows("supervisor", "test")]
    # the base arm and the tuned arm each check the same way, neither consulting the other's result
    base_seen = evaluate.verify_test_digest("supervisor", identities)
    tuned_seen = evaluate.verify_test_digest("supervisor", identities)
    assert base_seen["row_ids_sha256"] == tuned_seen["row_ids_sha256"] == canonical["row_ids_sha256"]


def test_a_mismatched_row_set_is_refused():
    identities = [evaluate.identity(r) for r in evaluate.load_rows("supervisor", "test")]
    raises(lambda: evaluate.verify_test_digest("supervisor", identities[:-1]),
           epoch_selection.SelectionError, "not the canonical supervisor test split")
    raises(lambda: evaluate.verify_test_digest("supervisor", list(reversed(identities))),
           epoch_selection.SelectionError, "not the canonical supervisor test split")


def test_a_corpus_edited_under_the_gate_is_caught_rather_than_blessed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "supervisor").mkdir(parents=True)
        real = evaluate.FINETUNE_V5 / "supervisor" / "selection_test.jsonl"
        with open(real, encoding="utf-8") as f:
            head = [next(f) for _ in range(5)]
        (root / "supervisor" / "selection_test.jsonl").write_text("".join(head), encoding="utf-8")
        manifest = json.loads((evaluate.FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        raises(lambda: evaluate.canonical_test_digest("supervisor", root=root),
               epoch_selection.SelectionError, "does not match the manifest")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 base test arm unit tests passed")


if __name__ == "__main__":
    main()
