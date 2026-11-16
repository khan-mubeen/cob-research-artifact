"""unit tests for the reports/v5 mode/model layout, the one-shot gates after the move, the descriptive
test path, and the generation stop rule. runnable directly (python ...) or under pytest.

the one-shot discipline these protect: epoch selection reads validation only; the base arm and the
selected epoch each get exactly one test read, through authorize_base_test and authorize_test; the
other epochs may be read on test afterwards only as descriptive reads, which never feed selection
and never touch the one-shot files
"""

import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import config
from src.experiments.v5 import epoch_selection, evaluate, evaluate_descriptive, stop_rule

ROOT = Path(config.PROJECT_ROOT)
RUN = "qwen-sealed-v5-r01"
ADAPTERS = {1: "models/training/sealed/qwen/qwen-sealed-v5-r01/epoch-1",
            2: "models/training/sealed/qwen/qwen-sealed-v5-r01/epoch-2",
            3: "models/training/sealed/qwen/qwen-sealed-v5-r01/epoch-3"}


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def write_selection(root, run=RUN, chosen=2):
    path = epoch_selection.run_dir(run, root) / epoch_selection.SELECTION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset_version": "v5", "run": run, "mode": "sealed", "split_used": "validation",
        "validation_rows": 100, "validation_row_ids_sha256": "abc",
        "ranking": [{"epoch": e, "adapter": a} for e, a in ADAPTERS.items()],
        "chosen": {"epoch": chosen, "adapter": ADAPTERS[chosen]}}), encoding="utf-8")
    return path


def guard_args(split="test", adapter="", epoch=0, run=RUN):
    return SimpleNamespace(run=run, split=split, adapter=adapter, epoch=epoch, smoke_rows=0, mode="sealed")


def result(split="validation", epoch=1, **extra):
    r = {"dataset_version": "v5", "mode": "sealed", "run": RUN, "split": split, "epoch": epoch, "arm": "tuned",
         "adapter": ADAPTERS.get(epoch, ""), "smoke": False, "rows": 3, "row_ids_sha256": "d",
         "summary": {s: {"decision_agreement_pct": 90.0, "json_valid_pct": 100.0, "mean_abs_offer_dev_pct": 1.0}
                     for s in epoch_selection.STAGES}}
    r.update(extra)
    return r


# ── the layout ──────────────────────────────────────────────────────────────

def test_a_canonical_run_lives_under_mode_then_model():
    assert epoch_selection.run_dir(RUN, "/r") == Path("/r/sealed/qwen")
    assert epoch_selection.run_dir("gemma-reverse_auction-v5-r01", "/r") == Path("/r/reverse_auction/gemma")
    assert epoch_selection.run_dir("pilot-granite-sealed", "/r") == Path("/r/_scratch/pilot-granite-sealed")
    assert epoch_selection.validation_path(RUN, 0, root="/r").name == "validation_base.json"
    assert epoch_selection.validation_path(RUN, 2, root="/r").name == "validation_epoch2.json"
    assert epoch_selection.test_result_path(RUN, "/r").name == "test_selected.json"
    assert epoch_selection.base_test_result_path(RUN, "/r").name == "test_base.json"
    assert epoch_selection.descriptive_test_path(RUN, 3, "/r").name == "test_descriptive_epoch3.json"


# ── the one-shot gates after the move ───────────────────────────────────────

def test_after_the_move_a_second_tuned_and_a_second_base_test_read_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        write_selection(tmp)
        tuned = guard_args(adapter=ADAPTERS[2], epoch=2)
        out, _ = evaluate.guard(tuned, root=tmp)
        assert out == Path(tmp) / "sealed" / "qwen" / "test_selected.json"
        out.write_text("{}", encoding="utf-8")
        raises(lambda: evaluate.guard(tuned, root=tmp), epoch_selection.SelectionError, "exactly once")
        base_out, _ = evaluate.guard(guard_args(), root=tmp)
        assert base_out == Path(tmp) / "sealed" / "qwen" / "test_base.json"
        base_out.write_text("{}", encoding="utf-8")
        raises(lambda: evaluate.guard(guard_args(), root=tmp), epoch_selection.SelectionError, "exactly once")


def test_every_migrated_cell_still_refuses_a_second_test_read():
    """the real moved files, read only: each cell's recorded one-shot reads block a second one"""
    checked = 0
    for mode in epoch_selection.MODES:
        for sel in sorted((epoch_selection.RESULT_ROOT / mode).glob("*/selection.json")):
            model = sel.parent.name
            run = f"{model}-{mode}-v5-r01"
            chosen = json.loads(sel.read_text(encoding="utf-8"))["chosen"]
            if (sel.parent / "test_selected.json").exists():
                raises(lambda: epoch_selection.authorize_test(run, chosen["adapter"], chosen["epoch"]),
                       epoch_selection.SelectionError, "exactly once")
                checked += 1
            if (sel.parent / "test_base.json").exists():
                raises(lambda: epoch_selection.authorize_base_test(run), epoch_selection.SelectionError, "exactly once")
                checked += 1
    assert checked >= 30, f"only {checked} one-shot reads found in the migrated layout"


def test_select_still_refuses_a_base_result_after_the_move():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for e, extra in ((0, {"arm": "base", "adapter": ""}), (2, {}), (3, {})):
            p = Path(tmp) / f"r{e}.json"
            p.write_text(json.dumps(result(epoch=e, **extra)), encoding="utf-8")
            paths.append(p)
        raises(lambda: epoch_selection.select_from_paths(paths, root=tmp), epoch_selection.SelectionError,
               "base-model result")


# ── select() and compare_epochs never take a test or descriptive file ─────────

def test_select_and_compare_epochs_refuse_test_and_descriptive_inputs():
    from src.experiments import compare_epochs
    with tempfile.TemporaryDirectory() as tmp:
        for bad in (result(split="test", epoch=1), result(epoch=1, descriptive=True),
                    result(split="test", epoch=1, descriptive=True, tag=evaluate_descriptive.DESCRIPTIVE_TAG)):
            paths = []
            for e in (1, 2, 3):
                p = Path(tmp) / f"e{e}.json"
                p.write_text(json.dumps(bad if e == 1 else result(epoch=e)), encoding="utf-8")
                paths.append(str(p))
            raises(lambda: epoch_selection.select_from_paths(paths, root=tmp), epoch_selection.SelectionError)
            # compare_epochs reads sys.argv; select() refuses before anything could be written
            import sys
            saved = sys.argv
            sys.argv = ["compare_epochs.py", "--arch", "qwen", "--epoch-1", paths[0], "--epoch-2", paths[1],
                        "--epoch-3", paths[2]]
            try:
                raises(compare_epochs.main, epoch_selection.SelectionError)
            finally:
                sys.argv = saved
        assert not list(Path(tmp).rglob("selection.json"))


# ── the descriptive test path ───────────────────────────────────────────────

def test_the_descriptive_read_is_refused_before_selection():
    with tempfile.TemporaryDirectory() as tmp:
        raises(lambda: evaluate_descriptive.authorize_descriptive(RUN, 1, ADAPTERS[1], root=tmp),
               epoch_selection.SelectionError, "no selection")


def test_the_descriptive_read_refuses_the_base_arm_and_the_selected_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        write_selection(tmp, chosen=2)
        raises(lambda: evaluate_descriptive.authorize_descriptive(RUN, 0, "", root=tmp),
               epoch_selection.SelectionError, "base arm")
        raises(lambda: evaluate_descriptive.authorize_descriptive(RUN, 2, ADAPTERS[2], root=tmp),
               epoch_selection.SelectionError, "selected epoch")


def test_the_descriptive_read_refuses_an_unranked_adapter_and_a_second_write():
    with tempfile.TemporaryDirectory() as tmp:
        sel = write_selection(tmp, chosen=2)
        raises(lambda: evaluate_descriptive.authorize_descriptive(RUN, 1, "models/other/epoch-1", root=tmp),
               epoch_selection.SelectionError, "not the epoch-1 adapter")
        out, auth = evaluate_descriptive.authorize_descriptive(RUN, 1, ADAPTERS[1], root=tmp)
        assert out.name == "test_descriptive_epoch1.json"
        assert auth["selection_sha256"] == epoch_selection.sha256_file(sel)
        assert auth["descriptive"] == "descriptive, not used for selection"
        out.write_text("{}", encoding="utf-8")
        raises(lambda: evaluate_descriptive.authorize_descriptive(RUN, 1, ADAPTERS[1], root=tmp),
               epoch_selection.SelectionError, "written once")
        # and neither one-shot slot was touched
        assert not epoch_selection.test_result_path(RUN, tmp).exists()
        assert not epoch_selection.base_test_result_path(RUN, tmp).exists()


def test_a_smoke_output_must_be_scratch_and_never_a_canonical_slot():
    with tempfile.TemporaryDirectory() as tmp:
        write_selection(tmp, chosen=2)
        a = lambda out: SimpleNamespace(run=RUN, epoch=1, adapter=ADAPTERS[1], smoke_rows=5, smoke_out=str(out))  # noqa: E731
        for slot in (epoch_selection.test_result_path(RUN, tmp), epoch_selection.descriptive_test_path(RUN, 1, tmp)):
            raises(lambda: evaluate_descriptive.resolve_output(a(slot), root=tmp),
                   epoch_selection.SelectionError, "canonical result slot")
        raises(lambda: evaluate_descriptive.resolve_output(a(ROOT / "reports" / "smoke.json"), root=tmp),
               epoch_selection.SelectionError, "must be inside")
        out, _ = evaluate_descriptive.resolve_output(a(Path(tmp) / "_scratch" / "smoke.json"), root=tmp)
        assert out == (Path(tmp) / "_scratch" / "smoke.json").resolve()


def test_the_descriptive_module_never_touches_the_one_shot_gates_or_their_files():
    src = (ROOT / "src" / "experiments" / "v5" / "evaluate_descriptive.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    forbidden = {"authorize_test", "authorize_base_test", "test_result_path", "base_test_result_path",
                 "TEST_SELECTED_FILE", "TEST_BASE_FILE", "guard"}
    assert not names & forbidden, names & forbidden
    # docstrings explain what the module never writes, so they are excluded; every other string is checked
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)}
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and id(n) not in docs]
    assert not [s for s in strings if "test_selected" in s or "test_base" in s or "test_result" in s]


def test_the_sharded_path_refuses_base_and_the_selected_epoch_on_test():
    from src.experiments.v5 import evaluate_sharded
    base = ["shard", "--mode", "sealed", "--split", "test", "--run", RUN, "--epoch", "0", "--adapter", "",
            "--shard", "0/4", "--stage-dir", "/nonexistent"]
    raises(lambda: evaluate_sharded.parse_args(base), SystemExit)
    # the selected epoch: refused by the descriptive gate before any row or model is loaded
    for mode in epoch_selection.MODES:
        for sel in sorted((epoch_selection.RESULT_ROOT / mode).glob("*/selection.json"))[:1]:
            chosen = json.loads(sel.read_text(encoding="utf-8"))["chosen"]
            run = f"{sel.parent.name}-{mode}-v5-r01"
            a = evaluate_sharded.parse_args(["shard", "--mode", mode, "--split", "test", "--run", run,
                                              "--epoch", str(chosen["epoch"]), "--adapter", chosen["adapter"],
                                              "--shard", "0/4", "--stage-dir", "/nonexistent"])
            raises(lambda: evaluate_sharded.shard(a), epoch_selection.SelectionError, "selected epoch")


# ── the generation stop rule ────────────────────────────────────────────────

class _Tok:
    """a chat template that ends an assistant turn with <eot> (id 7); <eos> is 9"""
    all_special_ids = [7, 9]
    added_tokens_decoder = {}
    eos_token_id = 7

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "".join(f"<s>{m['role']}\n{m['content']}<eot>\n" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        while text:
            for tok, i in (("<eot>", 7), ("<eos>", 9)):
                if text.startswith(tok):
                    ids.append(i)
                    text = text[len(tok):]
                    break
            else:
                ids.append(100)
                text = text[1:]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, i):
        return {7: "<eot>", 9: "<eos>"}.get(i, "x")


def _model(eos):
    return SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=eos))


def test_the_stop_rule_is_a_no_op_when_the_model_already_stops_on_end_of_turn():
    stops, info = stop_rule.stop_ids(_model([9, 7]), _Tok(), "phi", "sealed")
    assert stops is None and not info["added"] and info["end_of_turn_id"] == 7
    stops, _ = stop_rule.stop_ids(_model(7), _Tok(), "granite", "sealed")
    assert stops is None


def test_the_stop_rule_adds_end_of_turn_when_it_is_missing_and_respects_its_scope():
    stops, info = stop_rule.stop_ids(_model(9), _Tok(), "qwen", "reverse_auction")
    assert stops == [9, 7] and info["added"] and info["generation_stop_ids"] == [9]
    saved = stop_rule.SCOPE
    try:
        stop_rule.SCOPE = {("qwen", "reverse_auction")}
        assert stop_rule.stop_ids(_model(9), _Tok(), "qwen", "sealed")[0] is None
        assert stop_rule.stop_ids(_model(9), _Tok(), "qwen", "reverse_auction")[0] == [9, 7]
    finally:
        stop_rule.SCOPE = saved


def test_generate_all_passes_no_stop_ids_by_default_so_v4_is_unchanged():
    """the default call builds no eos_token_id keyword; only the v5 paths pass stop ids"""
    for module in ("eval_finetune", "eval_reverse_auction"):
        src = (ROOT / "src" / "experiments" / f"{module}.py").read_text(encoding="utf-8")
        fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "generate_all")
        assert fn.args.defaults and isinstance(fn.args.defaults[-1], ast.Constant) and fn.args.defaults[-1].value is None
        body = ast.unparse(fn)
        assert '{"eos_token_id": list(stop_ids)} if stop_ids else {}' in body.replace("'", '"')


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 layout, one-shot gate, descriptive and stop-rule unit tests passed")


if __name__ == "__main__":
    main()
