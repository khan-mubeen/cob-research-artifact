"""unit tests for the v5 training datasets and the train-only stratified subsample.
runnable directly (python ...) or under pytest.

the leakage guarantee is tested, not inferred from reusing the split file: a subsample drawn from
the real v5 supervisor train file only holds train orders, a single validation or test record makes
it raise, and the strata follow the split's own apportionment
"""

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import config
from src.experiments.v5 import split as split_module
from src.experiments.v5 import training_data as td

ROOT = Path(config.PROJECT_ROOT)
_CACHE = {}


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def supervisor_train():
    if "rows" not in _CACHE:
        with open(td.FINETUNE_V5 / "supervisor" / "selection_train.jsonl", encoding="utf-8") as f:
            _CACHE["rows"] = [{"order_id": json.loads(line)["order_id"], "n": i} for i, line in enumerate(f)]
    return _CACHE["rows"]


def first_record(relative):
    with open(td.FINETUNE_V5 / relative, encoding="utf-8") as f:
        return {"order_id": json.loads(f.readline())["order_id"]}


def test_the_v5_datasets_name_only_train_files_and_their_windows():
    for name in td.V5_TRAIN_SETS:
        paths = td.train_set_paths(name)
        assert paths and all(p.name.endswith("_train.jsonl") and p.exists() for p in paths)
    assert td.V5_WINDOWS == {"v5_sealed": 1024, "v5_reverse_auction": 1280, "v5_supervisor": 2048}
    raises(lambda: td.check_window("v5_reverse_auction", 1024), ValueError, "1280")
    td.check_window("v5_reverse_auction", 1280)


def test_the_windows_cover_the_measured_token_lengths():
    measured = json.loads((ROOT / "reports" / "v5" / "phase6_token_lengths.json").read_text(encoding="utf-8"))
    for mode, window in (("sealed", "v5_sealed"), ("reverse_auction", "v5_reverse_auction"), ("supervisor", "v5_supervisor")):
        assert measured["max_tokens_any_model"][mode] <= td.V5_WINDOWS[window], mode


def test_a_subsample_of_the_real_train_file_holds_only_train_orders_in_the_split_strata():
    split = td.read_split()
    taken, report = td.stratified_subsample(supervisor_train(), 5000)
    assert len(taken) == report["rows"] == 5000
    assert all(split[r["order_id"]][0] == "train" for r in taken)
    population = Counter(stratum for s, stratum in split.values() if s == "train")
    expected = split_module.apportion(population, 5000)
    achieved = Counter(split[r["order_id"]][1] for r in taken)
    if not report["shortfall_topped_up"]:
        assert dict(achieved) == {k: v for k, v in expected.items() if v}


def test_one_validation_or_test_record_makes_it_raise():
    rows = supervisor_train()[:2000]
    for relative in ("supervisor/selection_validation.jsonl", "supervisor/selection_test.jsonl",
                     "material_validation.jsonl", "reverse_auction/logistics_test.jsonl"):
        leaked = rows + [first_record(relative)]
        raises(lambda: td.stratified_subsample(leaked, 500), td.TrainSplitLeak, "outside the train split")
        raises(lambda: td.assert_train_only(leaked, td.read_split()), td.TrainSplitLeak)
    raises(lambda: td.assert_train_only([{"order_id": "ORD-99999999"}], td.read_split()), td.TrainSplitLeak,
           "not_in_split_file")


def test_every_real_v5_train_file_is_train_only():
    split = td.read_split()
    for name in td.V5_TRAIN_SETS:
        for path in td.train_set_paths(name):
            with open(path, encoding="utf-8") as f:
                ids = {line[14:28].split('"')[0] for line in f}
            assert all(split[i][0] == "train" for i in ids), path.name


def test_the_subsample_is_deterministic_and_its_apportionment_is_the_splits():
    a, _ = td.stratified_subsample(supervisor_train(), 3000)
    b, _ = td.stratified_subsample(supervisor_train(), 3000)
    assert [r["n"] for r in a] == [r["n"] for r in b]
    sizes = {("x", "u"): 7, ("y", "n"): 21, ("z", "f"): 2}
    assert td.apportion(sizes, 9) == split_module.apportion(sizes, 9)


def test_a_short_stratum_is_topped_up_and_reported():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "split.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["order_id", "split", "scenario_type", "priority"])
            for i in range(10):
                w.writerow([f"A{i}", "train", "a", "normal"])
            for i in range(10):
                w.writerow([f"B{i}", "train", "b", "urgent"])
            w.writerow(["V0", "validation", "a", "normal"])
        records = [{"order_id": f"A{i}"} for i in range(10)] + [{"order_id": "B0"}]
        taken, report = td.stratified_subsample(records, 8, split_path=path)
        assert len(taken) == 8 and report["shortfall_topped_up"] == {"b|urgent": 3}
        raises(lambda: td.stratified_subsample(records + [{"order_id": "V0"}], 8, split_path=path), td.TrainSplitLeak)


def test_finetune_lora_routes_v5_names_through_this_module_and_keeps_v4s_subsample():
    source = (ROOT / "src" / "experiments" / "finetune_lora.py").read_text(encoding="utf-8")
    assert "v5_training_data.stratified_subsample(records, args.subsample)" in source
    assert "records = stratified_subsample(records, args.subsample)" in source
    assert "v5_training_data.assert_train_only(records, v5_training_data.read_split())" in source
    assert "--include-edge does not apply to v5" in source
    assert 'default="adamw_8bit"' in source and "optim=args.optim" in source


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 training subsample unit tests passed")


if __name__ == "__main__":
    main()
