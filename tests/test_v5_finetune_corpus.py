"""unit tests for the generated v5 fine-tuning corpus: the split, leakage, the manifest and the
separation of supplementary sets. runnable directly (python ...) or under pytest.

the order-id scan reads only the first field of each line, which every v5 generator writes first,
so it covers every row of every main file without parsing the prompts
"""

import ast
import hashlib
import json
import re
from pathlib import Path

import config
from src.experiments.v5.common import FINETUNE_V5, SPLITS, read_split

ROOT = Path(config.PROJECT_ROOT)
ORDER_ID = re.compile(rb'^\{"order_id": "([^"]+)"')
MAIN = [f"{stage}_{split}.jsonl" for stage in ("material", "tailoring", "logistics") for split in SPLITS] \
    + [f"reverse_auction/{stage}_{split}.jsonl" for stage in ("material", "tailoring", "logistics") for split in SPLITS] \
    + [f"supervisor/selection_{split}.jsonl" for split in SPLITS]


def manifest():
    return json.loads((FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))


def test_the_split_is_exactly_80_10_10_of_50000_orders_and_every_stratum_is_in_every_split():
    meta = json.loads((FINETUNE_V5 / "split" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["source_order_count"] == 50000
    assert meta["counts"] == {"train": 40000, "validation": 5000, "test": 5000}
    assert all(stratum["in_every_split"] for stratum in meta["strata"])
    assert meta["disjoint"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert meta["stratify_on"] == ["scenario_type", "priority"] and meta["third_dimension_added"] is False


def test_no_order_id_appears_in_two_splits_across_every_stage_and_mode():
    split_of = read_split()
    seen = {s: set() for s in SPLITS}
    for relative in MAIN:
        split = next(s for s in SPLITS if relative.endswith(f"_{s}.jsonl"))
        with open(FINETUNE_V5 / relative, "rb") as f:
            for line in f:
                order_id = ORDER_ID.match(line).group(1).decode()
                assert split_of[order_id] == split, (relative, order_id)
                seen[split].add(order_id)
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not seen[a] & seen[b], (a, b)


def test_the_manifest_lists_every_file_with_its_real_hash_and_rows():
    stored = manifest()
    on_disk = sorted(str(p.relative_to(FINETUNE_V5)).replace("\\", "/") for p in FINETUNE_V5.rglob("*")
                     if p.is_file() and p.name != "manifest.json")
    assert [f["path"] for f in stored["files"]] == on_disk
    for entry in stored["files"]:
        path = FINETUNE_V5 / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
        if path.suffix == ".jsonl":
            with open(path, "rb") as f:
                assert sum(1 for line in f if line.strip()) == entry["rows"], entry["path"]


def test_supplementary_sets_are_never_counted_in_the_main_corpus():
    stored = manifest()
    main_total = sum(f["rows"] for f in stored["files"] if f["role"].startswith("main:"))
    assert stored["main_corpus_rows"]["all"] == main_total
    roles = {f["path"]: f["role"] for f in stored["files"]}
    assert all(role == "supplementary:robustness_eval" for p, role in roles.items() if p.startswith("edge/") and p.endswith(".jsonl"))
    assert all(role == "supplementary:diagnostic" for p, role in roles.items()
               if p.startswith("reverse_auction/floor_boundary/") and p.endswith(".jsonl"))
    assert stored["supplementary_rows_not_in_main_corpus"]["diagnostic"] == 1188


def test_the_validator_imports_nothing_it_validates():
    tree = ast.parse((ROOT / "src" / "experiments" / "v5" / "validate.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    banned = [m for m in imported if m and (m.startswith("src.common") or m.startswith("src.experiments"))]
    assert not banned, banned


def test_the_phase_5_validation_passed():
    result = json.loads((ROOT / "reports" / "v5" / "phase5_validation.json").read_text(encoding="utf-8"))
    assert result["passed"] is True, result["checks"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 fine-tuning corpus unit tests passed")


if __name__ == "__main__":
    main()
