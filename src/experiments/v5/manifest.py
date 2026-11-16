"""the provenance manifest for data/finetune/v5: every file, its role, sha256 and row count

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.manifest
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.manifest --verify

rows are counted from the files themselves: non-empty lines for jsonl, data lines for csv, and
none for json metadata. the main train/validation/test corpus is totalled separately from the
supplementary sets, which never count toward it. --verify recomputes every hash and row count and
fails on any difference
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from src.experiments.v5.common import FINETUNE_V5, SPLITS, rel, require_v5, sha256_file, write_json

MANIFEST_PATH = FINETUNE_V5 / "manifest.json"


def count_rows(path):
    if path.suffix == ".jsonl":
        with open(path, "rb") as f:
            return sum(1 for line in f if line.strip())
    if path.suffix == ".csv":
        with open(path, "rb") as f:
            return sum(1 for line in f if line.strip()) - 1
    return None


def role_of(relative):
    parts = relative.split("/")
    name = parts[-1]
    if name.endswith(".json"):
        return "metadata"
    if parts[0] == "split":
        return "split"
    if parts[0] == "edge":
        return "supplementary:robustness_eval"
    if parts[:2] == ["reverse_auction", "floor_boundary"]:
        return "supplementary:diagnostic"
    split = name.rsplit("_", 1)[-1].removesuffix(".jsonl")
    if split in SPLITS:
        mode = "supervisor" if parts[0] == "supervisor" else "reverse_auction" if parts[0] == "reverse_auction" else "sealed"
        return f"main:{mode}:{split}"
    return "other"


def build_manifest():
    files = []
    for path in sorted(p for p in FINETUNE_V5.rglob("*") if p.is_file() and p != MANIFEST_PATH):
        relative = str(path.relative_to(FINETUNE_V5)).replace("\\", "/")
        files.append({"path": relative, "role": role_of(relative), "sha256": sha256_file(path),
                      "bytes": path.stat().st_size, "rows": count_rows(path)})
    main = defaultdict(int)
    supplementary = defaultdict(int)
    for f in files:
        if f["role"].startswith("main:"):
            _, mode, split = f["role"].split(":")
            main[f"{mode}:{split}"] += f["rows"]
            main[f"{mode}:all"] += f["rows"]
            main["all"] += f["rows"]
        elif f["role"].startswith("supplementary:"):
            supplementary[f["role"].split(":", 1)[1]] += f["rows"]
    return {"root": rel(FINETUNE_V5), "file_count": len(files),
            "main_corpus_rows": dict(sorted(main.items())),
            "supplementary_rows_not_in_main_corpus": dict(sorted(supplementary.items())),
            "files": files}


def verify(stored):
    problems = []
    listed = {f["path"]: f for f in stored["files"]}
    on_disk = {str(p.relative_to(FINETUNE_V5)).replace("\\", "/") for p in FINETUNE_V5.rglob("*")
               if p.is_file() and p != MANIFEST_PATH}
    for missing in sorted(set(listed) - on_disk):
        problems.append(f"{missing}: listed but not on disk")
    for extra in sorted(on_disk - set(listed)):
        problems.append(f"{extra}: on disk but not listed")
    for relative in sorted(set(listed) & on_disk):
        path = FINETUNE_V5 / relative
        if sha256_file(path) != listed[relative]["sha256"]:
            problems.append(f"{relative}: sha256 differs")
        if count_rows(path) != listed[relative]["rows"]:
            problems.append(f"{relative}: row count differs")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    require_v5()
    if args.verify:
        import json
        stored = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
        problems = verify(stored)
        print(f"{stored['file_count']} files checked, {len(problems)} problems")
        for p in problems:
            print(f"  {p}")
        return 1 if problems else 0
    manifest = build_manifest()
    write_json(MANIFEST_PATH, manifest)
    print(f"{manifest['file_count']} files, main corpus rows {manifest['main_corpus_rows']}")
    print(f"supplementary rows {manifest['supplementary_rows_not_in_main_corpus']}")
    print(f"wrote {rel(MANIFEST_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
