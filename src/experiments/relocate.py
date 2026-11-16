"""hash-verified relocation engine for the repository reorganization

the file mapping comes from the reorganization plan rather than this script, so
the plan remains the source of truth for old paths, new paths, and dispositions.

contract per file:
  1. record SHA-256 and byte size at the old path
  2. move the file
  3. recompute SHA-256 at the new path
  4. abort the stage on a mismatch
  5. append the result to the relocation manifest

the JSONL manifest is appended and fsynced after each file, preserving verified
records if a run is interrupted. see reports/feasibility/hardware_incident_log.json.

mapping file format:
  [{"stage": 2, "old_path": "...", "new_path": "...", "disposition": "move"}, ...]
  disposition: "move" | "move+rename" | "delete" | "keep"
  move and move+rename run directly; delete requires --allow-delete.

usage:
  python src/experiments/relocate.py --mapping <plan_mapping.json> --stage 2 --dry-run
  python src/experiments/relocate.py --mapping <plan_mapping.json> --stage 2
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "reports" / "relocation_manifest.jsonl"

# paths excluded from relocation regardless of mapping contents
PROTECTED = {
    "data/synthetic/v4/orders.csv",
    "data/synthetic/v4/orders_metadata.json",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def log(record: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def is_protected(rel: str) -> bool:
    norm = rel.replace("\\", "/")
    return any(norm == p or norm.startswith(p) for p in PROTECTED)


def files_under(p: Path):
    """every regular file at or beneath p, as (absolute, relative-to-p)"""
    if p.is_file():
        yield p, Path(p.name)
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                yield f, f.relative_to(p)


def check_free_space(entries, dry_run: bool) -> None:
    """largest single move must fit; same-volume moves are renames but we do
    not assume that, because a cross-volume mapping would silently copy."""
    largest = 0
    for e in entries:
        src = REPO / e["old_path"]
        if not src.exists():
            continue
        total = sum(f.stat().st_size for f, _ in files_under(src))
        largest = max(largest, total)
    free = shutil.disk_usage(REPO).free
    print(f"largest single move: {largest / 1024**3:.2f} GiB | "
          f"free: {free / 1024**3:.2f} GiB")
    if largest > free and not dry_run:
        sys.exit("ABORT: insufficient free space for the largest single move")


def relocate(entries, dry_run: bool, allow_delete: bool) -> int:
    moved = verified = failed = skipped = 0
    for e in entries:
        disp = e.get("disposition", "move")
        old_rel, new_rel = e["old_path"], e.get("new_path")

        if is_protected(old_rel) or (new_rel and is_protected(new_rel)):
            print(f"REFUSED (protected): {old_rel}")
            log({"ts": datetime.now().isoformat(timespec="seconds"),
                 "old_path": old_rel, "new_path": new_rel,
                 "result": "refused_protected"})
            skipped += 1
            continue

        if disp == "delete" and not allow_delete:
            print(f"SKIP (delete needs --allow-delete): {old_rel}")
            skipped += 1
            continue
        if disp == "keep":
            skipped += 1
            continue

        src, dst = REPO / old_rel, REPO / new_rel
        if not src.exists():
            print(f"MISSING SOURCE: {old_rel}")
            log({"ts": datetime.now().isoformat(timespec="seconds"),
                 "old_path": old_rel, "new_path": new_rel,
                 "result": "missing_source"})
            failed += 1
            continue

        for abs_src, rel in files_under(src):
            abs_dst = dst / rel if src.is_dir() else dst

            if dry_run:
                # validate paths without hashing 173 GB during a dry run
                collision = "  !! DESTINATION EXISTS" if abs_dst.exists() else ""
                print(f"DRY  {abs_src.relative_to(REPO)} -> {abs_dst.relative_to(REPO)} "
                      f"({abs_src.stat().st_size} B){collision}")
                moved += 1
                if collision:
                    failed += 1
                continue

            pre_hash, pre_size = sha256_of(abs_src), abs_src.stat().st_size

            abs_dst.parent.mkdir(parents=True, exist_ok=True)
            if abs_dst.exists():
                print(f"REFUSED (destination exists): {abs_dst.relative_to(REPO)}")
                log({"ts": datetime.now().isoformat(timespec="seconds"),
                     "old_path": str(abs_src.relative_to(REPO)),
                     "new_path": str(abs_dst.relative_to(REPO)),
                     "result": "refused_destination_exists"})
                failed += 1
                continue

            shutil.move(str(abs_src), str(abs_dst))
            post_hash, post_size = sha256_of(abs_dst), abs_dst.stat().st_size
            ok = (post_hash == pre_hash and post_size == pre_size)

            log({"ts": datetime.now().isoformat(timespec="seconds"),
                 "old_path": str(abs_src.relative_to(REPO)),
                 "new_path": str(abs_dst.relative_to(REPO)),
                 "sha256": pre_hash, "sha256_after": post_hash,
                 "size": pre_size, "size_after": post_size,
                 "disposition": disp,
                 "result": "verified" if ok else "HASH_MISMATCH"})
            moved += 1
            if ok:
                verified += 1
            else:
                failed += 1
                print(f"HASH MISMATCH: {abs_dst.relative_to(REPO)}")
                print(f"  before {pre_hash} ({pre_size} B)")
                print(f"  after  {post_hash} ({post_size} B)")
                sys.exit("ABORT: hash verification failed, stage stopped")

        # remove the source tree only after every file has moved and verified
        if not dry_run and src.is_dir() and not any(src.rglob("*")):
            shutil.rmtree(src, ignore_errors=True)

    print(f"\nmoved={moved} verified={verified} failed={failed} skipped={skipped}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True,
                    help="JSON mapping derived from the plan's appendices")
    ap.add_argument("--stage", type=int, default=None,
                    help="run only this stage number from the mapping")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-delete", action="store_true",
                    help="required before any 'delete' disposition is executed")
    args = ap.parse_args()

    entries = json.load(open(args.mapping, encoding="utf-8"))
    if args.stage is not None:
        entries = [e for e in entries if e.get("stage") == args.stage]
    if not entries:
        sys.exit("no entries selected; check --stage")

    print(f"entries selected: {len(entries)}"
          + (f" (stage {args.stage})" if args.stage is not None else ""))
    check_free_space(entries, args.dry_run)
    failed = relocate(entries, args.dry_run, args.allow_delete)
    print(f"manifest: {MANIFEST}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
