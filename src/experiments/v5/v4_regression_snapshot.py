"""a byte-level snapshot of everything v4 that the v5 wiring must leave identical

  PYTHONPATH=. python -m src.experiments.v5.v4_regression_snapshot --label phase1
  PYTHONPATH=. python -m src.experiments.v5.v4_regression_snapshot --label phase2 --compare phase1

three arms, all written to reports/v5/wiring/v4_regression_<label>.json:
  files         sha256 of the protected source files, the golden fixtures, the v4 and v5
                synthetic data, the production calibration and the whole data/finetune/v4 tree
  census        the exhaustive v4 decision census digest, recomputed live over all 50,000 orders
  regeneration  the four v4 generators run unmodified into a scratch directory, their output
                pointed away from data/finetune/v4 by reassigning each module's OUT_DIR, and every
                file they write hashed. this covers every sealed, reverse-auction, supervisor and
                edge prompt and label the v4 pipeline produces, not only the golden sample

the regeneration uses the arguments the committed v4 sets record: the sealed and supervisor
defaults, and the reverse-auction set's --worded-formulas --floor-pairs 500. nothing here reads
COB_DATASET_VERSION, and it refuses to run with it set, so the snapshot is always the v4 default
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

from config import PROJECT_ROOT
from src.dataset.context import ENV_VAR

REPORT_DIR = PROJECT_ROOT / "reports" / "v5" / "wiring"
SCRATCH = PROJECT_ROOT / "data" / "finetune" / ".v4_regeneration_tmp" / "run"

PROTECTED_FILES = (
    "config.py",
    "src/common/bidding_formulas.py", "src/common/slm.py", "src/common/reverse_auction.py",
    "src/common/bidding_modes.py",
    "src/experiments/generate_finetune_data.py", "src/experiments/generate_reverse_auction_data.py",
    "src/experiments/generate_supervisor_data.py", "src/experiments/generate_edge_examples.py",
    "src/experiments/compare_epochs.py", "src/experiments/eval_finetune.py",
    "src/data_synthesis/synthesize_logistics.py", "src/data_synthesis/synthesize_manufacturers.py",
    "src/data_synthesis/synthesize_material_providers.py", "src/data_synthesis/synthesize_orders.py",
    "src/data_synthesis/synthesize_product.py",
    "data/processed/v5/logistics/agent_calibration.csv",
    "data/processed/v5/logistics/agent_calibration_reasoning.json",
)
PROTECTED_TREES = ("src/supervisor", "src/agents", "src/api", "tests/golden", "data/synthetic/v4",
                   "data/synthetic/v5", "data/finetune/v4")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree(root):
    """sha256 of every file under root, skipping bytecode caches"""
    out = {}
    for path in sorted(Path(PROJECT_ROOT / root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = sha256_file(path)
    return out


def file_arm():
    files = {name: sha256_file(PROJECT_ROOT / name) for name in PROTECTED_FILES}
    trees = {root: tree(root) for root in PROTECTED_TREES}
    return {"files": files, "trees": trees}


def census_arm():
    from tests.golden.v4_baseline import GOLDEN_DIR, census_digest, load_v4
    live = census_digest(load_v4())
    fixture = json.loads((GOLDEN_DIR / "v4_decision_census.json").read_text(encoding="utf-8"))
    return {"live": live, "fixture": fixture, "live_equals_fixture": live == fixture}


def run_quiet(fn, argv):
    saved = sys.argv
    sys.argv = argv
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            fn()
    finally:
        sys.argv = saved
    return buffer.getvalue()


def regeneration_arm(scratch):
    from src.experiments import (
        generate_edge_examples, generate_finetune_data, generate_reverse_auction_data, generate_supervisor_data,
    )
    scratch = Path(scratch)
    sealed_dir, auction_dir, supervisor_dir = scratch / "sealed", scratch / "reverse_auction", scratch / "supervisor"
    generate_finetune_data.OUT_DIR = sealed_dir
    generate_reverse_auction_data.OUT_DIR = auction_dir
    generate_reverse_auction_data.SEALED_DIR = sealed_dir
    generate_supervisor_data.OUT_DIR = supervisor_dir
    generate_edge_examples.OUT_DIR = sealed_dir
    logs = {
        "sealed": run_quiet(generate_finetune_data.main, ["generate_finetune_data"]),
        "reverse_auction": run_quiet(generate_reverse_auction_data.main,
                                     ["generate_reverse_auction_data", "--worded-formulas", "--floor-pairs", "500",
                                      "--show", "0"]),
        "supervisor": run_quiet(generate_supervisor_data.main, ["generate_supervisor_data"]),
        "edge": run_quiet(generate_edge_examples.main, ["generate_edge_examples"]),
    }
    hashes = {str(p.relative_to(scratch)).replace("\\", "/"): sha256_file(p)
              for p in sorted(scratch.rglob("*")) if p.is_file()}
    return {"files": hashes, "log_tail": {k: v.strip().splitlines()[-3:] for k, v in logs.items()}}


def compare(before, after):
    diffs = []

    def walk(a, b, where):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    diffs.append(f"{where}/{key}: only in {'after' if key not in a else 'before'}")
                else:
                    walk(a[key], b[key], f"{where}/{key}")
        elif a != b:
            diffs.append(f"{where}: {a!r} != {b!r}")

    for arm in ("files", "census", "regeneration"):
        section = {k: v for k, v in before[arm].items() if k != "log_tail"}
        other = {k: v for k, v in after[arm].items() if k != "log_tail"}
        walk(section, other, arm)
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--compare", default="", help="label of an earlier snapshot to compare against")
    parser.add_argument("--allow-changed", default="",
                        help="comma-separated protected paths expected to change, for example the two wired files")
    args = parser.parse_args()
    if os.getenv(ENV_VAR, "").strip():
        raise SystemExit(f"unset {ENV_VAR}: the v4 snapshot must run on the v4 default")

    snapshot = {"label": args.label, "files": file_arm()}
    print(f"[{args.label}] files hashed")
    snapshot["census"] = census_arm()
    print(f"[{args.label}] census live equals fixture: {snapshot['census']['live_equals_fixture']}")
    # the v4 reverse-auction generator records its outputs relative to the project root, so the
    # scratch lives inside the repo, under the gitignored data/finetune, at one fixed path: a
    # random temporary name would change the dataset_hashes.json it writes from run to run
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    try:
        snapshot["regeneration"] = regeneration_arm(SCRATCH)
    finally:
        shutil.rmtree(SCRATCH.parent, ignore_errors=True)
    print(f"[{args.label}] regenerated {len(snapshot['regeneration']['files'])} v4 files")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"v4_regression_{args.label}.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")

    if args.compare:
        before = json.loads((REPORT_DIR / f"v4_regression_{args.compare}.json").read_text(encoding="utf-8"))
        allowed = {f"files/files/{p}" for p in filter(None, args.allow_changed.split(","))}
        diffs = compare(before, snapshot)
        unexpected = [d for d in diffs if d.split(":")[0] not in allowed]
        result = {"compared": [args.compare, args.label], "differences": diffs, "allowed_to_change": sorted(allowed),
                  "unexpected_differences": unexpected, "identical_apart_from_allowed": not unexpected}
        out = REPORT_DIR / f"v4_regression_{args.compare}_vs_{args.label}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, indent=2))
        return 0 if not unexpected else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
