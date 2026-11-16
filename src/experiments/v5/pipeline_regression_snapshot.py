"""a before/after snapshot of the v4 training and evaluation pipeline the v5 support must leave identical

  PYTHONPATH=. python -m src.experiments.v5.pipeline_regression_snapshot --label before
  PYTHONPATH=. python -m src.experiments.v5.pipeline_regression_snapshot --label after --compare before \
      --allow-changed src/experiments/compare_epochs.py,src/experiments/eval_finetune.py,src/experiments/finetune_lora.py

arms, written to reports/v5/pipeline/pipeline_regression_<label>.json:
  files         sha256 of the pipeline sources, every published v4 epoch comparison, and every
                evaluation json those comparisons were built from
  comparisons   compare_epochs.py re-run for all six published v4 epoch-1 vs epoch-2 comparisons on
                their recorded inputs, written to a scratch directory instead of the published tree.
                the output json minus its generated_at stamp is hashed, and checked against the
                published file minus its stamp
  evaluation    eval_finetune.load_eval_sets() on the v4 data: every set, every prompt and label, hashed
  training      finetune_lora's v4 training sets: the resolved paths, the seeded record order of the
                sealed and reverse-auction sets, the 15,000-row stratified subsample the epoch-1 and
                epoch-2 protocol trained on, and the round-one edge records, all hashed
  cli           the --help text of the three scripts, so any change to an existing option shows

refuses to run with COB_DATASET_VERSION set: the v4 snapshot is always the v4 default
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT
from src.dataset.context import ENV_VAR

REPORT_DIR = PROJECT_ROOT / "reports" / "v5" / "pipeline"
SCRATCH = PROJECT_ROOT / "data" / "finetune" / ".pipeline_regression_tmp"
COMPARISONS = PROJECT_ROOT / "reports" / "epoch-2" / "production" / "comparison"
SOURCES = ("src/experiments/compare_epochs.py", "src/experiments/eval_finetune.py", "src/experiments/finetune_lora.py",
           "src/experiments/eval_reverse_auction.py", "src/experiments/eval_supervisor.py",
           "src/experiments/report_paths.py")
SUBSAMPLE_ROWS = 15000


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def canonical(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode())


def quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


def file_arm():
    files = {s: sha256_file(PROJECT_ROOT / s) for s in SOURCES}
    published = {}
    for path in sorted(COMPARISONS.glob("*_epoch1_vs_epoch2.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        published[path.name] = sha256_file(path)
        for key in ("epoch_1", "epoch_2", "base"):
            if blob["inputs"].get(key):
                published[blob["inputs"][key]] = sha256_file(PROJECT_ROOT / blob["inputs"][key])
    return {"sources": files, "published": published}


def comparison_arm():
    from src.experiments import compare_epochs
    out = {}
    scratch = SCRATCH / "comparison"
    compare_epochs.report_dir = lambda *args, **kwargs: scratch
    for path in sorted(COMPARISONS.glob("*_epoch1_vs_epoch2.json")):
        published = json.loads(path.read_text(encoding="utf-8"))
        inputs = published["inputs"]
        argv = ["compare_epochs", "--arch", published["architecture"], "--epoch-1", inputs["epoch_1"],
                "--epoch-2", inputs["epoch_2"]] + (["--base", inputs["base"]] if inputs.get("base") else [])
        saved, sys.argv = sys.argv, argv
        try:
            _, stdout = quiet(compare_epochs.main)
        finally:
            sys.argv = saved
        fresh = json.loads((scratch / path.name).read_text(encoding="utf-8"))
        fresh.pop("generated_at"), published.pop("generated_at")
        out[path.name] = {"output_without_stamp_sha256": canonical(fresh),
                          "stdout_sha256": sha256_bytes(stdout.encode()),
                          "reproduces_published_without_stamp": fresh == published}
    return out


def evaluation_arm():
    from src.experiments import eval_finetune
    sets, _ = quiet(eval_finetune.load_eval_sets)
    return {"sets": {name: {"rows": len(rows), "sha256": canonical(rows)} for name, rows in sorted(sets.items())},
            "all_sets_sha256": canonical(sets)}


def training_arm():
    from src.experiments import finetune_lora
    out = {"train_set_paths": {name: [str(Path(p).relative_to(PROJECT_ROOT)).replace("\\", "/")
                                      for p in fn()] for name, fn in sorted(finetune_lora.TRAIN_SETS.items())
                               if not name.startswith("v5")}}
    for name in ("sealed", "reverse_auction"):
        records, _ = quiet(finetune_lora.load_train_records, name)
        out[f"{name}_records"] = {"rows": len(records), "sha256": canonical(records)}
        if name == "sealed":
            subsample, stdout = quiet(finetune_lora.stratified_subsample, records, SUBSAMPLE_ROWS)
            out["sealed_stratified_subsample_15000"] = {"rows": len(subsample), "sha256": canonical(subsample),
                                                        "stdout_sha256": sha256_bytes(stdout.encode())}
    edge, _ = quiet(finetune_lora.load_edge_train_records, PROJECT_ROOT / "data" / "finetune" / "v4" / "round1_edge")
    out["round1_edge_records"] = {"rows": len(edge), "sha256": canonical(edge)}
    return out


def cli_arm():
    out = {}
    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    for script in ("src/experiments/compare_epochs.py", "src/experiments/eval_finetune.py",
                   "src/experiments/finetune_lora.py"):
        run = subprocess.run([sys.executable, script, "--help"], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        text = run.stdout[run.stdout.find("usage:"):] if "usage:" in run.stdout else run.stdout
        out[script] = {"exit": run.returncode, "help": text}
    return out


def compare(before, after, allowed):
    differences = []
    for arm in ("files", "comparisons", "evaluation", "training"):
        def walk(a, b, where):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in sorted(set(a) | set(b)):
                    if key not in a or key not in b:
                        differences.append(f"{where}/{key}: only in {'after' if key not in a else 'before'}")
                    else:
                        walk(a[key], b[key], f"{where}/{key}")
            elif a != b:
                differences.append(f"{where}: {str(a)[:80]} != {str(b)[:80]}")
        walk(before[arm], after[arm], arm)
    removed_help = {}
    for script, blob in before["cli"].items():
        old_lines = set(blob["help"].splitlines())
        new_lines = set(after["cli"][script]["help"].splitlines())
        removed_help[script] = sorted(line for line in old_lines - new_lines if line.strip())
    unexpected = [d for d in differences if d.split(":")[0] not in {f"files/sources/{p}" for p in allowed}]
    return {"differences": differences, "allowed_to_change": sorted(allowed), "unexpected_differences": unexpected,
            "help_lines_removed_or_changed": removed_help,
            "identical_apart_from_allowed": not unexpected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--compare", default="")
    parser.add_argument("--allow-changed", default="")
    args = parser.parse_args()
    if os.getenv(ENV_VAR, "").strip():
        raise SystemExit(f"unset {ENV_VAR}: the v4 pipeline snapshot must run on the v4 default")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    try:
        snapshot = {"label": args.label, "files": file_arm()}
        snapshot["comparisons"] = comparison_arm()
        print(f"[{args.label}] comparisons reproduce published: "
              f"{all(v['reproduces_published_without_stamp'] for v in snapshot['comparisons'].values())}")
        snapshot["evaluation"] = evaluation_arm()
        print(f"[{args.label}] evaluation sets hashed: {snapshot['evaluation']['all_sets_sha256'][:16]}")
        snapshot["training"] = training_arm()
        print(f"[{args.label}] training sets hashed")
        snapshot["cli"] = cli_arm()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"pipeline_regression_{args.label}.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    if args.compare:
        before = json.loads((REPORT_DIR / f"pipeline_regression_{args.compare}.json").read_text(encoding="utf-8"))
        result = compare(before, snapshot, set(filter(None, args.allow_changed.split(","))))
        out = REPORT_DIR / f"pipeline_regression_{args.compare}_vs_{args.label}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in result.items() if k != "differences"}, indent=2))
        return 0 if result["identical_apart_from_allowed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
