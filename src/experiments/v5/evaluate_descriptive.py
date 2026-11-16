"""descriptive test reads: the epochs that selection did NOT choose, read on test after selection

the one-shot discipline is untouched. epoch selection reads validation only; the base arm and the
selected epoch each get exactly one test read through evaluate.py and epoch_selection's
authorize_base_test / authorize_test. this module never imports or calls either of those, never
writes test_base.json or test_selected.json, and refuses both of those arms outright.

what it adds, so a test chart can show base and all three epochs: for each epoch other than the
selected one, one read of the same pinned test rows, written once to

  reports/v5/selection/<mode>/<model>/test_descriptive_epoch<N>.json

tagged "descriptive, not used for selection". select() and compare_epochs refuse such a file as input
(split is test, and descriptive is set), so it can never feed back into the choice.

refusals, each covered by tests/test_v5_layout_and_descriptive.py:
  no selection.json yet         the read happens only after the choice is recorded
  epoch 0 / no adapter          the base arm already has its one-shot read
  the selected epoch            it already has its one-shot read
  an adapter other than the one the selection ranked for that epoch
  an existing descriptive file  written once

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.evaluate_descriptive \
      --mode sealed --run qwen-sealed-v5-r01 --epoch 1 --adapter <epoch-1 dir>
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from src.experiments.v5 import evaluate, training_data
from src.experiments.v5.epoch_selection import (EPOCHS, MODES, RESULT_ROOT, SELECTION_FILE, SelectionError,
                                                descriptive_test_path, row_ids_digest, run_dir,
                                                selection_metrics, sha256_file)

DESCRIPTIVE_TAG = "descriptive, not used for selection"


def authorize_descriptive(run, epoch, adapter, root=RESULT_ROOT, allow_existing=False):
    """(output path, authorization record) for a descriptive test read; raises when not allowed"""
    selection_path = run_dir(run, root) / SELECTION_FILE
    if not selection_path.exists():
        raise SelectionError(f"no selection for run {run!r}: a descriptive test read happens only after "
                             f"validation has chosen an epoch")
    if epoch == 0 or not adapter:
        raise SelectionError("the base arm already has its one-shot test read; the descriptive path refuses it")
    if epoch not in EPOCHS:
        raise SelectionError(f"epoch must be one of {EPOCHS}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if epoch == selection["chosen"]["epoch"]:
        raise SelectionError(f"epoch {epoch} is the selected epoch of {run!r}; it already has its one-shot "
                             f"test read and the descriptive path refuses it")
    ranked = {e["epoch"]: e["adapter"] for e in selection["ranking"]}
    if ranked.get(epoch) != adapter:
        raise SelectionError(f"adapter {adapter!r} is not the epoch-{epoch} adapter the selection ranked "
                             f"({ranked.get(epoch)!r})")
    out = descriptive_test_path(run, epoch, root)
    if out.exists() and not allow_existing:
        raise SelectionError(f"{out} already exists; a descriptive test read is written once")
    return out, {"selection": str(selection_path), "selection_sha256": sha256_file(selection_path),
                 "descriptive": DESCRIPTIVE_TAG}


def build_result(a, rows, identities, canonical, window, authorization, summary, scored, stop_info, extra=None):
    result = {
        "dataset_version": "v5", "mode": a.mode, "split": "test", "run": a.run, "epoch": a.epoch,
        "arm": "tuned", "adapter": a.adapter, "tag": DESCRIPTIVE_TAG, "descriptive": True,
        "smoke": bool(a.smoke_rows), "rows": len(rows), "canonical_test": canonical,
        "row_ids_sha256": row_ids_digest(identities), "window": window,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "authorization": authorization,
        "summary": summary, "selection_metrics": selection_metrics(a.mode, summary),
        "generation_stop_rule": stop_info, "rows_detail": scored,
    }
    result.update(extra or {})
    return result


def write(out, result):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(out)


def resolve_output(a, root=RESULT_ROOT):
    """the canonical descriptive file, or, for a smoke run, the explicit scratch path it must name"""
    out, authorization = authorize_descriptive(a.run, a.epoch, a.adapter, root,
                                               allow_existing=bool(a.smoke_rows))
    if a.smoke_rows:
        if not a.smoke_out:
            raise SelectionError("a smoke descriptive read must name --smoke-out; it never writes the canonical file")
        out = smoke_output(a.smoke_out, root)
        if out.exists():
            raise SelectionError(f"{out} already exists")
    return out, authorization


def smoke_output(path, root=RESULT_ROOT):
    """a smoke read's path: inside <root>/_scratch or the system temporary directory, never a canonical slot"""
    out = Path(path).resolve()
    scratch = (Path(root) / "_scratch").resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    if not (out.is_relative_to(scratch) or out.is_relative_to(tmp)):
        raise SelectionError(f"--smoke-out {path} must be inside {scratch} or {tmp}")
    if out.is_relative_to(Path(root).resolve()) and not out.is_relative_to(scratch):
        raise SelectionError(f"--smoke-out {path} is a canonical result slot")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", required=True, type=int)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-seq-length", type=int, default=0)
    ap.add_argument("--smoke-rows", type=int, default=0, help="first N rows per file, for the equivalence proof only")
    ap.add_argument("--smoke-out", default="", help="where a smoke read is written; required with --smoke-rows")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    from src.dataset import context as dataset_context
    if dataset_context.resolve().version != "v5":
        raise SystemExit(f"v5 evaluation needs {dataset_context.ENV_VAR}=v5")
    a = parse_args(argv)
    out, authorization = resolve_output(a)
    rows = evaluate.load_rows(a.mode, "test", a.smoke_rows)
    identities = [evaluate.identity(r) for r in rows]
    # the same pinned test-row digest the one-shot reads check against; a smoke read covers a prefix
    canonical = evaluate.verify_test_digest(a.mode, identities) if not a.smoke_rows else None
    print(f"{a.run} {a.mode} test (descriptive) epoch {a.epoch}: {len(rows)} rows -> {out}")
    if a.dry_run:
        return 0
    window = a.max_seq_length or training_data.V5_WINDOWS[evaluate.MODE_DATASET[a.mode]]
    training_data.check_window(evaluate.MODE_DATASET[a.mode], window)
    ns = SimpleNamespace(mode=a.mode, split="test", run=a.run, epoch=a.epoch, adapter=a.adapter, tag="",
                         model_name="", base_model="", base_layout="registry", base_root="")
    checkpoint, revision = evaluate.resolve_checkpoint(ns)
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=checkpoint, revision=revision,
                                                         max_seq_length=window, load_in_4bit=False, dtype=None)
    FastLanguageModel.for_inference(model)
    summary, scored, stop_info = evaluate.generate_and_score(ns, model, tokenizer, rows)
    write(out, build_result(a, rows, identities, canonical, window, authorization, summary, scored, stop_info))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
