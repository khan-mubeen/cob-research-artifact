"""one result schema for every v5 measurement, written so a killed job can be resumed

  reports/v5/results/<family>/[<mode>/<model>/]<name>.jsonl      the records, one json object per line
  reports/v5/results/<family>/[<mode>/<model>/]<name>.csv        the same records, flat
  reports/v5/results/<family>/[<mode>/<model>/]<name>.meta.json  the provenance of the run itself

the evaluation layout uses the mode/model subdirectory: results/records/<mode>/<model>/records.*
for evaluation summaries, results/latency/<mode>/<model>/{base,selected}.* for latency and tokens

every record carries model, mode, epoch, checkpoint, git commit, corpus manifest hash, backend,
execution profile, hardware and timestamp, so two numbers from different machines can never sit in
one table without the difference being visible

resume safety, which a three-hour wall time makes a requirement rather than a nicety:
  append only        a record is appended with one write and flushed to disk, never rewritten
  keyed              each record carries a record_key; done_keys() reads what is already there, so
                     a resumed job skips finished work instead of counting it twice
  torn line tolerant a job killed mid-write can leave a partial last line. the reader stops at it
                     and the next append truncates it away, so the file is never left half-parsed
                     and never silently duplicated

nothing here imports torch, unsloth or a model
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT
from src.experiments.v5 import execution

RESULTS_ROOT = PROJECT_ROOT / "reports" / "v5" / "results"
MANIFEST_PATH = PROJECT_ROOT / "data" / "finetune" / "v5" / "manifest.json"

PROVENANCE_FIELDS = ("model", "mode", "epoch", "checkpoint", "git_commit", "git_dirty",
                     "corpus_manifest_sha256", "backend", "profile", "hardware", "recorded_at",
                     "epoch_comparability")

# carried on every record so the caveat travels with the numbers instead of living in someone's
# memory of a chat. it matters the moment a v4 epoch-1 number is put beside a v5 epoch-1 number
EPOCH_COMPARABILITY_NOTE = (
    "v5 epochs come from ONE continuous run saved at each epoch boundary "
    "(finetune_lora.py --save-epoch-adapters). the linear LR schedule decays across the whole "
    "3-epoch horizon, so a v5 epoch-1 or epoch-2 checkpoint is NOT directly comparable to v4's "
    "separately trained epoch-1 and epoch-2 families, which each decayed over their own shorter "
    "horizon. compare v5 epochs with each other, and compare v4 against v5 only at the final "
    "selected checkpoint")


def manifest_sha256(path=MANIFEST_PATH):
    """the corpus the measurement was taken against, by content not by name"""
    import hashlib
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def provenance(model, mode, epoch, checkpoint, backend=None, profile=None, hardware=None):
    """the provenance every record carries; hardware is measured unless one is supplied"""
    return {
        "model": model,
        "mode": mode,
        "epoch": epoch,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "git_commit": execution.git_commit(),
        "git_dirty": execution.dirty_worktree(),
        "corpus_manifest_sha256": manifest_sha256(),
        "backend": backend,
        "profile": profile,
        "hardware": hardware if hardware is not None else execution.hardware_id(),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "epoch_comparability": EPOCH_COMPARABILITY_NOTE,
    }


def _read_lines(path):
    """every complete line, and whether the file ends in a torn one

    a line is complete when it ends with a newline AND parses as json. anything else is the tail of
    a job that died mid-write"""
    if not Path(path).exists():
        return [], 0
    raw = Path(path).read_bytes()
    good, offset = [], 0
    for chunk in raw.split(b"\n"):
        line = chunk.decode("utf-8", errors="replace")
        if not line.strip():
            offset += len(chunk) + 1
            continue
        try:
            good.append(json.loads(line))
        except ValueError:
            break          # torn tail, everything after it is discarded on the next append
        offset += len(chunk) + 1
    return good, min(offset, len(raw))


class ResultSet:
    """one family of records, appended to safely and resumable

    family is the kind of measurement (evaluation, latency, training), run is the thing being
    measured (qwen-sealed-epoch1). both become directory and file names, so they are checked"""

    def __init__(self, family, run, root=RESULTS_ROOT, subdir=None):
        for name, value in (("family", family), ("run", run)):
            if not value or any(c in str(value) for c in "/\\ :"):
                raise ValueError(f"{name} {value!r} must be a simple name with no path separators or spaces")
        if subdir is not None:
            parts = str(subdir).split("/")
            if len(parts) != 2 or any(not p or any(c in p for c in "\\ :.") for p in parts):
                raise ValueError(f"subdir {subdir!r} must be <mode>/<model>")
        self.family, self.run, self.subdir = family, run, subdir
        self.dir = Path(root) / family / subdir if subdir else Path(root) / family
        self.path = self.dir / f"{run}.jsonl"
        self.csv_path = self.dir / f"{run}.csv"
        self.meta_path = self.dir / f"{run}.meta.json"

    # ── reading ──
    def records(self):
        records, _ = _read_lines(self.path)
        return records

    def done_keys(self):
        """the record keys already written, so a resumed job does the rest and nothing twice"""
        return {r.get("record_key") for r in self.records() if r.get("record_key") is not None}

    def torn_tail(self):
        """true when the file ends in a partially written line, the mark of a killed job"""
        if not self.path.exists():
            return False
        _, offset = _read_lines(self.path)
        return offset < len(self.path.read_bytes())

    # ── writing ──
    def _truncate_torn_tail(self):
        if not self.path.exists():
            return False
        _, offset = _read_lines(self.path)
        raw = self.path.read_bytes()
        if offset >= len(raw):
            return False
        with open(self.path, "r+b") as f:
            f.truncate(offset)
        return True

    def append(self, record, skip_done=True):
        """one record, appended and flushed

        returns True when written, False when its key was already present. the flush and fsync are
        what make a killed job lose at most the record it was writing"""
        key = record.get("record_key")
        if skip_done and key is not None and key in self.done_keys():
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        self._truncate_torn_tail()
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True

    def write_meta(self, payload):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")

    def write_csv(self):
        """the flat view, rebuilt from the jsonl rather than appended to

        rebuilding is deliberate: the jsonl is the append-only source of truth, and a csv rebuilt
        from it can never disagree with it or carry a torn row"""
        records = self.records()
        if not records:
            return None
        columns, seen = [], set()
        for record in records:
            for key in _flatten(record):
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(_flatten(record))
        return self.csv_path


def _flatten(record, prefix=""):
    """nested record to one flat row; lists and deep objects become json strings"""
    out = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value, default=str)
        else:
            out[name] = value
    return out


def record_key(*parts):
    """the identity of one measurement, stable across runs and machines"""
    return "/".join(str(p) for p in parts)
