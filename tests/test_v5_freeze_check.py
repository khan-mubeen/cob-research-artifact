"""unit tests for the freeze gate: candidate, authoritative freeze, and the remote launch gate.
runnable directly (python ...) or under pytest.

the point of most of these is the unhappy path. a gate that only passes when everything is already
right is not a gate, so a corrupted corpus hash, a moved commit, a drifted pin, a stale generated
script and a candidate-only record each get their own test. nothing here writes to the real report
paths or needs an accelerator
"""

import json
import tempfile
from pathlib import Path

import config
from src.experiments.v5 import execution, freeze_check, model_registry, results_store, slurm

ROOT = Path(config.PROJECT_ROOT)
RENDER_CONTEXT = execution.resolve("rocm", "lumi", root=freeze_check.RENDER_ROOT)


def rendered_script(model="qwen", mode="sealed", base_layout=None):
    """one generated script exactly as slurm.py would write it today

    base_layout defaults to None, which is what the generator itself passes: script_text then
    resolves it from the profile. a test about hub-pin drift asks for "registry" explicitly,
    because the lumi profile reads its weights from the base tree and so pins no revision"""
    args = freeze_check.args_from_script("#SBATCH --partition=small-g\n#SBATCH --time=72:\n"
                                         "#SBATCH --gpus-per-node=8\n#SBATCH --cpus-per-task=7\n"
                                         "--num-train-epochs 3", base_layout=base_layout)
    args.account = "test_account"
    return slurm.script_text(args, RENDER_CONTEXT, model, mode)


def frozen_record(tmp, commit=None, files=None, models=None, authoritative=True):
    """a minimal authoritative record, built from the real manifest so the hashes are real"""
    manifest = json.loads((freeze_check.FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))
    if files is None:
        files = {f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "rows": f["rows"]}
                 for f in manifest["files"][:3]}
    record = {
        "authoritative": authoritative,
        "mode": "freeze",
        "commit_recorded_before_writing": commit or execution.git_commit(),
        "git": {"commit": commit or execution.git_commit(), "dirty": False},
        "corpus": {"manifest_sha256": results_store.manifest_sha256(), "files": files},
        "models": models if models is not None else {},
        "checks": [], "passed": True,
    }
    path = Path(tmp) / "freeze_check_local.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path, record


# ── candidate mode ──────────────────────────────────────────────────────────

def test_candidate_says_inside_itself_that_it_is_not_authoritative():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "freeze_check_candidate.json"
        freeze_check.run_candidate(path=path)
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["authoritative"] is False and record["disposable"] is True
        assert record["mode"] == "candidate"
        # the staleness rule has to live in the file, not only in a report someone may not read
        assert "stale" in record["staleness_rule"] and "must not be reused" in record["staleness_rule"]
        assert "NOT what LUMI checks against" in record["what_this_is_not"]


def test_candidate_never_writes_the_authoritative_file():
    before = freeze_check.LOCAL_PATH.exists()
    before_bytes = freeze_check.LOCAL_PATH.read_bytes() if before else None
    with tempfile.TemporaryDirectory() as tmp:
        freeze_check.run_candidate(path=Path(tmp) / "candidate.json")
    assert freeze_check.LOCAL_PATH.exists() == before
    if before:
        assert freeze_check.LOCAL_PATH.read_bytes() == before_bytes


# ── freeze mode ─────────────────────────────────────────────────────────────

def test_freeze_refuses_on_a_dirty_tree_and_writes_nothing():
    saved = execution.worktree_entries
    execution.worktree_entries = lambda: ["M src/experiments/v5/freeze_check.py"]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "freeze_check_local.json"
            assert freeze_check.run_freeze(path=target) == 1
            # the refusal has to precede the write, or writing the report is what dirties the tree
            assert not target.exists(), "freeze wrote a record despite refusing"
    finally:
        execution.worktree_entries = saved


def test_freeze_refuses_when_git_cannot_report_the_worktree_at_all():
    saved = execution.worktree_entries
    execution.worktree_entries = lambda: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "freeze_check_local.json"
            assert freeze_check.run_freeze(path=target) == 1
            assert not target.exists()
    finally:
        execution.worktree_entries = saved


def test_a_clean_tree_produces_an_authoritative_record_that_pins_the_pre_write_commit():
    saved_entries, saved_commit = execution.worktree_entries, execution.git_commit
    execution.worktree_entries = lambda: []
    execution.git_commit = lambda: "a" * 40
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "freeze_check_local.json"
            code = freeze_check.run_freeze(path=target)
            record = json.loads(target.read_text(encoding="utf-8"))
            assert record["authoritative"] is True and record["mode"] == "freeze"
            assert record["commit_recorded_before_writing"] == "a" * 40
            assert "NOT committed" in record["artifact_rule"]
            assert code in (0, 1)          # the corpus and launch checks decide the exit code
    finally:
        execution.worktree_entries, execution.git_commit = saved_entries, saved_commit


# ── the remote gate ─────────────────────────────────────────────────────────

def test_the_remote_gate_refuses_when_only_a_candidate_record_exists():
    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "freeze_check_candidate.json"
        candidate.write_text(json.dumps({"authoritative": False, "mode": "candidate"}), encoding="utf-8")
        code = freeze_check.run_remote(local_path=Path(tmp) / "missing.json", candidate_path=candidate)
        assert code == 2, "a candidate must never be accepted as a launch gate"


def test_the_remote_gate_refuses_a_record_that_is_not_marked_authoritative():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = frozen_record(tmp, authoritative=False)
        assert freeze_check.run_remote(local_path=path) == 2


def test_the_remote_gate_catches_a_corrupted_corpus_hash():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = json.loads((freeze_check.FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))
        files = {f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "rows": f["rows"]}
                 for f in manifest["files"][:3]}
        victim = sorted(files)[0]
        files[victim]["sha256"] = "0" * 64          # one file, one wrong hash
        path, _ = frozen_record(tmp, files=files)
        assert freeze_check.run_remote(local_path=path) == 1


def test_the_remote_gate_catches_a_missing_corpus_file():
    with tempfile.TemporaryDirectory() as tmp:
        files = {"supervisor/not_transferred.jsonl": {"sha256": "0" * 64, "bytes": 1, "rows": 1}}
        path, _ = frozen_record(tmp, files=files)
        assert freeze_check.run_remote(local_path=path) == 1


def test_the_remote_gate_catches_a_moved_commit():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = frozen_record(tmp, commit="b" * 40, files={})
        assert freeze_check.run_remote(local_path=path) == 1


def test_the_remote_gate_checks_a_derived_copy_by_fingerprint_not_by_hub_revision():
    local = [name for name in model_registry.MODELS if model_registry.model(name)["local"]]
    assert local, "the registry no longer has a locally derived checkpoint"
    name = local[0]
    fingerprint = freeze_check.local_fingerprint(model_registry.model(name)["checkpoint"])
    if not fingerprint["present"]:
        return                                   # the derived copy is not on this machine, nothing to compare
    models = {name: {"kind": "local_derived_copy", "checkpoint": model_registry.model(name)["checkpoint"],
                     "revision": model_registry.model(name)["revision"], "fingerprint": fingerprint}}
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = frozen_record(tmp, files={}, models=models)
        assert freeze_check.run_remote(local_path=path) == 0
        # now corrupt one recorded provenance hash and require the gate to notice
        first = sorted(fingerprint["provenance_files"])[0]
        broken = json.loads(path.read_text(encoding="utf-8"))
        broken["models"][name]["fingerprint"]["provenance_files"][first]["sha256"] = "0" * 64
        path.write_text(json.dumps(broken), encoding="utf-8")
        assert freeze_check.run_remote(local_path=path) == 1


def test_the_remote_gate_compares_the_live_results_store_resolution():
    """part b's fourth check: results_store's own live values, not a second hardcoded copy"""
    with tempfile.TemporaryDirectory() as tmp:
        path, record = frozen_record(tmp, files={})
        broken = json.loads(path.read_text(encoding="utf-8"))
        broken["corpus"]["manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(broken), encoding="utf-8")
        assert freeze_check.run_remote(local_path=path) == 1
        live = results_store.provenance("x", "y", 0, None, hardware={})
        assert live["corpus_manifest_sha256"] == record["corpus"]["manifest_sha256"]


# ── drift and staleness ─────────────────────────────────────────────────────

def test_a_freshly_rendered_script_is_not_stale_but_an_edited_one_is():
    text = rendered_script()
    drift, stale, missing = freeze_check.inspect_script(text, "qwen", "sealed", RENDER_CONTEXT)
    assert not drift and not stale and not missing
    edited = text.replace("--num-train-epochs 3", "--num-train-epochs 2")
    _, stale_after, _ = freeze_check.inspect_script(edited, "qwen", "sealed", RENDER_CONTEXT)
    assert stale_after, "an edited generated script must not pass as aligned"


def test_pin_drift_between_the_registry_and_a_script_is_caught():
    """a hub id and its revision, edited in a script, are both caught

    this is the registry layout by construction: it is the one that carries a hub id and a
    --revision to drift in the first place, so it is asked for explicitly rather than inherited
    from whichever profile the render context happens to be"""
    text = rendered_script(base_layout="registry")
    wrong_revision = text.replace(model_registry.model("qwen")["revision"], "c" * 40)
    drift, _, _ = freeze_check.inspect_script(wrong_revision, "qwen", "sealed", RENDER_CONTEXT,
                                              base_layout="registry")
    assert any(item["field"] == "--revision" for item in drift)
    wrong_base = text.replace("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-7B")
    drift_base, _, _ = freeze_check.inspect_script(wrong_base, "qwen", "sealed", RENDER_CONTEXT,
                                                   base_layout="registry")
    assert any(item["field"] == "--base-model" for item in drift_base)


def test_submit_all_runs_the_freeze_gate_before_any_sbatch():
    with tempfile.TemporaryDirectory() as tmp:
        slurm.main(["--backend", "rocm", "--profile", "lumi", "--profile-root", "/scratch/test",
                    "--account", "test", "--out-dir", tmp])
        text = (Path(tmp) / "submit_all.sh").read_text(encoding="utf-8")
        gate_at = text.index("freeze_check --remote")
        first_sbatch = text.index("sbatch ")
        assert gate_at < first_sbatch, "the gate must run before anything is queued"
        assert "no jobs submitted" in text and "exit 1" in text


def test_the_gate_reuses_the_existing_verification_rather_than_reimplementing_it():
    source = (ROOT / "src" / "experiments" / "v5" / "freeze_check.py").read_text(encoding="utf-8")
    assert "manifest_module.verify(" in source, "manifest verification must be reused"
    assert "from src.experiments.v5.common import" in source and "sha256_file" in source
    assert "def sha256_file" not in source, "hashing must not be reimplemented here"


def test_worktree_entries_and_dirty_worktree_cannot_disagree():
    entries = execution.worktree_entries()
    dirty = execution.dirty_worktree()
    assert dirty == (None if entries is None else bool(entries))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 freeze check unit tests passed")


if __name__ == "__main__":
    main()
