"""unit tests for the pinned canonical test digests.
runnable directly (python ...) or under pytest.

the rule being enforced: the digest recorded in split/metadata.json is the frozen authority, and the
live derivation from the corpus is the cross-check that proves the pin still matches reality. a
disagreement is refused and never resolved by rewriting the pin.

nothing here writes to data/finetune/v5: every tampering case is built in a temporary root from
copies, so the frozen corpus and its manifest are untouched
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import config
from src.experiments.v5 import epoch_selection, evaluate

ROOT = Path(config.PROJECT_ROOT)
MODES = ("sealed", "reverse_auction", "supervisor")


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def temp_root(tmp, mode="supervisor"):
    """a throwaway corpus root holding one mode's test files, the manifest and the split metadata"""
    root = Path(tmp)
    (root / "split").mkdir(parents=True, exist_ok=True)
    for relative in evaluate.MODE_FILES[mode]("test"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(evaluate.FINETUNE_V5 / relative, target)
    shutil.copy(evaluate.FINETUNE_V5 / "manifest.json", root / "manifest.json")
    shutil.copy(evaluate.FINETUNE_V5 / "split" / "metadata.json", root / "split" / "metadata.json")
    return root


def set_manifest_hash(root, relative):
    """make the manifest agree with whatever is on disk for one file"""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == relative:
            entry["sha256"] = sha(root / relative)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ── the pin matches the corpus ──────────────────────────────────────────────

def test_every_pinned_mode_digest_equals_its_live_derived_value():
    pins = evaluate.pinned_test_digests()
    assert set(pins) == set(MODES), pins
    for mode in MODES:
        canonical = evaluate.canonical_test_digest(mode)
        assert canonical["pinned"] == canonical["derived_live"], mode
        assert canonical["row_ids_sha256"] == pins[mode]
        assert canonical["authority"] == "split/metadata.json"
        assert len(pins[mode]) == 64


def test_the_split_metadata_records_the_pins_and_how_they_were_derived():
    metadata = json.loads((evaluate.FINETUNE_V5 / "split" / "metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["test_row_ids_sha256"]) == set(MODES)
    note = metadata["test_row_ids_sha256_method"]
    assert "authority" in note and "cross-check" in note


def test_the_manifest_agrees_with_the_split_metadata_on_disk():
    manifest = json.loads((evaluate.FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))
    entry = next(f for f in manifest["files"] if f["path"] == "split/metadata.json")
    assert entry["sha256"] == sha(evaluate.FINETUNE_V5 / "split" / "metadata.json")


# ── tampering with the pin, corpus untouched ────────────────────────────────

def test_an_edited_pin_is_caught_as_metadata_integrity_not_as_corpus_drift():
    with tempfile.TemporaryDirectory() as tmp:
        root = temp_root(tmp)
        rows = evaluate.FINETUNE_V5 / "supervisor" / "selection_test.jsonl"
        before = sha(root / "supervisor" / "selection_test.jsonl")

        metadata = json.loads((root / "split" / "metadata.json").read_text(encoding="utf-8"))
        metadata["test_row_ids_sha256"]["supervisor"] = "0" * 64
        (root / "split" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        message = ""
        try:
            evaluate.canonical_test_digest("supervisor", root=root)
            raise AssertionError("the edited pin was not caught")
        except epoch_selection.SelectionError as e:
            message = str(e)
        # the chain names the metadata, and says plainly that this is not row drift
        assert "split/metadata.json does not match the corpus manifest" in message
        assert "not corpus-row drift" in message
        # and the corpus rows really are untouched, in the temp root and in the real corpus
        assert sha(root / "supervisor" / "selection_test.jsonl") == before
        assert sha(rows) == before


def test_a_missing_pin_is_refused_rather_than_filled_in_from_the_live_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        root = temp_root(tmp)
        metadata = json.loads((root / "split" / "metadata.json").read_text(encoding="utf-8"))
        del metadata["test_row_ids_sha256"]
        (root / "split" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        set_manifest_hash(root, "split/metadata.json")          # metadata integrity intact
        raises(lambda: evaluate.canonical_test_digest("supervisor", root=root),
               epoch_selection.SelectionError, "never derived on the fly")


# ── tampering with the rows, pin untouched: the pre-existing behaviour ──────

def test_truncated_corpus_rows_are_still_caught():
    with tempfile.TemporaryDirectory() as tmp:
        root = temp_root(tmp)
        target = root / "supervisor" / "selection_test.jsonl"
        lines = target.read_text(encoding="utf-8").splitlines(True)
        target.write_text("".join(lines[:-1]), encoding="utf-8")
        raises(lambda: evaluate.canonical_test_digest("supervisor", root=root),
               epoch_selection.SelectionError, "does not match the manifest")


def test_reordered_corpus_rows_are_still_caught():
    with tempfile.TemporaryDirectory() as tmp:
        root = temp_root(tmp)
        target = root / "supervisor" / "selection_test.jsonl"
        lines = target.read_text(encoding="utf-8").splitlines(True)
        target.write_text("".join([lines[1], lines[0]] + lines[2:]), encoding="utf-8")
        raises(lambda: evaluate.canonical_test_digest("supervisor", root=root),
               epoch_selection.SelectionError, "does not match the manifest")


def test_rows_that_pass_every_file_hash_are_still_caught_by_the_pin():
    """the case only the pin can catch: rows edited and the manifest updated to match them

    every file-level hash agrees, so the manifest chain is satisfied. the pinned row-identity digest
    is the only remaining disagreement, which is exactly why it is the authority"""
    with tempfile.TemporaryDirectory() as tmp:
        root = temp_root(tmp)
        target = root / "supervisor" / "selection_test.jsonl"
        lines = target.read_text(encoding="utf-8").splitlines(True)
        target.write_text("".join([lines[1], lines[0]] + lines[2:]), encoding="utf-8")
        set_manifest_hash(root, "supervisor/selection_test.jsonl")
        set_manifest_hash(root, "split/metadata.json")

        message = ""
        try:
            evaluate.canonical_test_digest("supervisor", root=root)
            raise AssertionError("drifted rows passed every check")
        except epoch_selection.SelectionError as e:
            message = str(e)
        assert "do not match the pinned digest" in message
        # both values are shown, and neither artifact is blamed automatically
        assert "pinned (the frozen authority)" in message and "derived live from the corpus" in message
        assert "either the corpus rows drifted or the pinned" in message
        # and nothing was rewritten to make the failure go away
        metadata = json.loads((root / "split" / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["test_row_ids_sha256"]["supervisor"] == evaluate.pinned_test_digests()["supervisor"]


def test_the_pin_is_never_regenerated_by_a_passing_read():
    """a successful call must not rewrite the metadata it just read"""
    before = sha(evaluate.FINETUNE_V5 / "split" / "metadata.json")
    evaluate.canonical_test_digest("supervisor")
    assert sha(evaluate.FINETUNE_V5 / "split" / "metadata.json") == before


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 pinned test digest unit tests passed")


if __name__ == "__main__":
    main()
