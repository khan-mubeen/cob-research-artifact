"""unit tests for the v5 source manifest and the tree it indexes.
runnable directly (python ...) or under pytest.

the manifest's job is to make provenance checkable rather than claimed. so these
tests check the tree matches what was declared, every hash matches the bytes on
disk, a tampered file is caught, a record missing a required field is caught, and
above all that the two hash meanings are written down in plain language where
nobody can mistake a record-integrity hash for proof about an external source
"""

import copy
import json
import shutil
import tempfile
from pathlib import Path

from src.dataset import manifest as manifest_reader
from src.dataset.manifest import ACQUISITION_MODES, ManifestError, entry_errors
from src.dataset.v5_sources import (
    DOWNLOADS, MANUAL_RECORDS, declared_artifacts, group_of,
)

RAW_V5 = Path(__file__).resolve().parent.parent / "data" / "raw" / "v5"
MANIFEST = RAW_V5 / "manifest.json"


def loaded():
    return manifest_reader.load(MANIFEST)


def raises(fn, fragment):
    try:
        fn()
    except ManifestError as e:
        assert fragment in str(e), f"expected {fragment!r} in the error, got {e}"
        return
    raise AssertionError(f"expected a ManifestError mentioning {fragment!r}")


# ── the tree is what was declared ───────────────────────────────────────────

def test_the_declared_folder_structure_exists():
    for folder in ("material/cotton", "material/polyester", "material/linen",
                   "material/denim", "material/leather", "fx", "logistics"):
        assert (RAW_V5 / folder).is_dir(), f"{folder} is missing"


def test_every_declared_artifact_is_on_disk_and_indexed():
    manifest = loaded()
    indexed = set(manifest["artifact_sha256"])
    for name, mode in declared_artifacts():
        assert (RAW_V5 / name).is_file(), f"{name} is declared but not stored"
        assert name in indexed, f"{name} is stored but not indexed"
        assert manifest_reader.artifact(manifest, name)["acquisition_mode"] == mode


def test_nothing_unindexed_is_sitting_in_the_tree():
    """a file nobody indexed has no provenance, which is the failure mode here

    data/raw/v5/logistics is indexed by its own manifest, so the files under it
    are allowed when that manifest indexes them, and by nothing else"""
    indexed = set(loaded()["artifact_sha256"])
    allowed = indexed | {"manifest.json"} | {
        f"{group}/source.json" for group, _ in
        ((group_of(name), mode) for name, mode in declared_artifacts())}
    logistics = manifest_reader.load(RAW_V5 / "logistics" / "manifest.json")
    allowed |= {f"logistics/{name}" for name in logistics["artifact_sha256"]}
    allowed |= {"logistics/manifest.json"} | {
        f"logistics/{name.split('/', 1)[0]}/source.json" for name in logistics["artifact_sha256"]}
    allowed |= {f"logistics/{item['group']}/source.json" for item in logistics["incomplete"]}
    found = {str(p.relative_to(RAW_V5)).replace("\\", "/")
             for p in RAW_V5.rglob("*") if p.is_file()}
    assert found <= allowed, f"unindexed files in data/raw/v5: {sorted(found - allowed)}"


def test_every_folder_has_a_source_file_listing_its_sources():
    groups = {group_of(name) for name, _ in declared_artifacts()}
    for group in groups:
        path = RAW_V5 / group / "source.json"
        assert path.is_file(), f"{group} has no source.json"
        with open(path, encoding="utf-8") as f:
            source = json.load(f)
        assert source["group"] == group
        assert source["sources"], group


def test_the_source_files_and_the_manifest_agree():
    manifest = loaded()
    for group in {group_of(name) for name, _ in declared_artifacts()}:
        with open(RAW_V5 / group / "source.json", encoding="utf-8") as f:
            source = json.load(f)
        for entry in source["sources"]:
            indexed = manifest_reader.artifact(manifest, entry["artifact"])
            assert indexed["sha256"] == entry["sha256"]
            assert indexed["acquisition_mode"] == entry["acquisition_mode"]


def test_logistics_is_indexed_by_its_own_manifest():
    """the logistics method is decided, so the folder is no longer a placeholder

    this replaces the check that logistics held only a pending README. the
    README is gone and the folder is a verified source tree of its own"""
    assert not (RAW_V5 / "logistics" / "README.md").exists()
    report = manifest_reader.verify(RAW_V5 / "logistics" / "manifest.json")
    assert report["problems"] == [], report["problems"]
    assert report["artifacts"] >= 9


def test_the_manifest_points_to_the_logistics_sub_manifest_rather_than_omitting_it():
    assert "pending" not in loaded()
    pointers = loaded()["sub_manifests"]
    assert any(p["group"] == "logistics" and p["manifest"] == "logistics/manifest.json"
               for p in pointers)


# ── hashes verify ───────────────────────────────────────────────────────────

def test_the_whole_manifest_verifies_against_the_stored_files():
    report = manifest_reader.verify(MANIFEST)
    assert report["problems"] == [], report["problems"]
    assert report["artifacts"] == len(declared_artifacts())


def test_a_tampered_artifact_is_caught():
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "v5"
        shutil.copytree(RAW_V5, copy)
        target = copy / "material" / "cotton" / "PCOTTINDUSDM.csv"
        target.write_bytes(target.read_bytes() + b"2030-01-01,999.0\n")
        report = manifest_reader.verify(copy / "manifest.json")
        assert any("PCOTTINDUSDM.csv" in p and "sha256 on disk" in p
                   for p in report["problems"]), report["problems"]
        raises(lambda: manifest_reader.require_verified(copy / "manifest.json"),
               "does not verify")


def test_a_tampered_manual_record_is_caught_too():
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "v5"
        shutil.copytree(RAW_V5, copy)
        target = copy / "material" / "linen" / "alliance_flax.json"
        record = json.loads(target.read_text(encoding="utf-8"))
        record["observations"][0]["value"] = 6.06
        target.write_text(json.dumps(record, indent=2), encoding="utf-8")
        report = manifest_reader.verify(copy / "manifest.json")
        assert any("alliance_flax.json" in p for p in report["problems"])


def test_a_missing_artifact_is_caught():
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "v5"
        shutil.copytree(RAW_V5, copy)
        (copy / "fx" / "ecb_eurofxref_hist.csv").unlink()
        report = manifest_reader.verify(copy / "manifest.json")
        assert any("indexed but not on disk" in p for p in report["problems"])


def test_the_index_and_the_entries_must_agree():
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "v5"
        shutil.copytree(RAW_V5, copy)
        path = copy / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["artifact_sha256"]["material/cotton/PCOTTINDUSDM.csv"] = "0" * 64
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report = manifest_reader.verify(path)
        assert any("disagree" in p for p in report["problems"])


# ── the acquisition mode schema ─────────────────────────────────────────────

def test_exactly_two_acquisition_modes_exist():
    assert ACQUISITION_MODES == ("downloaded", "manual_record")
    modes = {e["acquisition_mode"] for e in loaded()["artifacts"]}
    assert modes <= set(ACQUISITION_MODES)


def test_a_third_acquisition_mode_is_refused_at_load():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["acquisition_modes"]["scraped"] = "invented later"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raises(lambda: manifest_reader.load(path),
               "the schema allows exactly ['downloaded', 'manual_record']")


def test_an_entry_with_an_invented_mode_is_refused():
    entry = copy.deepcopy(manifest_reader.artifact(loaded(),
                                                   "material/linen/alliance_flax.json"))
    entry["acquisition_mode"] = "estimated"
    problems = entry_errors(entry)
    assert any("is not one of" in p for p in problems), problems


def test_an_entry_missing_a_required_field_is_refused():
    manifest = loaded()
    downloaded = manifest_reader.artifact(manifest, "material/cotton/PCOTTINDUSDM.csv")
    for field in ("retrieval_date", "unit_confirmed_from", "frequency", "source_url"):
        broken = copy.deepcopy(downloaded)
        broken.pop(field)
        assert any(f"missing {field}" in p for p in entry_errors(broken))

    manual = manifest_reader.artifact(manifest, "material/linen/alliance_flax.json")
    for field in ("access_date", "verification_note", "page_or_section", "source_url"):
        broken = copy.deepcopy(manual)
        broken.pop(field)
        assert any(f"missing {field}" in p for p in entry_errors(broken))


def test_a_hash_purpose_that_does_not_match_the_mode_is_refused():
    manual = copy.deepcopy(manifest_reader.artifact(
        loaded(), "material/linen/alliance_flax.json"))
    manual["hash_purpose"] = "source_artifact_integrity"
    problems = entry_errors(manual)
    assert any("a manual_record artifact's hash is record_integrity" in p
               for p in problems), problems


# ── the distinction the manifest exists to make ─────────────────────────────

def test_the_manifest_explains_both_hash_meanings_in_plain_language():
    manifest = loaded()
    downloaded = manifest["acquisition_modes"]["downloaded"]
    manual = manifest["acquisition_modes"]["manual_record"]

    assert "proves the pinned file" in downloaded
    assert "has not changed since it was retrieved" in downloaded

    assert "record integrity" in manual.lower()
    assert "not of anything external" in manual
    assert "NOT evidence that the original web page" in manual

    first = manifest["read_this_first"]
    assert "mean different things" in first
    assert "do not cite a record-integrity hash" in first


def test_every_downloaded_artifact_hashes_the_download_itself():
    for entry in loaded()["artifacts"]:
        if entry["acquisition_mode"] != "downloaded":
            continue
        assert entry["hash_purpose"] == "source_artifact_integrity"
        assert entry["retrieval_date"]
        assert entry["unit"] and entry["unit_confirmed_from"]


def test_every_manual_record_hashes_the_local_record_and_says_so():
    manual = manifest_reader.manual_records(loaded())
    assert len(manual) == len(MANUAL_RECORDS)
    for entry in manual:
        assert entry["hash_purpose"] == "record_integrity"
        assert entry["verification_note"].strip()
        assert entry["access_date"] and entry["access_date_basis"]
        with open(RAW_V5 / entry["artifact"], encoding="utf-8") as f:
            record = json.load(f)
        assert record["hash_purpose"] == "record_integrity"
        assert "NOT evidence" in record["hash_purpose_means"]
        assert record["source_url"] and record["page_or_section"]


def test_the_recorded_verification_notes_are_the_ones_that_were_given():
    """the notes are evidence, not prose to be reworded"""
    expected = {
        "material/cotton/tabb_jersey_knit.json":
            "checked and explicitly excluded as blends",
        "material/polyester/tabb_polyester_twill.json":
            "printed as '100% Texturized Polyester', not a blend",
        "material/denim/tabb_denim.json":
            "does not quantify them",
        "material/polyester/sunsirs_dty.json":
            "the daily-frequency claim rests on the source's own label only",
        "material/linen/alliance_flax.json":
            "do not interpolate",
        "material/linen/imaterial_linen.json":
            "convert using the FX layer built in Part 6, not a hardcoded rate",
        "material/leather/leather_unlimited.json":
            "do not substitute the value-priced bundles here",
        "material/leather/townsend_conversion_rule.json":
            "avoids introducing a separate, unsupported leather density assumption",
    }
    manifest = loaded()
    for name, fragment in expected.items():
        note = manifest_reader.artifact(manifest, name)["verification_note"]
        assert fragment in note, f"{name} note does not contain {fragment!r}"


# ── the values that were recorded ───────────────────────────────────────────

def test_the_recorded_catalog_values_are_exactly_the_ones_given():
    with open(RAW_V5 / "material" / "cotton" / "tabb_jersey_knit.json",
              encoding="utf-8") as f:
        jersey = json.load(f)
    assert {o["sku"]: o["value"] for o in jersey["observations"]} == {
        "JKCWI": 6.65, "JKCCI": 6.95, "JKCWD": 7.65, "JKCCD": 7.95}
    assert all(o["composition"] == "100% Cotton" for o in jersey["observations"])

    with open(RAW_V5 / "material" / "denim" / "tabb_denim.json", encoding="utf-8") as f:
        denim = json.load(f)
    assert {o["sku"]: o["value"] for o in denim["observations"]} == {
        "DEHI": 4.85, "DELI": 3.85, "DEHBLD": 7.50, "DEHBD": 7.50}

    with open(RAW_V5 / "material" / "linen" / "alliance_flax.json",
              encoding="utf-8") as f:
        flax = json.load(f)
    assert [(o["date"], o["value"]) for o in flax["observations"]] == [
        ("2023-06-01", 6.05), ("2023-11-01", 8.23),
        ("2024-03-01", 9.08), ("2026-03-01", 5.89)]
    assert flax["currency"] == "EUR" and flax["interpolate"] is False

    with open(RAW_V5 / "material" / "leather" / "townsend_conversion_rule.json",
              encoding="utf-8") as f:
        townsend = json.load(f)
    assert townsend["conversion"]["oz_to_kg"] == 0.0283495


def test_the_imaterial_cotton_skus_are_marked_documentation_only():
    with open(RAW_V5 / "material" / "linen" / "imaterial_linen.json",
              encoding="utf-8") as f:
        record = json.load(f)
    cotton = [o for o in record["observations"] if o["composition"] == "100% Cotton"]
    assert len(cotton) == 5
    assert all("documentation-only" in o["use"] for o in cotton)
    assert "not used to generate v5 cotton prices" in record["comparator_note"]


# ── the declared blocker ────────────────────────────────────────────────────

def test_sunsirs_is_recorded_as_provenance_with_no_observations():
    blocked = manifest_reader.blocked_records(loaded())
    assert [e["artifact"] for e in blocked] == ["material/polyester/sunsirs_dty.json"]
    with open(RAW_V5 / "material" / "polyester" / "sunsirs_dty.json",
              encoding="utf-8") as f:
        record = json.load(f)
    assert record["observations"] == []
    assert record["observations_status"] == "missing"
    assert "must not be re-fetched or reconstructed" in record["observations_status_note"]


def test_no_source_value_was_invented_for_a_record_that_has_none():
    for spec in MANUAL_RECORDS:
        record = spec["record"]
        if record.get("observations_status") == "missing":
            assert not record["observations"], (
                f"{spec['artifact']} declares its observations missing but carries "
                f"{len(record['observations'])} of them")


# ── the declaration and the downloads line up ───────────────────────────────

def test_every_download_records_the_url_that_was_actually_used():
    manifest = loaded()
    for spec in DOWNLOADS:
        entry = manifest_reader.artifact(manifest, spec["artifact"])
        assert entry["source_url"] == spec["url"]
        assert entry["source_page"] == spec["series_page"]


def test_the_extracted_ecb_csv_records_what_it_was_extracted_from():
    entry = manifest_reader.artifact(loaded(), "fx/ecb_eurofxref_hist.csv")
    assert entry["derived_from"] == "fx/eurofxref-hist.zip"
    assert "extracted" in entry["derivation"] and "unmodified" in entry["derivation"]
    assert entry["acquisition_mode"] == "downloaded"


def test_the_fred_files_are_the_series_they_claim_to_be():
    for name, series_id in (("material/cotton/PCOTTINDUSDM.csv", "PCOTTINDUSDM"),
                            ("material/denim/WPU03420104.csv", "WPU03420104"),
                            ("material/leather/PHIDEUSDM.csv", "PHIDEUSDM")):
        header = (RAW_V5 / name).read_text(encoding="utf-8").split("\n", 1)[0].strip()
        assert header == f"observation_date,{series_id}", header


def test_provenance_is_the_shape_a_caller_should_carry():
    blob = manifest_reader.provenance(loaded(), "material/cotton/PCOTTINDUSDM.csv")
    assert blob["acquisition_mode"] == "downloaded"
    assert blob["hash_purpose"] == "source_artifact_integrity"
    assert blob["unit"] == "U.S. Cents per Pound"
    assert len(blob["sha256"]) == 64 and blob["retrieved_or_accessed"]


def test_asking_for_an_artifact_that_is_not_indexed_raises():
    raises(lambda: manifest_reader.artifact(loaded(), "material/silk/anything.csv"),
           "no artifact")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} source manifest unit tests passed")


if __name__ == "__main__":
    main()
