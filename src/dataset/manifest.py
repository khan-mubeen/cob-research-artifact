"""read and verify a source manifest, offline

the manifest is only worth writing if something checks it. this module is the
reader every consumer of data/raw/v5 goes through, so a tampered artifact, a
missing required field or an invented acquisition mode is caught at load time
rather than discovered in a price six steps later

it takes explicit paths. it does not import config or the dataset context, and
it opens no sockets: verifying a manifest is a local operation over local files
"""

import hashlib
import json
from pathlib import Path

ACQUISITION_MODES = ("downloaded", "manual_record")

HASH_PURPOSES = {
    "downloaded": "source_artifact_integrity",
    "manual_record": "record_integrity",
}

# every artifact entry carries these whatever its mode
COMMON_REQUIRED = ("artifact", "acquisition_mode", "sha256", "hash_purpose",
                   "source", "source_url", "unit", "role")

MODE_REQUIRED = {
    "downloaded": ("retrieval_date", "unit_confirmed_from", "frequency"),
    "manual_record": ("access_date", "access_date_basis", "verification_note",
                      "page_or_section", "currency"),
}


class ManifestError(ValueError):
    """the manifest, or what it points at, does not hold up

    raised rather than warned: an unverifiable provenance record is worse than
    no provenance record, because it looks like one"""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(manifest_path):
    """the parsed manifest, with its structure checked but not its files"""
    path = Path(manifest_path)
    if not path.is_file():
        raise ManifestError(f"no manifest at {path}")
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise ManifestError(f"{path} is not readable json: {e}")

    for key in ("dataset_version", "hash_algorithm", "acquisition_modes",
                "artifact_sha256", "artifacts"):
        if key not in manifest:
            raise ManifestError(f"{path} has no {key!r}")
    if manifest["hash_algorithm"] != "sha256":
        raise ManifestError(
            f"{path} uses {manifest['hash_algorithm']!r}, this reader only "
            f"verifies sha256")

    declared_modes = set(manifest["acquisition_modes"])
    if declared_modes != set(ACQUISITION_MODES):
        raise ManifestError(
            f"{path} declares acquisition modes {sorted(declared_modes)}, "
            f"the schema allows exactly {list(ACQUISITION_MODES)}")
    return manifest


def entry_errors(entry):
    """every way one artifact entry fails the schema"""
    problems = []
    mode = entry.get("acquisition_mode")
    if mode not in ACQUISITION_MODES:
        problems.append(
            f"acquisition_mode {mode!r} is not one of {list(ACQUISITION_MODES)}")
        return problems

    for key in COMMON_REQUIRED + MODE_REQUIRED[mode]:
        if key not in entry:
            problems.append(f"missing {key}")
        elif entry[key] is None or (isinstance(entry[key], str)
                                    and not entry[key].strip()):
            # a null unit on a rule record is legitimate, a null hash is not
            if key not in ("unit", "currency"):
                problems.append(f"{key} is empty")

    expected_purpose = HASH_PURPOSES[mode]
    if entry.get("hash_purpose") != expected_purpose:
        problems.append(
            f"hash_purpose is {entry.get('hash_purpose')!r}, a {mode} artifact's "
            f"hash is {expected_purpose}")

    digest = entry.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64:
        problems.append(f"sha256 {digest!r} is not a sha-256 digest")
    return problems


def verify(manifest_path, root=None):
    """check the manifest against the files it indexes

    returns a report: {"artifacts": n, "problems": [...]}. an empty problem list
    means every indexed file exists and hashes to what the manifest recorded"""
    manifest = load(manifest_path)
    base = Path(root) if root is not None else Path(manifest_path).parent
    problems = []

    entries = manifest["artifacts"]
    index = manifest["artifact_sha256"]
    listed = {e.get("artifact") for e in entries}
    if listed != set(index):
        missing = sorted(set(index) - listed)
        extra = sorted(listed - set(index))
        if missing:
            problems.append(f"artifact_sha256 indexes unlisted artifacts: {missing}")
        if extra:
            problems.append(f"artifacts not in artifact_sha256: {extra}")

    for entry in entries:
        name = entry.get("artifact", "<unnamed>")
        for problem in entry_errors(entry):
            problems.append(f"{name}: {problem}")

        path = base / name
        if not path.is_file():
            problems.append(f"{name}: indexed but not on disk at {path}")
            continue
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            problems.append(
                f"{name}: sha256 on disk is {actual}, manifest records "
                f"{entry.get('sha256')}")
        if index.get(name) != entry.get("sha256"):
            problems.append(
                f"{name}: artifact_sha256 and the artifact entry disagree")

    return {"artifacts": len(entries), "problems": problems}


def require_verified(manifest_path, root=None):
    """the manifest, or a raise naming everything that is wrong with it"""
    report = verify(manifest_path, root)
    if report["problems"]:
        raise ManifestError(
            f"{manifest_path} does not verify:\n  " + "\n  ".join(report["problems"]))
    return load(manifest_path)


def artifact(manifest, name):
    """one entry by path, or a raise. no partial matching, no guessing"""
    for entry in manifest["artifacts"]:
        if entry["artifact"] == name:
            return entry
    raise ManifestError(
        f"no artifact {name!r} in the manifest, it indexes "
        f"{sorted(e['artifact'] for e in manifest['artifacts'])}")


def provenance(manifest, name, root=None):
    """the provenance a caller should carry beside a value it read from a file

    this is what the fx layer returns alongside a rate: which file, which hash,
    what that hash means, and where the file came from"""
    entry = artifact(manifest, name)
    return {
        "artifact": entry["artifact"],
        "source": entry["source"],
        "source_url": entry["source_url"],
        "acquisition_mode": entry["acquisition_mode"],
        "sha256": entry["sha256"],
        "hash_purpose": entry["hash_purpose"],
        "unit": entry.get("unit"),
        "retrieved_or_accessed": entry.get("retrieval_date") or entry.get("access_date"),
    }


def manual_records(manifest):
    """the hand-recorded entries, so a caller can see which values are manual"""
    return [e for e in manifest["artifacts"]
            if e["acquisition_mode"] == "manual_record"]


def blocked_records(manifest):
    """manual records whose observations are declared missing

    these are provenance without data. anything that tries to price from one
    should stop, not carry on with an empty series"""
    return [e for e in manual_records(manifest)
            if e.get("observations_status") == "missing"]
