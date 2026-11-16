"""build data/raw/v5 from the declaration in v5_sources.py

  PYTHONPATH=. python src/dataset/acquire_v5_sources.py
  PYTHONPATH=. python src/dataset/acquire_v5_sources.py --refresh   re-download
  PYTHONPATH=. python src/dataset/acquire_v5_sources.py --check     no network

the downloads are the only network access anywhere in this layer. once the tree
exists, every reader below (the historical lookup, the fx layer, and later the v5
material calibration) works from the pinned local files and never opens a socket

re-running without --refresh keeps the files already on disk and their recorded
retrieval dates, and rewrites only the metadata. that makes the step idempotent:
a second run produces the same tree, not a new set of retrieval dates

if a source cannot be reached the script stops and says so. it never writes a
placeholder, a reconstruction or a substitute for source data
"""

import argparse
import hashlib
import io
import json
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_ROOT
from src.dataset.v5_sources import (
    DOWNLOADS, FX_CROSS_CHECK, HASH_PURPOSE, HASH_PURPOSE_MEANING,
    MANUAL_ACCESS_DATE_BASIS, MANUAL_RECORD_ACCESS_DATE,
    MANUAL_RECORDS, group_of,
)

RAW_V5 = PROJECT_ROOT / "data" / "raw" / "v5"

# folders this builder owns. data/raw/v5/logistics is built by
# acquire_v5_logistics_sources.py and indexed by its own manifest
TREE = ("material/cotton", "material/polyester", "material/linen",
        "material/denim", "material/leather", "fx")

USER_AGENT = "chain-of-bid v5 source acquisition (academic research)"


class AcquisitionError(RuntimeError):
    """a source could not be retrieved or did not return what it should

    the caller stops here. nothing downstream may invent, reconstruct or
    substitute source data to get past it"""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    """stable json: same input, same bytes, so a re-run produces no diff"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path):
    if not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ssl_context():
    """certifi's roots, because the default store does not verify every source

    fred verifies against the system store here and the ecb does not. using one
    explicit bundle for both means the script behaves the same on any machine"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url, timeout=180):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout,
                                    context=_ssl_context()) as response:
            return response.read(), response.status
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AcquisitionError(f"could not retrieve {url}: {type(e).__name__}: {e}")


def check_payload(spec, data):
    """refuse anything that is not the artifact the declaration describes

    a series endpoint that quietly returns an html error page is the failure this
    catches, and it is the one that would otherwise be stored and hashed as if it
    were data"""
    if not data:
        raise AcquisitionError(f"{spec['url']} returned an empty body")
    if data[:1] == b"<":
        raise AcquisitionError(
            f"{spec['url']} returned html, not the expected artifact")

    header = spec.get("expected_header")
    if header:
        first = data.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
        if first != header:
            raise AcquisitionError(
                f"{spec['url']} first line is {first!r}, expected {header!r}")

    prefix = spec.get("expected_header_startswith")
    if prefix and "extract_member" not in spec:
        first = data.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
        if not first.startswith(prefix):
            raise AcquisitionError(
                f"{spec['url']} first line {first!r} does not start with {prefix!r}")

    if "extract_member" in spec:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise AcquisitionError(f"{spec['url']} is not a readable zip: {e}")
        names = archive.namelist()
        if spec["extract_member"] not in names:
            raise AcquisitionError(
                f"{spec['url']} has no member {spec['extract_member']!r}, "
                f"it contains {names}")
        member = archive.read(spec["extract_member"])
        prefix = spec.get("expected_header_startswith")
        first = member.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
        if prefix and not first.startswith(prefix):
            raise AcquisitionError(
                f"{spec['extract_member']} first line {first!r} does not start "
                f"with {prefix!r}")
        return member
    return None


def download_entries(refresh, offline, previous):
    """one manifest entry per downloaded artifact, plus any extracted member"""
    entries = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for spec in DOWNLOADS:
        path = RAW_V5 / spec["artifact"]
        prior = previous.get(spec["artifact"], {})
        member_bytes = None

        if path.is_file() and not refresh:
            data = path.read_bytes()
            retrieval_date = prior.get("retrieval_date")
            if retrieval_date is None:
                raise AcquisitionError(
                    f"{spec['artifact']} is on disk but no recorded retrieval date "
                    f"exists for it; re-run with --refresh to retrieve it properly")
            print(f"  keep     {spec['artifact']} (retrieved {retrieval_date})")
            if "extract_member" in spec:
                member_bytes = zipfile.ZipFile(io.BytesIO(data)).read(spec["extract_member"])
        elif offline:
            raise AcquisitionError(
                f"{spec['artifact']} is missing and --check forbids network access")
        else:
            print(f"  fetch    {spec['url']}")
            data, status = fetch(spec["url"])
            member_bytes = check_payload(spec, data)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            retrieval_date = today
            print(f"           {len(data)} bytes, http {status}, stored unmodified")

        entry = {
            "artifact": spec["artifact"],
            "acquisition_mode": "downloaded",
            "sha256": sha256_file(path),
            "hash_purpose": HASH_PURPOSE["downloaded"],
            "bytes": path.stat().st_size,
            "source": spec["source"],
            "source_url": spec["url"],
            "source_page": spec.get("series_page"),
            "series_id": spec.get("series_id"),
            "retrieval_date": retrieval_date,
            "unit": spec["unit"],
            "unit_confirmed_from": spec["unit_confirmed_from"],
            "frequency": spec["frequency"],
            "role": spec["role"],
            "licence_note": spec["licence_note"],
        }
        if spec.get("seasonal_adjustment"):
            entry["seasonal_adjustment"] = spec["seasonal_adjustment"]
        if spec.get("dataset_page"):
            entry["dataset_page"] = spec["dataset_page"]
        if spec["artifact"].startswith("fx/"):
            entry["cross_check"] = FX_CROSS_CHECK
        entries.append(entry)

        if member_bytes is not None:
            extracted = RAW_V5 / spec["extract_to"]
            extracted.parent.mkdir(parents=True, exist_ok=True)
            if not extracted.is_file() or extracted.read_bytes() != member_bytes:
                extracted.write_bytes(member_bytes)
                print(f"  extract  {spec['extract_to']} "
                      f"({len(member_bytes)} bytes, zip member "
                      f"{spec['extract_member']})")
            entries.append({
                "artifact": spec["extract_to"],
                "acquisition_mode": "downloaded",
                "sha256": sha256_bytes(member_bytes),
                "hash_purpose": HASH_PURPOSE["downloaded"],
                "bytes": len(member_bytes),
                "source": spec["source"],
                "source_url": spec["url"],
                "source_page": spec.get("series_page"),
                "series_id": spec.get("series_id"),
                "retrieval_date": retrieval_date,
                "unit": spec["unit"],
                "unit_confirmed_from": spec["unit_confirmed_from"],
                "frequency": spec["frequency"],
                "role": ("the readable form of the pinned download: the single "
                         "member of the stored zip, extracted byte for byte. this "
                         "is the file the fx layer reads"),
                "licence_note": spec["licence_note"],
                "derived_from": spec["artifact"],
                "derivation": (f"zip member {spec['extract_member']!r} extracted "
                               f"unmodified; no parsing, filtering or reformatting"),
                "cross_check": FX_CROSS_CHECK,
            })
    return entries


def manual_entries():
    """one manifest entry per hand-recorded value, hashing the local record file"""
    entries = []
    for spec in MANUAL_RECORDS:
        path = RAW_V5 / spec["artifact"]
        record = dict(spec["record"])
        record["acquisition_mode"] = "manual_record"
        record["access_date"] = MANUAL_RECORD_ACCESS_DATE
        record["access_date_basis"] = MANUAL_ACCESS_DATE_BASIS
        record["hash_purpose"] = HASH_PURPOSE["manual_record"]
        record["hash_purpose_means"] = HASH_PURPOSE_MEANING["record_integrity"]
        write_json(path, record)

        entry = {
            "artifact": spec["artifact"],
            "acquisition_mode": "manual_record",
            "sha256": sha256_file(path),
            "hash_purpose": HASH_PURPOSE["manual_record"],
            "bytes": path.stat().st_size,
            "source": record["source"],
            "source_url": record["source_url"],
            "page_or_section": record.get("page_or_section"),
            "access_date": MANUAL_RECORD_ACCESS_DATE,
            "access_date_basis": MANUAL_ACCESS_DATE_BASIS,
            "unit": record.get("unit"),
            "currency": record.get("currency"),
            "role": record.get("role"),
            "observation_count": len(record.get("observations", [])),
            "verification_note": record["verification_note"],
        }
        if record.get("observations_status"):
            entry["observations_status"] = record["observations_status"]
            entry["observations_status_note"] = record["observations_status_note"]
        entries.append(entry)
        print(f"  record   {spec['artifact']} "
              f"({entry['observation_count']} observations)")
    return entries


def write_source_files(entries):
    """one source.json per folder, holding every source stored in that folder

    a folder holds several distinct sources (cotton has a FRED series and a
    catalog), so one file per folder listing them beats a sidecar per artifact:
    there is one place to look, and the two hash meanings sit side by side where
    the difference is impossible to miss"""
    by_group = {}
    for entry in entries:
        by_group.setdefault(group_of(entry["artifact"]), []).append(entry)

    for group, group_entries in sorted(by_group.items()):
        write_json(RAW_V5 / group / "source.json", {
            "group": group,
            "hash_algorithm": "sha256",
            "hash_purpose_means": HASH_PURPOSE_MEANING,
            "sources": sorted(group_entries, key=lambda e: e["artifact"]),
        })
        print(f"  source   {group}/source.json ({len(group_entries)} sources)")


def write_manifest(entries):
    """the top-level index, following the orders_metadata.json convention

    orders_metadata.json indexes its four inputs as {name: sha256}. artifact_sha256
    is that same map for every stored artifact here, and artifacts carries the full
    per-source record beside it

    generated_at is carried forward when nothing on disk moved, so re-running the
    acquisition step is a true no-op rather than a one-line diff"""
    previous = read_json(RAW_V5 / "manifest.json") or {}
    sums = {e["artifact"]: e["sha256"] for e in sorted(entries, key=lambda e: e["artifact"])}
    if previous.get("artifact_sha256") == sums and previous.get("generated_at"):
        generated_at = previous["generated_at"]
    else:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "dataset_version": "v5",
        "generated_at": generated_at,
        "hash_algorithm": "sha256",
        "acquisition_modes": {
            "downloaded": HASH_PURPOSE_MEANING["source_artifact_integrity"],
            "manual_record": HASH_PURPOSE_MEANING["record_integrity"],
        },
        "read_this_first": (
            "the two acquisition modes carry hashes that mean different things, and "
            "the difference matters. a downloaded artifact's hash is source "
            "integrity: it proves the pinned file has not changed since it was "
            "retrieved. a manual record's hash is record integrity ONLY: it proves "
            "this repository's written record of a hand-read value has not been "
            "edited. it does NOT prove that the original web page, catalog or pdf "
            "still contains that value. do not cite a record-integrity hash as "
            "evidence about the state of an external source"),
        "artifact_sha256": sums,
        "artifacts": sorted(entries, key=lambda e: e["artifact"]),
        "sub_manifests": [{
            "group": "logistics",
            "manifest": "logistics/manifest.json",
            "status": "indexed by its own manifest",
            "reason": ("the v5 logistics method is decided (methodology section 7.1). its "
                       "tariff, line-haul, fuel-surcharge and distance sources are built by "
                       "src/dataset/acquire_v5_logistics_sources.py and verified through "
                       "logistics/manifest.json on the same conventions as this manifest"),
        }],
    }
    write_json(RAW_V5 / "manifest.json", payload)
    print(f"  manifest manifest.json ({len(entries)} artifacts)")
    return payload


def previous_entries():
    """the artifacts a previous run recorded, keyed by path

    used only to carry a retrieval date forward for a file that is already on
    disk, so re-running does not restamp an old download with today's date"""
    manifest = read_json(RAW_V5 / "manifest.json")
    if not manifest:
        return {}
    return {e["artifact"]: e for e in manifest.get("artifacts", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="re-download every source even if it is already stored")
    parser.add_argument("--check", action="store_true",
                        help="rebuild metadata from what is on disk, no network")
    args = parser.parse_args()

    for folder in TREE:
        (RAW_V5 / folder).mkdir(parents=True, exist_ok=True)

    try:
        entries = download_entries(args.refresh, args.check, previous_entries())
    except AcquisitionError as e:
        print(f"\nacquisition stopped: {e}", file=sys.stderr)
        print("nothing was fabricated, reconstructed or substituted. "
              "fix the access problem and re-run", file=sys.stderr)
        return 1

    entries.extend(manual_entries())
    write_source_files(entries)
    write_manifest(entries)

    missing = [e for e in entries if e.get("observations_status") == "missing"]
    if missing:
        print("\nblocked, provenance recorded but no observations stored:")
        for entry in missing:
            print(f"  {entry['artifact']}: {entry['observations_status_note']}")
    print("\ndata/raw/v5 is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
