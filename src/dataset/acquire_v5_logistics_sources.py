"""build data/raw/v5/logistics from the declaration in v5_logistics_sources.py

  PYTHONPATH=. python src/dataset/acquire_v5_logistics_sources.py

no network access. every tariff here is a manual record, and the one generated
artifact, distance/location_coordinates.json, is written beforehand by
route_distance.py from the pinned OSM extract. this step writes the record files,
one source.json per folder and the logistics manifest, and then verifies that
manifest with the same reader the material manifest uses. run again with nothing
changed it rewrites the same bytes

a table kept as printed text is parsed here and written as numbers beside the
text it came from, so every record shows both the evidence and its reading
"""

import re
import sys
from datetime import datetime, timezone

from config import PROJECT_ROOT
from src.dataset.acquire_v5_sources import read_json, sha256_file, write_json
from src.dataset.manifest import verify
from src.dataset.v5_logistics_sources import (
    COORDINATES_ARTIFACT, GROUP_FEEDS, MANUAL_ACCESS_DATE, MANUAL_RECORDS, NOT_STATED,
    UNREADABLE, group_of,
)
from src.dataset.v5_sources import HASH_PURPOSE, HASH_PURPOSE_MEANING

RAW_V5 = PROJECT_ROOT / "data" / "raw" / "v5"
RAW_LOGISTICS = RAW_V5 / "logistics"
GROUPS = tuple(GROUP_FEEDS)

DATED_PERCENT = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d+,\d+)\s*%")


class LogisticsSourceError(ValueError):
    """a declared table does not parse into the shape it was declared as"""


def parse_decimal(text):
    return float(text.replace(",", "."))


def parse_lavarahti_table(table_text, lower_text, upper_text):
    """the Lavarahti freight table: distance bands and one price row per weight bracket"""
    lowers = [int(x) for x in lower_text.split()]
    uppers = [int(x) for x in upper_text.split()]
    if len(lowers) != len(uppers):
        raise LogisticsSourceError("the distance band header rows differ in length")
    bands = [{"km_lower": lo, "km_upper": hi} for lo, hi in zip(lowers, uppers)]

    rows = []
    for line in table_text.strip().splitlines():
        tokens = line.split()
        if len(tokens) != 6 + len(bands) or tokens[1] != "-" or tokens[4] != "-":
            raise LogisticsSourceError(f"unexpected Lavarahti row: {line[:60]!r}")
        rows.append({"kg_lower": int(tokens[0]), "kg_upper": int(tokens[2]),
                     "lvm_lower": parse_decimal(tokens[3]),
                     "lvm_upper": parse_decimal(tokens[5]),
                     "eur_by_band": [parse_decimal(t) for t in tokens[6:]]})
    for earlier, later in zip(rows, rows[1:]):
        if later["kg_lower"] != earlier["kg_upper"] + 1:
            raise LogisticsSourceError(
                f"weight brackets are not contiguous at {earlier['kg_upper']} kg")
    return {"km_bands": bands, "rows": rows}


def parse_dated_percentages(text):
    """every 'd.m.yyyy nn,nn %' in a published list, as printed and as numbers"""
    out, seen = [], set()
    for match in DATED_PERCENT.finditer(text):
        day, month, year, percent = match.groups()
        iso = f"{year}-{int(month):02d}-{int(day):02d}"
        if iso in seen:
            raise LogisticsSourceError(f"the date {iso} appears twice in one series")
        seen.add(iso)
        out.append({"date": iso, "percent": parse_decimal(percent),
                    "printed_as": match.group(0)})
    return out


def build_record(spec):
    record = dict(spec["record"])
    record["acquisition_mode"] = "manual_record"
    record["access_date"] = MANUAL_ACCESS_DATE
    record["hash_purpose"] = HASH_PURPOSE["manual_record"]
    record["hash_purpose_means"] = HASH_PURPOSE_MEANING["record_integrity"]
    record["calibration_feeds"] = GROUP_FEEDS[group_of(spec["artifact"])]
    record["metadata_convention"] = {
        "not_stated": NOT_STATED,
        "unreadable": UNREADABLE,
        "meaning": ("a field reading 'not stated in source' was read and is absent from the "
                    "source; a field reading 'not verified' could not be read at all"),
    }
    if "freight_table_as_printed" in record:
        record["freight_table_parsed"] = parse_lavarahti_table(
            record["freight_table_as_printed"], record["km_band_lower_as_printed"],
            record["km_band_upper_as_printed"])
    if "series_as_printed" in record:
        record["series_parsed"] = {name: parse_dated_percentages(text)
                                   for name, text in record["series_as_printed"].items()}
    return record


def observation_count(record):
    if "freight_table_parsed" in record:
        table = record["freight_table_parsed"]
        return len(table["rows"]) * len(table["km_bands"])
    if "series_parsed" in record:
        return sum(len(s) for s in record["series_parsed"].values())
    return sum(len(o["rows"]) if "rows" in o else 1 for o in record.get("observations", []))


def manual_entry(spec, record):
    path = RAW_LOGISTICS / spec["artifact"]
    entry = {
        "artifact": spec["artifact"],
        "acquisition_mode": "manual_record",
        "sha256": sha256_file(path),
        "hash_purpose": HASH_PURPOSE["manual_record"],
        "bytes": path.stat().st_size,
        "source": record["source"],
        "source_url": record["source_url"],
        "page_or_section": record["page_or_section"],
        "access_date": record["access_date"],
        "access_date_basis": record["access_date_basis"],
        "unit": record.get("unit"),
        "currency": record.get("currency"),
        "role": record["role"],
        "calibration_feeds": record["calibration_feeds"],
        "reverification_status": record["reverification_status"],
        "observation_count": observation_count(record),
        "verification_note": record["verification_note"],
    }
    for optional in ("calibration_bound", "observations_status", "source_url_given_in_task"):
        if record.get(optional):
            entry[optional] = record[optional]
    return entry


def coordinates_entry():
    """the geocoded coordinates, when route_distance.py has produced them"""
    path = RAW_LOGISTICS / COORDINATES_ARTIFACT
    document = read_json(path)
    if document is None:
        return None
    prov = document["provenance"]
    return {
        "artifact": COORDINATES_ARTIFACT,
        "acquisition_mode": "downloaded",
        "sha256": sha256_file(path),
        "hash_purpose": HASH_PURPOSE["downloaded"],
        "bytes": path.stat().st_size,
        "source": prov["source"],
        "source_url": prov["source_url"],
        "retrieval_date": prov["retrieval_date"],
        "unit": prov["unit"],
        "unit_confirmed_from": prov["unit_confirmed_from"],
        "frequency": prov["frequency"],
        "role": prov["role"],
        "licence_note": prov["licence_note"],
        "calibration_feeds": GROUP_FEEDS["distance"],
        "derived_from": prov["derived_from"],
        "derivation": prov["derivation"],
        "location_count": len(document["locations"]),
        "unresolved_count": len(document["unresolved"]),
    }


def write_source_files(entries):
    for group in GROUPS:
        group_entries = sorted((e for e in entries if group_of(e["artifact"]) == group),
                               key=lambda e: e["artifact"])
        payload = {
            "group": f"logistics/{group}",
            "calibration_feeds": GROUP_FEEDS[group],
            "hash_algorithm": "sha256",
            "hash_purpose_means": HASH_PURPOSE_MEANING,
            "sources": group_entries,
        }
        if not group_entries:
            payload["status"] = ("incomplete: nothing was generated for this group, see "
                                 "incomplete in ../manifest.json")
        write_json(RAW_LOGISTICS / group / "source.json", payload)
        print(f"  source   {group}/source.json ({len(group_entries)} sources)")


def write_manifest(entries, incomplete):
    """the logistics index, on the conventions of data/raw/v5/manifest.json"""
    parent = read_json(RAW_V5 / "manifest.json")
    if parent is None:
        raise LogisticsSourceError("data/raw/v5/manifest.json is missing, build it first")

    previous = read_json(RAW_LOGISTICS / "manifest.json") or {}
    sums = {e["artifact"]: e["sha256"] for e in sorted(entries, key=lambda e: e["artifact"])}
    if previous.get("artifact_sha256") == sums and previous.get("generated_at"):
        generated_at = previous["generated_at"]
    else:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "dataset_version": "v5",
        "group": "logistics",
        "parent_manifest": "../manifest.json",
        "generated_at": generated_at,
        "hash_algorithm": "sha256",
        "acquisition_modes": parent["acquisition_modes"],
        "read_this_first": parent["read_this_first"],
        "logistics_notes": [
            ("every tariff record carries tariff_metadata with vat_status, effective_date, "
             "currency_as_published, unit_basis, service_category and "
             "fuel_surcharge_treatment, each with the source's own wording"),
            ("'not stated in source' means the source was read and does not state the "
             "field; 'not verified' means the source could not be read at all"),
            ("fuel_surcharge records are evidence of published mechanisms and percentages. "
             "none of them is a per-agent value, and no common value is implied"),
        ],
        "artifact_sha256": sums,
        "artifacts": sorted(entries, key=lambda e: e["artifact"]),
        "reverification_blocked": sorted(e["artifact"] for e in entries
                                         if str(e.get("reverification_status", ""))
                                         .startswith("blocked")),
        "incomplete": incomplete,
    }
    write_json(RAW_LOGISTICS / "manifest.json", payload)
    print(f"  manifest logistics/manifest.json ({len(entries)} artifacts)")


def main():
    for group in GROUPS:
        (RAW_LOGISTICS / group).mkdir(parents=True, exist_ok=True)

    entries = []
    for spec in MANUAL_RECORDS:
        record = build_record(spec)
        write_json(RAW_LOGISTICS / spec["artifact"], record)
        entries.append(manual_entry(spec, record))
        print(f"  record   {spec['artifact']} ({entries[-1]['observation_count']} values)")

    incomplete = []
    coordinates = coordinates_entry()
    if coordinates is None:
        incomplete.append({
            "group": "distance",
            "artifact": COORDINATES_ARTIFACT,
            "status": "not generated",
            "reason": ("src/dataset/route_distance.py geocode has not produced the "
                       "coordinates; see the report for why"),
        })
        print(f"  missing  {COORDINATES_ARTIFACT}, recorded as incomplete")
    else:
        entries.append(coordinates)
        print(f"  indexed  {COORDINATES_ARTIFACT} ({coordinates['location_count']} locations, "
              f"{coordinates['unresolved_count']} unresolved)")

    write_source_files(entries)
    write_manifest(entries, incomplete)

    report = verify(RAW_LOGISTICS / "manifest.json")
    if report["problems"]:
        print("\nlogistics manifest does not verify:\n  " + "\n  ".join(report["problems"]),
              file=sys.stderr)
        return 1
    print(f"\nlogistics manifest verifies: {report['artifacts']} artifacts, no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
