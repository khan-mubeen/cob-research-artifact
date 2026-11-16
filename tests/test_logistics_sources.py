"""unit tests for the v5 logistics source records and their manifest.
runnable directly (python ...) or under pytest.

the records are only useful if three things hold. every tariff figure carries the
six metadata fields it needs before any comparison, in the source's own words or
explicitly marked as absent or unreadable. the supplied values and verification
notes are exactly the ones given, not paraphrases. and the manifest verifies on
the same conventions as the material manifest, with the difference between a
record-integrity hash and a source-integrity hash kept intact
"""

import copy
import json
import shutil
import tempfile
from pathlib import Path

from src.dataset import manifest as manifest_reader
from src.dataset import osrm_engine
from src.dataset.acquire_v5_logistics_sources import (
    LogisticsSourceError, parse_dated_percentages, parse_lavarahti_table,
)
from src.dataset.v5_logistics_sources import MANUAL_RECORDS, NOT_STATED, UNREADABLE

LOGISTICS = Path(__file__).resolve().parent.parent / "data" / "raw" / "v5" / "logistics"
MANIFEST = LOGISTICS / "manifest.json"
REQUIRED_METADATA = ("vat_status", "effective_date", "currency_as_published", "unit_basis",
                     "service_category", "fuel_surcharge_treatment")
TARIFF_GROUPS = ("rate_calibration", "line_haul_rate", "fuel_surcharge")


def rec(artifact):
    with open(LOGISTICS / artifact, encoding="utf-8") as f:
        return json.load(f)


def raises(fn, exc, fragment):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in the error, got {e}"
        return
    raise AssertionError(f"expected {exc.__name__} mentioning {fragment!r}")


# ── the tree and its manifest ────────────────────────────────────────────────

def test_the_requested_folder_structure_exists():
    for group in ("rate_calibration", "line_haul_rate", "fuel_surcharge", "distance"):
        assert (LOGISTICS / group / "source.json").is_file(), group
    for name in ("rate_calibration/posti_parcel.json", "rate_calibration/posti_express_freight.json",
                 "rate_calibration/postnord_parcel_zones.json",
                 "rate_calibration/matkahuolto_lavarahti.json",
                 "line_haul_rate/upply_corridor_rates.json",
                 "fuel_surcharge/postnord_fuel_surcharge.json",
                 "distance/location_coordinates.json"):
        assert (LOGISTICS / name).is_file(), name


def test_the_logistics_manifest_verifies_against_the_stored_files():
    report = manifest_reader.verify(MANIFEST)
    assert report["problems"] == [], report["problems"]
    assert report["artifacts"] == len(MANUAL_RECORDS) + 1


def test_the_manifest_reuses_the_material_manifest_hash_conventions_exactly():
    logistics = manifest_reader.load(MANIFEST)
    parent = manifest_reader.load(LOGISTICS.parent / "manifest.json")
    assert logistics["acquisition_modes"] == parent["acquisition_modes"]
    assert logistics["read_this_first"] == parent["read_this_first"]
    assert set(logistics["acquisition_modes"]) == {"downloaded", "manual_record"}


def test_every_tariff_is_a_manual_record_and_only_the_coordinates_are_downloaded():
    for entry in manifest_reader.load(MANIFEST)["artifacts"]:
        if entry["artifact"] == "distance/location_coordinates.json":
            assert entry["acquisition_mode"] == "downloaded"
            assert entry["hash_purpose"] == "source_artifact_integrity"
        else:
            assert entry["acquisition_mode"] == "manual_record", entry["artifact"]
            assert entry["hash_purpose"] == "record_integrity"


def test_a_tampered_tariff_record_is_caught():
    with tempfile.TemporaryDirectory() as tmp:
        copy_dir = Path(tmp) / "logistics"
        shutil.copytree(LOGISTICS, copy_dir)
        target = copy_dir / "rate_calibration" / "posti_parcel.json"
        record = json.loads(target.read_text(encoding="utf-8"))
        record["observations"][0]["eur_per_item"] = 5.80
        target.write_text(json.dumps(record, indent=2), encoding="utf-8")
        report = manifest_reader.verify(copy_dir / "manifest.json")
        assert any("posti_parcel.json" in p for p in report["problems"]), report["problems"]


def test_the_blocked_reverification_is_listed_in_the_manifest():
    assert manifest_reader.load(MANIFEST)["reverification_blocked"] == [
        "rate_calibration/postnord_parcel_zones.json"]


# ── the metadata block every tariff needs ───────────────────────────────────

def test_every_tariff_figure_carries_all_six_metadata_fields():
    for spec in MANUAL_RECORDS:
        if spec["artifact"].split("/")[0] not in TARIFF_GROUPS:
            continue
        metadata = rec(spec["artifact"])["tariff_metadata"]
        for field in REQUIRED_METADATA:
            assert field in metadata, f"{spec['artifact']} has no {field}"
            value, wording = metadata[field]["value"], metadata[field]["source_wording"]
            assert isinstance(value, str) and value.strip(), f"{spec['artifact']} {field} value is blank"
            assert isinstance(wording, str) and wording.strip(), f"{spec['artifact']} {field} wording is blank"


def test_absent_fields_say_not_stated_and_unreadable_fields_say_not_verified():
    upply = rec("line_haul_rate/upply_corridor_rates.json")["tariff_metadata"]
    assert upply["vat_status"]["value"] == NOT_STATED
    assert upply["fuel_surcharge_treatment"]["value"] == NOT_STATED
    assert upply["payload_or_load_factor"]["value"] == NOT_STATED
    assert rec("rate_calibration/posti_parcel.json")["tariff_metadata"][
        "fuel_surcharge_treatment"]["value"] == NOT_STATED
    postnord = rec("rate_calibration/postnord_parcel_zones.json")["tariff_metadata"]
    assert postnord["vat_status"]["value"] == UNREADABLE
    assert postnord["effective_date"]["value"] == UNREADABLE


def test_vat_and_fuel_treatment_are_recorded_from_the_sources_own_wording():
    parcel = rec("rate_calibration/posti_parcel.json")["tariff_metadata"]
    assert parcel["vat_status"]["value"] == "exclusive"
    assert "added to prices during invoicing" in parcel["vat_status"]["source_wording"]
    freight = rec("rate_calibration/posti_express_freight.json")["tariff_metadata"]
    assert freight["vat_status"]["value"] == "exclusive"
    assert freight["fuel_surcharge_treatment"]["value"] == "charged separately"
    assert "veloitetaan erikseen" in freight["fuel_surcharge_treatment"]["source_wording"]
    lavarahti = rec("rate_calibration/matkahuolto_lavarahti.json")["tariff_metadata"]
    assert "alv 0%" in lavarahti["vat_status"]["source_wording"]
    assert lavarahti["effective_date"]["value"] == "2024-11-01"


# ── the supplied values and notes, exactly ──────────────────────────────────

def test_the_supplied_parcel_prices_are_recorded_exactly():
    parcel = {o["service"]: (o["eur_per_item"], o["eur_per_kg"])
              for o in rec("rate_calibration/posti_parcel.json")["observations"]}
    assert parcel == {"Postal Parcel": (5.90, 0.30), "Express Parcel": (8.90, 0.30),
                      "Home Parcel": (10.90, 0.30)}
    postnord = {o["zone"]: o["eur_per_parcel"]
                for o in rec("rate_calibration/postnord_parcel_zones.json")["observations"]}
    assert postnord == {"A": 12.00, "B": 14.00, "C": 19.00}


def test_the_given_verification_notes_are_verbatim():
    expected = {
        "rate_calibration/posti_parcel.json": (
            "lower-bound calibration evidence; these parcel services have substantially lower "
            "shipment-weight limits (25-35kg) than this simulation's typical 50-1000+ kg "
            "shipments, per docs/data_synthesis_methodology.md Section 7.1"),
        "rate_calibration/postnord_parcel_zones.json": (
            "lower-bound calibration evidence, same category caveat as Posti parcel; "
            "zone-and-weight-bracket structure, not a constant EUR/kg rate"),
        "rate_calibration/posti_express_freight.json": (
            "upper-bound calibration evidence; prices by transport-unit type and delivery zone "
            "(0-5), not by a constant per-kg rate; record whatever specific zone/unit prices are "
            "stated on this page with their exact wording, do not force them into an F+alpha "
            "shape here, that translation happens in Part 3"),
        "rate_calibration/matkahuolto_lavarahti.json": (
            "upper-bound calibration evidence for heavier shipments outside ordinary parcel "
            "weight limits; record the actual tariff structure found, do not assume it matches "
            "Posti's zone count or unit types"),
        "fuel_surcharge/postnord_fuel_surcharge.json": (
            "this is SOURCE EVIDENCE of a real fuel-surcharge mechanism, tied to the previous "
            "month's diesel consumer price index from Statistics Finland. It is not "
            "automatically the surcharge value for every simulated logistics agent. Huong "
            "expects this component to differ between agents; record the mechanism and whatever "
            "historical percentages are actually published, do not produce or imply a single "
            "common value here."),
    }
    for artifact, note in expected.items():
        assert rec(artifact)["verification_note"] == note, artifact


def test_the_corridor_record_holds_source_values_only():
    upply = rec("line_haul_rate/upply_corridor_rates.json")
    assert {(o["page"], o["rate_type"], o["corridor"], o["eur_per_km"]) for o in upply["observations"]} == {
        (22, "contract", "Warsaw to Duisburg", 1.45), (22, "contract", "Duisburg to Lille", 2.58),
        (22, "contract", "Duisburg to Prague", 1.73), (22, "contract", "Antwerp to Duisburg", 3.05),
        (25, "spot", "Warsaw to Duisburg", 1.51), (25, "spot", "Duisburg to Lille", 2.89),
        (25, "spot", "Duisburg to Prague", 2.00), (25, "spot", "Antwerp to Duisburg", 3.19)}
    allowed = {"page", "rate_type", "corridor", "index_points", "eur_per_km", "printed_as"}
    for obs in upply["observations"]:
        assert set(obs) == allowed, f"derived field in a raw observation: {set(obs) - allowed}"
        assert "€" in obs["printed_as"]
    assert not any(k for k in upply if "derivation" in k or "candidate" in k or "payload_t" in k)


# ── the tables parsed from printed text ─────────────────────────────────────

def test_the_lavarahti_table_parses_to_the_printed_cells():
    table = rec("rate_calibration/matkahuolto_lavarahti.json")["freight_table_parsed"]
    assert len(table["rows"]) == 60 and len(table["km_bands"]) == 30
    assert table["km_bands"][0] == {"km_lower": 1, "km_upper": 49}
    rows = {r["kg_lower"]: r for r in table["rows"]}
    assert rows[100]["eur_by_band"][0] == 34.72
    assert rows[950]["eur_by_band"][0] == 119.62
    assert rows[14000]["eur_by_band"][-1] == 2246.14
    assert rows[0]["kg_upper"] == 79 and rows[14000]["kg_upper"] == 14999


def test_the_posti_freight_table_holds_every_zone_and_unit_price():
    record = rec("rate_calibration/posti_express_freight.json")
    express = next(o for o in record["observations"] if o["service_printed_fi"] == "Express-rahti 14")
    assert [r["zone"] for r in express["rows"]] == [0, 1, 2, 3, 4, 5]
    assert express["rows"][0]["eur"]["pallet_first_unit"] == 89.59
    assert express["rows"][5]["eur"]["furniture_pallet_next_units"] == 374.26
    assert record["source_url_given_in_task"].endswith("/parcels-and-logistics/freight")


def test_dated_percentages_parse_as_printed_and_refuse_duplicates():
    parsed = parse_dated_percentages("2026 16.7.2026 18,21% 9.7.2026 18,08 % 2025 1.12.2025 13,64 %")
    assert [(p["date"], p["percent"]) for p in parsed] == [
        ("2026-07-16", 18.21), ("2026-07-09", 18.08), ("2025-12-01", 13.64)]
    assert parsed[0]["printed_as"] == "16.7.2026 18,21%"
    raises(lambda: parse_dated_percentages("1.1.2026 1,00 % 1.1.2026 2,00 %"),
           LogisticsSourceError, "appears twice")


def test_a_malformed_lavarahti_row_is_refused():
    raises(lambda: parse_lavarahti_table("0 - 79 0,00 - 0,24 1,00", "1 50", "49 99"),
           LogisticsSourceError, "unexpected Lavarahti row")
    raises(lambda: parse_lavarahti_table("0 - 79 0,00 - 0,24 1,00\n90 - 99 0,25 - 0,30 2,00", "1", "49"),
           LogisticsSourceError, "not contiguous")


# ── fuel surcharge is evidence, never an agent value ────────────────────────

def test_the_fuel_surcharge_series_are_recorded_in_full():
    matkahuolto = rec("fuel_surcharge/matkahuolto_fuel_surcharge.json")["series_parsed"]
    assert {k: len(v) for k, v in matkahuolto.items()} == {
        "Paketti- ja kuljetuspalvelut kotimaassa ja ulkomaille": 39, "Lavarahti": 79, "Rahti": 27}
    assert len(rec("fuel_surcharge/pakettipiste_fuel_surcharge.json")["observations"]) == 21
    postnord = rec("fuel_surcharge/postnord_fuel_surcharge.json")["observations"]
    assert postnord[0] == {"starting_from": "2026-02-01", "parcels_percent": 4.6,
                           "pallets_and_groupage_percent": 6.7}
    assert len(postnord) == 14


def test_no_fuel_record_assigns_or_implies_a_value_for_an_agent():
    for spec in MANUAL_RECORDS:
        if not spec["artifact"].startswith("fuel_surcharge/"):
            continue
        text = json.dumps(rec(spec["artifact"]))
        assert "agent_id" not in text and '"L01"' not in text, spec["artifact"]
        assert "not a per-agent value" in rec(spec["artifact"])["verification_note"] or \
            "not automatically the surcharge value for every simulated logistics agent" in \
            rec(spec["artifact"])["verification_note"]


# ── the geocoded coordinates ────────────────────────────────────────────────

def test_the_coordinates_cover_every_v4_location_with_provenance():
    from src.dataset.route_distance import unique_locations

    document = rec("distance/location_coordinates.json")
    assert document["unresolved"] == []
    names = {r["location"] for r in document["locations"]}
    assert names == set(unique_locations()["all_locations"])
    for r in document["locations"]:
        assert 59.5 < r["lat"] < 70.5 and 19.0 < r["lon"] < 32.0, r
        assert r["place"] == "city"
    derived = document["provenance"]["derived_from"]
    assert derived["sha256"] == osrm_engine.EXTRACT["sha256"]
    assert derived["published_md5"] == osrm_engine.EXTRACT["published_md5"]
    jyvaskyla = next(r for r in document["locations"] if r["location"] == "Jyvaskyla")
    assert jyvaskyla["osm_name"] == "Jyväskylä"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} logistics source unit tests passed")


if __name__ == "__main__":
    main()
