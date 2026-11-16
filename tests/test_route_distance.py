"""unit tests for the route-distance foundation.
runnable directly (python ...) or under pytest.

the selection rule and the location sets are checked on every machine. the matrix
and its provenance are checked whenever they exist. the live determinism check
needs the gitignored OSRM graph in tools/osrm; on a clone without it that one test
reports SKIP rather than pass, the same way the golden census does when shortened
"""

import hashlib
import json
from pathlib import Path

from src.dataset import osrm_engine
from src.dataset import route_distance as rd

MATRIX = rd.MATRIX_PATH
PROVENANCE = rd.PROVENANCE_PATH


def city(place, node_id=1):
    return {"place": place, "osm_node_id": node_id}


# ── selection rule ──────────────────────────────────────────────────────────

def test_accents_and_case_fold_to_the_same_key():
    assert rd.fold("Jyväskylä") == rd.fold("JYVASKYLA") == rd.fold("jyvaskyla")


def test_a_single_city_node_resolves():
    chosen, rule = rd.select_candidate("X", [city("city"), city("village", 2)])
    assert chosen["place"] == "city" and "place=city" in rule


def test_a_single_town_resolves_only_when_there_is_no_city():
    chosen, rule = rd.select_candidate("X", [city("town")])
    assert chosen["place"] == "town" and "place=town" in rule


def test_ambiguity_is_never_resolved_by_guessing():
    assert rd.select_candidate("X", [city("city", 1), city("city", 2)]) == (None, None)
    assert rd.select_candidate("X", [city("city", 1), city("city", 2), city("town", 3)]) == (None, None)
    assert rd.select_candidate("X", [city("town", 1), city("town", 2)]) == (None, None)
    assert rd.select_candidate("X", [city("village")]) == (None, None)


def test_the_location_sets_come_from_v4_read_only():
    locations = rd.unique_locations()
    expected = ["Espoo", "Helsinki", "Jyvaskyla", "Lahti", "Oulu", "Tampere", "Turku"]
    assert locations["manufacturer_locations"] == expected
    assert locations["delivery_cities"] == expected
    assert locations["all_locations"] == expected


# ── the stored matrix and provenance ────────────────────────────────────────

def matrix_rows():
    import csv
    with open(MATRIX, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_the_matrix_covers_every_manufacturer_and_delivery_pair():
    if not MATRIX.is_file():
        raise AssertionError(f"{MATRIX} missing; run route_distance.py all")
    rows = matrix_rows()
    assert len(rows) == 49
    assert tuple(rows[0].keys()) == rd.MATRIX_COLUMNS
    distance = {(r["origin_manufacturer_location"], r["destination_deliver_to"]): float(r["road_distance_km"])
                for r in rows}
    for (a, b), km in distance.items():
        assert (km == 0.0) == (a == b), (a, b, km)


def test_the_road_distances_are_plausible_finnish_intercity_distances():
    distance = {(r["origin_manufacturer_location"], r["destination_deliver_to"]): float(r["road_distance_km"])
                for r in matrix_rows()}
    assert 150 < distance[("Helsinki", "Tampere")] < 200
    assert 550 < distance[("Helsinki", "Oulu")] < 700
    assert 5 < distance[("Espoo", "Helsinki")] < 30
    assert 140 < distance[("Helsinki", "Turku")] < 190
    for (a, b), km in distance.items():
        if a != b:
            straight_line_floor = 5
            assert km > straight_line_floor, (a, b)


def test_the_provenance_records_every_pinned_input():
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["matrix"]["sha256"] == hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    assert provenance["osm_extract"]["source_url"] == osrm_engine.EXTRACT["source_url"]
    assert provenance["osm_extract"]["osm_data_timestamp"] == osrm_engine.EXTRACT["osm_data_timestamp"]
    assert provenance["osrm"]["version"] == osrm_engine.VERSION
    assert provenance["routing_profile"]["sha256_by_file"]["car.lua"] == osrm_engine.PROFILE_FILES["car.lua"]
    assert provenance["generation_date"]
    assert provenance["coordinates"]["sha256"] == hashlib.sha256(rd.COORDINATES_PATH.read_bytes()).hexdigest()
    assert "not assumed symmetric" in provenance["direction"]


def test_the_recorded_full_rebuild_produced_an_identical_matrix():
    check = json.loads(PROVENANCE.read_text(encoding="utf-8")).get("determinism_check")
    assert check is not None, "run route_distance.py determinism"
    assert check["identical"] is True
    assert check["matrix_sha256"] == check["rebuilt_matrix_sha256"] == \
        hashlib.sha256(MATRIX.read_bytes()).hexdigest()


def test_requerying_the_pinned_graph_reproduces_the_matrix_bytes():
    stamp = osrm_engine.DATA_DIR / osrm_engine.GRAPH_STAMP
    base = osrm_engine.DATA_DIR / osrm_engine.EXTRACT["file"].replace(".osm.pbf", ".osrm")
    if not stamp.is_file() or not Path(f"{base}.hsgr").is_file():
        print("    SKIP live requery, tools/osrm has no prepared graph on this machine")
        return
    coordinates = json.loads(rd.COORDINATES_PATH.read_text(encoding="utf-8"))
    with osrm_engine.routed(base, port=5057, announce=lambda *_: None) as url:
        rows, _, _ = rd.build_matrix(coordinates, coordinates["location_sets_from_v4"], url)
    assert rd.matrix_csv_bytes(rows) == MATRIX.read_bytes()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} route distance unit tests passed")


if __name__ == "__main__":
    main()
