"""pinned coordinates and road distances for every manufacturer location and
delivery city, from the pinned OSM extract and the pinned OSRM engine

  PYTHONPATH=. python src/dataset/route_distance.py geocode
  PYTHONPATH=. python src/dataset/route_distance.py matrix
  PYTHONPATH=. python src/dataset/route_distance.py all
  PYTHONPATH=. python src/dataset/route_distance.py determinism

geocoding reads place nodes straight out of the same extract the router is built
from, so the coordinates and the road graph cannot come from different maps. no
geocoding service is called, commercial or otherwise

the v4 location names are read from data/synthetic/v4, read-only. they are ascii
where OpenStreetMap carries the finnish spelling ("Jyvaskyla", "Jyväskylä"), so
a name matches when the two agree after accents are removed and case is folded.
a location resolves only when exactly one place=city node carries the name, or,
when there is no city node at all, exactly one place=town node. anything else is
written to unresolved with every candidate and no coordinate

this part is non-blocking by design: every failure ends in a clear message and a
non-zero exit, and the logistics source builder records whatever was not
generated instead of stopping
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.dataset import osrm_engine
from src.dataset.acquire_v5_sources import read_json, write_json

V4_DIR = PROJECT_ROOT / "data" / "synthetic" / "v4"
COORDINATES_PATH = PROJECT_ROOT / "data" / "raw" / "v5" / "logistics" / "distance" / "location_coordinates.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "v5" / "logistics"
MATRIX_PATH = PROCESSED_DIR / "manufacturer_to_city_distance_matrix.csv"
PROVENANCE_PATH = PROCESSED_DIR / "distance_provenance.json"

PLACE_PREFERENCE = ("city", "town")
MATRIX_COLUMNS = ("origin_manufacturer_location", "destination_deliver_to",
                  "road_distance_m", "road_distance_km")


class RouteDistanceError(RuntimeError):
    """the coordinates or the matrix could not be produced from pinned inputs"""


def rel(path):
    return str(Path(path).relative_to(PROJECT_ROOT)).replace("\\", "/")


def fold(name):
    return "".join(c for c in unicodedata.normalize("NFKD", name)
                   if not unicodedata.combining(c)).casefold()


def unique_locations(v4_dir=V4_DIR):
    """every location name v5 routing needs, from the v4 files, read-only"""
    manufacturers = pd.read_csv(Path(v4_dir) / "manufacturer_agents.csv", usecols=["location"])
    orders = pd.read_csv(Path(v4_dir) / "orders.csv", usecols=["deliver_to"])
    logistics = pd.read_csv(Path(v4_dir) / "logistics_agents.csv", usecols=["service_regions"])

    manufacturer_locations = sorted(map(str, manufacturers["location"].unique()))
    order_cities = sorted(map(str, orders["deliver_to"].unique()))
    regions = sorted({r for s in logistics["service_regions"] for r in str(s).split("|")})
    delivery = sorted(set(order_cities) | set(regions))
    return {
        "manufacturer_locations": manufacturer_locations,
        "order_deliver_to": order_cities,
        "logistics_service_regions": regions,
        "delivery_cities": delivery,
        "all_locations": sorted(set(manufacturer_locations) | set(delivery)),
        "read_from": [rel(Path(v4_dir) / "manufacturer_agents.csv"),
                      rel(Path(v4_dir) / "orders.csv"),
                      rel(Path(v4_dir) / "logistics_agents.csv")],
    }


def select_candidate(name, candidates):
    """(chosen, rule) or (None, None), by the documented selection rule

    a tier with more than one node is ambiguous and stops the search there: two
    city nodes with the same name are not rescued by a single town node"""
    for place in PLACE_PREFERENCE:
        tier = [c for c in candidates if c["place"] == place]
        if len(tier) == 1:
            return tier[0], (f"the only place={place} node whose name matches {name!r} "
                             f"after removing accents and folding case")
        if len(tier) > 1:
            return None, None
    return None, None


def geocode(names, pbf_path):
    """(resolved, unresolved) for the names, read from place nodes in the extract"""
    osrm_engine.resolve_pylib()
    import osmium

    wanted = {fold(n): n for n in names}
    if len(wanted) != len(names):
        raise RouteDistanceError(f"two location names fold to the same key: {names}")
    candidates = {n: [] for n in names}

    processor = osmium.FileProcessor(str(pbf_path), osmium.osm.NODE) \
        .with_filter(osmium.filter.KeyFilter("place"))
    for node in processor:
        osm_name = node.tags.get("name")
        if not osm_name or fold(osm_name) not in wanted:
            continue
        candidates[wanted[fold(osm_name)]].append({
            "osm_node_id": node.id,
            "osm_name": osm_name,
            "place": node.tags.get("place"),
            "population_tag": node.tags.get("population"),
            "wikidata": node.tags.get("wikidata"),
            "lat": round(node.location.lat, 7),
            "lon": round(node.location.lon, 7),
        })

    resolved, unresolved = [], []
    for name in sorted(names):
        found = sorted(candidates[name], key=lambda c: c["osm_node_id"])
        chosen, rule = select_candidate(name, found)
        if chosen is None:
            unresolved.append({
                "location": name,
                "reason": ("no single place=city node, and no single place=town node "
                           "where no city node exists, carries this name; no "
                           "coordinate is guessed"),
                "candidates": found,
            })
            continue
        resolved.append({"location": name, **chosen, "match_rule": rule,
                         "other_candidates": [c for c in found if c is not chosen]})
    return resolved, unresolved


def extract_header_timestamp(pbf_path):
    osrm_engine.resolve_pylib()
    import osmium
    reader = osmium.io.Reader(str(pbf_path), osmium.osm.osm_entity_bits.NOTHING)
    try:
        return reader.header().get("osmosis_replication_timestamp")
    finally:
        reader.close()


def coordinates_document(resolved, unresolved, locations, extract):
    """the raw distance artifact. no run timestamp inside, so it regenerates to
    the same bytes from the same extract"""
    return {
        "provenance": {
            "source": ("OpenStreetMap place nodes read from the Geofabrik Finland extract "
                       f"{extract['file']}"),
            "source_url": extract["source_url"],
            "retrieval_date": extract["retrieval_date"],
            "retrieval_date_basis": extract["retrieval_date_basis"],
            "unit": "WGS84 decimal degrees (EPSG:4326), rounded to 7 decimal places",
            "unit_confirmed_from": ("the OpenStreetMap data model stores node coordinates as "
                                    "WGS84 latitude and longitude at a precision of 7 "
                                    "decimal places"),
            "frequency": f"single snapshot, OSM data up to {extract['osm_data_timestamp']}",
            "role": ("pinned coordinates for every manufacturer location and delivery city, "
                     "the inputs to the road-distance matrix"),
            "licence_note": extract["licence_note"],
            "derived_from": {
                "file": extract["file"],
                "source_url": extract["source_url"],
                "published_md5": extract["published_md5"],
                "sha256": extract["sha256"],
                "bytes": extract["bytes"],
                "osm_data_timestamp": extract["osm_data_timestamp"],
                "stored_at": "tools/osrm/data, gitignored, fetched by src/dataset/osrm_engine.py",
            },
            "derivation": ("node id, name, place tag and coordinate copied from the matched "
                           "place node of the pinned extract; no coordinate is computed, "
                           "averaged or snapped here"),
            "lookup_method": ("every node carrying a place tag is scanned with pyosmium "
                              f"{osrm_engine.OSMIUM_WHEEL['requirement']}; no geocoding "
                              "service is queried"),
            "name_matching": ("a v4 name matches an OSM name when both agree after unicode "
                              "NFKD accent removal and case folding; the exact OSM name is "
                              "recorded"),
            "selection_rule": ("exactly one place=city node, or exactly one place=town node "
                               "when no city node exists; otherwise unresolved"),
            "generated_by": "src/dataset/route_distance.py geocode",
        },
        "location_sets_from_v4": locations,
        "locations": resolved,
        "unresolved": unresolved,
    }


def build_matrix(coordinates, locations, base_url):
    """one row per manufacturer location and delivery city, both directions kept"""
    by_name = {r["location"]: r for r in coordinates["locations"]}
    origins = locations["manufacturer_locations"]
    destinations = locations["delivery_cities"]
    missing = sorted({*origins, *destinations} - set(by_name))
    if missing:
        raise RouteDistanceError(
            f"no pinned coordinate for {missing}; the matrix is not built with a gap")

    points = sorted({*origins, *destinations})
    index = {name: i for i, name in enumerate(points)}
    path = "/table/v1/driving/" + ";".join(
        f"{by_name[n]['lon']:.7f},{by_name[n]['lat']:.7f}" for n in points)
    query = (f"?sources={';'.join(str(index[n]) for n in origins)}"
             f"&destinations={';'.join(str(index[n]) for n in destinations)}"
             "&annotations=distance")
    response = osrm_engine.query_json(base_url + path + query)
    if response.get("code") != "Ok":
        raise RouteDistanceError(f"osrm table query failed: {response.get('code')} "
                                 f"{response.get('message')}")

    rows = []
    for i, origin in enumerate(origins):
        for j, destination in enumerate(destinations):
            meters = response["distances"][i][j]
            if meters is None:
                raise RouteDistanceError(f"no road route from {origin} to {destination}")
            rows.append({"origin_manufacturer_location": origin,
                         "destination_deliver_to": destination,
                         "road_distance_m": round(float(meters), 1),
                         "road_distance_km": round(float(meters) / 1000.0, 3)})

    snapped = {
        "sources": [{"location": n, "snapped_lon_lat": w["location"],
                     "snap_distance_m": w["distance"], "road_name": w.get("name", "")}
                    for n, w in zip(origins, response["sources"])],
        "destinations": [{"location": n, "snapped_lon_lat": w["location"],
                          "snap_distance_m": w["distance"], "road_name": w.get("name", "")}
                         for n, w in zip(destinations, response["destinations"])],
    }
    return rows, snapped, path + query


def matrix_csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MATRIX_COLUMNS)
    for r in rows:
        writer.writerow([r["origin_manufacturer_location"], r["destination_deliver_to"],
                         f"{r['road_distance_m']:.1f}", f"{r['road_distance_km']:.3f}"])
    return buffer.getvalue().encode("utf-8")


def asymmetry(rows):
    distance = {(r["origin_manufacturer_location"], r["destination_deliver_to"]):
                r["road_distance_m"] for r in rows}
    worst = None
    for (a, b), meters in distance.items():
        back = distance.get((b, a))
        if back is None or a >= b:
            continue
        gap = abs(meters - back)
        if worst is None or gap > worst["abs_difference_m"]:
            worst = {"pair": [a, b], "a_to_b_m": meters, "b_to_a_m": back,
                     "abs_difference_m": round(gap, 1)}
    return worst


def provenance_document(rows, snapped, query_path, extract, engine, profile, matrix_bytes):
    return {
        "generation_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "matrix": {"artifact": rel(MATRIX_PATH),
                   "sha256": hashlib.sha256(matrix_bytes).hexdigest(),
                   "rows": len(rows), "columns": list(MATRIX_COLUMNS)},
        "osm_extract": {k: extract[k] for k in ("file", "source_url", "published_md5", "sha256",
                                                "bytes", "osm_data_timestamp", "provider",
                                                "licence_note", "retrieval_date",
                                                "retrieval_date_basis")},
        "osrm": {"version": engine["version"], "reported_version": engine["reported_version"],
                 "engine_asset": engine["asset"], "engine_asset_sha256": engine["asset_sha256"],
                 "preparation": "osrm-extract with the profile below, then osrm-contract",
                 "algorithm": osrm_engine.ALGORITHM,
                 "service": "table/v1/driving with annotations=distance, on 127.0.0.1 only",
                 "query_path": query_path},
        "routing_profile": profile,
        "coordinates": {"artifact": rel(COORDINATES_PATH),
                        "sha256": osrm_engine.sha256_of(COORDINATES_PATH)},
        "route_semantics": ("road_distance_m is the length of the fastest route under the car "
                            "profile between the snapped coordinates, as the OSRM table "
                            "service returns it; it is not a shortest-distance route"),
        "direction": ("origin is a manufacturer location and destination an order deliver_to "
                      "city; the matrix is not assumed symmetric and both directions are "
                      "stored as routed"),
        "largest_direction_difference": asymmetry(rows),
        "snapped_waypoints": snapped,
    }


# ── commands ─────────────────────────────────────────────────────────────────

def command_geocode():
    locations = unique_locations()
    extract = osrm_engine.resolve_extract()
    pbf = osrm_engine.extract_path()
    header_timestamp = extract_header_timestamp(pbf)
    if header_timestamp != extract["osm_data_timestamp"]:
        raise RouteDistanceError(
            f"{pbf.name} header timestamp is {header_timestamp}, the pin says "
            f"{extract['osm_data_timestamp']}")
    resolved, unresolved = geocode(locations["all_locations"], pbf)
    write_json(COORDINATES_PATH, coordinates_document(resolved, unresolved, locations, extract))
    for r in resolved:
        print(f"  resolved {r['location']:10s} -> {r['osm_name']} node {r['osm_node_id']} "
              f"({r['lat']}, {r['lon']}) place={r['place']}")
    for u in unresolved:
        print(f"  UNRESOLVED {u['location']}: {len(u['candidates'])} candidates")
    print(f"  wrote {rel(COORDINATES_PATH)}")
    return 3 if unresolved else 0


def route_rows(data_dir=None, port=5055):
    coordinates = read_json(COORDINATES_PATH)
    if coordinates is None:
        raise RouteDistanceError(f"{rel(COORDINATES_PATH)} does not exist, run geocode first")
    if coordinates["unresolved"]:
        raise RouteDistanceError(
            f"unresolved locations {[u['location'] for u in coordinates['unresolved']]}")
    base = osrm_engine.prepare_graph(data_dir)
    with osrm_engine.routed(base, port=port) as url:
        return build_matrix(coordinates, coordinates["location_sets_from_v4"], url)


def command_matrix():
    rows, snapped, query_path = route_rows()
    matrix_bytes = matrix_csv_bytes(rows)
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.write_bytes(matrix_bytes)
    document = provenance_document(
        rows, snapped, query_path, osrm_engine.resolve_extract(),
        osrm_engine.resolve_engine(), osrm_engine.resolve_profile(), matrix_bytes)
    previous = read_json(PROVENANCE_PATH)
    if previous and previous.get("determinism_check", {}).get("matrix_sha256") == \
            document["matrix"]["sha256"]:
        document["determinism_check"] = previous["determinism_check"]
    write_json(PROVENANCE_PATH, document)
    print(f"  wrote {rel(MATRIX_PATH)} ({len(rows)} rows) and {rel(PROVENANCE_PATH)}")
    return 0


def command_determinism():
    """rebuild the graph from nothing in a second directory and compare bytes"""
    if not MATRIX_PATH.is_file():
        raise RouteDistanceError(f"{rel(MATRIX_PATH)} does not exist, run matrix first")
    rebuild_dir = osrm_engine.TOOLS_DIR / "data_rebuild"
    shutil.rmtree(rebuild_dir, ignore_errors=True)
    rows, _, _ = route_rows(data_dir=rebuild_dir, port=5056)
    rebuilt = matrix_csv_bytes(rows)
    stored = MATRIX_PATH.read_bytes()
    identical = rebuilt == stored
    provenance = read_json(PROVENANCE_PATH)
    provenance["determinism_check"] = {
        "method": ("a second graph was built from nothing by osrm-extract and osrm-contract "
                   "in tools/osrm/data_rebuild from the same pinned extract, profile and "
                   "engine, queried with the same pinned coordinates, and its matrix bytes "
                   "compared with the stored matrix"),
        "check_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "identical": identical,
        "matrix_sha256": hashlib.sha256(stored).hexdigest(),
        "rebuilt_matrix_sha256": hashlib.sha256(rebuilt).hexdigest(),
    }
    write_json(PROVENANCE_PATH, provenance)
    print(f"  determinism: rebuilt matrix identical to stored matrix: {identical}")
    return 0 if identical else 4


COMMANDS = {"geocode": command_geocode, "matrix": command_matrix,
            "determinism": command_determinism}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    args = parser.parse_args()
    steps = ["geocode", "matrix"] if args.command == "all" else [args.command]
    try:
        for step in steps:
            code = COMMANDS[step]()
            if code:
                return code
    except (osrm_engine.OsrmToolError, RouteDistanceError, OSError) as exc:
        print(f"route distance incomplete at {steps}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
