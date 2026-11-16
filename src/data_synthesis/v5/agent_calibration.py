"""part 1: per-agent logistics coefficients, drawn from the pinned calibration evidence

  PYTHONPATH=. python -m src.data_synthesis.v5.agent_calibration

writes data/processed/v5/logistics/agent_calibration.csv and
agent_calibration_reasoning.json. every F_i and alpha_i here is a reading already
computed in data/processed/v5/logistics/rate_calibration_range.json, taken whole.
F_i and alpha_i always come from the same reading: they are never mixed across
readings, averaged, or interpolated between corners

two fixed rules over the agent's v4 specialty text, the only per-agent description
the project has, decide each coefficient pair:
  corner   parcel-like (low F, high alpha) or freight-like (high F, low alpha)
  reading  which real reading inside that corner

three things are the same for every agent, each for a recorded reason:
  beta_i       0.07163 EUR/tonne-km, see settings.BETA_BASIS
  fuel series  settings.SHARED_FUEL_SERIES_ID, see settings.SHARED_FUEL_BASIS
  freight      no weight cap

a parcel-corner agent is capped at the weight its own parcel service documents,
read from the pinned Posti parcel record rather than rounded to one number:
Postal Parcel 25 kg, Home and Express Parcel 25 kg or 35 kg with the pinned
Oversized fee added
"""

import json
import re
import sys

import pandas as pd

from src.data_synthesis.v5 import fuel
from src.data_synthesis.v5.settings import (
    AGENT_CALIBRATION_PATH, BETA_BASIS, BETA_EUR_PER_TONNE_KM, PROCESSED_V5, RAW_V5,
    SHARED_FUEL_BASIS, SHARED_FUEL_SERIES_ID, V4_DIR, WINDOW_DAYS, rel, sha256_file, write_csv,
    write_json,
)

RANGE_PATH = PROCESSED_V5 / "logistics" / "rate_calibration_range.json"
REASONING_PATH = PROCESSED_V5 / "logistics" / "agent_calibration_reasoning.json"
POSTI_PARCEL_PATH = RAW_V5 / "logistics" / "rate_calibration" / "posti_parcel.json"

# corner rule: any wording about medium, large or bulk loads means freight; only
# wording about small or light loads means parcel; no size wording means freight,
# because the v5 median shipment is 257 kg, far above any parcel weight limit
FREIGHT_WORDS = ("medium", "large", "bulk")
PARCEL_WORDS = ("small", "light", "low-load")


class CalibrationError(ValueError):
    """the pinned evidence does not say what an assignment needs it to say"""


def load_ranges():
    with open(RANGE_PATH, encoding="utf-8") as f:
        return json.load(f)


def reading_catalogue(ranges):
    """every reading this assignment may use, each an F and alpha pair from one reading"""
    readings = {}
    for i, r in enumerate(ranges["lower_bound_evidence"]["posti_parcel"]):
        rid = "posti_" + r["service"].lower().replace(" ", "_")
        readings[rid] = {
            "reading_id": rid, "corner": "parcel", "evidence_carrier": "Posti",
            "service": f"{r['service']}, digital contract prices",
            "fixed_charge_eur": r["F_equivalent_eur"],
            "alpha_eur_per_kg": r["alpha_equivalent_eur_per_kg"],
            "arithmetic": r["arithmetic"],
            "source": f"{rel(RANGE_PATH)} lower_bound_evidence.posti_parcel[{i}]",
            "valid_weight_domain_kg": r["valid_weight_domain_kg"],
            "pickup": "hand-in at a Posti outlet; the evidence holds no parcel pickup reading",
        }

    freight = ranges["upper_bound_evidence"]["posti_express_freight"]
    pickup = freight["pickup_normalization"]["separate_pickup_one_unit_eur"]
    zone0 = freight["by_zone"][0]
    for key, rid, domain in (
            ("secant_loose_package_to_half_pallet",
             "posti_express_freight_zone0_loose_to_half_pallet_with_pickup", [35, 500]),
            ("secant_half_pallet_to_pallet",
             "posti_express_freight_zone0_half_pallet_to_pallet_with_pickup", [500, 1000])):
        s = zone0[key]
        fixed = round(s["F_equivalent_eur"] + pickup, 10)
        readings[rid] = {
            "reading_id": rid, "corner": "freight", "evidence_carrier": "Posti",
            "service": "Express Freight 14, zone 0, door to door",
            "fixed_charge_eur": fixed,
            "alpha_eur_per_kg": s["alpha_equivalent_eur_per_kg"],
            "arithmetic": (f"{s['arithmetic']}; with the separate pickup the route needs at the "
                           f"manufacturer: F = {s['F_equivalent_eur']:.4f} + {pickup:.2f} = "
                           f"{fixed:.4f} EUR"),
            "source": (f"{rel(RANGE_PATH)} upper_bound_evidence.posti_express_freight.by_zone[0]."
                       f"{key} and pickup_normalization.separate_pickup_one_unit_eur"),
            "valid_weight_domain_kg": domain,
            "pickup": f"included: Posti's separate pickup of one transport unit, {pickup:.2f} EUR",
        }

    lavarahti = ranges["upper_bound_evidence"]["matkahuolto_lavarahti"]
    for i, s in enumerate(lavarahti["secants"]):
        if not s["pickup_included"]:
            continue
        low = s["points"][0].split(" ")[0].replace("-", "_")
        rid = f"matkahuolto_lavarahti_1_49km_{low}_to_950_999kg_with_pickup"
        readings[rid] = {
            "reading_id": rid, "corner": "freight", "evidence_carrier": "Matkahuolto",
            "service": "Lavarahti pallet freight, 1-49 km band, mandatory pickup included",
            "fixed_charge_eur": s["F_equivalent_eur"],
            "alpha_eur_per_kg": s["alpha_equivalent_eur_per_kg"],
            "arithmetic": s["arithmetic"],
            "source": f"{rel(RANGE_PATH)} upper_bound_evidence.matkahuolto_lavarahti.secants[{i}]",
            "valid_weight_domain_kg": [float(s["points"][0].split(" ")[4].rstrip(",")), 974.5],
            "pickup": "included: Lavarahti pickup is always part of the service",
        }
    return readings


EXCLUDED_READINGS = {
    "matkahuolto_lavarahti secants without pickup": (
        "Lavarahti pickup is mandatory ('Palveluun kuuluu aina nouto'), so a reading without it "
        "leaves out a charge every shipment pays"),
    "posti_express_freight zone 0 secants without pickup": (
        "the route starts at the manufacturer, so collection is needed; the with-pickup reading "
        "of the same secant is used instead"),
    "posti_express_freight zones 1 to 5": (
        "the zones are not defined in the price list and their prices rise with zone, so they "
        "carry distance that beta_i already represents"),
    "posti_express_freight marginal full pallet": (
        "an alpha-only reading with no fixed charge, so it is not a coherent F and alpha pair"),
    "posti_express_freight whole-unit prices with alpha 0": (
        "valid only up to one unit's gross weight; as an F and alpha pair it would price every "
        "heavier shipment at the same amount"),
    "posti Express Parcel zone fee variants": (
        "the fee depends on the destination postal code zone, and the postal-code zone list is not "
        "pinned, so it cannot be mapped to the seven cities"),
    "postnord_parcel_zones": (
        "not identifiable: one weight bracket per zone cannot separate F from alpha, and its "
        "metadata could not be read (HTTP 403)"),
}


def corner_for(specialty):
    text = specialty.lower()
    freight = [w for w in FREIGHT_WORDS if w in text]
    parcel = [w for w in PARCEL_WORDS if w in text]
    if freight:
        return "freight", f"specialty mentions {freight}"
    if parcel:
        return "parcel", f"specialty mentions only small or light loads {parcel}"
    return "freight", ("specialty has no shipment-size wording; freight is the default because "
                       "the v5 median shipment is 257 kg, far above the 25-35 kg parcel limit")


def reading_for(corner, specialty):
    text = specialty.lower()
    if corner == "parcel":
        if "express" in text or "fastest" in text:
            return "posti_express_parcel", "specialty mentions express or fastest"
        if "budget" in text or "cheapest" in text or "lowest base cost" in text:
            return "posti_postal_parcel", "specialty mentions budget or cheapest"
        return "posti_home_parcel", "no budget or express wording: the standard doorstep parcel"
    if "bulk" in text or "high capacity" in text:
        return ("posti_express_freight_zone0_loose_to_half_pallet_with_pickup",
                "specialty mentions bulk or high capacity: the lowest-alpha reading")
    if "large" in text or "premium" in text:
        return ("posti_express_freight_zone0_half_pallet_to_pallet_with_pickup",
                "specialty mentions large or premium: the pallet reading")
    if "small" in text:
        return ("matkahuolto_lavarahti_1_49km_100_149_to_950_999kg_with_pickup",
                "freight agent whose specialty also mentions small orders: the lower-F Lavarahti reading")
    return ("matkahuolto_lavarahti_1_49km_0_79_to_950_999kg_with_pickup",
            "freight agent with medium or no size wording: the lower-alpha Lavarahti reading")


def _number(pattern, text, what):
    match = re.search(pattern, text)
    if match is None:
        raise CalibrationError(f"{rel(POSTI_PARCEL_PATH)} no longer states {what}: {text!r}")
    return float(match.group(1))


def parcel_weight_limits(record):
    """the weight each parcel reading is eligible to, read from the pinned Posti record

    the base limit covers all three services. the Oversized additional service raises
    Home and Express Parcel, and leaves Postal Parcel where it was. its fee is printed
    in the same record, so where the service extends a limit the fee is charged for
    the extra weight instead of being absorbed"""
    printed = record["weight_limits_as_printed"]
    base_text = printed["Postal Parcel, Home Parcel, Express Parcel and Return"]
    oversized_text = printed["with Oversized additional service"]
    fee_text = printed["Oversized additional service price"]
    base = _number(r"^maximum weight (\d+(?:\.\d+)?) kg$", base_text, "the base maximum weight")
    postal = _number(r"postal parcel’s maximum weight (\d+(?:\.\d+)?) kg", oversized_text,
                     "Postal Parcel's Oversized maximum")
    home_express = _number(r"Home Parcel’s, Express Parcel’s and Return’s maximum weight (\d+(?:\.\d+)?) kg",
                           oversized_text, "Home and Express Parcel's Oversized maximum")
    fee = _number(r"^€ (\d+(?:\.\d+)?)$", fee_text, "the Oversized additional service price")
    quoted = {"base": base_text, "with_oversized": oversized_text, "oversized_fee": fee_text,
              "record": rel(POSTI_PARCEL_PATH), "page": "weight limits page 6"}

    def limit(service, oversized_limit):
        if oversized_limit > base:
            return {"max_shipment_weight_kg": oversized_limit, "oversized_above_kg": base,
                    "oversized_fee_eur": fee,
                    "rule": (f"{service} carries up to {base:g} kg, and up to {oversized_limit:g} kg with the "
                             f"Oversized additional service, whose EUR {fee:.2f} fee is added to the price "
                             f"of any shipment above {base:g} kg"),
                    "source_wording": quoted}
        return {"max_shipment_weight_kg": base, "oversized_above_kg": None, "oversized_fee_eur": None,
                "rule": (f"{service} carries up to {base:g} kg; the Oversized additional service does not "
                         f"raise its weight limit ({oversized_limit:g} kg), so it is capped at {base:g} kg"),
                "source_wording": quoted}

    return {"posti_postal_parcel": limit("Postal Parcel", postal),
            "posti_home_parcel": limit("Home Parcel", home_express),
            "posti_express_parcel": limit("Express Parcel", home_express)}


def shared_fuel_check(series_by_id):
    """the shared series must be usable, and the coverage claim behind it must hold"""
    coverage = {sid: fuel.window_coverage(s) for sid, s in series_by_id.items()}
    shared = coverage[SHARED_FUEL_SERIES_ID]
    if not shared["usable"]:
        raise CalibrationError(f"the shared fuel series {SHARED_FUEL_SERIES_ID} cannot price the v5 window")
    usable = {sid: c for sid, c in coverage.items() if c["usable"]}
    most_distinct = max(c["distinct_percentages_across_window_days"] for c in usable.values())
    check = {
        "shared_series": SHARED_FUEL_SERIES_ID,
        "basis": SHARED_FUEL_BASIS,
        "observations_in_window": shared["observations_in_window"],
        "distinct_percentages_across_window_days": shared["distinct_percentages_across_window_days"],
        "window_days_after_last_observation": shared["window_days_after_last_observation"],
        "has_the_most_distinct_percentages_of_the_usable_series":
            shared["distinct_percentages_across_window_days"] == most_distinct,
        "has_no_day_after_its_last_observation": shared["window_days_after_last_observation"] == 0,
        "usable_series_compared": {
            sid: {k: c[k] for k in ("observations_in_window", "distinct_percentages_across_window_days",
                                    "window_days_after_last_observation")}
            for sid, c in usable.items()},
        "unusable_series": {sid: c["reason"] for sid, c in coverage.items() if not c["usable"]},
    }
    return check, coverage


def build():
    ranges = load_ranges()
    readings = reading_catalogue(ranges)
    with open(POSTI_PARCEL_PATH, encoding="utf-8") as f:
        limits = parcel_weight_limits(json.load(f))
    series_by_id = fuel.load_series(RAW_V5 / "logistics")
    fuel_check, coverage = shared_fuel_check(series_by_id)
    v4 = pd.read_csv(V4_DIR / "logistics_agents.csv").to_dict("records")

    rows, reasoning = [], []
    for agent in sorted(v4, key=lambda a: a["agent_id"]):
        corner, corner_rule = corner_for(agent["specialty"])
        reading_id, reading_rule = reading_for(corner, agent["specialty"])
        reading = readings[reading_id]
        limit = limits.get(reading_id) if corner == "parcel" else None
        if corner == "parcel" and limit is None:
            raise CalibrationError(f"{agent['agent_id']}: parcel reading {reading_id} has no documented weight limit")
        rows.append({
            "agent_id": agent["agent_id"],
            "calibration_corner": corner,
            "calibration_reading_id": reading_id,
            "fixed_charge_eur": reading["fixed_charge_eur"],
            "alpha_eur_per_kg": reading["alpha_eur_per_kg"],
            "beta_eur_per_tonne_km": BETA_EUR_PER_TONNE_KM,
            "fuel_surcharge_series_id": SHARED_FUEL_SERIES_ID,
            "max_shipment_weight_kg": limit["max_shipment_weight_kg"] if limit else None,
            "oversized_above_kg": limit["oversized_above_kg"] if limit else None,
            "oversized_fee_eur": limit["oversized_fee_eur"] if limit else None,
        })
        weight_domain = reading["valid_weight_domain_kg"]
        reasoning.append({
            "agent_id": agent["agent_id"],
            "v4_specialty": agent["specialty"],
            "v4_location": agent["location"],
            "corner": corner, "corner_rule": corner_rule,
            "reading_id": reading_id, "reading_rule": reading_rule,
            "reading": reading,
            "weight_limit": limit or {
                "max_shipment_weight_kg": None,
                "rule": "freight-corner agent: eligible across its existing range, no weight cap"},
            "evidence_domain_caveat": (
                f"the reading is evidenced for about {weight_domain[0]} to {weight_domain[1]} kg"
                + ("; the weight cap keeps the agent inside what its own service documents" if limit
                   else "; v5 shipments outside that range use it as an extrapolation")),
            "beta_eur_per_tonne_km": BETA_EUR_PER_TONNE_KM,
            "fuel_surcharge_series_id": SHARED_FUEL_SERIES_ID,
            "fuel_rule": "every agent uses the shared series, see shared_fuel_series in this file",
            "unchanged_from_v4": {k: agent[k] for k in (
                "location", "service_regions", "default_completion_same_city_days",
                "default_completion_intercity_days", "delay_probability", "damage_probability",
                "reputation_score")},
        })

    frame = pd.DataFrame(rows)
    write_csv(frame, AGENT_CALIBRATION_PATH)
    write_json(REASONING_PATH, {
        "purpose": ("the per-agent F_i, alpha_i, beta_i, weight limit and fuel series behind "
                    "agent_calibration.csv, with the rule and evidence for every value"),
        "status": ("every F_i and alpha_i is one whole reading from the calibration evidence. the fuel "
                   "percentage is not stored: every agent resolves it from one shared series at each "
                   "order's own date"),
        "inputs": {rel(RANGE_PATH): sha256_file(RANGE_PATH),
                   rel(POSTI_PARCEL_PATH): sha256_file(POSTI_PARCEL_PATH),
                   rel(V4_DIR / "logistics_agents.csv"): sha256_file(V4_DIR / "logistics_agents.csv")},
        "rules": {"corner": {"freight_words": list(FREIGHT_WORDS), "parcel_words": list(PARCEL_WORDS),
                             "default": "freight"},
                  "reading": "see reading_for in src/data_synthesis/v5/agent_calibration.py",
                  "weight_limit": ("a parcel-corner agent is capped at the weight its own calibration "
                                   "service documents in the pinned Posti record; freight has no cap"),
                  "fuel": "one shared series for every agent"},
        "shared_fuel_series": fuel_check,
        "fuel_mapping_change": (
            "this replaces the earlier per-agent carrier-series mapping. that mapping made the "
            "carriers' incomparable percentage bases decide which agents won, an artifact of the "
            "evidence rather than a real difference between agents; agent-specific fuel variation is "
            "not undesirable in principle"),
        "parcel_weight_limits": limits,
        "beta": {"value_eur_per_tonne_km": BETA_EUR_PER_TONNE_KM, "basis": BETA_BASIS},
        "readings_available": readings,
        "readings_excluded": EXCLUDED_READINGS,
        "fuel_series_window_coverage": coverage,
        "window_days": [WINDOW_DAYS[0].isoformat(), WINDOW_DAYS[-1].isoformat()],
        "agents": reasoning,
    })
    return frame, reasoning, fuel_check


def main():
    frame, _, fuel_check = build()
    print(frame.to_string(index=False))
    print(f"\nshared fuel series {fuel_check['shared_series']}: "
          f"{fuel_check['observations_in_window']} observations in window, "
          f"{fuel_check['distinct_percentages_across_window_days']} distinct percentages, "
          f"{fuel_check['window_days_after_last_observation']} days after its last observation")
    print(f"wrote {rel(AGENT_CALIBRATION_PATH)} and {rel(REASONING_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
