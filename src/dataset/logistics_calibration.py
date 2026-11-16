"""calibration research for the v5 logistics coefficients: a traced candidate for
beta_i, and a documented evidence range for F_i and alpha_i

  PYTHONPATH=. python src/dataset/logistics_calibration.py

writes data/processed/v5/logistics/beta_candidate_derivation.json and
data/processed/v5/logistics/rate_calibration_range.json. both are research
artifacts, not synthesis inputs. beta_i = 0.07163 EUR/tonne-km is used because
the project supervisor specified it, whatever this module finds, and no per-agent
F_i, alpha_i, beta_i or fuel-surcharge value is produced here

every input number is read from the manual records in data/raw/v5/logistics and
every derived number carries the arithmetic that produced it. nothing is
interpolated into a smooth range: the range file reports the discrete readings
the evidence supports, how the sources differ before they can be compared, and
where the evidence has gaps
"""

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.dataset.acquire_v5_sources import read_json, write_json
from src.dataset.v5_logistics_sources import NOT_STATED

RAW_LOGISTICS = PROJECT_ROOT / "data" / "raw" / "v5" / "logistics"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "v5" / "logistics"
METHODOLOGY = PROJECT_ROOT / "docs" / "data_synthesis_methodology.md"
V4_DIR = PROJECT_ROOT / "data" / "synthetic" / "v4"

BETA_SPECIFIED = 0.07163
PAYLOADS_REQUESTED_T = (24.0, 24.15, 25.0, 26.0, 28.0, 32.8)

# how close a quotient has to land to count as consistent with 0.07163. a
# declared threshold, not a finding: the report shows every combination either way
CLOSE_MATCH_TOLERANCE_PERCENT = 1.0

# gross weight limits the Posti 30.9.2024 english list prints for each unit
POSTI_UNIT_CAPACITY_KG = {"loose_package": 35, "half_pallet": 500, "pallet": 1000,
                          "furniture_pallet": 1200}

PARCEL_WEIGHT_DOMAIN_KG = (1, 25)
PARCEL_OVERSIZED_LIMIT_KG = 35


class CalibrationError(ValueError):
    """a record needed for the derivation is missing or not in the declared shape"""


def rel(path):
    return str(Path(path).relative_to(PROJECT_ROOT)).replace("\\", "/")


def record(artifact):
    path = RAW_LOGISTICS / artifact
    document = read_json(path)
    if document is None:
        raise CalibrationError(f"{rel(path)} is missing, build the logistics sources first")
    return document, {"artifact": rel(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def r4(value):
    return f"{value:.4f}"


# ── beta ─────────────────────────────────────────────────────────────────────

def methodology_payload_figures(path=METHODOLOGY):
    """tonnage figures stated in section 7 of the methodology, if there are any"""
    text = Path(path).read_text(encoding="utf-8")
    start = text.index("## 7. Logistics Synthesis")
    end = text.index("## 8.", start)
    section = text[start:end]
    found = re.findall(r"(\d+(?:\.\d+)?)\s*(?:t|tonnes?|tons?)\b(?!-)", section)
    return {"section_scanned": "7. Logistics Synthesis",
            "pattern": "a number followed by t, tonne, tonnes, ton or tons",
            "figures_found_t": sorted({float(f) for f in found})}


def beta_combinations(corridor_record, payloads):
    rows = []
    for obs in corridor_record["observations"]:
        for payload in payloads:
            quotient = obs["eur_per_km"] / payload
            difference = (quotient - BETA_SPECIFIED) / BETA_SPECIFIED * 100.0
            rows.append({
                "corridor": obs["corridor"], "rate_type": obs["rate_type"],
                "page": obs["page"], "eur_per_km": obs["eur_per_km"],
                "payload_t": payload,
                "eur_per_tonne_km": quotient,
                "arithmetic": f"{r4(obs['eur_per_km'])} / {r4(payload)} = {r4(quotient)}",
                "difference_from_0_07163_percent": round(difference, 4),
                "abs_difference_percent": round(abs(difference), 4),
            })
    rows.sort(key=lambda r: (r["abs_difference_percent"], r["corridor"], r["rate_type"],
                             r["payload_t"]))
    return rows


def derivation_status(closest_abs_percent, document_states_payload,
                      tolerance=CLOSE_MATCH_TOLERANCE_PERCENT):
    """the only three outcomes, and never confirmed on arithmetic alone

    a close quotient can move the status to candidate. a payload stated in the
    source document does not confirm anything by itself either: it only flags that
    the stated figure has to be compared with the candidate by hand"""
    if closest_abs_percent > tolerance:
        return "unresolved"
    if document_states_payload:
        return "candidate, document states a payload that needs checking by hand"
    return "candidate"


def beta_derivation():
    corridor, source = record("line_haul_rate/upply_corridor_rates.json")
    scan = methodology_payload_figures()
    payloads = sorted(set(PAYLOADS_REQUESTED_T) | set(scan["figures_found_t"]))
    rows = beta_combinations(corridor, payloads)
    closest = rows[0]
    within = [r for r in rows if r["abs_difference_percent"] <= CLOSE_MATCH_TOLERANCE_PERCENT]

    payload_field = corridor["tariff_metadata"]["payload_or_load_factor"]
    document_states_payload = payload_field["value"] != NOT_STATED
    status = derivation_status(closest["abs_difference_percent"], document_states_payload)

    return {
        "status": status,
        "status_meaning": {
            "candidate": ("a combination of a printed corridor rate and a tested payload is "
                          "numerically consistent with 0.07163. that shows consistency, not "
                          "how the figure was actually computed: the method behind 0.07163 "
                          "was never stated, only its source document"),
            "confirmed": ("reserved for the case where the source document itself states the "
                          "payload or rate relationship that produces 0.07163. it does not, so "
                          "this status is not used"),
            "unresolved": ("no tested combination lands within the tolerance, the same "
                           "declared-blocker pattern as the SunSirs polyester observations"),
        },
        "beta_used": {
            "value": BETA_SPECIFIED,
            "unit": "EUR/tonne-km",
            "basis": ("used as directly specified by the project supervisor, independent of "
                      "whether this candidate derivation is ever confirmed"),
        },
        "source_record": source,
        "dimensional_reading": (
            "the corridor rates are printed in EUR per km per truck. dividing by a payload in "
            "tonnes gives EUR per tonne-km for a truck carrying that payload over the whole "
            "distance, which is the unit of beta_i"),
        "assumptions_the_arithmetic_makes": [
            "the printed per-km rate is the price of one truck, not of a part load",
            "the truck carries the tested payload for every km of the lane",
            "the rate carries no separate fuel surcharge; the source does not state either way",
            ("an international EU headhaul lane rate stands in for Finnish domestic line-haul "
             "work; the source covers no Finnish lane"),
        ],
        "document_states_payload": {"value": document_states_payload,
                                    "record_field": payload_field},
        "payloads_tested_t": payloads,
        "payload_sources": {"requested_in_task": list(PAYLOADS_REQUESTED_T),
                            "methodology_scan": scan},
        "close_match_tolerance_percent": CLOSE_MATCH_TOLERANCE_PERCENT,
        "closest_combination": {
            **closest,
            "exact_eur_per_tonne_km": closest["eur_per_tonne_km"],
            "rounded_4dp": r4(closest["eur_per_tonne_km"]),
            "match": (f"{closest['arithmetic']}, which differs from 0.07163 by "
                      f"{closest['difference_from_0_07163_percent']:+.4f}%"),
        },
        "combinations_within_tolerance": len(within),
        "within_tolerance": within,
        "uniqueness_note": (
            f"{len(within)} of {len(rows)} combinations land within "
            f"{CLOSE_MATCH_TOLERANCE_PERCENT}% of 0.07163, so a close quotient is not unique "
            "to one corridor and payload. the closest one is recorded as the candidate; the "
            "others are why it stays a candidate"),
        "combination_count": len(rows),
        "all_combinations": rows,
    }


# ── F_i and alpha_i ──────────────────────────────────────────────────────────

def secant(point_a, point_b):
    """(F, alpha, arithmetic) for the straight line through two (kg, EUR) points"""
    (w1, p1), (w2, p2) = point_a, point_b
    alpha = (p2 - p1) / (w2 - w1)
    fixed = p1 - alpha * w1
    arithmetic = (f"alpha = ({r4(p2)} - {r4(p1)}) / ({r4(w2)} - {r4(w1)}) = {r4(alpha)} EUR/kg; "
                  f"F = {r4(p1)} - {r4(alpha)} x {r4(w1)} = {r4(fixed)} EUR")
    return fixed, alpha, arithmetic


def posti_parcel_readings(rec):
    readings = []
    for obs in rec["observations"]:
        reading = {
            "source": "rate_calibration/posti_parcel.json",
            "service": obs["service"],
            "method": "direct: the tariff is already a fixed charge plus a per-kg charge",
            "F_equivalent_eur": obs["eur_per_item"],
            "alpha_equivalent_eur_per_kg": obs["eur_per_kg"],
            "arithmetic": (f"F = {obs['eur_per_item']:.2f} (EUR/item as printed); "
                           f"alpha = {obs['eur_per_kg']:.2f} (EUR/kg as printed)"),
            "valid_weight_domain_kg": list(PARCEL_WEIGHT_DOMAIN_KG),
            "domain_note": (f"maximum weight 25 kg; {PARCEL_OVERSIZED_LIMIT_KG} kg for Home and "
                            "Express Parcel with the EUR 3.50 Oversized service"),
        }
        if obs["service"] == "Express Parcel":
            reading["zone_fee_variants"] = {
                "zone 1": {"F_equivalent_eur": round(obs["eur_per_item"] + 0.60, 2),
                           "arithmetic": f"{obs['eur_per_item']:.2f} + 0.60"},
                "zone 2": {"F_equivalent_eur": round(obs["eur_per_item"] + 1.20, 2),
                           "arithmetic": f"{obs['eur_per_item']:.2f} + 1.20"},
            }
        readings.append(reading)
    return readings


def postnord_readings(rec):
    readings = []
    for obs in rec["observations"]:
        low, high = (float(x) for x in obs["weight_bracket_kg"].split("-"))
        price = obs["eur_per_parcel"]
        readings.append({
            "source": "rate_calibration/postnord_parcel_zones.json",
            "zone": obs["zone"],
            "method": ("not identifiable: one weight bracket per zone gives one price, and a "
                       "fixed charge and a per-kg charge cannot both be recovered from one "
                       "price"),
            "F_equivalent_eur": None,
            "alpha_equivalent_eur_per_kg": None,
            "all_in_eur_per_kg_at_bracket_bounds": {
                f"at {low} kg": price / low, f"at {high} kg": price / high},
            "arithmetic": (f"{price:.2f} / {r4(low)} = {r4(price / low)} EUR/kg; "
                           f"{price:.2f} / {r4(high)} = {r4(price / high)} EUR/kg"),
            "valid_weight_domain_kg": [low, high],
            "caveat": "metadata unverified: the source returned HTTP 403, see the record",
        })
    return readings


def posti_freight_readings(rec):
    table = next(o for o in rec["observations"] if o["service_printed_fi"] == "Express-rahti 14")
    pickup = rec["separate_pickup_as_printed"]["eur_by_transport_unit_count"]["1"]
    by_zone = []
    for row in table["rows"]:
        eur = row["eur"]
        loose_half = secant((POSTI_UNIT_CAPACITY_KG["loose_package"], eur["loose_package"]),
                            (POSTI_UNIT_CAPACITY_KG["half_pallet"], eur["half_pallet_first_unit"]))
        half_pallet = secant((POSTI_UNIT_CAPACITY_KG["half_pallet"], eur["half_pallet_first_unit"]),
                             (POSTI_UNIT_CAPACITY_KG["pallet"], eur["pallet_first_unit"]))
        marginal = eur["pallet_next_units"] / POSTI_UNIT_CAPACITY_KG["pallet"]
        by_zone.append({
            "zone": row["zone"],
            "whole_unit_prices_as_F_within_capacity": {
                "loose_package_up_to_35_kg": eur["loose_package"],
                "half_pallet_up_to_500_kg": eur["half_pallet_first_unit"],
                "pallet_up_to_1000_kg": eur["pallet_first_unit"],
                "furniture_pallet_up_to_1200_kg": eur["furniture_pallet_first_unit"],
                "method": ("a unit price does not change with weight inside the unit's "
                           "capacity, so read on its own it is a fixed charge with alpha = 0 "
                           "up to that capacity"),
            },
            "secant_loose_package_to_half_pallet": {
                "F_equivalent_eur": loose_half[0], "alpha_equivalent_eur_per_kg": loose_half[1],
                "arithmetic": loose_half[2],
                "points": "(35 kg, loose package price), (500 kg, half pallet first-unit price)"},
            "secant_half_pallet_to_pallet": {
                "F_equivalent_eur": half_pallet[0], "alpha_equivalent_eur_per_kg": half_pallet[1],
                "arithmetic": half_pallet[2],
                "points": "(500 kg, half pallet first-unit price), (1000 kg, pallet first-unit price)"},
            "marginal_full_pallet_eur_per_kg": {
                "value": marginal,
                "arithmetic": f"{r4(eur['pallet_next_units'])} / 1000 = {r4(marginal)} EUR/kg",
                "meaning": "the per-kg cost of each additional fully loaded pallet to the same address"},
        })
    return {
        "source": "rate_calibration/posti_express_freight.json",
        "service": "Express-rahti 14 (Express Freight)",
        "translation_method": (
            "the tariff is a price per transport unit per zone. two honest readings are given "
            "per zone: each unit price as a fixed charge valid up to that unit's gross weight "
            "limit, and a straight line through two consecutive unit types placed at their "
            "gross weight limits, whose slope is an alpha-equivalent and whose intercept is an "
            "F-equivalent. placing each price at the unit's maximum weight gives the lowest "
            "per-kg reading the unit allows; a lighter load pays the same price"),
        "capacity_assumption": (
            "the half pallet column is headed 'Rolltainer/Half pallet'; the half pallet limit of "
            "500 kg is used, a rolltainer's limit is 250 kg, which would roughly double that "
            "column's per-kg reading"),
        "least_distance_reading": "zone 0",
        "zone_caveat": ("zones 0-5 are not defined in the price list (the zone list is a link), "
                        "so they cannot be mapped to km. prices rise with zone, which suggests "
                        "zones carry distance, and only zone 0 is used as the least-distance "
                        "F/alpha reading; zones 1-5 are kept to show that entanglement"),
        "pickup_normalization": {
            "separate_pickup_one_unit_eur": pickup,
            "reason": "prices run from a Posti outlet; collection from the sender is a separate charge",
            "door_to_door_zone0_secant_half_pallet_to_pallet_F_eur": by_zone[0]["secant_half_pallet_to_pallet"]["F_equivalent_eur"] + pickup,
            "arithmetic": (f"{r4(by_zone[0]['secant_half_pallet_to_pallet']['F_equivalent_eur'])} + "
                           f"{r4(pickup)} = "
                           f"{r4(by_zone[0]['secant_half_pallet_to_pallet']['F_equivalent_eur'] + pickup)} EUR"),
        },
        "by_zone": by_zone,
    }


def pickup_price(pickup_rows, kg_upper):
    for limit, price in pickup_rows:
        if kg_upper <= limit:
            return limit, price
    raise CalibrationError(f"no Lavarahti pickup price covers {kg_upper} kg")


def lavarahti_readings(rec):
    table = rec["freight_table_parsed"]
    band = table["km_bands"][0]
    pickup_rows = rec["pickup_table_as_printed"]["rows"]
    rows = table["rows"]

    def point(kg_lower, with_pickup):
        row = next(r for r in rows if r["kg_lower"] == kg_lower)
        midpoint = (row["kg_lower"] + row["kg_upper"]) / 2.0
        price = row["eur_by_band"][0]
        note = f"{row['kg_lower']}-{row['kg_upper']} kg bracket, midpoint {midpoint}, {price:.2f} EUR"
        if with_pickup:
            limit, fee = pickup_price(pickup_rows, row["kg_upper"])
            price += fee
            note += f" + pickup up to {limit} kg {fee:.2f} EUR"
        return (midpoint, price), note

    secants = []
    for low_kg in (0, 100):
        for with_pickup in (False, True):
            a, note_a = point(low_kg, with_pickup)
            b, note_b = point(950, with_pickup)
            fixed, alpha, arithmetic = secant(a, b)
            secants.append({
                "points": [note_a, note_b],
                "pickup_included": with_pickup,
                "F_equivalent_eur": fixed, "alpha_equivalent_eur_per_kg": alpha,
                "arithmetic": arithmetic,
            })

    cap = 3000
    return {
        "source": "rate_calibration/matkahuolto_lavarahti.json",
        "service": "Lavarahti, terminal to recipient",
        "distance_band_used": band,
        "translation_method": (
            "the tariff is a price per weight bracket per distance band. only the shortest band, "
            "1-49 km, is read for F and alpha, because every longer band adds distance cost that "
            "beta_i already represents. each bracket price is placed at its bracket midpoint "
            "and a straight line is drawn between two brackets that span the simulation's "
            "shipment weights; the slope is the alpha-equivalent and the intercept the "
            "F-equivalent. two low brackets are shown so the sensitivity to that choice is "
            "visible"),
        "pickup_normalization": ("pickup is always part of the service but priced separately, so "
                                 "every secant is given both as published and with the pickup "
                                 "fee for each bracket's upper weight added"),
        "secants": secants,
        "discrete_points_1_to_49_km": [
            {"kg_lower": r["kg_lower"], "kg_upper": r["kg_upper"], "eur": r["eur_by_band"][0]}
            for r in rows if r["kg_lower"] < cap],
        "discrete_points_note": (f"brackets up to the service's {cap} kg shipment limit; the "
                                 "table itself continues to 14999 kg"),
    }


def normalization(parcel, postnord, freight, lavarahti):
    def meta(rec, key):
        return rec["tariff_metadata"][key]["value"]
    sources = []
    for name, rec in (("posti_parcel", parcel), ("postnord_parcel_zones", postnord),
                      ("posti_express_freight", freight),
                      ("matkahuolto_lavarahti", lavarahti)):
        pickup = rec["tariff_metadata"].get("pickup_treatment", {}).get("value", NOT_STATED)
        sources.append({
            "source": name,
            "vat_status": meta(rec, "vat_status"),
            "effective_date": meta(rec, "effective_date"),
            "currency_as_published": meta(rec, "currency_as_published"),
            "fuel_surcharge_treatment": meta(rec, "fuel_surcharge_treatment"),
            "pickup_treatment": pickup,
        })
    return {
        "sources": sources,
        "vat": ("Posti parcel, Posti Express Freight and Matkahuolto Lavarahti all state prices "
                "without VAT, so they compare directly. PostNord's VAT status could not be read "
                "(HTTP 403); its figures are compared only with that flag attached"),
        "fuel_surcharge": (
            "both upper-bound sources state the fuel surcharge is charged separately, so their "
            "readings are ex-fuel, which is the shape the parametric model needs because it "
            "applies fuel_surcharge_i(t) on top. the Posti digital parcel list does not mention "
            "fuel and PostNord could not be read. no adjustment is applied, because applying one "
            "would assume an inclusion the sources do not state. the consequence is recorded "
            "instead: if either parcel tariff did include fuel, its ex-fuel F and alpha would be "
            "lower than read here, by up to Posti's 21.10% parcel surcharge of September 2026 "
            "for that source"),
        "pickup": ("the parcel lists price hand-in at an outlet, the Posti freight list prices "
                   "outlet to recipient with pickup extra, and Lavarahti prices terminal to "
                   "recipient with mandatory pickup extra. freight readings are therefore shown "
                   "with and without pickup, and parcel readings are hand-in only"),
        "dates": ("effective dates differ: Lavarahti from 1.11.2024, the Posti freight list from "
                  "16.11.2025, the Posti parcel list 'until further notice' (file 202501), "
                  "PostNord unverified. no price-level adjustment between those dates is applied"),
    }


def shipment_weight_context(v4_dir=V4_DIR):
    """what the simulation actually ships, read-only from the v4 corpus"""
    orders = pd.read_csv(Path(v4_dir) / "orders.csv", usecols=["product_id", "material_type", "quantity"])
    products = pd.read_csv(Path(v4_dir) / "products.csv")
    merged = orders.merge(products, on=["product_id", "material_type"], how="left")
    weight = merged["quantity"] * merged["finished_weight_kg_per_unit"]
    return {
        "read_from": [rel(Path(v4_dir) / "orders.csv"), rel(Path(v4_dir) / "products.csv")],
        "definition": "quantity x finished_weight_kg_per_unit, as the supervisor computes shipment_weight_kg",
        "orders": int(weight.count()),
        "min_kg": round(float(weight.min()), 3),
        "p5_kg": round(float(weight.quantile(0.05)), 3),
        "median_kg": round(float(weight.median()), 3),
        "p95_kg": round(float(weight.quantile(0.95)), 3),
        "max_kg": round(float(weight.max()), 3),
        "share_at_or_below_35_kg": round(float((weight <= 35).mean()), 4),
        "share_above_1000_kg": round(float((weight > 1000).mean()), 4),
        "share_above_3000_kg": round(float((weight > 3000).mean()), 4),
    }


def rate_calibration_range():
    parcel, parcel_src = record("rate_calibration/posti_parcel.json")
    postnord, postnord_src = record("rate_calibration/postnord_parcel_zones.json")
    freight, freight_src = record("rate_calibration/posti_express_freight.json")
    lavarahti, lavarahti_src = record("rate_calibration/matkahuolto_lavarahti.json")

    parcel_r = posti_parcel_readings(parcel)
    postnord_r = postnord_readings(postnord)
    freight_r = posti_freight_readings(freight)
    lavarahti_r = lavarahti_readings(lavarahti)
    zone0 = freight_r["by_zone"][0]

    freight_F = [zone0["secant_loose_package_to_half_pallet"]["F_equivalent_eur"],
                 zone0["secant_half_pallet_to_pallet"]["F_equivalent_eur"],
                 *[s["F_equivalent_eur"] for s in lavarahti_r["secants"]]]
    freight_alpha = [zone0["secant_loose_package_to_half_pallet"]["alpha_equivalent_eur_per_kg"],
                     zone0["secant_half_pallet_to_pallet"]["alpha_equivalent_eur_per_kg"],
                     zone0["marginal_full_pallet_eur_per_kg"]["value"],
                     *[s["alpha_equivalent_eur_per_kg"] for s in lavarahti_r["secants"]]]
    parcel_F = [r["F_equivalent_eur"] for r in parcel_r]
    parcel_alpha = sorted({r["alpha_equivalent_eur_per_kg"] for r in parcel_r})

    lowest_parcel = min(parcel_r, key=lambda r: r["F_equivalent_eur"])
    crossovers = []
    for label, fixed, alpha in (
            ("Posti Express Freight zone 0, half pallet to pallet secant",
             zone0["secant_half_pallet_to_pallet"]["F_equivalent_eur"],
             zone0["secant_half_pallet_to_pallet"]["alpha_equivalent_eur_per_kg"]),
            ("Lavarahti 1-49 km, 100-149 to 950-999 kg secant, pickup included",
             lavarahti_r["secants"][3]["F_equivalent_eur"],
             lavarahti_r["secants"][3]["alpha_equivalent_eur_per_kg"])):
        weight = (fixed - lowest_parcel["F_equivalent_eur"]) / \
                 (lowest_parcel["alpha_equivalent_eur_per_kg"] - alpha)
        crossovers.append({
            "freight_reading": label,
            "crossover_kg": weight,
            "arithmetic": (f"({r4(fixed)} - {r4(lowest_parcel['F_equivalent_eur'])}) / "
                           f"({r4(lowest_parcel['alpha_equivalent_eur_per_kg'])} - {r4(alpha)}) "
                           f"= {r4(weight)} kg"),
        })

    return {
        "purpose": ("documents a method and the bounds the Finnish carrier evidence supports for "
                    "F_i and alpha_i. it is not a table of agent values and contains none"),
        "not_produced_here": ("per-agent F_i, alpha_i, beta_i or fuel-surcharge values for any of "
                              "the twelve logistics agents; that is v5 synthesis work"),
        "model": ("base_transport_cost_eur_i(t) = [F_i + alpha_i x weight_kg + beta_i x "
                  "(weight_kg / 1000) x distance_km] x [1 + fuel_surcharge_i(t)], "
                  "docs/data_synthesis_methodology.md section 7.1"),
        "source_records": [parcel_src, postnord_src, freight_src, lavarahti_src],
        "normalization": normalization(parcel, postnord, freight, lavarahti),
        "lower_bound_evidence": {"posti_parcel": parcel_r, "postnord_parcel_zones": postnord_r},
        "upper_bound_evidence": {"posti_express_freight": freight_r,
                                 "matkahuolto_lavarahti": lavarahti_r},
        "supported_readings": {
            "F_equivalent_eur": {
                "parcel_direct_readings": parcel_F,
                "freight_least_distance_secant_readings": {"min": min(freight_F),
                                                           "max": max(freight_F),
                                                           "values": freight_F},
                "freight_whole_unit_prices_zone0": [
                    zone0["whole_unit_prices_as_F_within_capacity"]["loose_package_up_to_35_kg"],
                    zone0["whole_unit_prices_as_F_within_capacity"]["furniture_pallet_up_to_1200_kg"]],
            },
            "alpha_equivalent_eur_per_kg": {
                "parcel_direct_readings": parcel_alpha,
                "freight_least_distance_readings": {"min": min(freight_alpha),
                                                    "max": max(freight_alpha),
                                                    "values": freight_alpha},
            },
            "these_are_discrete_readings": ("min and max are the smallest and largest readings "
                                            "actually computed from the evidence, not ends of a "
                                            "fitted or interpolated distribution"),
        },
        "finding_on_bound_direction": (
            "the evidence supports parcel tariffs as the lower bound for F but not for alpha. "
            "parcel tariffs carry the lowest fixed charge and the highest per-kg charge "
            f"({parcel_alpha[0]:.2f} EUR/kg), while freight tariffs carry higher fixed charges and "
            f"per-kg readings of about {min(freight_alpha):.3f} to {max(freight_alpha):.3f} EUR/kg, "
            "because freight prices capacity by the unit rather than by the kilogram. the "
            "evidence therefore describes two corners, low-F and high-alpha parcel against "
            "high-F and low-alpha freight, rather than one range in which both coefficients rise "
            "together"),
        "parcel_freight_crossover": {
            "parcel_reading": f"{lowest_parcel['service']} F = {lowest_parcel['F_equivalent_eur']:.2f}, "
                              f"alpha = {lowest_parcel['alpha_equivalent_eur_per_kg']:.2f}",
            "crossovers": crossovers,
            "note": (f"both crossovers lie above the {PARCEL_OVERSIZED_LIMIT_KG} kg parcel limit, "
                     "so the parcel line is not actually available at the weight where freight "
                     "becomes the cheaper reading"),
        },
        "gaps": [
            ("PostNord's tariff metadata could not be read (HTTP 403, no archive capture), and its "
             "single weight bracket per zone cannot separate F from alpha"),
            "Posti's delivery zones 0-5 are not defined in the price list, so they cannot be mapped to km",
            ("the parcel evidence stops at 25 kg (35 kg oversized); the simulation's shipments are "
             "mostly far heavier, see simulation_shipment_weights"),
            ("between 35 kg and a half pallet only Lavarahti's lowest brackets give evidence; there "
             "is no parcel-style F + alpha tariff for that range"),
            "whether either parcel tariff includes fuel is not stated or could not be read",
            ("the sources' effective dates span November 2024 to November 2025 and no price-level "
             "adjustment between them is applied"),
        ],
        "simulation_shipment_weights": shipment_weight_context(),
    }


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    beta = beta_derivation()
    write_json(PROCESSED / "beta_candidate_derivation.json", beta)
    closest = beta["closest_combination"]
    print(f"  beta     status={beta['status']}; closest {closest['match']} "
          f"({closest['corridor']}, {closest['rate_type']}, page {closest['page']}); "
          f"{beta['combinations_within_tolerance']} of {beta['combination_count']} within "
          f"{CLOSE_MATCH_TOLERANCE_PERCENT}%")
    ranges = rate_calibration_range()
    write_json(PROCESSED / "rate_calibration_range.json", ranges)
    supported = ranges["supported_readings"]
    print(f"  range    F parcel {supported['F_equivalent_eur']['parcel_direct_readings']}, "
          f"F freight secants {supported['F_equivalent_eur']['freight_least_distance_secant_readings']['min']:.2f}"
          f"-{supported['F_equivalent_eur']['freight_least_distance_secant_readings']['max']:.2f}; "
          f"alpha parcel {supported['alpha_equivalent_eur_per_kg']['parcel_direct_readings']}, "
          f"alpha freight {supported['alpha_equivalent_eur_per_kg']['freight_least_distance_readings']['min']:.4f}"
          f"-{supported['alpha_equivalent_eur_per_kg']['freight_least_distance_readings']['max']:.4f}")
    print(f"  wrote {rel(PROCESSED / 'beta_candidate_derivation.json')} and "
          f"{rel(PROCESSED / 'rate_calibration_range.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
