"""the seven sourced freight candidates, reused exactly as the prior proposal task wrote them

  1  Matkahuolto Lavarahti, 1-49 km column, [0, 1000) kg
  2  Matkahuolto Lavarahti, 1-49 km column, [1000, 1500) kg
  3  Matkahuolto Lavarahti, 1-49 km column, [1500, 2000) kg
  4  Matkahuolto Lavarahti, 1-49 km column, [2000, 2500) kg
  5  Matkahuolto Lavarahti, 1-49 km column, [2500, 3000) kg
  6  Posti Express Freight 14, zone 0, loose package to half pallet, [35, 500] kg
  7  Posti Express Freight 14, zone 0, half pallet to pallet, [500, 1000] kg

each candidate's F, alpha and weight domain is read from the prior calibration-proposal
task's output, data/processed/v5/logistics/agent_calibration_proposal.json, and nothing is
re-derived. verify_candidates recomputes each one from the pinned raw records only to
prove that reuse is exact; any difference raises CandidateDiscrepancy and the task stops
"""

import json
import re

from src.data_synthesis.v5.settings import PROCESSED_V5, RAW_V5, rel

RATES = RAW_V5 / "logistics" / "rate_calibration"
LAVARAHTI_PATH = RATES / "matkahuolto_lavarahti.json"
POSTI_FREIGHT_PATH = RATES / "posti_express_freight.json"
PRIOR_PROPOSAL_PATH = PROCESSED_V5 / "logistics" / "agent_calibration_proposal.json"

REUSED_FIELDS = ("fixed_charge_eur", "alpha_eur_per_kg", "min_shipment_weight_kg", "max_shipment_weight_kg",
                 "max_weight_inclusive")


def segment_id(low, high):
    return f"matkahuolto_lavarahti_1_49km_{low}_to_{high}kg_with_pickup"


POSTI_LOOSE_TO_HALF = "posti_express_freight_zone0_loose_to_half_pallet_with_pickup"
POSTI_HALF_TO_PALLET = "posti_express_freight_zone0_half_pallet_to_pallet_with_pickup"

CANDIDATES = {
    1: (segment_id(0, 1000), "Matkahuolto Lavarahti [0, 1000) kg"),
    2: (segment_id(1000, 1500), "Matkahuolto Lavarahti [1000, 1500) kg"),
    3: (segment_id(1500, 2000), "Matkahuolto Lavarahti [1500, 2000) kg"),
    4: (segment_id(2000, 2500), "Matkahuolto Lavarahti [2000, 2500) kg"),
    5: (segment_id(2500, 3000), "Matkahuolto Lavarahti [2500, 3000) kg"),
    6: (POSTI_LOOSE_TO_HALF, "Posti Express Freight [35, 500] kg, loose package to half pallet"),
    7: (POSTI_HALF_TO_PALLET, "Posti Express Freight [500, 1000] kg, half pallet to pallet"),
}


class CandidateDiscrepancy(ValueError):
    """a reused candidate no longer matches its pinned source"""


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_candidates(path=PRIOR_PROPOSAL_PATH):
    """the seven readings exactly as written by the prior task, keyed by candidate number"""
    readings = load_json(path)["readings"]
    return {number: {**readings[reading_id], "candidate": number, "label": label}
            for number, (reading_id, label) in CANDIDATES.items()}


def domains(candidates):
    return {number: {"min_kg": c["min_shipment_weight_kg"], "max_kg": c["max_shipment_weight_kg"],
                     "max_inclusive": c["max_weight_inclusive"]}
            for number, c in candidates.items()}


# ── recomputation from the pinned records, used only to verify the reuse ─────

def posti_unit_limit_kg(record, unit):
    printed = record["transport_unit_limits_as_printed_en_20240930"][unit]
    match = re.search(r"Maximum gross weight (\d+) kg", printed)
    if match is None:
        raise CandidateDiscrepancy(f"{rel(POSTI_FREIGHT_PATH)} no longer prints a gross weight for {unit}: {printed!r}")
    return float(match.group(1))


def posti_freight_secant(record, low, high):
    express = next(o for o in record["observations"] if o["service_printed_fi"] == "Express-rahti 14")
    zone0 = next(row for row in express["rows"] if row["zone"] == 0)["eur"]
    pickup = record["separate_pickup_as_printed"]["eur_by_transport_unit_count"]["1"]
    (low_col, low_unit), (high_col, high_unit) = low, high
    x1, x2 = posti_unit_limit_kg(record, low_unit), posti_unit_limit_kg(record, high_unit)
    alpha = (zone0[high_col] - zone0[low_col]) / (x2 - x1)
    return {"fixed_charge_eur": zone0[low_col] - alpha * x1 + pickup, "alpha_eur_per_kg": alpha,
            "min_shipment_weight_kg": x1, "max_shipment_weight_kg": x2, "max_weight_inclusive": True}


def lavarahti_pickup(record, kg_upper):
    for limit, eur in record["pickup_table_as_printed"]["rows"]:
        if kg_upper <= limit:
            return eur
    raise CandidateDiscrepancy(f"no Lavarahti pickup row covers {kg_upper} kg")


def lavarahti_secant(record, low_kg, high_kg):
    table = record["freight_table_parsed"]
    if table["km_bands"][0] != {"km_lower": 1, "km_upper": 49}:
        raise CandidateDiscrepancy("the first Lavarahti distance band is no longer 1-49 km")
    inside = [r for r in table["rows"] if r["kg_lower"] >= low_kg and r["kg_upper"] < high_kg]
    first, last = inside[0], inside[-1]
    x1, x2 = (first["kg_lower"] + first["kg_upper"]) / 2, (last["kg_lower"] + last["kg_upper"]) / 2
    y1 = first["eur_by_band"][0] + lavarahti_pickup(record, first["kg_upper"])
    y2 = last["eur_by_band"][0] + lavarahti_pickup(record, last["kg_upper"])
    alpha = (y2 - y1) / (x2 - x1)
    return {"fixed_charge_eur": y1 - alpha * x1, "alpha_eur_per_kg": alpha, "min_shipment_weight_kg": float(low_kg),
            "max_shipment_weight_kg": float(high_kg), "max_weight_inclusive": False}


def recompute(number, lavarahti, posti):
    if number <= 5:
        low, high = ((0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000))[number - 1]
        return lavarahti_secant(lavarahti, low, high)
    if number == 6:
        return posti_freight_secant(posti, ("loose_package", "Loose colli"), ("half_pallet_first_unit", "Half pallet"))
    return posti_freight_secant(posti, ("half_pallet_first_unit", "Half pallet"), ("pallet_first_unit", "Pallet"))


def verify_candidates(candidates):
    """every reused field must equal its recomputation from the pinned records exactly"""
    lavarahti, posti = load_json(LAVARAHTI_PATH), load_json(POSTI_FREIGHT_PATH)
    mismatches = []
    for number, candidate in candidates.items():
        fresh = recompute(number, lavarahti, posti)
        mismatches += [(number, field, candidate[field], fresh[field])
                       for field in REUSED_FIELDS if candidate[field] != fresh[field]]
    if mismatches:
        raise CandidateDiscrepancy(f"reused candidates differ from the pinned sources: {mismatches}")
    return {"candidates_checked": len(candidates), "fields_checked": list(REUSED_FIELDS),
            "comparison": "exact equality, python ==", "discrepancies": 0,
            "sources": [rel(LAVARAHTI_PATH), rel(POSTI_FREIGHT_PATH)], "reused_from": rel(PRIOR_PROPOSAL_PATH)}


def candidate_1_lower_bound():
    """the lower-bound tension inside the Lavarahti source, quoted from the pinned record"""
    record = load_json(LAVARAHTI_PATH)
    first = record["freight_table_parsed"]["rows"][0]
    service_text = next(line for line in record["service_limits_as_printed"] if "35 kg" in line)
    return {
        "used_kg": 0.0,
        "reading_0_kg": (f"the price table's first printed bracket is {first['kg_lower']}-{first['kg_upper']} kg "
                         f"(printed as '{record['freight_table_as_printed'][:6]}'), and the table has no "
                         f"minimum-weight column"),
        "reading_35_kg": f"the service's own description says '{service_text}' (for shipments over 35 kg or large shipments)",
        "reason": ("0 kg is used because the service text itself admits shipments under 35 kg when they are large "
                   "('tai isokokoisille'), the tariff prices them from 0 kg, and the simulation has no size field "
                   "that could tell a large 20 kg shipment from a small one"),
        "source": rel(LAVARAHTI_PATH),
    }
