"""unit tests for the logistics calibration research.
runnable directly (python ...) or under pytest.

the beta file has to report a candidate as a candidate and state that 0.07163 is
used regardless. the range file has to show its arithmetic, flag the differences
between sources before comparing them, and contain no agent values at all. every
expected number below is recomputed from the raw records by hand, not copied from
the output it checks
"""

import json
from pathlib import Path

import pandas as pd

from src.dataset import logistics_calibration as lc

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "v5" / "logistics"
RAW = ROOT / "data" / "raw" / "v5" / "logistics"


def stored(name):
    with open(PROCESSED / name, encoding="utf-8") as f:
        return json.load(f)


def raw(artifact):
    with open(RAW / artifact, encoding="utf-8") as f:
        return json.load(f)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ── beta ─────────────────────────────────────────────────────────────────────

def test_beta_is_used_as_specified_whatever_the_derivation_finds():
    beta = stored("beta_candidate_derivation.json")
    assert beta["beta_used"]["value"] == 0.07163
    assert beta["beta_used"]["unit"] == "EUR/tonne-km"
    assert "independent of whether this candidate derivation is ever confirmed" in \
        beta["beta_used"]["basis"]


def test_the_closest_combination_is_recomputed_by_hand():
    rates = [(o["corridor"], o["rate_type"], o["eur_per_km"])
             for o in raw("line_haul_rate/upply_corridor_rates.json")["observations"]]
    payloads = (24, 24.15, 25, 26, 28, 32.8)
    combos = [(abs(r / p - 0.07163) / 0.07163 * 100, c, t, r, p)
              for c, t, r in rates for p in payloads]
    combos.sort()
    best = combos[0]
    beta = stored("beta_candidate_derivation.json")
    closest = beta["closest_combination"]
    assert (closest["corridor"], closest["rate_type"], closest["eur_per_km"], closest["payload_t"]) == \
        (best[1], best[2], best[3], best[4]) == ("Duisburg to Prague", "contract", 1.73, 24.15)
    assert close(closest["exact_eur_per_tonne_km"], 1.73 / 24.15)
    assert closest["arithmetic"] == "1.7300 / 24.1500 = 0.0716"
    assert close(closest["difference_from_0_07163_percent"], round((1.73 / 24.15 - 0.07163) / 0.07163 * 100, 4))
    assert beta["combination_count"] == len(combos) == 48
    assert beta["combinations_within_tolerance"] == sum(1 for c in combos if c[0] <= 1.0) == 3


def test_a_close_match_stays_a_candidate_because_the_document_states_no_payload():
    beta = stored("beta_candidate_derivation.json")
    assert beta["status"] == "candidate"
    assert beta["document_states_payload"]["value"] is False
    text = json.dumps(beta)
    assert '"status": "confirmed"' not in text


def test_the_status_rule_has_only_three_outcomes_and_never_confirms_on_arithmetic():
    assert lc.derivation_status(0.0078, False) == "candidate"
    assert lc.derivation_status(1.0, False) == "candidate"
    assert lc.derivation_status(1.0001, False) == "unresolved"
    assert lc.derivation_status(5.0, True) == "unresolved"
    assert lc.derivation_status(0.0, True).startswith("candidate")
    assert "confirmed" not in lc.derivation_status(0.0, True)


def test_the_methodology_scan_found_no_further_payload_figure():
    scan = lc.methodology_payload_figures()
    assert scan["figures_found_t"] == []
    assert stored("beta_candidate_derivation.json")["payloads_tested_t"] == [24.0, 24.15, 25.0, 26.0, 28.0, 32.8]


def test_the_beta_file_regenerates_identically():
    assert json.loads(json.dumps(lc.beta_derivation())) == stored("beta_candidate_derivation.json")


# ── F_i and alpha_i ──────────────────────────────────────────────────────────

def test_the_range_file_contains_no_agent_values():
    text = json.dumps(stored("rate_calibration_range.json"))
    for token in ('"agent_id"', '"L01"', '"L12"', "per_agent"):
        assert token not in text, token
    assert "not a table of agent values" in stored("rate_calibration_range.json")["purpose"]


def test_parcel_readings_are_the_printed_tariff():
    readings = stored("rate_calibration_range.json")["lower_bound_evidence"]["posti_parcel"]
    assert [(r["service"], r["F_equivalent_eur"], r["alpha_equivalent_eur_per_kg"]) for r in readings] == [
        ("Postal Parcel", 5.90, 0.30), ("Express Parcel", 8.90, 0.30), ("Home Parcel", 10.90, 0.30)]


def test_postnord_is_reported_as_not_identifiable_rather_than_forced():
    for r in stored("rate_calibration_range.json")["lower_bound_evidence"]["postnord_parcel_zones"]:
        assert r["F_equivalent_eur"] is None and r["alpha_equivalent_eur_per_kg"] is None
        assert r["method"].startswith("not identifiable")


def test_the_lavarahti_secant_is_recomputed_by_hand():
    table = raw("rate_calibration/matkahuolto_lavarahti.json")["freight_table_parsed"]
    rows = {r["kg_lower"]: r for r in table["rows"]}
    p1, p2 = rows[100]["eur_by_band"][0], rows[950]["eur_by_band"][0]
    alpha = (p2 - p1) / (974.5 - 124.5)
    fixed = p1 - alpha * 124.5
    secants = stored("rate_calibration_range.json")["upper_bound_evidence"]["matkahuolto_lavarahti"]["secants"]
    as_published = next(s for s in secants if not s["pickup_included"] and s["points"][0].startswith("100-149"))
    assert close(as_published["alpha_equivalent_eur_per_kg"], alpha)
    assert close(as_published["F_equivalent_eur"], fixed)
    with_pickup = next(s for s in secants if s["pickup_included"] and s["points"][0].startswith("100-149"))
    alpha_p = ((p2 + 46.46) - (p1 + 22.18)) / 850
    assert close(with_pickup["alpha_equivalent_eur_per_kg"], alpha_p)
    assert close(with_pickup["F_equivalent_eur"], (p1 + 22.18) - alpha_p * 124.5)


def test_the_posti_freight_secant_is_recomputed_by_hand():
    freight = raw("rate_calibration/posti_express_freight.json")
    zone0 = next(o for o in freight["observations"] if o["service_printed_fi"] == "Express-rahti 14")["rows"][0]["eur"]
    alpha = (zone0["pallet_first_unit"] - zone0["half_pallet_first_unit"]) / (1000 - 500)
    reading = stored("rate_calibration_range.json")["upper_bound_evidence"]["posti_express_freight"]["by_zone"][0]
    assert close(reading["secant_half_pallet_to_pallet"]["alpha_equivalent_eur_per_kg"], alpha)
    assert close(reading["secant_half_pallet_to_pallet"]["F_equivalent_eur"],
                 zone0["half_pallet_first_unit"] - alpha * 500)


def test_the_reported_extremes_are_computed_readings_not_a_fitted_range():
    supported = stored("rate_calibration_range.json")["supported_readings"]
    freight_f = supported["F_equivalent_eur"]["freight_least_distance_secant_readings"]
    assert freight_f["min"] == min(freight_f["values"]) and freight_f["max"] == max(freight_f["values"])
    freight_a = supported["alpha_equivalent_eur_per_kg"]["freight_least_distance_readings"]
    assert freight_a["min"] == min(freight_a["values"]) and freight_a["max"] == max(freight_a["values"])


def test_the_bound_direction_finding_follows_from_the_numbers():
    ranges = stored("rate_calibration_range.json")
    supported = ranges["supported_readings"]
    assert max(supported["alpha_equivalent_eur_per_kg"]["parcel_direct_readings"]) > \
        supported["alpha_equivalent_eur_per_kg"]["freight_least_distance_readings"]["max"]
    assert min(supported["F_equivalent_eur"]["parcel_direct_readings"]) < \
        supported["F_equivalent_eur"]["freight_least_distance_secant_readings"]["min"]
    assert "not for alpha" in ranges["finding_on_bound_direction"]
    for crossover in ranges["parcel_freight_crossover"]["crossovers"]:
        assert crossover["crossover_kg"] > 35


def test_normalization_flags_every_difference_before_comparison():
    by_source = {s["source"]: s for s in stored("rate_calibration_range.json")["normalization"]["sources"]}
    assert by_source["posti_express_freight"]["fuel_surcharge_treatment"] == "charged separately"
    assert by_source["matkahuolto_lavarahti"]["fuel_surcharge_treatment"].startswith("charged separately")
    assert by_source["posti_parcel"]["fuel_surcharge_treatment"] == "not stated in source"
    assert by_source["postnord_parcel_zones"]["vat_status"].startswith("not verified")


def test_the_shipment_weight_context_matches_an_independent_read_of_v4():
    v4 = ROOT / "data" / "synthetic" / "v4"
    orders = pd.read_csv(v4 / "orders.csv", usecols=["product_id", "material_type", "quantity"])
    products = pd.read_csv(v4 / "products.csv")
    weights = orders.merge(products, on=["product_id", "material_type"])
    weights = weights["quantity"] * weights["finished_weight_kg_per_unit"]
    context = stored("rate_calibration_range.json")["simulation_shipment_weights"]
    assert context["orders"] == len(weights) == 50000
    assert context["median_kg"] == round(float(weights.median()), 3)
    assert context["max_kg"] == round(float(weights.max()), 3)


def test_the_range_file_regenerates_identically():
    assert json.loads(json.dumps(lc.rate_calibration_range())) == stored("rate_calibration_range.json")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} logistics calibration unit tests passed")


if __name__ == "__main__":
    main()
