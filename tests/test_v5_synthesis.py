"""unit tests for the v5 synthesis subtree and the dataset it writes.
runnable directly (python ...) or under pytest.

four things are checked against something outside the code under test: the v4
constants the subtree copies still match the v4 files they were copied from, parsed
rather than imported; every per-agent logistics value is one whole reading from the
calibration evidence, and every parcel weight limit is the one the pinned Posti record
prints; every price in data/synthetic/v5 is recomputed from the pinned sources by
hand; and the dataset regenerates to the same bytes
"""

import ast
import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.dataset.fx import load_ecb_rates
from src.dataset.material_pricing import FIXED_BASIS, MaterialSources
from src.data_synthesis.v5 import (
    agent_calibration, build, fuel, logistics, manufacturers, materials, orders, products, settings,
)

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "data" / "synthetic" / "v4"
V5 = ROOT / "data" / "synthetic" / "v5"
SUBTREE = ROOT / "src" / "data_synthesis" / "v5"
POSTAL, CAPPED_WITH_FEE = {"L03", "L06"}, {"L07", "L10", "L12"}

_CACHE = {}


def cached(key, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def csv(path):
    return pd.read_csv(path)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def v4_constant(filename, name):
    tree = ast.parse((ROOT / "src" / "data_synthesis" / filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {filename}")


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def route():
    return cached("route", logistics.RouteCost)


def lavarahti_percent(day):
    """the shared series as printed in the raw record, latest row on or before the day"""
    record = json.loads((settings.RAW_V5 / "logistics" / "fuel_surcharge" / "matkahuolto_fuel_surcharge.json")
                        .read_text(encoding="utf-8"))
    rows = [r for r in record["series_parsed"]["Lavarahti"] if r["date"] <= day]
    return max(rows, key=lambda r: r["date"])["percent"]


def eur_to_usd(day):
    return cached("fx", lambda: load_ecb_rates(settings.RAW_V5 / "fx" / "ecb_eurofxref_hist.csv")).rate(
        "EUR", "USD", day).rate


def distance(origin, destination):
    matrix = csv(logistics.DISTANCE_MATRIX_PATH).set_index(["origin_manufacturer_location", "destination_deliver_to"])
    return matrix.loc[(origin, destination), "road_distance_km"]


# ── v4 stays untouched and uncoupled ────────────────────────────────────────

def test_no_v5_module_imports_a_v4_generator():
    for path in SUBTREE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert "synthesize_" not in name, f"{path.name} imports {name}"
                assert name != "src.data_synthesis" and not (
                    name.startswith("src.data_synthesis.") and not name.startswith("src.data_synthesis.v5")), \
                    f"{path.name} imports {name}"
                assert "bidding_formulas" not in name, f"{path.name} imports {name}"


def test_the_copied_v4_order_design_constants_still_match_the_v4_generator():
    for name in ("WEEKDAY_WEIGHTS", "SCENARIO_MIX", "PRIORITY_WEIGHTS", "QUANTITY_BANDS",
                 "DEADLINE_DAYS", "BUDGET_MULTIPLIERS", "DEFECT_TOLERANCE", "REPUTATION_REQUIREMENTS",
                 "AVG_WORKLOAD_MARKUP"):
        assert getattr(orders, name) == v4_constant("synthesize_orders.py", name), name
    assert settings.PRIORITY_MARKUP == v4_constant("synthesize_orders.py", "PRIORITY_MARKUP")
    source = (ROOT / "src" / "data_synthesis" / "synthesize_orders.py").read_text(encoding="utf-8")
    assert '"budget_override":   (0.93, 1.03)' in source and orders.TIGHT_LOW_BUDGET_BAND == (0.93, 1.03)


def test_the_copied_unit_cost_bands_and_cost_ratios_still_match():
    assert manufacturers.UNIT_COST_BANDS_EUR_V4 == v4_constant("synthesize_manufacturers.py", "UNIT_COSTS")
    specs = v4_constant("synthesize_material_providers.py", "AGENT_SPECS")
    copied = {s["agent_id"]: {m: v["cost_ratio"] for m, v in s["materials"].items()} for s in specs}
    assert materials.COST_RATIOS_V4 == copied


def test_the_carried_ratios_are_the_ones_behind_the_v4_material_csv():
    """v4 cost / ratio is one base price per material, up to v4's 3-decimal rounding"""
    v4 = csv(V4 / "material_agents.csv")
    for material in settings.ALL_MATERIALS:
        implied = [row[f"{material}_base_cost_eur_per_kg"] / materials.COST_RATIOS_V4[row["agent_id"]][material]
                   for _, row in v4.iterrows() if not pd.isna(row[f"{material}_base_cost_eur_per_kg"])]
        assert max(implied) - min(implied) < 0.01, (material, implied)


# ── part 1: per-agent logistics calibration ─────────────────────────────────

def calibration():
    return csv(settings.AGENT_CALIBRATION_PATH)


def reasoning():
    return json.loads(agent_calibration.REASONING_PATH.read_text(encoding="utf-8"))


def test_every_agent_pair_is_one_whole_reading_from_the_evidence():
    ranges = json.loads(agent_calibration.RANGE_PATH.read_text(encoding="utf-8"))
    parcel = {r["service"]: r for r in ranges["lower_bound_evidence"]["posti_parcel"]}
    zone0 = ranges["upper_bound_evidence"]["posti_express_freight"]["by_zone"][0]
    pickup = ranges["upper_bound_evidence"]["posti_express_freight"]["pickup_normalization"][
        "separate_pickup_one_unit_eur"]
    secants = ranges["upper_bound_evidence"]["matkahuolto_lavarahti"]["secants"]
    expected = {
        "posti_postal_parcel": (parcel["Postal Parcel"]["F_equivalent_eur"], parcel["Postal Parcel"]["alpha_equivalent_eur_per_kg"]),
        "posti_express_parcel": (parcel["Express Parcel"]["F_equivalent_eur"], parcel["Express Parcel"]["alpha_equivalent_eur_per_kg"]),
        "posti_home_parcel": (parcel["Home Parcel"]["F_equivalent_eur"], parcel["Home Parcel"]["alpha_equivalent_eur_per_kg"]),
        "posti_express_freight_zone0_loose_to_half_pallet_with_pickup": (
            zone0["secant_loose_package_to_half_pallet"]["F_equivalent_eur"] + pickup,
            zone0["secant_loose_package_to_half_pallet"]["alpha_equivalent_eur_per_kg"]),
        "posti_express_freight_zone0_half_pallet_to_pallet_with_pickup": (
            zone0["secant_half_pallet_to_pallet"]["F_equivalent_eur"] + pickup,
            zone0["secant_half_pallet_to_pallet"]["alpha_equivalent_eur_per_kg"]),
        "matkahuolto_lavarahti_1_49km_0_79_to_950_999kg_with_pickup": (
            secants[1]["F_equivalent_eur"], secants[1]["alpha_equivalent_eur_per_kg"]),
        "matkahuolto_lavarahti_1_49km_100_149_to_950_999kg_with_pickup": (
            secants[3]["F_equivalent_eur"], secants[3]["alpha_equivalent_eur_per_kg"]),
    }
    assert secants[1]["pickup_included"] and secants[3]["pickup_included"]
    for row in calibration().itertuples():
        f, a = expected[row.calibration_reading_id]
        assert close(row.fixed_charge_eur, f) and close(row.alpha_eur_per_kg, a), row.agent_id


def test_the_corner_structure_holds_and_no_pair_is_a_midpoint():
    frame = calibration()
    parcel, freight = frame[frame.calibration_corner == "parcel"], frame[frame.calibration_corner == "freight"]
    assert len(parcel) + len(freight) == 12 and len(parcel) > 0 and len(freight) > 0
    assert parcel.fixed_charge_eur.max() < freight.fixed_charge_eur.min()
    assert parcel.alpha_eur_per_kg.min() > freight.alpha_eur_per_kg.max()
    readings = set(reasoning()["readings_available"])
    assert set(frame.calibration_reading_id) <= readings


def test_beta_is_uniform_and_its_candidate_status_is_unchanged():
    assert set(calibration().beta_eur_per_tonne_km) == {0.07163}
    beta = json.loads((settings.PROCESSED_V5 / "logistics" / "beta_candidate_derivation.json").read_text(encoding="utf-8"))
    assert beta["status"] == "candidate"


def test_every_agent_uses_the_one_shared_fuel_series():
    assert set(calibration().fuel_surcharge_series_id) == {"matkahuolto_lavarahti"}
    assert set(csv(V5 / "logistics_agents.csv").fuel_surcharge_series_id) == {"matkahuolto_lavarahti"}
    assert not hasattr(agent_calibration, "fuel_for") and not hasattr(fuel, "usable_ids")
    coverage = fuel.window_coverage(fuel.load_series(settings.RAW_V5 / "logistics")["matkahuolto_lavarahti"])
    assert coverage["usable"] and coverage["window_days_after_last_observation"] == 0
    assert coverage["distinct_percentages_across_window_days"] == 22
    check = reasoning()["shared_fuel_series"]
    assert check["has_the_most_distinct_percentages_of_the_usable_series"]
    assert set(check["unusable_series"]) == {"posti_parcel", "posti_freight", "matkahuolto_rahti"}
    assert all(a["fuel_surcharge_series_id"] == "matkahuolto_lavarahti" for a in reasoning()["agents"])


def test_the_shared_series_resolves_by_each_orders_own_date():
    early, late = date(2025, 10, 20), date(2026, 3, 20)
    for agent in route().agents.values():
        sid = agent["fuel_surcharge_series_id"]
        assert route().fuel_lookup(sid, early).observation_date <= early
        assert route().fuel_lookup(sid, late).observation_date <= late
    assert route().fuel_lookup("matkahuolto_lavarahti", early).value == lavarahti_percent("2025-10-20") == 17.33
    assert route().fuel_lookup("matkahuolto_lavarahti", late).value == lavarahti_percent("2026-03-20") == 29.49


def test_parcel_weight_limits_are_the_ones_the_posti_record_prints():
    record = json.loads(agent_calibration.POSTI_PARCEL_PATH.read_text(encoding="utf-8"))["weight_limits_as_printed"]
    assert record["Postal Parcel, Home Parcel, Express Parcel and Return"] == "maximum weight 25 kg"
    assert "postal parcel’s maximum weight 25 kg" in record["with Oversized additional service"]
    assert "Home Parcel’s, Express Parcel’s and Return’s maximum weight 35 kg" in record["with Oversized additional service"]
    assert record["Oversized additional service price"] == "€ 3.50"
    frame = calibration().set_index("agent_id")
    for agent, row in frame.iterrows():
        if row.calibration_corner == "freight":
            assert pd.isna(row.max_shipment_weight_kg) and pd.isna(row.oversized_fee_eur), agent
        elif agent in POSTAL:
            assert row.calibration_reading_id == "posti_postal_parcel"
            assert row.max_shipment_weight_kg == 25.0 and pd.isna(row.oversized_above_kg) and pd.isna(row.oversized_fee_eur)
        else:
            assert agent in CAPPED_WITH_FEE and row.calibration_reading_id in ("posti_home_parcel", "posti_express_parcel")
            assert (row.max_shipment_weight_kg, row.oversized_above_kg, row.oversized_fee_eur) == (35.0, 25.0, 3.5)
    assert set(frame[frame.calibration_corner == "parcel"].index) == POSTAL | CAPPED_WITH_FEE
    for agent in reasoning()["agents"]:
        if agent["corner"] == "parcel":
            assert agent["weight_limit"]["max_shipment_weight_kg"] == frame.loc[agent["agent_id"], "max_shipment_weight_kg"]
            assert "kg" in agent["weight_limit"]["rule"] and agent["weight_limit"]["source_wording"]


def test_v4_service_fields_are_unchanged_on_the_v5_logistics_profile():
    v4, v5 = csv(V4 / "logistics_agents.csv").set_index("agent_id"), csv(V5 / "logistics_agents.csv").set_index("agent_id")
    for column in ("location", "service_regions", "default_completion_same_city_days",
                   "default_completion_intercity_days", "delay_probability", "damage_probability",
                   "reputation_score", "min_profit_margin", "initial_markup"):
        assert (v4[column] == v5[column]).all(), column
    assert set(v5.price_excludes) == {"initial_markup|min_profit_margin"}
    assert logistics.PRICE_EXCLUDES == ("initial_markup", "min_profit_margin")


def test_a_freight_price_applies_fuel_damage_and_priority_by_hand():
    got = route().price("L08", 850.0, "Tampere", "Oulu", "2025-11-20", "urgent")
    row = csv(V5 / "logistics_agents.csv").set_index("agent_id").loc["L08"]
    percent = lavarahti_percent("2025-11-20")
    freight = row.fixed_charge_eur + row.alpha_eur_per_kg * 850 + 0.07163 * 0.85 * distance("Tampere", "Oulu")
    expected = freight * (1 + percent / 100) * (1 + row.damage_probability) * 1.30 * eur_to_usd("2025-11-20")
    assert percent == 18.28 and got["fuel_percent"] == percent and got["oversized_fee_eur"] == 0.0
    assert close(got["price_usd"], expected)
    assert got["damage_probability"] == row.damage_probability > 0 and got["priority_markup"] == 0.30


def test_priority_scales_the_whole_logistics_price():
    day = date(2026, 1, 14)
    normal = route().price_for_distance("L04", 400.0, 180.0, day, "normal")
    assert close(route().price_for_distance("L04", 400.0, 180.0, day, "urgent"), normal * 1.30)
    assert close(route().price_for_distance("L04", 400.0, 180.0, day, "flexible"), normal * 0.85)


def test_the_oversized_fee_is_added_after_fuel_and_before_damage_and_priority():
    origin, destination = csv(logistics.DISTANCE_MATRIX_PATH).iloc[0][["origin_manufacturer_location",
                                                                        "destination_deliver_to"]]
    row = csv(V5 / "logistics_agents.csv").set_index("agent_id").loc["L10"]
    percent, rate, km = lavarahti_percent("2026-02-10"), eur_to_usd("2026-02-10"), distance(origin, destination)

    def by_hand(kg, fee):
        freight = 10.9 + 0.3 * kg + 0.07163 * kg / 1000 * km
        return (freight * (1 + percent / 100) + fee) * (1 + row.damage_probability) * rate

    for kg, fee in ((25.0, 0.0), (25.01, 3.5), (35.0, 3.5)):
        got = route().price("L10", kg, origin, destination, "2026-02-10", "normal")
        assert got["oversized_fee_eur"] == fee and close(got["price_usd"], by_hand(kg, fee)), kg


def test_no_capped_agent_can_be_priced_above_its_limit():
    day = date(2026, 1, 14)
    raises(lambda: route().price_for_distance("L03", 25.01, 100.0, day, "normal"), logistics.WeightOutsideDomain, "25")
    raises(lambda: route().price_for_distance("L07", 35.01, 100.0, day, "normal"), logistics.WeightOutsideDomain, "35")
    assert route().eligible_weight("L03", 25.0) and not route().eligible_weight("L06", 25.01)
    assert route().eligible_weight("L12", 35.0) and not route().eligible_weight("L12", 35.01)
    assert route().eligible_weight("L08", 20000.0)
    route().price_for_distance("L08", 20000.0, 100.0, day, "normal")


# ── part 2: manufacturer usd conversion ─────────────────────────────────────

def test_manufacturer_costs_are_v4_eur_times_one_reference_rate():
    rate = load_ecb_rates(settings.RAW_V5 / "fx" / "ecb_eurofxref_hist.csv").rate("EUR", "USD", date(2025, 10, 15))
    assert rate.rate == 1.1622 and rate.is_exact
    v4, v5 = csv(V4 / "manufacturer_agents.csv"), csv(V5 / "manufacturer_agents.csv")
    for product in settings.ALL_PRODUCTS:
        for eur, usd in zip(v4[f"unit_cost_{product}"], v5[f"unit_cost_usd_{product}"]):
            assert (pd.isna(eur) and pd.isna(usd)) or usd == round(eur * 1.1622, 4)
    for column in ("agent_id", "product_types", "daily_capacity_units", "current_workload_units",
                   "defect_probability", "reputation_score", "initial_markup", "location"):
        assert (v4[column] == v5[column]).all(), column
    assert set(v5.fx_eur_to_usd_reference) == {1.1622} and set(v5.pricing_date) == {"2025-10-15"}


# ── part 3: materials, products, orders ─────────────────────────────────────

def material_sources():
    return cached("material_sources", lambda: MaterialSources(settings.RAW_V5))


def test_reference_prices_come_from_material_pricing_on_each_day():
    table = csv(materials.REFERENCE_PATH).set_index(["material", "date"])
    for material in settings.ALL_MATERIALS:
        for day in (date(2025, 10, 15), date(2026, 1, 9), date(2026, 3, 2), date(2026, 4, 12)):
            assert close(table.loc[(material, day.isoformat()), "usd_per_kg"],
                         material_sources().price(material, day)["usd_per_kg"]), (material, day)
    counts = csv(materials.REFERENCE_PATH).groupby("material").usd_per_kg.nunique().to_dict()
    assert counts["cotton"] >= 6 and counts["denim"] >= 6 and counts["leather"] >= 6 and counts["linen"] == 2
    assert counts["polyester"] == 1


def test_polyester_is_priced_at_its_fixed_anchor_and_flagged_on_every_row():
    reference = csv(materials.REFERENCE_PATH)
    rows = reference[reference.material == "polyester"]
    assert len(rows) == settings.CREATION_WINDOW_DAYS and set(rows.price_basis) == {FIXED_BASIS}
    assert set(rows.usd_per_kg) == {material_sources().anchor("polyester")["usd_per_kg"]}
    assert round(rows.usd_per_kg.iloc[0], 4) == 14.6887
    assert rows[["series", "series_observation_date", "series_days_carried", "series_ratio"]].isna().all().all()
    dated = reference[reference.material != "polyester"]
    assert set(dated.price_basis) == {"anchor_times_dated_series"} and dated.series.notna().all()
    assert set(settings.FIXED_PRICE_MATERIALS) == {"polyester"}


def test_the_material_profile_restores_every_v4_agent_and_material():
    profile, v4 = csv(V5 / "material_agents.csv"), csv(V4 / "material_agents.csv")
    assert list(profile.agent_id) == list(v4.agent_id) and len(profile) == 15
    assert "blocked_material_types" not in profile.columns
    for row in profile.to_dict("records"):
        supplied = set(row["material_types"].split("|"))
        assert supplied == set(materials.COST_RATIOS_V4[row["agent_id"]]), row["agent_id"]
    polyester = profile[profile.polyester_price_ratio_to_reference.notna()].set_index("agent_id")
    assert set(polyester.index) == {"M01", "M04", "M05", "M06", "M07", "M11"}
    for agent, row in polyester.iterrows():
        assert row.polyester_price_ratio_to_reference == materials.COST_RATIOS_V4[agent]["polyester"]
        assert row.polyester_available_kg == v4.set_index("agent_id").loc[agent, "polyester_available_kg"]


def test_an_agents_base_cost_is_its_ratio_times_the_reference_on_the_pricing_date():
    profile = csv(V5 / "material_agents.csv")
    reference = csv(materials.REFERENCE_PATH).set_index(["material", "date"]).usd_per_kg
    for row in profile.to_dict("records"):
        for material in row["material_types"].split("|"):
            expected = round(row[f"{material}_price_ratio_to_reference"] * reference[(material, "2025-10-15")], 4)
            assert close(row[f"{material}_base_cost_usd_per_kg"], expected, 1e-12), (row["agent_id"], material)


def test_products_are_the_v4_table_unchanged():
    v4, v5 = csv(V4 / "products.csv"), csv(V5 / "products.csv")
    assert len(v5) == 12 and v4.equals(v5)
    assert (V4 / "products.csv").read_bytes().replace(b"\r\n", b"\n") == (V5 / "products.csv").read_bytes()


def test_orders_cover_all_five_materials_and_stay_in_the_window():
    frame = csv(V5 / "orders.csv")
    assert len(frame) == 50000 and set(frame.material_type) == set(settings.ALL_MATERIALS)
    pairs = set(zip(frame.product_id, frame.material_type))
    assert pairs == set(zip(csv(V4 / "products.csv").product_id, csv(V4 / "products.csv").material_type))
    created = pd.to_datetime(frame.order_created).dt.date
    assert created.min() >= date(2025, 10, 15) and created.max() <= date(2026, 4, 12)
    assert (frame.pricing_date == frame.order_created.str[:10]).all()


def test_metadata_states_polyester_is_fixed_and_every_provisional_decision():
    meta = json.loads((V5 / "orders_metadata.json").read_text(encoding="utf-8"))
    coverage = meta["material_coverage"]
    assert coverage["all_v4_materials_covered"] is True and coverage["materials"] == list(settings.ALL_MATERIALS)
    assert coverage["product_material_pairs"] == {"v4": 12, "v5": 12}
    notice = coverage["FIXED_PRICE_NOTICE"]
    assert "FIXED" in notice and "14.6887" in notice and "does not move with time" in notice
    assert coverage["price_basis"]["polyester"].startswith("fixed")
    assert coverage["fixed_prices_usd_per_kg"]["polyester"] == material_sources().anchor("polyester")["usd_per_kg"]
    assert coverage["orders_by_material"] == coverage["v4_orders_by_material"]
    for stale in ("WARNING", "blocked_materials", "v5_orders_with_a_blocked_material"):
        assert stale not in coverage
    assert "provisional" in meta["window"]["status"]
    price = meta["logistics_price"]
    assert "provisional" in price["status"]
    assert set(price["excludes"]) == {"initial_markup", "min_profit_margin"}
    assert "damage_probability" in price["applies"] and "priority markup" in price["applies"]
    assert price["shared_fuel_series"]["id"] == "matkahuolto_lavarahti"
    assert set(price["parcel_weight_limits"]) == POSTAL | CAPPED_WITH_FEE
    assert meta["seed"] == 42 and meta["base_date"] == "2025-10-15"


def test_every_v5_file_passes_its_schema():
    assert {path: problems for path, problems in build.validate().items() if problems} == {}


def test_budgets_sit_in_their_band_of_an_independently_recomputed_anchor():
    mat, ref = csv(V5 / "material_agents.csv"), csv(materials.REFERENCE_PATH).set_index(["material", "date"]).usd_per_kg
    mfr, log, prod = csv(V5 / "manufacturer_agents.csv"), csv(V5 / "logistics_agents.csv"), csv(V5 / "products.csv")
    f_mat = (1 + mat.defect_probability.mean()) * (1 + mat.initial_markup.mean())
    f_mfr = (1 + mfr.defect_probability.mean()) * (1 + mfr.initial_markup.mean()) * 1.03
    caps = log.set_index("agent_id").max_shipment_weight_kg
    for row in csv(V5 / "orders.csv").iloc[::997].to_dict("records"):
        pair = prod[(prod.product_id == row["product_id"]) & (prod.material_type == row["material_type"])].iloc[0]
        m, weight = row["material_type"], row["quantity"] * pair.finished_weight_kg_per_unit
        material = row["quantity"] * pair.material_kg_per_unit * ref[(m, row["pricing_date"])] \
            * mat[f"{m}_price_ratio_to_reference"].dropna().mean() * f_mat
        manufacturing = row["quantity"] * mfr[f"unit_cost_usd_{row['product_id']}"].dropna().mean() * f_mfr
        makers = mfr[mfr.product_types.str.split("|").apply(lambda t: row["product_id"] in t)].location
        km = sum(route().distance_km[(loc, row["deliver_to"])] for loc in makers) / len(makers)
        serving = [a for a in log[log.service_regions.str.split("|").apply(lambda r: row["deliver_to"] in r)].agent_id
                   if pd.isna(caps[a]) or weight <= caps[a]]
        day = date.fromisoformat(row["pricing_date"])
        shipping = sum(route().price_for_distance(a, weight, km, day, row["priority"]) for a in serving) / len(serving)
        market = (material + manufacturing) * (1 + settings.PRIORITY_MARKUP[row["priority"]]) + shipping
        band = {"low_budget": orders.BUDGET_MULTIPLIERS["low_budget"],
                "tight_low_budget": orders.TIGHT_LOW_BUDGET_BAND}.get(row["scenario_type"],
                                                                     orders.BUDGET_MULTIPLIERS[row["priority"]])
        assert band[0] - 1e-4 <= row["budget_usd"] / market <= band[1] + 1e-4, (row["order_id"], row["budget_usd"] / market)


def test_orders_and_material_prices_regenerate_to_identical_bytes():
    frame, _, _ = orders.generate_frame()
    assert frame.to_csv(index=False, lineterminator="\n").encode() == (V5 / "orders.csv").read_bytes()
    reference = materials.reference_prices(material_sources())
    assert reference.to_csv(index=False, lineterminator="\n").encode() == materials.REFERENCE_PATH.read_bytes()
    profile, _ = materials.profile(material_sources())
    assert profile.to_csv(index=False, lineterminator="\n").encode() == materials.PROFILE_PATH.read_bytes()


# ── part 4: diagnostics ─────────────────────────────────────────────────────

def test_the_diagnostics_report_records_its_checks_and_warnings():
    report = json.loads((settings.REPORTS_V5 / "bid_competition_v5.json").read_text(encoding="utf-8"))
    assert report["orders"] == 50000 and set(report["orders_by_material"]) == set(settings.ALL_MATERIALS)
    assert isinstance(report["warnings"], list)
    assert set(report["fuel_date_pattern"]) == {"matkahuolto_lavarahti"}
    assert report["fuel_date_pattern"]["matkahuolto_lavarahti"]["distinct_percentages_over_orders"] > 1
    assert report["material_movement"]["polyester"]["price_basis"] == FIXED_BASIS


def test_no_parcel_agent_wins_above_its_weight_limit():
    report = json.loads((settings.REPORTS_V5 / "bid_competition_v5.json").read_text(encoding="utf-8"))
    assert report["weight_domain_violations"] == 0
    domains = report["logistics_weight_domains_kg"]
    for agent, heaviest in report["logistics_heaviest_win_kg"].items():
        if domains[agent]["max"] is not None:
            assert heaviest <= domains[agent]["max"], (agent, heaviest)
    assert {a for a, d in domains.items() if d["max"] is not None} == POSTAL | CAPPED_WITH_FEE
    for band in ("500-1000 kg", "1000-1500 kg", "1500-2000 kg", "2000-2500 kg", "2500 kg and over"):
        assert report["logistics_corner_wins_by_weight"].get(band, {}).get("parcel", 0) == 0, band


def test_the_fuel_counterfactual_shows_no_swing_across_shared_series():
    result = json.loads((settings.REPORTS_V5 / "fuel_counterfactual_v5.json").read_text(encoding="utf-8"))
    assert result["stored_mapping"] == "shared matkahuolto_lavarahti"
    assert result["largest_swing_across_shared_series_points"] == 0.0
    stored = result["variants"]["shared matkahuolto_lavarahti"]["lowest_cost_logistics_win_share"]
    report = json.loads((settings.REPORTS_V5 / "bid_competition_v5.json").read_text(encoding="utf-8"))
    assert stored == report["win_share_lowest_cost"]["logistics"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 synthesis unit tests passed")


if __name__ == "__main__":
    main()
