"""unit tests for the v5 material pricing module.
runnable directly (python ...) or under pytest.

every anchor is recomputed here from the raw manual records with the
methodology's own constants, and every dated price is recomputed from the raw
series, so the module is checked against arithmetic rather than against itself.
polyester must be its fixed anchor on every date and say so, dates must never read
a later observation, and nothing may depend on the clock or the network
"""

import ast
import inspect
import json
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path

from src.dataset import history, material_pricing
from src.dataset.fx import load_ecb_rates
from src.dataset.history import HistoryError, MalformedDate, from_fred_csv, from_manual_record
from src.dataset.material_pricing import (
    ANCHOR_DATES, DATED_BASIS, FIXED_BASIS, MaterialPricingError, MaterialSources, NoTemporalSeries,
    UnknownMaterial, sample_calculations,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "v5"
SAMPLE = ROOT / "data" / "processed" / "v5" / "material" / "sample_price_calculations.json"

_SOURCES = None


def sources():
    global _SOURCES
    if _SOURCES is None:
        _SOURCES = MaterialSources(RAW)
    return _SOURCES


def raw(relative):
    with open(RAW / relative, encoding="utf-8") as f:
        return json.load(f)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in the error, got {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


# ── anchors recomputed by hand ──────────────────────────────────────────────

def test_the_cotton_anchor_is_the_mean_of_the_four_jersey_knit_prices():
    prices = [o["value"] for o in raw("material/cotton/tabb_jersey_knit.json")["observations"]]
    assert prices == [6.65, 6.95, 7.65, 7.95]
    assert close(sources().anchor("cotton")["usd_per_kg"], sum(prices) / 4 * 2.20462)


def test_the_denim_and_polyester_anchors_use_the_linear_yard_conversion():
    def usd_per_kg(price, oz, width):
        return price / (oz * width / 36 * 28.3495 / 1000)
    denim = [usd_per_kg(4.85, 14.5, 60), usd_per_kg(3.85, 10, 60),
             usd_per_kg(7.50, 14.5, 68), usd_per_kg(7.50, 13.3, 68)]
    assert close(sources().anchor("denim")["usd_per_kg"], sum(denim) / 4)
    assert close(sources().anchor("polyester")["usd_per_kg"], usd_per_kg(4.65, 6.7, 60))


def test_the_leather_anchor_uses_the_midpoint_of_the_sold_weight_range():
    anchor = sources().anchor("leather")
    assert close(anchor["usd_per_kg"], 89.95 / (20 * 2.5 * 28.3495 / 1000))
    bounds = anchor["products"][0]["usd_per_kg_at_range_bounds"]
    assert close(bounds["2.0 oz/sqft"], 89.95 / (20 * 2.0 * 0.0283495))
    assert close(bounds["3.0 oz/sqft"], 89.95 / (20 * 3.0 * 0.0283495))


def test_the_linen_anchor_converts_zar_at_the_anchor_date_through_the_fx_layer():
    rates = load_ecb_rates(RAW / "fx" / "ecb_eurofxref_hist.csv")
    zar_to_usd = rates.rate("ZAR", "USD", ANCHOR_DATES["linen"][0]).rate
    linen = [o for o in raw("material/linen/imaterial_linen.json")["observations"] if o["use"] == "linen"]
    expected = sum(o["value"] / ((o["width_cm"] / 100) * o["gsm"] / 1000) * zar_to_usd for o in linen) / len(linen)
    assert len(linen) == 4
    assert close(sources().anchor("linen")["usd_per_kg"], expected)


def test_every_anchor_is_reproduced_exactly_at_its_anchor_date():
    for material in ("cotton", "polyester", "linen", "denim", "leather"):
        anchor = sources().anchor(material)
        at_anchor = sources().price(material, anchor["anchor_date"])
        assert at_anchor["usd_per_kg"] == anchor["usd_per_kg"], material
        assert at_anchor["series_ratio"] == 1.0


# ── dated prices recomputed from the raw series ─────────────────────────────

def test_a_cotton_price_follows_the_fred_series_ratio():
    series = from_fred_csv(RAW / "material" / "cotton" / "PCOTTINDUSDM.csv", "cents")
    at, anchor = series.as_of("2025-06-17"), series.as_of(ANCHOR_DATES["cotton"][0])
    assert at.observation_date == date(2025, 6, 1)
    expected = sources().anchor("cotton")["usd_per_kg"] * (at.value / 100 * 2.20462) / (anchor.value / 100 * 2.20462)
    assert close(sources().price("cotton", "2025-06-17")["usd_per_kg"], expected)


def test_a_denim_price_follows_the_index_ratio_without_currency_conversion():
    series = from_fred_csv(RAW / "material" / "denim" / "WPU03420104.csv", "index")
    ratio = series.as_of("2025-02-10").value / series.as_of(ANCHOR_DATES["denim"][0]).value
    assert close(sources().price("denim", "2025-02-10")["usd_per_kg"],
                 sources().anchor("denim")["usd_per_kg"] * ratio)


def test_linen_carries_the_last_flax_observation_forward_and_never_interpolates():
    flax = from_manual_record(RAW / "material" / "linen" / "alliance_flax.json")
    rates = load_ecb_rates(RAW / "fx" / "ecb_eurofxref_hist.csv")
    early = sources().price("linen", "2024-06-01")
    late = sources().price("linen", "2026-02-28")
    assert early["usd_per_kg"] == late["usd_per_kg"]
    assert early["series_at_requested_date"]["observation_date"] == "2024-03-01"
    usd_obs = 9.08 * rates.rate("EUR", "USD", "2024-03-01").rate
    anchor_obs = flax.as_of(ANCHOR_DATES["linen"][0])
    usd_anchor = anchor_obs.value * rates.rate("EUR", "USD", anchor_obs.observation_date).rate
    assert close(early["usd_per_kg"], sources().anchor("linen")["usd_per_kg"] * usd_obs / usd_anchor)


def test_an_earlier_date_never_reads_a_later_observation():
    for material in ("cotton", "linen", "denim", "leather"):
        for when in ("2024-12-31", "2025-07-15", "2026-06-30"):
            used = sources().price(material, when)["series_at_requested_date"]["observation_date"]
            assert used <= when, (material, when, used)


def test_the_v4_window_is_flat_because_every_dated_source_ends_before_it():
    document = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert set(document["window_movement"]) == set(material_pricing.MATERIALS)
    for material, movement in document["window_movement"].items():
        assert movement["distinct_prices_across_window"] == 1, material
        if material == "polyester":
            assert movement["price_basis"] == FIXED_BASIS
            assert movement["all_carried_past_last_observation"] is None
        else:
            assert movement["price_basis"] == DATED_BASIS, material
            assert movement["all_carried_past_last_observation"] is True, material


# ── polyester: a fixed anchor, never a dated price ──────────────────────────

def test_polyester_is_the_tabb_twill_anchor_on_every_date_with_no_fx_step():
    record = raw("material/polyester/tabb_polyester_twill.json")
    [twill] = record["observations"]
    assert (record["currency"], twill["sku"], twill["value"], twill["width_inches"]) == ("USD", "TWPWD", 4.65, 60)
    assert "6.7oz" in twill["description"]
    expected = 4.65 / (6.7 * 60 / 36 * 28.3495 / 1000)
    assert round(expected, 4) == 14.6887
    anchor = sources().anchor("polyester")
    assert "fx" not in anchor["products"][0] and anchor["reference_currency"] == "USD"
    for when in ("2019-03-01", "2025-10-15", "2026-01-31", "2026-04-12", "2031-12-31"):
        price = sources().price("polyester", when)
        assert price["usd_per_kg"] == anchor["usd_per_kg"] and close(price["usd_per_kg"], expected), when
        assert price["price_basis"] == FIXED_BASIS and price["series_ratio"] == 1.0
        assert price["series_at_requested_date"] is None and price["temporal_series_status"] == "missing"


def test_polyester_records_that_no_temporal_series_exists():
    assert not sources().has_temporal_series("polyester")
    assert all(sources().has_temporal_series(m) for m in ("cotton", "linen", "denim", "leather"))
    raises(lambda: sources().series_usd("polyester", "2026-08-01"), NoTemporalSeries, "observations_status='missing'")
    assert sources().price("cotton", "2025-10-15")["price_basis"] == DATED_BASIS


def test_a_polyester_series_record_that_stops_declaring_missing_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(RAW / "material", root / "material")
        shutil.copytree(RAW / "fx", root / "fx", ignore=shutil.ignore_patterns("*.zip"))
        path = root / "material" / "polyester" / "sunsirs_dty.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["observations_status"] = "supplied"
        path.write_text(json.dumps(record), encoding="utf-8")
        raises(lambda: MaterialSources(root), MaterialPricingError, "revisited")


# ── what must fail ──────────────────────────────────────────────────────────


def test_unknown_material_and_bad_dates_raise_typed_errors():
    raises(lambda: sources().price("silk", "2026-08-01"), UnknownMaterial, "silk")
    raises(lambda: sources().price("cotton", None), MalformedDate, "missing")
    raises(lambda: sources().price("cotton", "soon"), MalformedDate, "not an iso date")
    raises(lambda: sources().price("linen", "2023-05-31"), HistoryError, "no observation on or before")


# ── purity ──────────────────────────────────────────────────────────────────

def test_prices_do_not_move_with_the_system_clock():
    original = history.datetime
    results = []
    for pretend in ("2019-01-01T00:00:00", "2026-09-13T12:00:00", "2040-06-30T23:00:00"):
        fixed = datetime.strptime(pretend, "%Y-%m-%dT%H:%M:%S")

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed

            @classmethod
            def today(cls):
                return fixed

        history.datetime = _Frozen
        try:
            results.append(json.dumps([sources().price(m, "2025-09-01")
                                       for m in material_pricing.MATERIALS], sort_keys=True))
        finally:
            history.datetime = original
    assert len(set(results)) == 1


def test_the_module_reads_no_clock_opens_no_socket_and_imports_no_config_at_module_level():
    tree = ast.parse(inspect.getsource(material_pricing))
    module_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_imports.append(node.module)
    for banned in ("config", "src.dataset.context", "urllib", "socket", "requests"):
        assert not any(m == banned or m.startswith(banned + ".") for m in module_imports), banned
    code = ast.unparse(tree)
    for banned in ("datetime.now(", "date.today(", "time.time("):
        assert banned not in code, banned


def test_the_sample_file_regenerates_identically():
    document = json.loads(SAMPLE.read_text(encoding="utf-8"))
    rebuilt = sample_calculations(sources(), document["order_window"], document["spot_check_dates"])
    assert json.loads(json.dumps(rebuilt)) == document


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} material pricing unit tests passed")


if __name__ == "__main__":
    main()
