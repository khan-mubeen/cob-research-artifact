"""unit tests for the date-matched fx layer.
runnable directly (python ...) or under pytest.

three things have to hold. the direction has to be right, checked against real
rows of the pinned ECB file and against the three independently verified
cross-check values, not against an assertion in a docstring. a pair with no EUR
side has to cross correctly through the published EUR legs. and there must be no
fallback rate anywhere: a missing pair, a missing date or a date the file cannot
answer raises, which is exactly what v4's hardcoded 0.92 did not do
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

from src.dataset import fx, history
from src.dataset.fx import (
    FxError, MissingRate, UnknownCurrency, load_ecb_rates,
)
from src.dataset.history import MalformedDate, MalformedSeries

RAW_V5 = Path(__file__).resolve().parent.parent / "data" / "raw" / "v5"
ECB_CSV = RAW_V5 / "fx" / "ecb_eurofxref_hist.csv"

# the three values verified by hand during the methodology work. they are a
# sanity check on the downloaded series, never the source of truth
CROSS_CHECK = (("USD", "2024", 1.0824), ("ZAR", "2024", 19.8297),
               ("USD", "2026-08", 1.1593))

_RATES = None


def rates():
    global _RATES
    if _RATES is None:
        _RATES = load_ecb_rates(ECB_CSV)
    return _RATES


def raises(fn, exc, fragment):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in the error, got {e}"
        return
    raise AssertionError(f"expected {exc.__name__} mentioning {fragment!r}")


# ── the pinned file ─────────────────────────────────────────────────────────

def test_the_pinned_ecb_file_loads_offline_and_covers_what_v5_needs():
    table = rates()
    assert table.covers_currency("USD") and table.covers_currency("ZAR")
    assert table.covers_currency("EUR"), "EUR is the base and is always available"
    assert not table.covers_currency("XXX")
    assert len(table.currencies) > 30


def test_the_file_is_quoted_as_units_per_one_euro():
    """1 EUR buys more than 1 USD and far more than 1 ZAR, in that order

    if the file were inverted these three would all be wrong together, which is
    the point of checking magnitudes rather than a single number"""
    usd = rates().rate("EUR", "USD", "2024-06-14").rate
    zar = rates().rate("EUR", "ZAR", "2024-06-14").rate
    assert 0.8 < usd < 1.6, f"EUR/USD of {usd} is not a plausible units-per-euro rate"
    assert 10 < zar < 30, f"EUR/ZAR of {zar} is not a plausible units-per-euro rate"
    assert zar > usd


def test_the_downloaded_series_reproduces_the_hand_verified_cross_checks():
    """the only use these values have: confirming the download is the right one"""
    table = rates()
    for currency, period, expected in CROSS_CHECK:
        series = table._series[currency]
        values = [o.value for o in series.observations
                  if o.date.isoformat().startswith(period)]
        assert values, f"no {currency} observations in {period}"
        average = sum(values) / len(values)
        assert abs(average - expected) < 0.0001, (
            f"EUR/{currency} over {period} averages {average:.4f}, the verified "
            f"cross-check is {expected}")


def test_the_cross_check_values_are_recorded_on_the_fx_source_not_used_as_data():
    with open(RAW_V5 / "fx" / "source.json", encoding="utf-8") as f:
        source = json.load(f)
    entry = next(s for s in source["sources"] if s["artifact"].endswith(".csv"))
    note = entry["cross_check"]["verification_note"]
    assert "never as the FX layer's actual source of truth" in note


# ── direction, both ways and through the cross ──────────────────────────────

def test_a_eur_pair_is_read_straight_off_the_file():
    resolved = rates().rate("EUR", "USD", "2024-06-14")
    assert resolved.rate_date == date(2024, 6, 14) and resolved.is_exact
    assert resolved.via == "ecb reference rate"
    assert abs(resolved.rate - 1.0686) < 1e-9


def test_the_inverse_pair_is_the_reciprocal():
    forward = rates().rate("EUR", "USD", "2024-06-14").rate
    back = rates().rate("USD", "EUR", "2024-06-14")
    assert abs(back.rate - 1.0 / forward) < 1e-12
    assert back.via == "inverted EUR/USD"


def test_zar_to_usd_is_eur_to_usd_divided_by_eur_to_zar():
    """the formula the methodology states, checked as arithmetic"""
    when = "2024-06-14"
    eur_usd = rates().rate("EUR", "USD", when).rate
    eur_zar = rates().rate("EUR", "ZAR", when).rate
    crossed = rates().zar_to_usd(when)
    assert abs(crossed.rate - (eur_usd / eur_zar)) < 1e-12
    assert crossed.via == "EUR/USD divided by EUR/ZAR"


def test_the_cross_rate_is_the_right_way_round():
    """a rand is worth a few US cents, not a few dollars"""
    rate = rates().zar_to_usd("2024-06-14").rate
    assert 0.03 < rate < 0.10, f"ZAR to USD of {rate} is upside down"
    assert abs(rates().rate("USD", "ZAR", "2024-06-14").rate * rate - 1.0) < 1e-12


def test_a_known_conversion_comes_out_where_it_should():
    """265 ZAR per metre, the iMaterial IMA10 linen price, in 2024 dollars"""
    amount, resolved = rates().convert(265.00, "ZAR", "USD", "2024-06-14")
    assert 12.0 < amount < 18.0, f"265 ZAR converted to {amount} USD"
    assert abs(amount - 265.00 * resolved.rate) < 1e-12


def test_both_legs_of_a_cross_are_taken_from_the_same_date():
    """a friday EUR/USD must never be mixed with a thursday EUR/ZAR"""
    saturday = "2024-06-15"
    crossed = rates().zar_to_usd(saturday)
    usd = rates().rate("EUR", "USD", saturday)
    zar = rates().rate("EUR", "ZAR", saturday)
    assert usd.rate_date == zar.rate_date == date(2024, 6, 14)
    assert crossed.rate_date == date(2024, 6, 14)
    assert abs(crossed.rate - usd.rate / zar.rate) < 1e-12


# ── already-USD amounts ─────────────────────────────────────────────────────

def test_a_usd_amount_passes_through_without_touching_the_file():
    resolved = rates().rate("USD", "USD", "2024-06-14")
    assert resolved.rate == 1.0 and resolved.via == "identity"
    assert resolved.rate_date == resolved.requested_date and resolved.is_exact
    amount, used = rates().to_usd(1234.56, "USD", "2024-06-14")
    assert amount == 1234.56 and used.via == "identity"


def test_an_identity_conversion_works_on_a_day_the_ecb_did_not_publish():
    """christmas day, and a date past the end of the file

    an identity that quietly depended on a published row would make a USD amount
    in a USD dataset fail on a holiday"""
    for when in ("2024-12-25", "2099-01-01"):
        assert rates().rate("USD", "USD", when).rate == 1.0
        assert rates().to_usd(10.0, "USD", when)[0] == 10.0


def test_a_eur_identity_also_passes_through():
    assert rates().rate("EUR", "EUR", "2024-06-14").rate == 1.0


# ── weekends, holidays and the ends of the file ─────────────────────────────

def test_a_weekend_takes_the_previous_working_day_and_never_the_next():
    saturday = rates().rate("EUR", "USD", "2024-06-15")
    sunday = rates().rate("EUR", "USD", "2024-06-16")
    assert saturday.rate_date == sunday.rate_date == date(2024, 6, 14)
    assert saturday.days_carried == 1 and sunday.days_carried == 2
    assert not saturday.is_exact


def test_a_target_holiday_carries_forward_too():
    """1 january is a TARGET closing day every year"""
    resolved = rates().rate("EUR", "USD", "2024-01-01")
    assert resolved.rate_date < date(2024, 1, 1)
    assert resolved.days_carried > 0


def test_a_date_before_the_file_starts_raises_rather_than_extrapolating():
    raises(lambda: rates().rate("EUR", "USD", "1998-12-31"), MissingRate,
           "no observation on or before")


def test_a_carry_forward_limit_makes_a_stale_rate_an_error():
    bounded = load_ecb_rates(ECB_CSV, max_carry_forward_days=5)
    assert bounded.rate("EUR", "USD", "2024-06-15").days_carried == 1
    raises(lambda: bounded.rate("EUR", "USD", "2099-01-01"), MissingRate,
           "carry-forward limit of 5 days")


# ── nothing falls back ──────────────────────────────────────────────────────

def test_a_missing_date_raises_and_says_there_is_no_current_rate():
    raises(lambda: rates().rate("EUR", "USD", None), MissingRate,
           "no fallback rate in this layer")


def test_a_malformed_date_raises_a_typed_error():
    raises(lambda: rates().rate("EUR", "USD", "sometime"), MalformedDate,
           "is not an iso date")
    raises(lambda: rates().rate("EUR", "USD", ""), MalformedDate, "is empty")


def test_a_missing_or_malformed_currency_raises():
    raises(lambda: rates().rate(None, "USD", "2024-06-14"), FxError,
           "base currency is missing")
    raises(lambda: rates().rate("EUR", None, "2024-06-14"), FxError,
           "quote currency is missing")
    raises(lambda: rates().rate("EUR", "dollars", "2024-06-14"), FxError,
           "not a three-letter code")
    raises(lambda: rates().rate("EUR", 840, "2024-06-14"), FxError,
           "must be a code")


def test_a_currency_the_file_does_not_carry_raises():
    raises(lambda: rates().rate("EUR", "XBT", "2024-06-14"), UnknownCurrency,
           "is not in ecb_eurofxref_hist.csv")
    raises(lambda: rates().to_usd(10.0, "XBT", "2024-06-14"), UnknownCurrency,
           "is not in")


def test_a_currency_the_ecb_retired_is_still_answered_inside_its_own_range():
    """the file keeps withdrawn columns as N/A after the changeover; those rows
    are skipped rather than read as a rate, and the series simply ends"""
    table = rates()
    assert table.covers_currency("HRK")
    assert table.rate("EUR", "HRK", "2015-06-15").rate > 1
    raises(lambda: table.rate("EUR", "HRK", "2015-06-15").convert("lots"), FxError,
           "amount must be a number")


def _module_code(module):
    """the module's executable code, with docstrings and comments removed

    the prose in fx.py names the v4 constant it replaces, so a plain text search
    would flag its own explanation. these checks are about what runs"""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree), tree


def test_no_fixed_rate_constant_appears_anywhere_in_the_module():
    """the v4 value this layer exists to remove"""
    import ast

    code, tree = _module_code(fx)
    numbers = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)]
    assert 0.92 not in numbers, "the v4 fixed rate is a constant in fx.py"
    for banned in (r"FX_USD_TO_EUR", r"DEFAULT_RATE", r"FALLBACK"):
        assert not re.search(banned, code), f"{banned} appears in fx.py code"


def test_the_fx_layer_opens_no_sockets_and_reads_no_clock():
    code, _ = _module_code(fx)
    for banned in ("urllib", "requests", "socket", "urlopen",
                   "datetime.now(", "date.today(", "time.time("):
        assert banned not in code, f"{banned} appears in fx.py code"


def test_a_malformed_rate_file_fails_at_load_not_at_the_first_lookup():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.csv"
        empty.write_text("", encoding="utf-8")
        raises(lambda: load_ecb_rates(empty), MalformedSeries, "the file is empty")

        wrong = Path(tmp) / "wrong.csv"
        wrong.write_text("observation_date,PCOTTINDUSDM\n2024-01-01,80.0\n",
                         encoding="utf-8")
        raises(lambda: load_ecb_rates(wrong), MalformedSeries,
               "whose first column is Date")

        blanks = Path(tmp) / "blanks.csv"
        blanks.write_text("Date,USD\n2024-01-02,N/A\n", encoding="utf-8")
        raises(lambda: load_ecb_rates(blanks), MalformedSeries,
               "no currency column had any observation")

    raises(lambda: load_ecb_rates(Path(tmp) / "gone.csv"), FxError,
           "no ECB reference-rate file at")


# ── provenance travels with the number ──────────────────────────────────────

def test_every_rate_carries_its_own_provenance():
    resolved = rates().rate("ZAR", "USD", "2024-06-15")
    blob = resolved.as_dict()
    assert blob["requested_date"] == "2024-06-15"
    assert blob["rate_date"] == "2024-06-14"
    assert blob["days_carried"] == 1
    assert blob["via"] == "EUR/USD divided by EUR/ZAR"
    assert blob["source"].endswith("ecb_eurofxref_hist.csv")
    assert len(blob["source_sha256"]) == 64


def test_the_recorded_source_hash_is_the_hash_of_the_file_that_was_read():
    import hashlib

    expected = hashlib.sha256(ECB_CSV.read_bytes()).hexdigest()
    assert rates().source_sha256 == expected


def test_the_source_hash_matches_what_the_manifest_recorded():
    from src.dataset import manifest as manifest_reader

    manifest = manifest_reader.load(RAW_V5 / "manifest.json")
    recorded = manifest_reader.artifact(manifest, "fx/ecb_eurofxref_hist.csv")
    assert rates().source_sha256 == recorded["sha256"]
    assert recorded["hash_purpose"] == "source_artifact_integrity"


def test_a_conversion_returns_the_amount_and_the_rate_that_made_it():
    amount, resolved = rates().to_usd(100.0, "EUR", "2024-06-14")
    assert abs(amount - 100.0 * resolved.rate) < 1e-12
    assert resolved.base == "EUR" and resolved.quote == "USD"


# ── determinism ─────────────────────────────────────────────────────────────

def test_the_same_pair_and_date_resolve_identically_whatever_day_it_is():
    original = history.datetime
    results = []
    for pretend_now in ("2019-02-02T00:00:00", "2026-09-12T12:00:00",
                        "2040-11-30T23:00:00"):
        fixed = datetime.strptime(pretend_now, "%Y-%m-%dT%H:%M:%S")

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed

            @classmethod
            def today(cls):
                return fixed

        history.datetime = _Frozen
        try:
            table = load_ecb_rates(ECB_CSV)
            results.append(json.dumps([table.rate(b, q, w).as_dict()
                                       for b, q, w in (("EUR", "USD", "2024-06-14"),
                                                       ("ZAR", "USD", "2024-06-15"),
                                                       ("USD", "USD", "2024-12-25"))],
                                      sort_keys=True))
        finally:
            history.datetime = original
    assert len(set(results)) == 1, "an fx rate moved with the reported system time"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} fx unit tests passed")


if __name__ == "__main__":
    main()
