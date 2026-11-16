"""unit tests for the as-of-date historical lookup.
runnable directly (python ...) or under pytest.

the rule under test is one sentence: the observation that applies on a date is
the most recent one whose own date is not after it. everything below is a way of
that going wrong. the ones that matter most are the two that would be invisible
in a finished dataset: reading a future observation because it happened to be
closer, and producing a different price depending on what day the code ran
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

from src.dataset import history
from src.dataset.history import (
    MalformedDate, MalformedSeries, ObservationNotFound, Series, from_fred_csv,
    from_manual_record, parse_date,
)

RAW_V5 = Path(__file__).resolve().parent.parent / "data" / "raw" / "v5"

# a deliberately awkward series: irregular gaps, and listed newest first
UNSORTED_ROWS = [
    ("2026-03-01", 5.89),
    ("2024-03-01", 9.08),
    ("2023-11-01", 8.23),
    ("2023-06-01", 6.05),
]


def linen():
    return Series("linen test", UNSORTED_ROWS, "EUR/kg")


def raises(fn, exc, fragment):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in the error, got {e}"
        return
    raise AssertionError(f"expected {exc.__name__} mentioning {fragment!r}")


# ── the rule ─────────────────────────────────────────────────────────────────

def test_an_exact_observation_date_returns_that_observation():
    for when, value in (("2023-06-01", 6.05), ("2023-11-01", 8.23),
                        ("2024-03-01", 9.08), ("2026-03-01", 5.89)):
        found = linen().as_of(when)
        assert found.value == value
        assert found.observation_date == parse_date(when)
        assert found.days_carried == 0 and found.is_exact


def test_a_date_between_two_observations_carries_the_earlier_one_forward():
    found = linen().as_of("2023-08-15")
    assert found.value == 6.05
    assert found.observation_date == date(2023, 6, 1)
    assert found.days_carried == 75
    assert not found.is_exact


def test_the_day_before_an_observation_still_reads_the_previous_one():
    """off by one in the other direction is the expensive mistake"""
    found = linen().as_of("2023-10-31")
    assert found.value == 6.05 and found.observation_date == date(2023, 6, 1)
    assert linen().as_of("2023-11-01").value == 8.23


def test_an_earlier_date_never_reads_a_later_observation():
    series = linen()
    for when in ("2023-06-02", "2023-09-09", "2024-02-29", "2025-12-31"):
        found = series.as_of(when)
        assert found.observation_date <= parse_date(when), (
            f"{when} read an observation dated {found.observation_date}")


def test_a_closer_later_observation_is_still_not_used():
    """2024-02-28 is two days from the 2024-03-01 observation and nine months
    from the one that actually applies. proximity is not the rule"""
    found = linen().as_of("2024-02-28")
    assert found.observation_date == date(2023, 11, 1) and found.value == 8.23


def test_an_oddly_ordered_source_resolves_the_same_as_a_sorted_one():
    """the ECB file lists newest first, so this is not hypothetical"""
    forwards = Series("sorted", sorted(UNSORTED_ROWS), "EUR/kg")
    backwards = Series("reversed", UNSORTED_ROWS, "EUR/kg")
    for when in ("2023-06-01", "2023-12-25", "2024-03-01", "2026-06-30"):
        assert forwards.as_of(when).as_dict()["value"] == backwards.as_of(when).as_dict()["value"]
        assert (forwards.as_of(when).observation_date
                == backwards.as_of(when).observation_date)


def test_nothing_is_interpolated():
    """every resolved value is one of the stored ones, never a blend"""
    series = linen()
    stored = {o.value for o in series.observations}
    for day in range(1, 1200, 7):
        when = date.fromordinal(date(2023, 6, 1).toordinal() + day)
        assert series.as_of(when).value in stored


# ── the two ends of the series ──────────────────────────────────────────────

def test_a_date_before_the_first_observation_raises():
    raises(lambda: linen().as_of("2023-05-31"), ObservationNotFound,
           "no observation on or before")
    raises(lambda: linen().as_of("1999-01-01"), ObservationNotFound,
           "the series starts at 2023-06-01")


def test_the_error_says_there_is_no_policy_rather_than_guessing_one():
    try:
        linen().as_of("2020-01-01")
    except ObservationNotFound as e:
        assert "no policy" in str(e)
        return
    raise AssertionError("a date before the series resolved to something")


def test_covers_answers_the_same_question_without_raising():
    assert linen().covers("2023-06-01") and linen().covers("2030-01-01")
    assert not linen().covers("2023-05-31")


def test_a_date_after_the_last_observation_is_carried_but_flagged():
    found = linen().as_of("2026-09-12")
    assert found.value == 5.89
    assert found.observation_date == date(2026, 3, 1)
    assert found.after_last_observation is True
    assert found.days_carried == 195


def test_a_date_inside_the_series_is_not_flagged_as_past_the_end():
    assert linen().as_of("2024-06-01").after_last_observation is False


def test_a_carry_forward_limit_turns_a_stale_read_into_an_error():
    bounded = Series("bounded", UNSORTED_ROWS, "EUR/kg", max_carry_forward_days=90)
    assert bounded.as_of("2023-08-15").value == 6.05
    raises(lambda: bounded.as_of("2026-09-12"), ObservationNotFound,
           "carry-forward limit of 90 days")


def test_without_a_limit_nothing_is_silently_bounded():
    assert linen().max_carry_forward_days is None
    assert linen().as_of("2099-01-01").value == 5.89


# ── the system clock ────────────────────────────────────────────────────────

class _NoClock(datetime):
    """a datetime whose now() and today() are unavailable

    parsing still works because strptime and fromisoformat are inherited"""

    @classmethod
    def now(cls, tz=None):
        raise AssertionError("the historical lookup read the system clock")

    @classmethod
    def today(cls):
        raise AssertionError("the historical lookup read the system clock")

    @classmethod
    def utcnow(cls):
        raise AssertionError("the historical lookup read the system clock")


def test_the_lookup_never_reads_the_system_clock():
    original = history.datetime
    history.datetime = _NoClock
    try:
        series = linen()
        assert series.as_of("2024-06-01").value == 9.08
        assert series.as_of("2026-09-12").after_last_observation
        raises(lambda: series.as_of("2020-01-01"), ObservationNotFound, "no policy")
    finally:
        history.datetime = original


def test_the_same_series_and_date_resolve_identically_whatever_day_it_is():
    """vary what the clock reports across years and confirm nothing moves"""
    original = history.datetime
    results = []
    for pretend_now in ("2020-01-01T00:00:00", "2024-07-04T12:00:00",
                        "2026-09-12T23:59:59", "2031-12-31T06:30:00"):
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
            series = linen()
            results.append(json.dumps(
                [series.as_of(w).as_dict() for w in
                 ("2023-06-01", "2023-12-25", "2024-03-01", "2026-09-12")],
                sort_keys=True))
        finally:
            history.datetime = original
    assert len(set(results)) == 1, "the lookup moved with the reported system time"


def test_no_clock_call_appears_in_the_module_source():
    import inspect

    source = inspect.getsource(history)
    for banned in (r"\bdatetime\.now\(", r"\bdate\.today\(", r"\btime\.time\(",
                   r"\butcnow\("):
        found = [m for m in re.findall(banned, source)]
        assert not found, f"{banned} appears in history.py"


# ── malformed input ─────────────────────────────────────────────────────────

def test_a_missing_or_unparseable_date_raises_a_typed_error():
    series = linen()
    raises(lambda: series.as_of(None), MalformedDate, "is missing")
    raises(lambda: series.as_of(""), MalformedDate, "is empty")
    raises(lambda: series.as_of("not-a-date"), MalformedDate, "is not an iso date")
    raises(lambda: series.as_of(20240601), MalformedDate, "must be an iso date string")


def test_datetimes_and_dates_are_both_accepted_as_the_same_day():
    series = linen()
    assert (series.as_of("2024-06-01").value
            == series.as_of("2024-06-01T13:45:00").value
            == series.as_of(date(2024, 6, 1)).value
            == series.as_of(datetime(2024, 6, 1, 13, 45)).value)


def test_an_offset_aware_stamp_is_read_as_its_wall_clock_date():
    assert linen().as_of("2024-03-01T00:30:00+03:00").observation_date == date(2024, 3, 1)


def test_an_empty_or_duplicated_or_non_numeric_series_raises():
    raises(lambda: Series("empty", [], "EUR/kg"), MalformedSeries,
           "at least one observation")
    raises(lambda: Series("dupe", [("2024-01-01", 1.0), ("2024-01-01", 2.0)], "x"),
           MalformedSeries, "share the date")
    raises(lambda: Series("text", [("2024-01-01", "about five")], "x"),
           MalformedSeries, "non-numeric value")
    raises(lambda: Series("inf", [("2024-01-01", float("inf"))], "x"),
           MalformedSeries, "non-finite value")
    raises(lambda: Series("neg", [("2024-01-01", 1.0)], "x", max_carry_forward_days=-1),
           MalformedSeries, "must not be negative")


# ── the pinned sources load through the same rule ───────────────────────────

def test_a_pinned_fred_series_loads_and_resolves():
    series = from_fred_csv(RAW_V5 / "material" / "cotton" / "PCOTTINDUSDM.csv",
                           "U.S. Cents per Pound")
    assert len(series) > 300
    assert series.first.date == date(1992, 1, 1)
    # monthly observations are stamped on the first, so mid-month carries forward
    mid = series.as_of("2024-06-17")
    assert mid.observation_date == date(2024, 6, 1) and mid.days_carried == 16
    assert series.as_of("2024-06-01").is_exact


def test_every_pinned_fred_series_is_monthly_and_ordered():
    for folder, name, unit in (("cotton", "PCOTTINDUSDM", "U.S. Cents per Pound"),
                               ("denim", "WPU03420104", "Index Dec 1994=100"),
                               ("leather", "PHIDEUSDM", "U.S. Cents per Pound")):
        series = from_fred_csv(RAW_V5 / "material" / folder / f"{name}.csv", unit)
        dates = [o.date for o in series.observations]
        assert dates == sorted(dates)
        assert all(d.day == 1 for d in dates), f"{name} is not stamped monthly"


def test_the_pinned_linen_record_carries_forward_and_does_not_interpolate():
    series = from_manual_record(RAW_V5 / "material" / "linen" / "alliance_flax.json")
    assert len(series) == 4 and series.unit == "EUR per kg"
    assert series.as_of("2024-03-01").value == 9.08
    # the two-year gap between the 2024 and 2026 observations is carried, not filled
    assert series.as_of("2025-06-01").value == 9.08
    assert series.as_of("2026-03-01").value == 5.89


def test_a_record_with_no_observations_refuses_to_become_an_empty_series():
    raises(lambda: from_manual_record(
        RAW_V5 / "material" / "polyester" / "sunsirs_dty.json"),
        MalformedSeries, "observations_status=missing")


def test_a_catalog_price_level_is_not_a_time_series():
    raises(lambda: from_manual_record(
        RAW_V5 / "material" / "cotton" / "tabb_jersey_knit.json"),
        MalformedSeries, "no dated observations")


def test_a_fred_loader_pointed_at_the_wrong_file_says_so():
    raises(lambda: from_fred_csv(RAW_V5 / "fx" / "ecb_eurofxref_hist.csv", "x"),
           MalformedSeries, "expected a FRED export")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} historical lookup unit tests passed")


if __name__ == "__main__":
    main()
