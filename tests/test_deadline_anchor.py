"""unit tests for what "now" means when the supervisor measures a deadline.
runnable directly (python -m tests.test_deadline_anchor) or under pytest.

the rule these pin down: outside dynamic mode the supervisor anchors a deadline
on the order's own order_created, never on the day the run happens. the same
saved order therefore has to give the same remaining-days answer whenever it is
replayed, including long after its deadline has passed on the real calendar,
which is exactly the case that used to collapse to the one-day clamp.
"""

from datetime import datetime, timedelta

from src.supervisor.supervisor import (
    DYNAMIC_RESOURCES, days_until_deadline, order_reference_now, remaining_days_now,
)


def order(order_id, created, deadline):
    return {"order_id": order_id, "order_created": created, "deadline": deadline}


# an order from well before any plausible run date, so "its deadline has already
# passed on the real clock" stays true however long this test lives
STALE = order("ORD-STALE", "2020-03-02T09:30:00", "2020-03-14")
STALE_INTERVAL = 11 + 14.5 / 24          # 2020-03-02 09:30 -> 2020-03-14 00:00


def expected_interval(o):
    created = datetime.fromisoformat(o["order_created"])
    deadline = datetime.strptime(o["deadline"], "%Y-%m-%d")
    return (deadline - created).total_seconds() / 86400.0


# ── the case the fix exists for ──────────────────────────────────────────────

def test_stale_order_is_not_measured_against_the_real_calendar():
    # the precondition: this deadline really is in the past as the test runs
    assert datetime.strptime(STALE["deadline"], "%Y-%m-%d") < datetime.now()
    assert remaining_days_now(STALE) == STALE_INTERVAL


def test_stale_order_is_neither_negative_nor_clamped():
    # before the anchor fix this was about -2000 raw and 1 after the clamp
    assert remaining_days_now(STALE) > 0
    assert days_until_deadline(STALE) == 11
    assert days_until_deadline(STALE) != 1


def test_a_stale_order_still_loses_a_deadline_it_never_had():
    # anchoring is not the same as being generous: an order created two days
    # before its own deadline gets two days, not the eleven above
    tight = order("ORD-TIGHT", "2020-03-12T00:00:00", "2020-03-14")
    assert days_until_deadline(tight) == 2


# ── the fixture from the audit ───────────────────────────────────────────────

def test_ord_00001_gets_its_stored_eight_day_interval():
    # created 2026-07-17T11:00, deadline 2026-07-26, the eight whole days the
    # saved corpus intends. a static run after 2026-07-26 used to give it one
    first = order("ORD-00001", "2026-07-17T11:00:00", "2026-07-26")
    assert remaining_days_now(first) == 8 + 13.0 / 24
    assert days_until_deadline(first) == 8


# ── the property that makes a static run reproducible ────────────────────────

def test_the_interval_is_the_stored_one_for_every_sampled_offset():
    # each order is created a different distance from the same real "today", so
    # any dependence on the run date would show up as a spread across these
    for offset_days in (-900, -30, -1, 0, 1, 30, 900):
        created = datetime.now() + timedelta(days=offset_days)
        deadline = (created + timedelta(days=9)).date()
        o = order(f"ORD-{offset_days}", created.isoformat(), deadline.isoformat())
        assert remaining_days_now(o) == expected_interval(o)


def test_static_mode_reads_the_same_field_dynamic_mode_seeds_its_clock_from():
    assert order_reference_now(STALE) == datetime.fromisoformat(STALE["order_created"])


# ── the dynamic path is untouched: an explicit reading still wins ────────────

def test_an_explicit_now_still_overrides_the_anchor():
    # this is how dynamic mode calls in, passing the walking clock's reading
    sim_now = datetime(2020, 3, 10, 12, 0, 0)
    assert remaining_days_now(STALE, sim_now) == 3.5
    assert days_until_deadline(STALE, sim_now) == 3


def test_the_clamp_still_applies_to_a_reading_past_the_deadline():
    # dynamic mode can legitimately walk past a deadline, and that still floors
    # at one day rather than turning negative
    assert days_until_deadline(STALE, datetime(2020, 4, 1)) == 1


# ── degenerate orders keep their old behavior rather than raising ────────────

def test_an_order_with_no_creation_time_falls_back_to_the_real_clock():
    o = {"order_id": "ORD-NONE", "deadline": "2020-03-14"}
    assert order_reference_now(o) is None
    assert remaining_days_now(o) < 0        # the real clock, as it always was

def test_an_unreadable_creation_time_falls_back_rather_than_raising():
    assert order_reference_now(order("ORD-JUNK", "not-a-timestamp", "2020-03-14")) is None


def test_an_offset_aware_creation_time_is_measured_not_rejected():
    o = order("ORD-TZ", "2020-03-02T09:30:00+03:00", "2020-03-14")
    assert remaining_days_now(o) == STALE_INTERVAL


def main():
    print(f"DYNAMIC_RESOURCES={DYNAMIC_RESOURCES}, real now={datetime.now().isoformat(timespec='seconds')}")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} deadline anchor unit tests passed")


if __name__ == "__main__":
    main()
