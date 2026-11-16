"""as-of-date lookup over a pinned observation series

one lookup rule, used by every dated source regardless of shape: a monthly FRED
series, the ECB's working-day reference rates, and the four irregular linen
observations published inside quarterly press releases all resolve the same way.

    the observation that applies on a date is the most recent one
    whose own date is not after it

that is carry-forward, and it is the whole rule. nothing here interpolates,
because no methodology in this project calls for interpolation, and a price
invented between two real observations is indistinguishable from a real one once
it reaches a bid.

two things this module deliberately does not do:

  it never reads the system clock. not as a default, not as a fallback, not for
  a "current" lookup. a price that depends on what day the code ran is not a
  historical price, and the v4 supervisor's real-clock deadline fallback is
  exactly the pattern being kept out of pricing

  it never resolves a context, a config or a path on its own. it is handed a
  series and a date. that keeps config, the dataset context and this layer from
  forming an import cycle, and it is why everything here is testable without any
  of them

a date before the first observation raises. there is no approved policy for
pricing something the source does not cover, and guessing one would be silent
extrapolation backwards
"""

import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


class HistoryError(ValueError):
    """anything wrong with a series, a lookup date or the pairing of the two"""


class MalformedSeries(HistoryError):
    """the observations are unusable: empty, duplicated, or not numbers"""


class MalformedDate(HistoryError):
    """a date that cannot be parsed, or is missing where one is required"""


class ObservationNotFound(HistoryError):
    """the series does not cover the requested date

    raised for a date before the first observation, and for a date past the last
    one when the caller set a carry-forward limit"""


def parse_date(value, field="date") -> date:
    """an iso date or datetime, as a date. anything else raises

    None and the empty string raise too: a missing date is the error this layer
    exists to surface, not something to fill in"""
    if value is None:
        raise MalformedDate(f"{field} is missing")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise MalformedDate(f"{field} must be an iso date string, got {value!r}")

    text = value.strip()
    if not text:
        raise MalformedDate(f"{field} is empty")
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # an offset-aware stamp is a wall-clock date here, the offset is not a price input
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        raise MalformedDate(f"{field} {value!r} is not an iso date or datetime")


@dataclass(frozen=True)
class Observation:
    date: date
    value: float
    note: str = ""


@dataclass(frozen=True)
class Lookup:
    """one resolved observation, with everything needed to explain it

    the caller gets the value and, in the same object, which observation it came
    from and how far it was carried. a number on its own cannot be audited"""

    series: str
    requested_date: date
    observation_date: date
    value: float
    unit: str
    days_carried: int
    after_last_observation: bool
    source: str = ""

    @property
    def is_exact(self) -> bool:
        return self.days_carried == 0

    def as_dict(self):
        return {
            "series": self.series,
            "requested_date": self.requested_date.isoformat(),
            "observation_date": self.observation_date.isoformat(),
            "value": self.value,
            "unit": self.unit,
            "days_carried": self.days_carried,
            "after_last_observation": self.after_last_observation,
            "source": self.source,
        }


class Series:
    """an immutable, sorted, deduplicated set of dated observations

    the series is validated once at construction rather than at every lookup, so
    a malformed source fails where it is loaded and names the file, not three
    layers down inside a price"""

    def __init__(self, name, observations, unit, frequency="", source="",
                 max_carry_forward_days=None):
        if not observations:
            raise MalformedSeries(f"{name}: a series needs at least one observation")

        parsed = []
        for raw in observations:
            if isinstance(raw, Observation):
                parsed.append(raw)
                continue
            when, value = raw[0], raw[1]
            note = raw[2] if len(raw) > 2 else ""
            parsed.append(Observation(parse_date(when, f"{name} observation date"),
                                      _number(value, name, when), note))

        parsed.sort(key=lambda o: o.date)
        for earlier, later in zip(parsed, parsed[1:]):
            if earlier.date == later.date:
                raise MalformedSeries(
                    f"{name}: two observations share the date {earlier.date}, "
                    f"values {earlier.value} and {later.value}")

        if max_carry_forward_days is not None and max_carry_forward_days < 0:
            raise MalformedSeries(
                f"{name}: max_carry_forward_days must not be negative")

        self.name = name
        self.unit = unit
        self.frequency = frequency
        self.source = source
        self.max_carry_forward_days = max_carry_forward_days
        self._observations = tuple(parsed)

    # ── the series itself ───────────────────────────────────────────────────

    @property
    def observations(self):
        return self._observations

    @property
    def first(self) -> Observation:
        return self._observations[0]

    @property
    def last(self) -> Observation:
        return self._observations[-1]

    def __len__(self):
        return len(self._observations)

    def __repr__(self):
        return (f"Series({self.name!r}, {len(self)} observations, "
                f"{self.first.date} to {self.last.date}, unit={self.unit!r})")

    def covers(self, when) -> bool:
        return parse_date(when) >= self.first.date

    # ── the lookup ──────────────────────────────────────────────────────────

    def as_of(self, when) -> Lookup:
        """the observation that applies on this date

        an exact match returns that observation. anything else returns the most
        recent observation strictly before it. a later observation is never
        returned, whatever the input ordering was or how close its value is"""
        target = parse_date(when, f"{self.name} lookup date")

        if target < self.first.date:
            raise ObservationNotFound(
                f"{self.name}: no observation on or before {target}, the series "
                f"starts at {self.first.date}. there is no policy for pricing a "
                f"date the source does not cover")

        found = _latest_not_after(self._observations, target)
        days_carried = (target - found.date).days
        after_last = target > self.last.date

        limit = self.max_carry_forward_days
        if limit is not None and days_carried > limit:
            raise ObservationNotFound(
                f"{self.name}: {target} is {days_carried} days after the last "
                f"applicable observation {found.date}, past this series' "
                f"carry-forward limit of {limit} days")

        return Lookup(
            series=self.name,
            requested_date=target,
            observation_date=found.date,
            value=found.value,
            unit=self.unit,
            days_carried=days_carried,
            after_last_observation=after_last,
            source=self.source,
        )

    def value_as_of(self, when) -> float:
        """the bare number, for callers that carry provenance some other way"""
        return self.as_of(when).value


def _number(value, series_name, when):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(str(value).strip())
        except (TypeError, ValueError):
            raise MalformedSeries(
                f"{series_name}: observation {when} has a non-numeric value {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise MalformedSeries(
            f"{series_name}: observation {when} has a non-finite value {value!r}")
    return value


def _latest_not_after(observations, target):
    """binary search for the last observation whose date is <= target

    the series is sorted at construction, so this cannot be fooled by a source
    file that happens to list its rows newest first, as the ECB's does"""
    low, high = 0, len(observations) - 1
    found = None
    while low <= high:
        mid = (low + high) // 2
        if observations[mid].date <= target:
            found = observations[mid]
            low = mid + 1
        else:
            high = mid - 1
    return found


# ── loaders ──────────────────────────────────────────────────────────────────
# one per stored shape. each one only adapts a file into a Series; the lookup
# rule above stays the single implementation

def from_rows(name, rows, unit, **kwargs) -> Series:
    """(date, value) pairs from anywhere"""
    return Series(name, list(rows), unit, **kwargs)


def from_fred_csv(path, unit, name=None, value_column=None, **kwargs) -> Series:
    """a FRED series export: observation_date plus one value column

    a blank cell is a genuine gap in a FRED series and is dropped rather than
    read as a zero"""
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if len(columns) < 2 or columns[0] != "observation_date":
            raise MalformedSeries(
                f"{path}: expected a FRED export with an observation_date column, "
                f"got columns {columns}")
        column = value_column or columns[1]
        if column not in columns:
            raise MalformedSeries(f"{path}: no column {column!r}, has {columns}")

        series_name = name or column
        rows = []
        for line in reader:
            raw = (line.get(column) or "").strip()
            if raw in ("", ".", "NA", "N/A"):
                continue
            rows.append((line["observation_date"], raw))

    if not rows:
        raise MalformedSeries(f"{path}: column {column!r} has no usable observations")
    return Series(series_name, rows, unit, source=str(path), **kwargs)


def from_manual_record(path, name=None, **kwargs) -> Series:
    """a hand-recorded json source from data/raw/v5

    a record whose observations are declared missing raises here rather than
    returning an empty series: provenance without data must not read as data"""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        record = json.load(f)

    if record.get("observations_status") == "missing":
        raise MalformedSeries(
            f"{path}: this record declares observations_status=missing. it carries "
            f"provenance only, and no value may be resolved from it. "
            f"{record.get('observations_status_note', '')}")

    observations = record.get("observations") or []
    dated = [o for o in observations if o.get("date")]
    if not dated:
        raise MalformedSeries(
            f"{path}: no dated observations. catalog records are a price level, "
            f"not a time series, and have no as-of lookup")

    rows = [(o["date"], o["value"], o.get("published_as", "")) for o in dated]
    return Series(name or path.stem, rows,
                  record.get("unit") or "",
                  frequency=record.get("observation_kind", ""),
                  source=str(path), **kwargs)
