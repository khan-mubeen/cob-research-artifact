"""date-matched currency conversion from the pinned ECB reference rates

v4 converted one commodity price with a single hardcoded FX_USD_TO_EUR = 0.92 and
stamped that constant onto every material profile row. this layer is what
replaces it, and the replacement has to hold one line: there is no fallback rate,
anywhere, under any condition. a missing pair or a missing date raises

what the ECB publishes is one row per TARGET working day giving units of each
currency per 1 EUR. that makes EUR the base of every quoted pair and is why a
non-EUR pair is resolved through EUR:

    1 EUR = r_quote(date) quote
    1 EUR = r_base(date)  base
    so    1 base = (r_quote / r_base) quote

for ZAR to USD that is EUR_to_USD(date) / EUR_to_ZAR(date), which is the formula
the methodology states. the direction is verified against real ECB rows in the
tests, not asserted here

rates are resolved as of a date through the same carry-forward lookup every other
dated source uses, so a weekend or a TARGET holiday takes the last published
working day and never a later one. no interpolation

this module opens no sockets, reads no clock, and imports neither config nor the
dataset context. it is handed a file path and a date
"""

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.dataset.history import (
    HistoryError, MalformedSeries, Series, parse_date,
)

# the ECB quotes every rate as units of currency per one euro, so EUR is the
# base of the published file and its own rate is exactly 1
ECB_BASE = "EUR"

# values the ECB writes for a currency it did not publish that day
_ECB_BLANK = ("", "N/A", "NA", ".", "-")


class FxError(ValueError):
    """anything that stops a conversion being resolved from pinned data"""


class UnknownCurrency(FxError):
    """a currency the pinned file does not carry"""


class MissingRate(FxError):
    """a date the pinned file cannot answer for"""


@dataclass(frozen=True)
class Rate:
    """one resolved rate and its provenance, together

    a bare float is what made v4's 0.92 impossible to trace. every conversion
    here answers with the rate, the date the rate actually came from, the file it
    came from and that file's hash"""

    base: str
    quote: str
    rate: float
    requested_date: date
    rate_date: date
    source: str
    source_sha256: str
    via: str = ""
    days_carried: int = 0

    @property
    def is_exact(self) -> bool:
        return self.days_carried == 0

    def convert(self, amount: float) -> float:
        """amount in base -> amount in quote, unrounded"""
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise FxError(f"amount must be a number, got {amount!r}")
        return float(amount) * self.rate

    def as_dict(self):
        return {
            "base": self.base,
            "quote": self.quote,
            "rate": self.rate,
            "requested_date": self.requested_date.isoformat(),
            "rate_date": self.rate_date.isoformat(),
            "days_carried": self.days_carried,
            "via": self.via,
            "source": self.source,
            "source_sha256": self.source_sha256,
        }


class EcbReferenceRates:
    """the pinned ECB historical reference rates, as a currency table

    load it once and query it many times. loading parses the whole file, so a
    truncated or reordered download fails at load rather than at the first
    awkward date. the stored file happens to list newest first; Series sorts, so
    the ordering of the file is not part of the contract"""

    def __init__(self, path, max_carry_forward_days=None):
        path = Path(path)
        if not path.is_file():
            raise FxError(f"no ECB reference-rate file at {path}")

        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise MalformedSeries(f"{path}: the file is empty")
            if not header or header[0].strip() != "Date":
                raise MalformedSeries(
                    f"{path}: expected an ECB eurofxref file whose first column is "
                    f"Date, got {header[:3]}")

            currencies = [c.strip() for c in header[1:]]
            columns = {c: [] for c in currencies if c}
            for row in reader:
                if not row or not row[0].strip():
                    continue
                when = row[0].strip()
                for position, currency in enumerate(currencies, start=1):
                    if not currency or position >= len(row):
                        continue
                    raw = row[position].strip()
                    if raw in _ECB_BLANK:
                        continue
                    columns[currency].append((when, raw))

        self.path = path
        self.source = str(path)
        self.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self._series = {}
        for currency, rows in columns.items():
            if not rows:
                continue
            self._series[currency] = Series(
                f"ECB EUR/{currency}", rows, f"{currency} per 1 EUR",
                frequency="TARGET working days", source=self.source,
                max_carry_forward_days=max_carry_forward_days)

        if ECB_BASE in self._series:
            raise MalformedSeries(
                f"{path}: the file carries a EUR column, which the ECB reference "
                f"rates do not; this is not the file this reader expects")
        if not self._series:
            raise MalformedSeries(f"{path}: no currency column had any observation")

    # ── what the file covers ────────────────────────────────────────────────

    @property
    def currencies(self):
        """every currency this file can convert, EUR included as the base"""
        return tuple(sorted(self._series)) + (ECB_BASE,)

    def covers_currency(self, currency) -> bool:
        return currency == ECB_BASE or currency in self._series

    def __repr__(self):
        return (f"EcbReferenceRates({self.path.name}, "
                f"{len(self._series)} currencies plus {ECB_BASE})")

    # ── the one lookup everything else is built on ──────────────────────────

    def _eur_leg(self, currency, when):
        """units of currency per 1 EUR, as of a date"""
        if currency not in self._series:
            raise UnknownCurrency(
                f"{currency!r} is not in {self.path.name}; it carries "
                f"{len(self._series)} currencies plus {ECB_BASE}")
        try:
            return self._series[currency].as_of(when)
        except HistoryError as e:
            raise MissingRate(str(e))

    def rate(self, base, quote, when) -> Rate:
        """one unit of base, expressed in quote, as of a date

        both legs are resolved as of the same date, so a cross rate can never mix
        a friday EUR/USD with a thursday EUR/ZAR"""
        base = _currency_code(base, "base")
        quote = _currency_code(quote, "quote")
        target = _require_date(when)

        if base == quote:
            # an identity is answered as an identity, not by dividing a rate by
            # itself: a USD amount in a USD dataset must not depend on whether the
            # ECB happened to publish that day
            return Rate(base=base, quote=quote, rate=1.0, requested_date=target,
                        rate_date=target, source=self.source,
                        source_sha256=self.source_sha256, via="identity",
                        days_carried=0)

        if base == ECB_BASE:
            leg = self._eur_leg(quote, target)
            return Rate(base=base, quote=quote, rate=leg.value,
                        requested_date=target, rate_date=leg.observation_date,
                        source=self.source, source_sha256=self.source_sha256,
                        via="ecb reference rate", days_carried=leg.days_carried)

        if quote == ECB_BASE:
            leg = self._eur_leg(base, target)
            if leg.value == 0:
                raise MissingRate(
                    f"EUR/{base} is zero on {leg.observation_date}, cannot invert")
            return Rate(base=base, quote=quote, rate=1.0 / leg.value,
                        requested_date=target, rate_date=leg.observation_date,
                        source=self.source, source_sha256=self.source_sha256,
                        via=f"inverted EUR/{base}", days_carried=leg.days_carried)

        # neither side is EUR: cross through the published EUR legs
        base_leg = self._eur_leg(base, target)
        quote_leg = self._eur_leg(quote, target)
        if base_leg.value == 0:
            raise MissingRate(
                f"EUR/{base} is zero on {base_leg.observation_date}, cannot cross")
        return Rate(
            base=base, quote=quote,
            rate=quote_leg.value / base_leg.value,
            requested_date=target,
            rate_date=max(base_leg.observation_date, quote_leg.observation_date),
            source=self.source, source_sha256=self.source_sha256,
            via=f"EUR/{quote} divided by EUR/{base}",
            days_carried=max(base_leg.days_carried, quote_leg.days_carried))

    def convert(self, amount, base, quote, when):
        """(converted amount, the Rate that did it)

        the rate comes back with the amount on purpose. a converted number that
        has been separated from its rate date and source hash cannot be checked
        later, which is the position v4's 0.92 left every material price in"""
        resolved = self.rate(base, quote, when)
        return resolved.convert(amount), resolved

    # ── the two conversions v5 actually needs ───────────────────────────────

    def to_usd(self, amount, currency, when):
        """any source-currency amount as USD, as of a date

        a USD amount passes through untouched rather than being multiplied by a
        USD/USD rate read out of the file"""
        return self.convert(amount, currency, "USD", when)

    def zar_to_usd(self, when) -> Rate:
        """the cross the iMaterial linen pricelist needs

        EUR_to_USD(date) / EUR_to_ZAR(date), which is what rate() computes for a
        pair with no EUR side"""
        return self.rate("ZAR", "USD", when)


def _currency_code(value, label):
    if value is None:
        raise FxError(f"{label} currency is missing")
    if not isinstance(value, str):
        raise FxError(f"{label} currency must be a code, got {value!r}")
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise FxError(f"{label} currency {value!r} is not a three-letter code")
    return code


def _require_date(when):
    if when is None:
        raise MissingRate(
            "a conversion date is required; there is no current rate and no "
            "fallback rate in this layer")
    return parse_date(when, "fx date")


def load_ecb_rates(path, max_carry_forward_days=None) -> EcbReferenceRates:
    """the pinned ECB file at an explicit path, with no network access"""
    return EcbReferenceRates(path, max_carry_forward_days=max_carry_forward_days)
