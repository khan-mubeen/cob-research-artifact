"""the v5 logistics profile and the route cost every v5 logistics price comes from

  PYTHONPATH=. python -m src.data_synthesis.v5.logistics

the price of one shipment by one agent, in the order methodology section 7.5 sets:

  freight_eur  = F_i + alpha_i x weight_kg + beta_i x (weight_kg / 1000) x distance_km
  with_fuel    = freight_eur x (1 + fuel_surcharge(order date))
  subtotal     = with_fuel + Oversized fee, only for a parcel shipment above its base limit
  eur          = subtotal x (1 + damage_probability) x (1 + priority markup)
  usd          = eur x EUR_to_USD(order date)

distance runs from the candidate manufacturer's location to the order's deliver_to
city, from the pinned OSRM matrix. every agent reads the same shared fuel series, at
the order's own date, and the rate is read at the same date

F_i, alpha_i and beta_i come from real carrier market prices, so "no markup, no floor"
applies to exactly two fields: initial_markup is not applied on top of the market-level
price, and min_profit_margin is not used as a floor, because that floor needs a cost
basis no source provides. both stay on the profile for reference. damage-risk and
priority adjustments are applied. a proper cost-basis methodology is future work

the Oversized fee is an additional service rather than freight, so the fuel
surcharge is not applied to it. an agent is never priced outside the weight range its
reading is evidenced for: asking raises WeightOutsideDomain. the range is the optional
min_shipment_weight_kg (inclusive) up to max_shipment_weight_kg, inclusive unless
max_weight_inclusive says otherwise; a profile without those columns has no bound
"""

import math
import sys

import pandas as pd

from src.dataset.fx import load_ecb_rates
from src.dataset.history import parse_date
from src.dataset.schemas import Field, RecordSchema
from src.data_synthesis.v5 import fuel
from src.data_synthesis.v5.settings import (
    AGENT_CALIBRATION_PATH, DISTANCE_MATRIX_PATH, PRIORITY_MARKUP, RAW_V5, SHARED_FUEL_SERIES_ID,
    V4_DIR, V5_DIR, rel, write_csv,
)

PROFILE_PATH = V5_DIR / "logistics_agents.csv"

PRICE_EXCLUDES = ("initial_markup", "min_profit_margin")
PRICE_APPLIES = ("shared fuel surcharge by order date", "Oversized fee above a parcel base limit",
                 "damage_probability", "priority markup", "EUR_to_USD by order date")
PRICE_RULE = ("usd = ((F + alpha x kg + beta x kg / 1000 x km) x (1 + fuel(order date)) + oversized fee) "
              "x (1 + damage_probability) x (1 + priority markup) x EUR_to_USD(order date)")

LOGISTICS_SCHEMA_V5 = RecordSchema(
    name="logistics_agent", version="v5", currency="USD",
    fields=(
        Field("agent_id", "str"),
        Field("stage", "enum", choices=("logistics",)),
        Field("location", "str"),
        Field("service_regions", "str", note="pipe separated"),
        Field("fixed_charge_eur", "number", unit="per_shipment", currency="EUR", minimum=0.0),
        Field("alpha_eur_per_kg", "number", unit="per_kg", currency="EUR", minimum=0.0),
        Field("beta_eur_per_tonne_km", "number", unit="per_tonne_km", currency="EUR", minimum=0.0),
        Field("fuel_surcharge_series_id", "enum", choices=(SHARED_FUEL_SERIES_ID,)),
        Field("max_shipment_weight_kg", "number", unit="kg", nullable=True, minimum=0.0,
              note="empty means no cap"),
        Field("oversized_above_kg", "number", unit="kg", nullable=True, minimum=0.0),
        Field("oversized_fee_eur", "number", unit="per_shipment", currency="EUR", nullable=True, minimum=0.0),
        Field("calibration_corner", "enum", choices=("parcel", "freight")),
        Field("calibration_reading_id", "str"),
        Field("default_completion_same_city_days", "number", unit="days", minimum=0),
        Field("default_completion_intercity_days", "number", unit="days", minimum=0),
        Field("delay_probability", "number", unit="probability", minimum=0.0, maximum=1.0),
        Field("damage_probability", "number", unit="probability", minimum=0.0, maximum=1.0),
        Field("min_profit_margin", "number", unit="ratio", minimum=0.0),
        Field("initial_markup", "number", unit="ratio", minimum=0.0),
        Field("reputation_score", "number", unit="score", minimum=0.0, maximum=1.0),
        Field("dataset_version", "enum", choices=("v5",)),
        Field("coefficient_currency", "enum", choices=("EUR",)),
        Field("price_currency", "enum", choices=("USD",)),
        Field("price_excludes", "str"),
    ),
    allow_extra_columns=True,
    note=("v5 logistics profile for the parametric model. the foundation's placeholder v5 "
          "logistics schema in src/dataset/schemas.py predates the method decision and is "
          "superseded by this one"),
)


class WeightOutsideDomain(ValueError):
    """a shipment outside the weight range the agent's pricing evidence covers"""


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _optional(value):
    return None if _missing(value) else float(value)


def _inclusive(value):
    return True if _missing(value) else bool(value)


class RouteCost:
    """prices one shipment for one agent on one date, with every component traceable"""

    def __init__(self, profile_path=PROFILE_PATH, matrix_path=DISTANCE_MATRIX_PATH, raw_v5_dir=RAW_V5):
        self.agents = {}
        for r in pd.read_csv(profile_path).to_dict("records"):
            self.agents[r["agent_id"]] = {
                "fixed_charge_eur": float(r["fixed_charge_eur"]),
                "alpha_eur_per_kg": float(r["alpha_eur_per_kg"]),
                "beta_eur_per_tonne_km": float(r["beta_eur_per_tonne_km"]),
                "fuel_surcharge_series_id": r["fuel_surcharge_series_id"],
                "damage_probability": float(r["damage_probability"]),
                "min_shipment_weight_kg": _optional(r.get("min_shipment_weight_kg")),
                "max_shipment_weight_kg": _optional(r["max_shipment_weight_kg"]),
                "max_weight_inclusive": _inclusive(r.get("max_weight_inclusive")),
                "oversized_above_kg": _optional(r["oversized_above_kg"]),
                "oversized_fee_eur": _optional(r["oversized_fee_eur"]),
                "calibration_corner": r["calibration_corner"],
            }
        self.series = fuel.load_series(raw_v5_dir / "logistics")
        self.fx = load_ecb_rates(raw_v5_dir / "fx" / "ecb_eurofxref_hist.csv")
        matrix = pd.read_csv(matrix_path)
        self.distance_km = {(r.origin_manufacturer_location, r.destination_deliver_to): r.road_distance_km
                            for r in matrix.itertuples()}
        self._fuel, self._rate = {}, {}

    def fuel_lookup(self, series_id, day):
        key = (series_id, day)
        if key not in self._fuel:
            self._fuel[key] = self.series[series_id].as_of(day)
        return self._fuel[key]

    def eur_to_usd(self, day):
        if day not in self._rate:
            self._rate[day] = self.fx.rate("EUR", "USD", day)
        return self._rate[day]

    def eligible_weight(self, agent_id, weight_kg):
        a = self.agents[agent_id]
        low, cap = a["min_shipment_weight_kg"], a["max_shipment_weight_kg"]
        if low is not None and weight_kg < low:
            return False
        if cap is None:
            return True
        return weight_kg <= cap if a["max_weight_inclusive"] else weight_kg < cap

    def weight_domain(self, agent_id):
        """(min kg or None, max kg or None, whether max is inclusive)"""
        a = self.agents[agent_id]
        return a["min_shipment_weight_kg"], a["max_shipment_weight_kg"], a["max_weight_inclusive"]

    def _components(self, agent_id, weight_kg, distance_km, day, priority):
        a = self.agents[agent_id]
        if not self.eligible_weight(agent_id, weight_kg):
            low, cap, inclusive = self.weight_domain(agent_id)
            raise WeightOutsideDomain(
                f"{agent_id} is evidenced for {low if low is not None else 0:g} kg up to "
                f"{'no cap' if cap is None else f'{cap:g} kg'}{'' if inclusive else ' (exclusive)'}, "
                f"not {weight_kg:g} kg")
        freight = a["fixed_charge_eur"] + a["alpha_eur_per_kg"] * weight_kg \
            + a["beta_eur_per_tonne_km"] * (weight_kg / 1000.0) * distance_km
        lookup = self.fuel_lookup(a["fuel_surcharge_series_id"], day)
        with_fuel = freight * (1.0 + lookup.value / 100.0)
        oversized = (a["oversized_fee_eur"] if a["oversized_fee_eur"] is not None
                     and weight_kg > a["oversized_above_kg"] else 0.0)
        subtotal = with_fuel + oversized
        eur = subtotal * (1.0 + a["damage_probability"]) * (1.0 + PRIORITY_MARKUP[priority])
        return freight, lookup, with_fuel, oversized, subtotal, eur

    def price_for_distance(self, agent_id, weight_kg, distance_km, day, priority):
        """the price in USD for a known distance, the fast path"""
        eur = self._components(agent_id, weight_kg, distance_km, day, priority)[-1]
        return eur * self.eur_to_usd(day).rate

    def price(self, agent_id, weight_kg, origin, destination, when, priority):
        """every component of one route price, for audit and tests"""
        day = parse_date(when, "logistics pricing date")
        distance = self.distance_km[(origin, destination)]
        freight, lookup, with_fuel, oversized, subtotal, eur = self._components(
            agent_id, weight_kg, distance, day, priority)
        a = self.agents[agent_id]
        rate = self.eur_to_usd(day)
        return {
            "agent_id": agent_id, "date": day.isoformat(), "weight_kg": weight_kg, "priority": priority,
            "origin": origin, "destination": destination, "distance_km": distance,
            "fixed_charge_eur": a["fixed_charge_eur"], "alpha_eur_per_kg": a["alpha_eur_per_kg"],
            "beta_eur_per_tonne_km": a["beta_eur_per_tonne_km"],
            "freight_eur": freight,
            "fuel_series_id": a["fuel_surcharge_series_id"], "fuel_percent": lookup.value,
            "fuel_observation_date": lookup.observation_date.isoformat(),
            "fuel_days_carried": lookup.days_carried,
            "with_fuel_eur": with_fuel, "oversized_fee_eur": oversized, "subtotal_eur": subtotal,
            "damage_probability": a["damage_probability"], "priority_markup": PRIORITY_MARKUP[priority],
            "price_eur": eur, "eur_to_usd": rate.rate, "rate_date": rate.rate_date.isoformat(),
            "price_usd": eur * rate.rate, "rule": PRICE_RULE, "excludes": list(PRICE_EXCLUDES),
        }


def build(calibration_path=AGENT_CALIBRATION_PATH, profile_path=PROFILE_PATH):
    calibration = pd.read_csv(calibration_path)
    v4 = pd.read_csv(V4_DIR / "logistics_agents.csv")
    kept = v4.drop(columns=["base_delivery_cost", "cost_per_kg"])
    frame = kept.merge(calibration, on="agent_id", how="left", validate="one_to_one")
    if frame["fixed_charge_eur"].isna().any():
        raise ValueError("a v4 logistics agent has no v5 calibration row")
    frame["dataset_version"] = "v5"
    frame["coefficient_currency"] = "EUR"
    frame["price_currency"] = "USD"
    frame["price_excludes"] = "|".join(PRICE_EXCLUDES)
    lower = ["min_shipment_weight_kg"] if "min_shipment_weight_kg" in frame.columns else []
    inclusive = ["max_weight_inclusive"] if "max_weight_inclusive" in frame.columns else []
    order = ["agent_id", "stage", "location", "specialty", "service_regions", "calibration_corner",
             "calibration_reading_id", "fixed_charge_eur", "alpha_eur_per_kg", "beta_eur_per_tonne_km",
             "fuel_surcharge_series_id", *lower, "max_shipment_weight_kg", *inclusive, "oversized_above_kg",
             "oversized_fee_eur", "default_completion_same_city_days", "default_completion_intercity_days",
             "delay_probability", "damage_probability", "min_profit_margin", "initial_markup", "reputation_score",
             "dataset_version", "coefficient_currency", "price_currency", "price_excludes"]
    frame = frame[order]
    write_csv(frame, profile_path)
    problems = LOGISTICS_SCHEMA_V5.validate_rows(frame.to_dict("records"))
    if problems:
        raise ValueError(f"v5 logistics profile fails its schema: {problems[:3]}")
    return frame


def main():
    frame = build()
    print(frame[["agent_id", "calibration_corner", "fixed_charge_eur", "alpha_eur_per_kg",
                 "max_shipment_weight_kg", "oversized_fee_eur"]].to_string(index=False))
    route = RouteCost()
    for args in (("L08", 850.0, "Tampere", "Oulu", "2025-11-20", "urgent"),
                 ("L10", 30.0, "Helsinki", "Tampere", "2026-02-10", "normal")):
        example = route.price(*args)
        print({k: example[k] for k in ("agent_id", "weight_kg", "freight_eur", "fuel_percent",
                                       "oversized_fee_eur", "damage_probability", "priority_markup",
                                       "price_usd")})
    print(f"wrote {rel(PROFILE_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
