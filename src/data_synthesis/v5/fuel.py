"""the pinned carrier fuel-surcharge series, as dated series

every series here is read from a manual record under
data/raw/v5/logistics/fuel_surcharge/ and resolved with src/dataset/history.py,
so a percentage is always the one published on or before the order's own date,
carried forward, never interpolated and never taken from a later date

a series is usable for v5 only if it resolves on the first day of the v5 window.
three do not: Posti publishes only its two most recent values (August and
September 2026) and Matkahuolto's Rahti list starts on 1.1.2026, so none of them
can price an order created in October 2025. they stay listed with that reason
rather than disappearing

every v5 logistics agent uses the same one of these, settings.SHARED_FUEL_SERIES_ID.
the rest stay declared so that choice can be checked against them and the diagnostics
can replay the same orders on another shared series
"""

import json
from pathlib import Path

from src.dataset.history import HistoryError, Series
from src.data_synthesis.v5.settings import WINDOW_DAYS

FUEL_SERIES = {
    "postnord_parcels": {
        "record": "fuel_surcharge/postnord_fuel_surcharge.json",
        "carrier": "PostNord Finland", "series_as_published": "Fuel surcharge parcels",
        "service_class": "parcel",
        "rows": lambda r: [(o["starting_from"], o["parcels_percent"]) for o in r["observations"]],
    },
    "pakettipiste_domestic": {
        "record": "fuel_surcharge/pakettipiste_fuel_surcharge.json",
        "carrier": "Pakettipiste", "series_as_published": "Domestic parcel and logistics services",
        "service_class": "parcel",
        "rows": lambda r: [(o["starting"], o["percent"]) for o in r["observations"]],
    },
    "matkahuolto_parcel_and_transport": {
        "record": "fuel_surcharge/matkahuolto_fuel_surcharge.json",
        "carrier": "Matkahuolto",
        "series_as_published": "Paketti- ja kuljetuspalvelut kotimaassa ja ulkomaille",
        "service_class": "parcel",
        "rows": lambda r: [(o["date"], o["percent"]) for o in
                           r["series_parsed"]["Paketti- ja kuljetuspalvelut kotimaassa ja ulkomaille"]],
    },
    "postnord_pallets_and_groupage": {
        "record": "fuel_surcharge/postnord_fuel_surcharge.json",
        "carrier": "PostNord Finland", "series_as_published": "Fuel surcharge pallets and groupage",
        "service_class": "freight",
        "rows": lambda r: [(o["starting_from"], o["pallets_and_groupage_percent"])
                           for o in r["observations"]],
    },
    "matkahuolto_lavarahti": {
        "record": "fuel_surcharge/matkahuolto_fuel_surcharge.json",
        "carrier": "Matkahuolto", "series_as_published": "Lavarahti",
        "service_class": "freight",
        "rows": lambda r: [(o["date"], o["percent"]) for o in r["series_parsed"]["Lavarahti"]],
    },
    "posti_parcel": {
        "record": "fuel_surcharge/posti_fuel_surcharge.json",
        "carrier": "Posti", "series_as_published": "Parcel",
        "service_class": "parcel",
        "rows": lambda r: [(o["from"], o["percent"]) for o in r["observations"]
                           if o["series"] == "Parcel"],
    },
    "posti_freight": {
        "record": "fuel_surcharge/posti_fuel_surcharge.json",
        "carrier": "Posti", "series_as_published": "Freight, Express freight and Transport units",
        "service_class": "freight",
        "rows": lambda r: [(o["from"], o["percent"]) for o in r["observations"]
                           if o["series"] == "Freight, Express freight and Transport units"],
    },
    "matkahuolto_rahti": {
        "record": "fuel_surcharge/matkahuolto_fuel_surcharge.json",
        "carrier": "Matkahuolto", "series_as_published": "Rahti",
        "service_class": "freight",
        "rows": lambda r: [(o["date"], o["percent"]) for o in r["series_parsed"]["Rahti"]],
    },
}


def load_series(raw_logistics_dir):
    """every declared series, usable or not, keyed by id"""
    records, out = {}, {}
    for series_id, spec in FUEL_SERIES.items():
        path = Path(raw_logistics_dir) / spec["record"]
        if path not in records:
            with open(path, encoding="utf-8") as f:
                records[path] = json.load(f)
        out[series_id] = Series(series_id, spec["rows"](records[path]), "percent",
                                source=f"data/raw/v5/logistics/{spec['record']}")
    return out


def window_coverage(series, days=WINDOW_DAYS):
    """whether a series can price every order day, and how much of it is real"""
    first, last = days[0], days[-1]
    in_window = [o for o in series.observations if first <= o.date <= last]
    try:
        at_start = series.as_of(first)
    except HistoryError as exc:
        return {"usable": False,
                "reason": f"no observation on or before the window start {first}: {exc}",
                "first_observation": series.first.date.isoformat(),
                "last_observation": series.last.date.isoformat(),
                "observations_in_window": len(in_window)}
    at_end = series.as_of(last)
    distinct = sorted({series.as_of(d).value for d in days})
    return {
        "usable": True,
        "first_observation": series.first.date.isoformat(),
        "last_observation": series.last.date.isoformat(),
        "observations_in_window": len(in_window),
        "at_window_start": {"percent": at_start.value,
                            "observation_date": at_start.observation_date.isoformat()},
        "at_last_window_day": {"percent": at_end.value,
                               "observation_date": at_end.observation_date.isoformat(),
                               "days_carried": at_end.days_carried,
                               "after_last_observation": at_end.after_last_observation},
        "distinct_percentages_across_window_days": len(distinct),
        "window_days_after_last_observation": sum(
            1 for d in days if series.as_of(d).after_last_observation),
    }
