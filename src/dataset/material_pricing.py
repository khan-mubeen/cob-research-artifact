"""v5 material prices in USD per kg for a material and a date, from the sources
already pinned in data/raw/v5/material

  PYTHONPATH=. python src/dataset/material_pricing.py

the method is the one docs/data_synthesis_methodology.md section 5.1 documents.
a finished-material reference sets the price level at an anchor date, and a
historical series moves it:

    price_usd_per_kg(t) = anchor_usd_per_kg x series_usd(t) / series_usd(anchor)

cotton, denim and leather move with a monthly FRED series. linen moves with the
four irregular Alliance flax observations, carried forward and never
interpolated, each converted to USD at its own observation date.

polyester has no temporal series: its SunSirs observations are still missing, and
that status is unchanged. section 5.1 covers exactly this case, "if no credible
historical series is available for a material, the sourced price remains fixed",
so polyester is priced at its finished-fabric anchor, the Tabb Textile TWPWD twill,
on every date. every polyester result says so in price_basis and
temporal_series_status, so a fixed price can never be mistaken for a dated one.
asking for polyester's series itself raises NoTemporalSeries

three choices the methodology leaves open are made here explicitly and recorded
on every result, so they can be reviewed rather than discovered later:
  the anchor date of each reference, ANCHOR_DATES
  the anchor level when a reference lists several products: the arithmetic mean
    of their individual USD/kg values, every product's value kept beside it
  the leather weight when the product is sold as a 2-3 oz range: the midpoint

pure and offline. it imports neither config nor the dataset context, never reads
the clock, and never opens a socket: it is handed the raw v5 directory
"""

import hashlib
import json
import re
import sys
from pathlib import Path

from src.dataset.fx import load_ecb_rates
from src.dataset.history import from_fred_csv, from_manual_record, parse_date

# the physical constants the methodology prints, used exactly as printed
KG_PER_LB_FACTOR = 2.20462        # USD/kg = USD/lb x 2.20462
KG_PER_OZ = 28.3495 / 1000        # kg = oz x 28.3495 / 1000
LEATHER_OZ_PER_SQFT = (2.0, 3.0)  # "2-3 oz", as the product is sold

MATERIALS = ("cotton", "polyester", "linen", "denim", "leather")

DATED_BASIS = "anchor_times_dated_series"
FIXED_BASIS = "fixed_anchor_no_temporal_series"

ANCHOR_DATES = {
    "cotton": ("2024-11-01",
               "Tabb Textile 2024 catalog. it prints no price date; its url is filed under "
               "uploads/2024/11, the earliest month the catalog is evidenced as published"),
    "denim": ("2024-11-01",
              "Tabb Textile 2024 catalog, same basis as cotton"),
    "polyester": ("2024-11-01",
                  "Tabb Textile 2024 catalog, same basis as cotton. with no temporal series the "
                  "anchor date does not change the price"),
    "linen": ("2024-01-01",
              "iMaterial 'January 2024 Fabric Pricelist', the first day of the edition month"),
    "leather": ("2026-09-11",
                "the Leather Unlimited product page carries no price date; the access date "
                "recorded on the manual record is used"),
}

SERIES = {
    "cotton": {"name": "FRED PCOTTINDUSDM", "file": "material/cotton/PCOTTINDUSDM.csv",
               "unit": "U.S. Cents per Pound",
               "to_usd": "(US cents per lb / 100) x 2.20462 = USD per kg"},
    "denim": {"name": "FRED WPU03420104", "file": "material/denim/WPU03420104.csv",
              "unit": "Index Dec 1994=100",
              "to_usd": "a unitless index, used as its level; only its ratio enters the price"},
    "leather": {"name": "FRED PHIDEUSDM", "file": "material/leather/PHIDEUSDM.csv",
                "unit": "U.S. Cents per Pound",
                "to_usd": "(US cents per lb / 100) x 2.20462 = USD per kg"},
    "linen": {"name": "Alliance flax observations", "file": "material/linen/alliance_flax.json",
              "unit": "EUR per kg",
              "to_usd": "EUR per kg x EUR_to_USD(observation date) = USD per kg"},
    "polyester": {"name": "SunSirs Polyester DTY", "file": "material/polyester/sunsirs_dty.json",
                  "unit": None, "to_usd": None},
}

REFERENCES = {
    "cotton": "material/cotton/tabb_jersey_knit.json",
    "denim": "material/denim/tabb_denim.json",
    "polyester": "material/polyester/tabb_polyester_twill.json",
    "linen": "material/linen/imaterial_linen.json",
    "leather": "material/leather/leather_unlimited.json",
}

FX_FILE = "fx/ecb_eurofxref_hist.csv"


class MaterialPricingError(ValueError):
    """a price that cannot be produced from the pinned sources"""


class UnknownMaterial(MaterialPricingError):
    """a material the v5 methodology does not price"""


class NoTemporalSeries(MaterialPricingError):
    """a series was asked for a material that has none; its price is fixed instead"""


def _f(value, places=6):
    return f"{value:.{places}f}"


class MaterialSources:
    """the pinned material and fx files, loaded once from an explicit directory"""

    def __init__(self, raw_v5_dir):
        self.root = Path(raw_v5_dir)
        self.references = {m: self._json(p) for m, p in REFERENCES.items()}
        self.polyester_series_record = self._json(SERIES["polyester"]["file"])
        self.series = {
            "cotton": from_fred_csv(self.root / SERIES["cotton"]["file"], SERIES["cotton"]["unit"]),
            "denim": from_fred_csv(self.root / SERIES["denim"]["file"], SERIES["denim"]["unit"]),
            "leather": from_fred_csv(self.root / SERIES["leather"]["file"], SERIES["leather"]["unit"]),
            "linen": from_manual_record(self.root / SERIES["linen"]["file"]),
        }
        if self.polyester_series_record.get("observations_status") != "missing":
            raise MaterialPricingError(
                "sunsirs_dty.json no longer declares its observations missing; polyester's fixed "
                "price must be revisited before it is used")
        self.fx = load_ecb_rates(self.root / FX_FILE)
        used = sorted({*REFERENCES.values(), *(s["file"] for s in SERIES.values()), FX_FILE})
        self.source_sha256 = {
            f"data/raw/v5/{p}": hashlib.sha256((self.root / p).read_bytes()).hexdigest()
            for p in used}
        self._anchors = {}

    def _json(self, relative):
        with open(self.root / relative, encoding="utf-8") as f:
            return json.load(f)

    def has_temporal_series(self, material):
        self._require_known(material)
        return material in self.series

    # ── the price level ─────────────────────────────────────────────────────

    def product_anchors(self, material):
        """every reference product converted to USD/kg with its arithmetic"""
        self._require_known(material)
        anchor_date = ANCHOR_DATES[material][0]
        rec = self.references[material]
        products = []

        if material == "cotton":
            for o in rec["observations"]:
                usd = o["value"] * KG_PER_LB_FACTOR
                products.append({"sku": o["sku"], "source_price": o["value"], "source_unit": o["unit"],
                                 "usd_per_kg": usd,
                                 "arithmetic": f"{o['value']:.2f} USD/lb x 2.20462 = {_f(usd)} USD/kg"})

        elif material in ("denim", "polyester"):
            for o in rec["observations"]:
                oz = float(re.search(r"([\d.]+)oz", o["description"]).group(1))
                width = o["width_inches"]
                yard_oz = oz * width / 36
                yard_kg = yard_oz * KG_PER_OZ
                usd = o["value"] / yard_kg
                products.append({
                    "sku": o["sku"], "source_price": o["value"], "source_unit": o["unit"],
                    "oz_per_sq_yd": oz, "width_inches": width, "usd_per_kg": usd,
                    "arithmetic": (f"{oz} oz/yd2 x {width} in / 36 = {_f(yard_oz)} oz per linear "
                                   f"yard; x 28.3495 / 1000 = {_f(yard_kg)} kg; "
                                   f"{o['value']:.2f} USD / {_f(yard_kg)} kg = {_f(usd)} USD/kg")})

        elif material == "linen":
            for o in (x for x in rec["observations"] if x["use"] == "linen"):
                kg_per_metre = (o["width_cm"] / 100) * o["gsm"] / 1000
                zar_per_kg = o["value"] / kg_per_metre
                usd, rate = self.fx.to_usd(zar_per_kg, "ZAR", anchor_date)
                products.append({
                    "sku": o["sku"], "source_price": o["value"], "source_unit": o["unit"],
                    "gsm": o["gsm"], "width_cm": o["width_cm"], "usd_per_kg": usd,
                    "fx": rate.as_dict(),
                    "arithmetic": (f"{o['width_cm'] / 100} m x {o['gsm']} / 1000 = "
                                   f"{_f(kg_per_metre)} kg per metre; {o['value']:.2f} ZAR / "
                                   f"{_f(kg_per_metre)} kg = {_f(zar_per_kg)} ZAR/kg; x ZAR_to_USD "
                                   f"{_f(rate.rate, 8)} ({rate.via}, rate date "
                                   f"{rate.rate_date.isoformat()}) = {_f(usd)} USD/kg")})

        elif material == "leather":
            for o in rec["observations"]:
                low, high = LEATHER_OZ_PER_SQFT
                mid = (low + high) / 2
                total_kg = o["area_sqft"] * mid * KG_PER_OZ
                usd = o["value"] / total_kg
                bounds = {f"{w} oz/sqft": o["value"] / (o["area_sqft"] * w * KG_PER_OZ)
                          for w in (low, high)}
                products.append({
                    "product": o["product"], "source_price": o["value"], "source_unit": o["unit"],
                    "area_sqft": o["area_sqft"], "oz_per_sqft_used": mid,
                    "oz_per_sqft_as_sold": o["weight_oz_per_sqft_range"], "usd_per_kg": usd,
                    "usd_per_kg_at_range_bounds": bounds,
                    "arithmetic": (f"{o['area_sqft']} sqft x {mid} oz/sqft = {_f(o['area_sqft'] * mid)} "
                                   f"oz; x 28.3495 / 1000 = {_f(total_kg)} kg; {o['value']:.2f} USD / "
                                   f"{_f(total_kg)} kg = {_f(usd)} USD/kg")})

        if not products:
            raise MaterialPricingError(f"{REFERENCES[material]} yielded no reference products")
        return products

    def anchor(self, material):
        if material not in self._anchors:
            products = self.product_anchors(material)
            date, basis = ANCHOR_DATES[material]
            self._anchors[material] = {
                "material": material,
                "anchor_date": date,
                "anchor_date_basis": basis,
                "usd_per_kg": sum(p["usd_per_kg"] for p in products) / len(products),
                "aggregation": "arithmetic mean of the per-product USD/kg values",
                "reference_source": self.references[material]["source"],
                "reference_record": f"data/raw/v5/{REFERENCES[material]}",
                "reference_currency": self.references[material]["currency"],
                "products": products,
            }
        return self._anchors[material]

    # ── the temporal movement ───────────────────────────────────────────────

    def series_usd(self, material, when):
        """the series value on a date, in USD where it is a price, with its lookup"""
        self._require_known(material)
        if not self.has_temporal_series(material):
            raise NoTemporalSeries(
                f"{material} has no temporal series: data/raw/v5/{SERIES[material]['file']} declares "
                f"observations_status={self.polyester_series_record.get('observations_status')!r}. "
                f"its price is the fixed finished-material anchor, see price()")
        lookup = self.series[material].as_of(when)
        entry = {"series": SERIES[material]["name"], "requested_date": lookup.requested_date.isoformat(),
                 "observation_date": lookup.observation_date.isoformat(),
                 "observation_value": lookup.value, "observation_unit": SERIES[material]["unit"],
                 "days_carried": lookup.days_carried,
                 "after_last_observation": lookup.after_last_observation}

        if material in ("cotton", "leather"):
            usd = lookup.value / 100 * KG_PER_LB_FACTOR
            entry["arithmetic"] = f"({lookup.value} / 100) x 2.20462 = {_f(usd)} USD/kg"
        elif material == "denim":
            usd = lookup.value
            entry["arithmetic"] = f"index level {lookup.value}, unitless"
        else:
            rate = self.fx.rate("EUR", "USD", lookup.observation_date)
            usd = lookup.value * rate.rate
            entry["fx"] = rate.as_dict()
            entry["arithmetic"] = (f"{lookup.value} EUR/kg x EUR_to_USD {rate.rate} (rate date "
                                   f"{rate.rate_date.isoformat()}) = {_f(usd)} USD/kg")
        entry["value_usd"] = usd
        return entry

    # ── the price ───────────────────────────────────────────────────────────

    def price(self, material, when):
        self._require_known(material)
        requested = parse_date(when, "material price date")
        anchor = self.anchor(material)

        if not self.has_temporal_series(material):
            return {
                "material": material,
                "requested_date": requested.isoformat(),
                "usd_per_kg": anchor["usd_per_kg"],
                "price_basis": FIXED_BASIS,
                "method": ("fixed finished-material anchor: no credible historical series is available, "
                           "so the sourced price stays fixed (methodology section 5.1)"),
                "arithmetic": f"{_f(anchor['usd_per_kg'])} USD/kg, the same on every date",
                "anchor_usd_per_kg": anchor["usd_per_kg"],
                "anchor_date": anchor["anchor_date"],
                "series_ratio": 1.0,
                "series_at_requested_date": None,
                "series_at_anchor_date": None,
                "carried_past_last_observation": False,
                "temporal_series_status": self.polyester_series_record.get("observations_status"),
                "temporal_series_record": f"data/raw/v5/{SERIES[material]['file']}",
            }

        at_date = self.series_usd(material, requested)
        at_anchor = self.series_usd(material, anchor["anchor_date"])
        ratio = at_date["value_usd"] / at_anchor["value_usd"]
        usd_per_kg = anchor["usd_per_kg"] * ratio
        return {
            "material": material,
            "requested_date": requested.isoformat(),
            "usd_per_kg": usd_per_kg,
            "price_basis": DATED_BASIS,
            "method": "anchor_usd_per_kg x series_usd(t) / series_usd(anchor)",
            "arithmetic": (f"{_f(anchor['usd_per_kg'])} x {_f(at_date['value_usd'])} / "
                           f"{_f(at_anchor['value_usd'])} = {_f(usd_per_kg)} USD/kg"),
            "anchor_usd_per_kg": anchor["usd_per_kg"],
            "anchor_date": anchor["anchor_date"],
            "series_ratio": ratio,
            "series_at_requested_date": at_date,
            "series_at_anchor_date": at_anchor,
            "carried_past_last_observation": at_date["after_last_observation"],
            "temporal_series_status": "available",
        }

    def _require_known(self, material):
        if material not in MATERIALS:
            raise UnknownMaterial(f"{material!r} is not one of {list(MATERIALS)}")


def load_material_sources(raw_v5_dir):
    return MaterialSources(raw_v5_dir)


# ── the verification artifact ────────────────────────────────────────────────

def sample_calculations(sources, window, spot_dates):
    anchors, reproduction, spot, movement = {}, {}, {}, {}
    for material in MATERIALS:
        anchors[material] = sources.anchor(material)
        at_anchor = sources.price(material, anchors[material]["anchor_date"])
        reproduction[material] = {
            "anchor_date": at_anchor["anchor_date"],
            "anchor_usd_per_kg": at_anchor["anchor_usd_per_kg"],
            "price_at_anchor_date": at_anchor["usd_per_kg"],
            "exactly_equal": at_anchor["usd_per_kg"] == at_anchor["anchor_usd_per_kg"],
            "price_basis": at_anchor["price_basis"],
        }
        results = [sources.price(material, d) for d in spot_dates]
        spot[material] = results
        if not sources.has_temporal_series(material):
            movement[material] = {
                "price_basis": FIXED_BASIS,
                "distinct_prices_across_window": len({r["usd_per_kg"] for r in results}),
                "series_observations_used": [],
                "all_carried_past_last_observation": None,
            }
            continue
        observations = sorted({r["series_at_requested_date"]["observation_date"] for r in results})
        movement[material] = {
            "price_basis": DATED_BASIS,
            "distinct_prices_across_window": len({r["usd_per_kg"] for r in results}),
            "series_observations_used": observations,
            "all_carried_past_last_observation": all(r["carried_past_last_observation"] for r in results),
        }

    return {
        "purpose": ("verification artifact for src/dataset/material_pricing.py: it proves the "
                    "module reproduces its anchors and resolves dated prices from the pinned "
                    "files. it is not a replacement for data/synthetic/v4/material_agents.csv, "
                    "which this task does not touch"),
        "method_reference": "docs/data_synthesis_methodology.md section 5.1",
        "choices_made_here": {
            "anchor_dates": {m: {"date": d, "basis": b} for m, (d, b) in ANCHOR_DATES.items()},
            "multi_product_anchor": "arithmetic mean of per-product USD/kg",
            "leather_oz_per_sqft": "midpoint of the 2-3 oz range the product is sold as",
            "materials_without_a_temporal_series": {
                "polyester": ("fixed at its Tabb Textile finished-fabric anchor, because its SunSirs "
                              "observations are missing (methodology section 5.1)")},
        },
        "source_sha256": sources.source_sha256,
        "order_window": window,
        "spot_check_dates": spot_dates,
        "anchors": anchors,
        "anchor_reproduction": reproduction,
        "spot_checks": spot,
        "window_movement": movement,
    }


def main():
    import pandas as pd

    from config import PROJECT_ROOT
    from src.dataset.acquire_v5_sources import write_json

    v4 = PROJECT_ROOT / "data" / "synthetic" / "v4"
    with open(v4 / "orders_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    created = pd.read_csv(v4 / "orders.csv", usecols=["order_created"])["order_created"]
    window = {"base_date": metadata["base_date"],
              "creation_window_days": metadata["creation_window_days"],
              "first_order_created": str(created.min()), "last_order_created": str(created.max()),
              "read_from": ["data/synthetic/v4/orders_metadata.json", "data/synthetic/v4/orders.csv"]}
    spot_dates = [window["first_order_created"][:10], "2026-08-15", "2026-09-11", "2026-10-15",
                  "2026-11-15", "2026-12-15", window["last_order_created"][:10]]

    sources = MaterialSources(PROJECT_ROOT / "data" / "raw" / "v5")
    document = sample_calculations(sources, window, spot_dates)
    out = PROJECT_ROOT / "data" / "processed" / "v5" / "material" / "sample_price_calculations.json"
    write_json(out, document)

    for material in MATERIALS:
        rep = document["anchor_reproduction"][material]
        prices = [f"{r['usd_per_kg']:.4f}" for r in document["spot_checks"][material]]
        print(f"  {material:9s} anchor {rep['anchor_usd_per_kg']:.4f} USD/kg on {rep['anchor_date']}, "
              f"{rep['price_basis']}, reproduced exactly: {rep['exactly_equal']}; window prices {prices}")
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
