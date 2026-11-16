"""the v5 material profile and the dated reference prices it is priced from

  PYTHONPATH=. python -m src.data_synthesis.v5.materials

a v5 material price can move with the order date, so it cannot sit in one static
column the way v4's did. two files carry it instead:

  material_reference_prices.csv  one USD/kg reference price per material per window
                                 day, from src/dataset/material_pricing.py
  material_agents.csv            each agent's physical stock and service terms from
                                 v4, and its price position against that reference

an agent's price on a day is its {material}_price_ratio_to_reference multiplied by
the reference price for that material on that day. {material}_base_cost_usd_per_kg
is that same product on the profile's pricing_date, kept so the file satisfies the
v5 schema contract; it is not a second price

the ratios are v4's per-agent cost_ratio design values, copied from
src/data_synthesis/synthesize_material_providers.py. they are v4's synthetic
differentiation between suppliers, carried over unchanged, not new evidence

all five v4 materials and all fifteen v4 agents are present. polyester's reference
price is fixed: the price_basis column says so on every one of its rows, and its
series columns stay empty, so it cannot be read as a dated price
"""

import sys

import pandas as pd

from src.dataset.material_pricing import FIXED_BASIS, MaterialSources
from src.data_synthesis.v5.settings import (
    ALL_MATERIALS, FIXED_PRICE_MATERIALS, RAW_V5, V4_DIR, V5_DIR, WINDOW_DAYS, WINDOW_START, rel,
    write_csv,
)

PROFILE_PATH = V5_DIR / "material_agents.csv"
REFERENCE_PATH = V5_DIR / "material_reference_prices.csv"

# copied verbatim from the cost_ratio values in AGENT_SPECS in
# src/data_synthesis/synthesize_material_providers.py; tests check the copy still matches
COST_RATIOS_V4 = {
    "M01": {"cotton": 1.04, "polyester": 1.00},
    "M02": {"cotton": 1.15},
    "M03": {"cotton": 0.88, "denim": 0.82},
    "M04": {"polyester": 0.85},
    "M05": {"polyester": 1.18},
    "M06": {"cotton": 1.0, "polyester": 0.99},
    "M07": {"polyester": 0.88, "denim": 0.90, "leather": 0.93},
    "M08": {"linen": 0.90, "leather": 1.00},
    "M09": {"linen": 1.12, "leather": 1.10},
    "M10": {"cotton": 1.02, "linen": 0.88},
    "M11": {"polyester": 1.08, "linen": 1.08},
    "M12": {"linen": 0.98, "denim": 0.97},
    "M13": {"denim": 1.12, "leather": 1.05},
    "M14": {"leather": 0.95, "denim": 0.94},
    "M15": {"leather": 0.88},
}

BASE_COST_DECIMALS = 4


def sources():
    return MaterialSources(RAW_V5)


def reference_prices(material_sources):
    rows = []
    for day in WINDOW_DAYS:
        for material in ALL_MATERIALS:
            price = material_sources.price(material, day)
            series = price["series_at_requested_date"]
            rows.append({
                "date": day.isoformat(),
                "material": material,
                "usd_per_kg": price["usd_per_kg"],
                "price_basis": price["price_basis"],
                "anchor_usd_per_kg": price["anchor_usd_per_kg"],
                "anchor_date": price["anchor_date"],
                "series": series["series"] if series else None,
                "series_observation_date": series["observation_date"] if series else None,
                "series_days_carried": series["days_carried"] if series else None,
                "series_ratio": price["series_ratio"] if series else None,
            })
    return pd.DataFrame(rows)


def fixed_price_evidence(material_sources, reference):
    """prove each fixed price really is the anchor, unchanged on every day, with no fx step"""
    out = {}
    for material in FIXED_PRICE_MATERIALS:
        rows = reference[reference["material"] == material]
        values = rows["usd_per_kg"].unique()
        anchor = material_sources.anchor(material)
        if len(values) != 1 or set(rows["price_basis"]) != {FIXED_BASIS} or values[0] != anchor["usd_per_kg"]:
            raise RuntimeError(f"{material} is declared fixed but its reference prices are not its anchor")
        if anchor["reference_currency"] != "USD":
            raise RuntimeError(f"{material}'s anchor is not USD-native; a fixed price would need an fx date")
        out[material] = {
            "usd_per_kg": float(values[0]),
            "window_days": int(len(rows)),
            "price_basis": FIXED_BASIS,
            "reference_record": anchor["reference_record"],
            "reference_currency": anchor["reference_currency"],
            "fx_applied": False,
            "product_arithmetic": [p["arithmetic"] for p in anchor["products"]],
            "temporal_series_status": material_sources.price(material, WINDOW_START)["temporal_series_status"],
            "basis": FIXED_PRICE_MATERIALS[material],
        }
    return out


def profile(material_sources):
    v4 = pd.read_csv(V4_DIR / "material_agents.csv").to_dict("records")
    at_pricing_date = {m: material_sources.price(m, WINDOW_START)["usd_per_kg"] for m in ALL_MATERIALS}
    rows = []
    for agent in v4:
        supplied = str(agent["material_types"]).split("|")
        row = {k: agent[k] for k in ("agent_id", "stage", "location", "specialty")}
        row["material_types"] = "|".join(sorted(supplied))
        for k in ("default_completion_days", "lead_time_variance", "min_profit_margin", "initial_markup",
                  "defect_probability", "reputation_score"):
            row[k] = agent[k]
        for material in ALL_MATERIALS:
            if material in supplied:
                ratio = COST_RATIOS_V4[agent["agent_id"]][material]
                row[f"{material}_available_kg"] = agent[f"{material}_available_kg"]
                row[f"{material}_price_ratio_to_reference"] = ratio
                row[f"{material}_base_cost_usd_per_kg"] = round(ratio * at_pricing_date[material],
                                                                BASE_COST_DECIMALS)
                row[f"{material}_restock_amount_kg"] = agent[f"{material}_restock_amount_kg"]
                row[f"{material}_restock_time_days"] = agent[f"{material}_restock_time_days"]
            else:
                for suffix in ("available_kg", "price_ratio_to_reference", "base_cost_usd_per_kg",
                               "restock_amount_kg", "restock_time_days"):
                    row[f"{material}_{suffix}"] = None
        row.update({"dataset_version": "v5", "currency": "USD",
                    "pricing_date": WINDOW_START.isoformat(), "unit": "per_kg"})
        rows.append(row)
    return pd.DataFrame(rows), at_pricing_date


def build():
    material_sources = sources()
    reference = reference_prices(material_sources)
    frame, at_pricing_date = profile(material_sources)
    write_csv(reference, REFERENCE_PATH)
    write_csv(frame, PROFILE_PATH)
    return {"profile": frame, "reference": reference,
            "reference_at_pricing_date": at_pricing_date,
            "fixed_prices": fixed_price_evidence(material_sources, reference),
            "source_sha256": material_sources.source_sha256}


def main():
    result = build()
    reference = result["reference"]
    for material in ALL_MATERIALS:
        rows = reference[reference["material"] == material]
        print(f"  {material:9s} {rows['usd_per_kg'].nunique():3d} distinct USD/kg across {len(rows)} days, "
              f"{rows['usd_per_kg'].min():.4f} to {rows['usd_per_kg'].max():.4f}, {rows['price_basis'].iloc[0]}")
    print(f"  fixed: {result['fixed_prices']}")
    print(f"wrote {rel(PROFILE_PATH)} ({len(result['profile'])} agents) and {rel(REFERENCE_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
