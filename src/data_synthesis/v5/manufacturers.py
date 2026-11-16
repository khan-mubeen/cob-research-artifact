"""part 2: the v5 manufacturer profile, v4's unit costs relabelled from EUR to USD

  PYTHONPATH=. python -m src.data_synthesis.v5.manufacturers

currency relabelling only. capacity, workload, defect, reputation, product range,
markup and location are v4's, unchanged, and no date-aware manufacturer pricing is
introduced: every unit cost is converted at one fixed ECB rate, read on
FX_REFERENCE_DATE. the v4 synthesis script is not imported or modified; the v4
profile is read, not rebuilt
"""

import sys

import pandas as pd

from src.dataset.fx import load_ecb_rates
from src.data_synthesis.v5.settings import (
    ALL_PRODUCTS, FX_REFERENCE_BASIS, FX_REFERENCE_DATE, PROCESSED_V5, RAW_V5, V4_DIR, V5_DIR, rel,
    sha256_file, write_csv, write_json,
)

OUTPUT_PATH = V5_DIR / "manufacturer_agents.csv"
CONVERSION_PATH = PROCESSED_V5 / "manufacturer" / "usd_conversion.json"

# copied verbatim from UNIT_COSTS in src/data_synthesis/synthesize_manufacturers.py,
# "mid-range unit costs per product in eur"; tests check the copy still matches
UNIT_COST_BANDS_EUR_V4 = {
    "shirt":  {"min": 3.5, "max": 6.0},
    "hoodie": {"min": 6.0, "max": 10.0},
    "jacket": {"min": 9.0, "max": 14.0},
    "jeans":  {"min": 5.0, "max": 9.0},
    "skirt":  {"min": 3.0, "max": 5.5},
}

# four decimals keeps the relabelled value within 0.00005 USD of the exact product
USD_DECIMALS = 4


def reference_rate():
    rates = load_ecb_rates(RAW_V5 / "fx" / "ecb_eurofxref_hist.csv")
    return rates.rate("EUR", "USD", FX_REFERENCE_DATE)


def build():
    rate = reference_rate()
    v4 = pd.read_csv(V4_DIR / "manufacturer_agents.csv")
    out = v4.drop(columns=[f"unit_cost_{p}" for p in ALL_PRODUCTS]).copy()
    per_agent = []
    for product in ALL_PRODUCTS:
        eur = v4[f"unit_cost_{product}"]
        out[f"unit_cost_usd_{product}"] = (eur * rate.rate).round(USD_DECIMALS)
    for _, row in v4.iterrows():
        per_agent.append({
            "agent_id": row["agent_id"],
            "unit_cost_eur_v4": {p: (None if pd.isna(row[f"unit_cost_{p}"]) else float(row[f"unit_cost_{p}"]))
                                 for p in ALL_PRODUCTS},
            "unit_cost_usd_v5": {p: (None if pd.isna(row[f"unit_cost_{p}"])
                                     else round(float(row[f"unit_cost_{p}"]) * rate.rate, USD_DECIMALS))
                                 for p in ALL_PRODUCTS},
        })

    out["fx_eur_to_usd_reference"] = rate.rate
    out["fx_reference_date"] = rate.rate_date.isoformat()
    out["dataset_version"] = "v5"
    out["currency"] = "USD"
    out["pricing_date"] = FX_REFERENCE_DATE.isoformat()
    out["unit"] = "per_unit"
    write_csv(out, OUTPUT_PATH)

    write_json(CONVERSION_PATH, {
        "purpose": ("currency relabelling of the v4 manufacturer unit costs for v5. the pricing "
                    "methodology is unchanged and nothing here is date-aware"),
        "rate": {**rate.as_dict(), "basis": FX_REFERENCE_BASIS},
        "arithmetic": (f"unit_cost_usd = unit_cost_eur x {rate.rate} (ECB EUR/USD on "
                       f"{rate.rate_date.isoformat()}), rounded to {USD_DECIMALS} decimals"),
        "bands": {p: {"eur_v4": band,
                      "usd_v5": {k: round(v * rate.rate, USD_DECIMALS) for k, v in band.items()}}
                  for p, band in UNIT_COST_BANDS_EUR_V4.items()},
        "bands_source": ("UNIT_COSTS in src/data_synthesis/synthesize_manufacturers.py, read, not "
                         "imported"),
        "agents": per_agent,
        "inputs": {rel(V4_DIR / "manufacturer_agents.csv"): sha256_file(V4_DIR / "manufacturer_agents.csv"),
                   "data/raw/v5/fx/ecb_eurofxref_hist.csv": rate.source_sha256},
        "output": rel(OUTPUT_PATH),
    })
    return out, rate


def main():
    out, rate = build()
    print(f"EUR/USD {rate.rate} on {rate.rate_date} ({rate.via}, carried {rate.days_carried} days)")
    print(out[["agent_id", *[f"unit_cost_usd_{p}" for p in ALL_PRODUCTS]]].to_string(index=False))
    print(f"wrote {rel(OUTPUT_PATH)} and {rel(CONVERSION_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
