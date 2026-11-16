from pathlib import Path

import pandas as pd

from config import SYNTHETIC_DATA_DIR, DATA_VERSION

ALL_MATERIALS = ["cotton", "polyester", "linen", "denim", "leather"]
ALL_PRODUCTS = ["shirt", "hoodie", "jacket", "jeans", "skirt"]
ALL_CITIES = ["Oulu", "Helsinki", "Tampere", "Turku", "Jyvaskyla", "Lahti", "Espoo"]


def load_all_data():
    # keep derived columns identical across text and chart diagnostics
    materials = pd.read_csv(SYNTHETIC_DATA_DIR / "material_agents.csv")
    manufacturers = pd.read_csv(SYNTHETIC_DATA_DIR / "manufacturer_agents.csv")
    logistics = pd.read_csv(SYNTHETIC_DATA_DIR / "logistics_agents.csv")
    products = pd.read_csv(SYNTHETIC_DATA_DIR / "products.csv")
    orders = pd.read_csv(SYNTHETIC_DATA_DIR / "orders.csv")

    manufacturers["workload_ratio"] = (
        manufacturers["current_workload_units"] / manufacturers["daily_capacity_units"]
    )
    manufacturers["avg_unit_cost"] = manufacturers[
        [f"unit_cost_{p}" for p in ALL_PRODUCTS]
    ].mean(axis=1, skipna=True)

    materials["avg_cost_eur_per_kg"] = materials[
        [f"{m}_base_cost_eur_per_kg" for m in ALL_MATERIALS]
    ].mean(axis=1, skipna=True)
    materials["avg_available_kg"] = materials[
        [f"{m}_available_kg" for m in ALL_MATERIALS]
    ].mean(axis=1, skipna=True)

    return materials, manufacturers, logistics, products, orders


def find_dominated_agents(df, minimize_cols, maximize_cols, label_col="agent_id"):
    # another agent dominates only if it is no worse on every axis and better on one
    dominated_pairs = []
    rows = df.to_dict("records")

    for a in rows:
        for b in rows:
            if a[label_col] == b[label_col]:
                continue

            better_or_equal = True
            strictly_better = False

            for col in minimize_cols:
                if pd.isna(a.get(col)) or pd.isna(b.get(col)):
                    better_or_equal = False
                    break
                if b[col] > a[col]:
                    better_or_equal = False
                    break
                if b[col] < a[col]:
                    strictly_better = True

            if not better_or_equal:
                continue

            for col in maximize_cols:
                if pd.isna(a.get(col)) or pd.isna(b.get(col)):
                    better_or_equal = False
                    break
                if b[col] < a[col]:
                    better_or_equal = False
                    break
                if b[col] > a[col]:
                    strictly_better = True

            if better_or_equal and strictly_better:
                dominated_pairs.append((a[label_col], b[label_col]))

    return dominated_pairs


def print_dominance_report(label, dominated_pairs):
    print(f"\n{label} dominance check:")
    if not dominated_pairs:
        print("  no dominated agents found, every agent wins in at least one scenario")
        return
    for loser, winner in dominated_pairs:
        print(f"  {loser} is dominated by {winner}, no reason to ever pick {loser} over {winner}")
