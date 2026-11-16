import pandas as pd

from config import DATA_VERSION
from src.visualization.data_utils import (
    load_all_data,
    find_dominated_agents,
    print_dominance_report,
    ALL_MATERIALS,
    ALL_PRODUCTS,
    ALL_CITIES,
)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def main():
    materials, manufacturers, logistics, products, orders = load_all_data()

    section(f"OVERVIEW - version {DATA_VERSION}")
    print(f"material agents:     {len(materials)}")
    print(f"manufacturer agents: {len(manufacturers)}")
    print(f"logistics agents:    {len(logistics)}")
    print(f"orders:              {len(orders)}")

    section("MATERIAL COVERAGE")
    for mat in ALL_MATERIALS:
        count = materials[f"{mat}_available_kg"].notna().sum()
        print(f"  {mat:10s}: {count} agents")

    section("PRODUCT COVERAGE")
    for prod in ALL_PRODUCTS:
        count = manufacturers[f"unit_cost_{prod}"].notna().sum()
        print(f"  {prod:8s}: {count} agents")

    section("CITY COVERAGE (logistics)")
    for city in ALL_CITIES:
        count = logistics["service_regions"].str.contains(city).sum()
        print(f"  {city:10s}: {count} agents")

    section("MATERIAL AGENT STATS")
    print(materials[[
        "avg_cost_eur_per_kg", "avg_available_kg", "defect_probability",
        "default_completion_days", "initial_markup", "lead_time_variance",
        "reputation_score",
    ]].describe().to_string())

    section("MANUFACTURER AGENT STATS")
    print(manufacturers[[
        "avg_unit_cost", "daily_capacity_units", "current_workload_units",
        "workload_ratio", "defect_probability", "initial_markup",
        "reputation_score",
    ]].describe().to_string())

    section("LOGISTICS AGENT STATS")
    print(logistics[[
        "cost_per_kg", "base_delivery_cost", "damage_probability",
        "delay_probability", "initial_markup",
        "reputation_score",
    ]].describe().to_string())

    section("ORDER STATS")
    print(orders[[
        "quantity", "budget", "max_defect_rate", "min_reputation_score",
    ]].describe().to_string())

    section("MATERIAL FEATURE CORRELATION")
    print(materials[[
        "avg_cost_eur_per_kg", "avg_available_kg", "defect_probability",
        "default_completion_days", "initial_markup", "lead_time_variance",
    ]].corr().to_string())

    section("MANUFACTURER FEATURE CORRELATION")
    print(manufacturers[[
        "avg_unit_cost", "daily_capacity_units", "current_workload_units",
        "workload_ratio", "defect_probability", "initial_markup",
    ]].corr().to_string())

    section("LOGISTICS FEATURE CORRELATION")
    print(logistics[[
        "cost_per_kg", "base_delivery_cost", "damage_probability",
        "delay_probability", "initial_markup",
    ]].corr().to_string())

    section("ORDER FEATURE CORRELATION")
    print(orders[[
        "quantity", "budget", "max_defect_rate", "min_reputation_score",
    ]].corr().to_string())

    section("ORDER PRIORITY DISTRIBUTION")
    print(orders["priority"].value_counts().to_string())

    section("ORDER SCENARIO TYPE DISTRIBUTION")
    print(orders["scenario_type"].value_counts().to_string())

    section("ORDER PRODUCT-MATERIAL PAIR DISTRIBUTION")
    print((orders["product_id"] + " - " + orders["material_type"]).value_counts().to_string())

    section("DOMINANCE CHECKS - MATERIAL (per material type)")
    for mat in ALL_MATERIALS:
        cost_col = f"{mat}_base_cost_eur_per_kg"
        stock_col = f"{mat}_available_kg"
        sub = materials[["agent_id", cost_col, "defect_probability", stock_col,
                          "default_completion_days"]].dropna()
        if len(sub) < 2:
            continue
        pairs = find_dominated_agents(
            sub,
            minimize_cols=[cost_col, "defect_probability", "default_completion_days"],
            maximize_cols=[stock_col],
        )
        print_dominance_report(f"material - {mat}", pairs)

    section("DOMINANCE CHECKS - MANUFACTURER (per product)")
    for prod in ALL_PRODUCTS:
        cost_col = f"unit_cost_{prod}"
        sub = manufacturers[["agent_id", cost_col, "defect_probability",
                              "daily_capacity_units", "workload_ratio"]].dropna()
        if len(sub) < 2:
            continue
        pairs = find_dominated_agents(
            sub,
            minimize_cols=[cost_col, "defect_probability", "workload_ratio"],
            maximize_cols=["daily_capacity_units"],
        )
        print_dominance_report(f"manufacturer - {prod}", pairs)

    section("DOMINANCE CHECK - LOGISTICS")
    pairs = find_dominated_agents(
        logistics[["agent_id", "cost_per_kg", "damage_probability", "delay_probability", "base_delivery_cost"]],
        minimize_cols=["cost_per_kg", "damage_probability", "delay_probability", "base_delivery_cost"],
        maximize_cols=[],
    )
    print_dominance_report("logistics", pairs)

    print(f"\n{'=' * 70}")
    print("\ndiagnostics complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
