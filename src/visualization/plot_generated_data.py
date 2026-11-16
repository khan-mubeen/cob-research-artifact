# %%
# set SHOW_FIGS below based on how this file runs
#   True: run cells in vs code with the jupyter extension
#   False: save pngs with `make figures`

from pathlib import Path
import os
import tempfile

SHOW_FIGS = False
if not SHOW_FIGS:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import pandas as pd
import matplotlib
if not SHOW_FIGS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_VERSION
from src.visualization.data_utils import (
    load_all_data,
    find_dominated_agents,
    print_dominance_report,
    ALL_MATERIALS,
    ALL_PRODUCTS,
    ALL_CITIES,
)

plt.rcParams.update({
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.dpi":   120,
})

SAVE_FIGS = True
FIGS_DIR  = Path("reports/reference/dataset/figures") / DATA_VERSION


def save_or_show(fig, name):
    if SAVE_FIGS:
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"saved {name}.png")
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_correlation_heatmap(df, cols, title, name):
    # negative values mean two features move in opposite directions
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title(title)
    fig.tight_layout()
    save_or_show(fig, name)


def plot_agent_dimension_heatmap(pivot_df, value_label, title, name, fmt="{:.2f}"):
    # one row per agent, one column per material or product
    # faster to scan than grouped bar charts with many agents
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot_df) * 0.4)))
    im = ax.imshow(pivot_df.values.astype(float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticks(range(len(pivot_df)))
    ax.set_yticklabels(pivot_df.index)
    for i in range(len(pivot_df)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, fmt.format(float(val)), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=value_label)
    ax.set_title(title)
    fig.tight_layout()
    save_or_show(fig, name)


# %%
# load everything

materials, manufacturers, logistics, products, orders = load_all_data()

print(f"version:             {DATA_VERSION}")
print(f"material agents:     {len(materials)}")
print(f"manufacturer agents: {len(manufacturers)}")
print(f"logistics agents:    {len(logistics)}")
print(f"orders:              {len(orders)}")

print("\nmaterial coverage:")
for mat in ALL_MATERIALS:
    count = materials[f"{mat}_available_kg"].notna().sum()
    print(f"  {mat:10s}: {count} agents")

print("\nproduct coverage:")
for prod in ALL_PRODUCTS:
    count = manufacturers[f"unit_cost_{prod}"].notna().sum()
    print(f"  {prod:8s}: {count} agents")


# %%
# material: how many agents cover each material type

coverage_mat = {
    mat: materials[f"{mat}_available_kg"].notna().sum()
    for mat in ALL_MATERIALS
}
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(coverage_mat.keys(), coverage_mat.values(), color="tab:blue")
ax.set_ylabel("number of agents")
ax.set_title(f"material coverage by material type ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "material_coverage")


# %%
# material: stock per agent, grouped bar

fig, ax = plt.subplots(figsize=(10, 5))
for mat in ALL_MATERIALS:
    col = f"{mat}_available_kg"
    sub = materials[["agent_id", col]].dropna()
    ax.bar(sub["agent_id"] + f"\n{mat[:3]}", sub[col], label=mat)

ax.set_ylabel("available stock (kg)")
ax.set_title(f"material stock per agent ({DATA_VERSION})")
ax.tick_params(axis="x", rotation=90)
fig.tight_layout()
save_or_show(fig, "material_stock")


# %%
# material: stock heatmap, easier to scan than the bar chart above

stock_pivot = materials.set_index("agent_id")[[f"{m}_available_kg" for m in ALL_MATERIALS]]
stock_pivot.columns = ALL_MATERIALS
plot_agent_dimension_heatmap(
    stock_pivot, "kg",
    f"material stock heatmap, agent x material ({DATA_VERSION})",
    "material_stock_heatmap",
    fmt="{:.0f}",
)


# %%
# material: cost heatmap, blank cells mean agent does not handle that material

cost_pivot = materials.set_index("agent_id")[[f"{m}_base_cost_eur_per_kg" for m in ALL_MATERIALS]]
cost_pivot.columns = ALL_MATERIALS
plot_agent_dimension_heatmap(
    cost_pivot, "EUR per kg",
    f"material cost heatmap, agent x material ({DATA_VERSION})",
    "material_cost_heatmap",
)


# %%
# material: cost vs defect, spread shows the cost-quality tradeoff

fig, ax = plt.subplots(figsize=(7, 6))
for mat in ALL_MATERIALS:
    cost_col = f"{mat}_base_cost_eur_per_kg"
    sub = materials[["agent_id", cost_col, "defect_probability"]].dropna()
    ax.scatter(sub[cost_col], sub["defect_probability"], label=mat, s=70)
    for _, row in sub.iterrows():
        ax.annotate(row["agent_id"], (row[cost_col], row["defect_probability"]), fontsize=7)

ax.set_xlabel("cost per kg (EUR)")
ax.set_ylabel("defect probability")
ax.set_title(f"material cost vs defect probability ({DATA_VERSION})")
ax.legend()
fig.tight_layout()
save_or_show(fig, "material_cost_vs_defect")

for mat in ALL_MATERIALS:
    cost_col  = f"{mat}_base_cost_eur_per_kg"
    stock_col = f"{mat}_available_kg"
    sub = materials[["agent_id", cost_col, "defect_probability", stock_col, "default_completion_days"]].dropna()
    if len(sub) < 2:
        continue
    pairs = find_dominated_agents(
        sub,
        minimize_cols=[cost_col, "defect_probability", "default_completion_days"],
        maximize_cols=[stock_col],
    )
    print_dominance_report(f"material - {mat}", pairs)


# %%
# material: correlation heatmap

plot_correlation_heatmap(
    materials,
    ["avg_cost_eur_per_kg", "avg_available_kg", "defect_probability",
     "default_completion_days", "initial_markup", "lead_time_variance"],
    f"material agent feature correlation ({DATA_VERSION})",
    "material_correlation",
)


# %%
# manufacturer: how many agents make each product

coverage_prod = {
    prod: manufacturers[f"unit_cost_{prod}"].notna().sum()
    for prod in ALL_PRODUCTS
}
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(coverage_prod.keys(), coverage_prod.values(), color="tab:green")
ax.set_ylabel("number of agents")
ax.set_title(f"manufacturer product coverage ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "manufacturer_product_coverage")


# %%
# manufacturer: capacity vs workload, shows who is free and who is already busy

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(manufacturers))
ax.bar(x, manufacturers["daily_capacity_units"], label="daily capacity")
ax.bar(x, manufacturers["current_workload_units"], label="current workload", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(manufacturers["agent_id"], rotation=45)
ax.set_ylabel("units")
ax.set_title(f"manufacturer capacity vs workload ({DATA_VERSION})")
ax.legend()
fig.tight_layout()
save_or_show(fig, "manufacturer_capacity_workload")


# %%
# manufacturer: workload ratio colored by tier
# green = free to take orders, orange = getting busy, red = nearly full

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["tab:red" if r > 0.9 else "tab:orange" if r > 0.5 else "tab:green"
          for r in manufacturers["workload_ratio"]]
ax.bar(manufacturers["agent_id"], manufacturers["workload_ratio"], color=colors)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.axhline(0.9, color="gray", linestyle="--", linewidth=1)
ax.set_ylabel("workload ratio")
ax.set_title(f"manufacturer starting workload ratio ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "manufacturer_workload_ratio")


# %%
# manufacturer: unit cost heatmap, blank means agent does not make that product

manu_cost_pivot = manufacturers.set_index("agent_id")[[f"unit_cost_{p}" for p in ALL_PRODUCTS]]
manu_cost_pivot.columns = ALL_PRODUCTS
plot_agent_dimension_heatmap(
    manu_cost_pivot, "EUR per unit",
    f"manufacturer unit cost heatmap, agent x product ({DATA_VERSION})",
    "manufacturer_unit_cost_heatmap",
)


# %%
# manufacturer: high-capacity, low-defect agents may dominate

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(manufacturers["daily_capacity_units"], manufacturers["defect_probability"], s=70)
for _, row in manufacturers.iterrows():
    ax.annotate(row["agent_id"], (row["daily_capacity_units"], row["defect_probability"]), fontsize=8)

ax.set_xlabel("daily capacity (units)")
ax.set_ylabel("defect probability")
ax.set_title(f"manufacturer capacity vs defect probability ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "manufacturer_capacity_vs_defect")

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


# %%
# manufacturer: correlation heatmap

plot_correlation_heatmap(
    manufacturers,
    ["avg_unit_cost", "daily_capacity_units", "current_workload_units",
     "workload_ratio", "defect_probability", "initial_markup"],
    f"manufacturer agent feature correlation ({DATA_VERSION})",
    "manufacturer_correlation",
)


# %%
# logistics: how many agents cover each city

city_coverage = {city: logistics["service_regions"].str.contains(city).sum() for city in ALL_CITIES}

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(city_coverage.keys(), city_coverage.values(), color="tab:blue")
ax.set_ylabel("number of agents covering this city")
ax.set_title(f"logistics city coverage ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "logistics_city_coverage")


# %%
# logistics: service-region heatmap, shows exactly which agent covers which city

service_matrix = pd.DataFrame(index=logistics["agent_id"], columns=ALL_CITIES)
for _, row in logistics.iterrows():
    regions = str(row["service_regions"]).split("|")
    for city in ALL_CITIES:
        service_matrix.loc[row["agent_id"], city] = 1 if city in regions else 0
service_matrix = service_matrix.astype(float)

plot_agent_dimension_heatmap(
    service_matrix, "covered",
    f"logistics service-region heatmap ({DATA_VERSION})",
    "logistics_service_region_heatmap",
    fmt="{:.0f}",
)


# %%
# logistics: lower cost should come with higher damage risk

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(logistics["cost_per_kg"], logistics["damage_probability"], s=70)
for _, row in logistics.iterrows():
    ax.annotate(row["agent_id"], (row["cost_per_kg"], row["damage_probability"]), fontsize=8)

ax.set_xlabel("cost per kg (EUR)")
ax.set_ylabel("damage probability")
ax.set_title(f"logistics cost vs damage probability ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "logistics_cost_vs_damage")

pairs = find_dominated_agents(
    logistics[["agent_id", "cost_per_kg", "damage_probability", "delay_probability", "base_delivery_cost"]],
    minimize_cols=["cost_per_kg", "damage_probability", "delay_probability", "base_delivery_cost"],
    maximize_cols=[],
)
print_dominance_report("logistics", pairs)


# %%
# logistics: base cost vs per-kg cost
# a higher fixed fee can offset a lower per-kg rate

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(logistics["base_delivery_cost"], logistics["cost_per_kg"], s=70)
for _, row in logistics.iterrows():
    ax.annotate(row["agent_id"], (row["base_delivery_cost"], row["cost_per_kg"]), fontsize=8)

ax.set_xlabel("base delivery cost (EUR)")
ax.set_ylabel("cost per kg (EUR)")
ax.set_title(f"logistics base cost vs per-kg cost ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "logistics_base_vs_perkg")


# %%
# logistics: correlation heatmap

plot_correlation_heatmap(
    logistics,
    ["cost_per_kg", "base_delivery_cost", "damage_probability",
     "delay_probability", "initial_markup"],
    f"logistics agent feature correlation ({DATA_VERSION})",
    "logistics_correlation",
)


# %%
# logistics extension field outside the official schema

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(logistics["agent_id"], logistics["delay_probability"], color="tab:purple")
ax.set_ylabel("delay probability")
ax.set_title(f"logistics delay probability, extension field ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "logistics_delay_probability_extension")


# %%
# orders: priority distribution

fig, ax = plt.subplots(figsize=(6, 5))
orders["priority"].value_counts().plot(kind="bar", ax=ax, color="tab:blue")
ax.set_ylabel("number of orders")
ax.set_title(f"order priority distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_priority")


# %%
# orders: quantity distribution

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(orders["quantity"], bins=20, color="tab:green", edgecolor="black")
ax.set_xlabel("quantity")
ax.set_ylabel("number of orders")
ax.set_title(f"order quantity distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_quantity")


# %%
# orders: budget distribution

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(orders["budget"], bins=20, color="tab:red", edgecolor="black")
ax.set_xlabel("budget (EUR)")
ax.set_ylabel("number of orders")
ax.set_title(f"order budget distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_budget")


# %%
# orders: days until deadline from order creation

orders["order_created_dt"]   = pd.to_datetime(orders["order_created"])
orders["deadline_dt"]         = pd.to_datetime(orders["deadline"])
orders["days_until_deadline"] = (orders["deadline_dt"] - orders["order_created_dt"]).dt.days

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(orders["days_until_deadline"], bins=20, color="tab:orange", edgecolor="black")
ax.set_xlabel("days until deadline")
ax.set_ylabel("number of orders")
ax.set_title(f"order deadline distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_deadline_days")


# %%
# orders: scenario type distribution

fig, ax = plt.subplots(figsize=(8, 5))
orders["scenario_type"].value_counts().plot(kind="bar", ax=ax, color="tab:purple")
ax.set_ylabel("number of orders")
ax.set_title(f"order scenario type distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_scenario_type")


# %%
# orders: which city orders are delivered to
# compare with logistics coverage to check whether demand and supply align

fig, ax = plt.subplots(figsize=(8, 5))
orders["deliver_to"].value_counts().plot(kind="bar", ax=ax, color="tab:blue")
ax.set_ylabel("number of orders")
ax.set_title(f"order delivery city distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_delivery_city")


# %%
# orders: product-material pair bar chart

fig, ax = plt.subplots(figsize=(8, 5))
pair_counts = (orders["product_id"] + " - " + orders["material_type"]).value_counts()
pair_counts.plot(kind="bar", ax=ax, color="tab:cyan")
ax.set_ylabel("number of orders")
ax.set_title(f"order product-material pair distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_product_material_pairs")


# %%
# orders: product-material pair heatmap, easier to scan than the bar chart

pair_matrix = pd.crosstab(orders["product_id"], orders["material_type"])
plot_agent_dimension_heatmap(
    pair_matrix, "orders",
    f"order product-material heatmap ({DATA_VERSION})",
    "orders_product_material_heatmap",
    fmt="{:.0f}",
)


# %%
# orders: larger quantities should have larger budgets

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(orders["quantity"], orders["budget"], s=30, alpha=0.7, color="tab:blue")
ax.set_xlabel("quantity")
ax.set_ylabel("budget (EUR)")
ax.set_title(f"order quantity vs budget ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_quantity_vs_budget")


# %%
# orders: defect tolerance for material and manufacturing quality

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(orders["max_defect_rate"], bins=20, color="tab:red", edgecolor="black")
ax.set_xlabel("max defect rate")
ax.set_ylabel("number of orders")
ax.set_title(f"order defect tolerance distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_max_defect_rate")


# %%
# orders: reputation requirement distribution
# all material and logistics agents start at 0.5, so orders above 0.5 filter them out

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(orders["min_reputation_score"], bins=20, color="tab:green", edgecolor="black")
ax.set_xlabel("minimum reputation score")
ax.set_ylabel("number of orders")
ax.set_title(f"order reputation requirement distribution ({DATA_VERSION})")
fig.tight_layout()
save_or_show(fig, "orders_min_reputation_score")


# %%
# orders: feature correlation

plot_correlation_heatmap(
    orders,
    ["quantity", "budget", "max_defect_rate", "min_reputation_score"],
    f"order feature correlation ({DATA_VERSION})",
    "orders_correlation",
)

print(f"\ndone. {len(list(FIGS_DIR.glob('*.png')) if SAVE_FIGS else [])} charts saved to reports/reference/dataset/figures/{DATA_VERSION}/")
