from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR, SYNTHETIC_DATA_DIR

RAW_DATA_PATH = RAW_DATA_DIR / "garments_worker_productivity.csv"
OUTPUT_PATH   = SYNTHETIC_DATA_DIR / "manufacturer_agents.csv"

MIN_PRODUCT_COVERAGE = 3

REQUIRED_COLUMNS = {
    "team", "department", "no_of_workers",
    "actual_productivity", "targeted_productivity", "smv", "wip",
}

NUMERIC_COLUMNS = [
    "team", "no_of_workers", "actual_productivity",
    "targeted_productivity", "smv", "wip",
]

DEPARTMENT_ALIASES = {
    "sweing":    "sewing",
    "sewing":    "sewing",
    "finishing": "finishing",
}

# mid-range unit costs per product in eur
UNIT_COSTS = {
    "shirt":  {"min": 3.5, "max": 6.0},
    "hoodie": {"min": 6.0, "max": 10.0},
    "jacket": {"min": 9.0, "max": 14.0},
    "jeans":  {"min": 5.0, "max": 9.0},
    "skirt":  {"min": 3.0, "max": 5.5},
}

# each agent specializes in different product combinations
AGENT_PRODUCT_SPECIALIZATIONS = [
    ["shirt", "hoodie", "jacket"],
    ["shirt", "hoodie"],
    ["jacket", "jeans"],
    ["shirt", "jeans", "skirt"],
    ["hoodie", "jacket"],
    ["shirt", "skirt"],
    ["shirt", "hoodie", "jacket", "jeans"],
    ["jeans", "skirt"],
    ["shirt", "hoodie"],
    ["jacket", "jeans", "skirt"],
    ["shirt", "hoodie", "jacket"],
    ["shirt", "jeans"],
]

LOCATIONS = ["Oulu", "Helsinki", "Tampere", "Turku", "Jyvaskyla", "Lahti", "Espoo"]

# markups keep budget and premium niches separate from base unit cost
# t12 stays low for budget bids; t9 stays above t2 to keep their roles distinct
MARKUPS = [0.28, 0.29, 0.31, 0.30, 0.33, 0.32, 0.30, 0.32, 0.32, 0.31, 0.30, 0.22]

# cost ratios place agents within product price bands
# lower ratios mark budget agents; higher ratios leave room for premium tradeoffs
AGENT_COST_RATIOS = [
    0.92,  # T1:  normal quality, low cost, low workload
    0.88,  # T2:  budget across products
    0.82,  # T3:  cheapest jacket/jeans, compensates busy backlog
    0.92,  # T4:  affordable, wins urgent via free capacity
    0.78,  # T5:  hoodie/jacket quality mid-range
    0.87,  # T6:  budget shirt/skirt flexible orders
    0.74,  # T7:  absolute cheapest wins large low-budget orders
    0.73,  # T8:  cheapest jeans/skirt, beats T4 and T6 on price
    # t2 and t9 make the same products, so no ratio gives both a lowest-cost niche
    0.86,  # T9:  urgent large orders, almost empty capacity
    1.00,  # T10: premium jacket, worth the wait for quality
    0.98,  # T11: expensive but most available, wins urgent premium
    0.92,  # T12: budget shirt/jeans steady state
]

AGENT_WORKLOAD_FRACTIONS = [
    0.20,  # T1:  free after large order
    0.55,  # T2:  steady normal
    0.85,  # T3:  busy, compensates with price
    0.15,  # T4:  very free, urgent specialist
    0.70,  # T5:  moderately loaded
    0.35,  # T6:  light workload
    0.40,  # low workload keeps t7's cost advantage eligible
    0.50,  # T8:  normal steady state
    0.08,  # T9:  almost empty, urgent specialist
    0.08,  # T10: low workload premium jacket
    0.05,  # T11: most available in system, wins urgent premium
    0.75,  # T12: moderately busy
]

AGENT_SPECIALTIES = [
    "premium multi-product, low defect, free capacity after large order",
    "budget shirt and hoodie with great quality, often cheapest good option",
    "cheapest jacket and jeans producer, compensates busy backlog with price",
    "very free capacity, affordable, best choice for urgent mid-quality orders",
    "hoodie and jacket specialist, very cheap but higher defect risk",
    "budget shirt and skirt, light workload, wins for flexible orders",
    "absolute lowest cost across products, near capacity, high defect only option",
    "mid-range jeans and skirt, normal workload, reliable choice",
    "almost empty capacity, urgent large order specialist, competitive pricing",
    "premium jacket and jeans, busy but worth it for quality-critical orders",
    "most expensive but best quality and service across all product types",
    "budget shirt and jeans, lowest markup in system, wins price-sensitive orders",
]

def normalize_department(dept):
    clean = str(dept).strip().lower()
    return DEPARTMENT_ALIASES.get(clean, clean)


def get_scale(total_workers):
    if total_workers <= 25:
        return "small"
    if total_workers <= 60:
        return "medium"
    return "large"


def calculate_daily_capacity(avg_productivity, avg_smv, avg_workers):
    # 8 hour shift = 480 mins
    # capacity = workers * (shift_minutes / smv) * actual_productivity
    shift_minutes = 480

    if pd.isna(avg_smv) or avg_smv <= 0:
        return 0

    capacity = avg_workers * (shift_minutes / avg_smv) * avg_productivity
    return int(round(max(0, capacity)))


def estimate_defect_probability(avg_productivity, avg_targeted):
    # teams falling short of target tend to have higher defect rates
    if pd.isna(avg_productivity) or pd.isna(avg_targeted) or avg_targeted <= 0:
        return 0.10

    ratio = avg_productivity / avg_targeted
    return round(max(0.02, min(0.25, 1 - ratio)), 3)


def derive_reputation(avg_productivity):
    # reputation derived from actual productivity ratio, clamped between 0 and 1
    return round(max(0, min(1, avg_productivity)), 3)


def weighted_average(values_and_weights, fallback=0):
    total_weight = 0
    weighted_sum = 0

    for value, weight in values_and_weights:
        if pd.isna(value) or pd.isna(weight) or weight <= 0:
            continue

        weighted_sum += value * weight
        total_weight += weight

    if total_weight <= 0:
        return fallback

    return weighted_sum / total_weight


def get_unit_cost(product_name, product_types, cost_ratio=1.0):
    if product_name not in product_types:
        return None
    cost = UNIT_COSTS[product_name]
    midpoint = (cost["min"] + cost["max"]) / 2
    # apply agent cost position, clamp so no agent goes outside realistic range
    adjusted = max(cost["min"], min(cost["max"], midpoint * cost_ratio))
    return round(adjusted, 2)


def validate_columns(df):
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")


def validate_product_coverage():
    coverage = {product: 0 for product in UNIT_COSTS}

    for product_types in AGENT_PRODUCT_SPECIALIZATIONS:
        for product in product_types:
            if product not in UNIT_COSTS:
                raise ValueError(f"unknown product type in specialization: {product}")

            coverage[product] += 1

    print("product coverage:")
    all_ok = True

    for product, count in coverage.items():
        status = "ok" if count >= MIN_PRODUCT_COVERAGE else f"FAIL need at least {MIN_PRODUCT_COVERAGE}"
        print(f"  {product:8s}: {count} agents  [{status}]")

        if count < MIN_PRODUCT_COVERAGE:
            all_ok = False

    if not all_ok:
        raise ValueError("product coverage check failed, fix manufacturer specializations")

    print("all products covered by enough manufacturer agents")


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    validate_columns(df)

    # normalize department names before anything else
    df["department"] = df["department"].apply(normalize_department)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 0)

    # finishing had no wip recorded in raw data so set it to 0
    df.loc[df["department"] == "finishing", "wip"] = 0

    return df


def build_agent(team_id, team_df, agent_index):
    sewing_df    = team_df[team_df["department"] == "sewing"]
    finishing_df = team_df[team_df["department"] == "finishing"]

    sew_workers      = sewing_df["no_of_workers"].mean() if not sewing_df.empty else 0
    sew_productivity = sewing_df["actual_productivity"].mean() if not sewing_df.empty else 0
    sew_targeted     = sewing_df["targeted_productivity"].mean() if not sewing_df.empty else 0
    sew_smv          = sewing_df["smv"].mean() if not sewing_df.empty else 0

    fin_workers      = finishing_df["no_of_workers"].mean() if not finishing_df.empty else 0
    fin_productivity = finishing_df["actual_productivity"].mean() if not finishing_df.empty else 0
    fin_targeted     = finishing_df["targeted_productivity"].mean() if not finishing_df.empty else 0
    fin_smv          = finishing_df["smv"].mean() if not finishing_df.empty else 0

    total_workers = int(round(sew_workers + fin_workers))

    # sewing is the bottleneck in garment production
    sewing_capacity    = calculate_daily_capacity(sew_productivity, sew_smv, sew_workers)
    finishing_capacity = calculate_daily_capacity(fin_productivity, fin_smv, fin_workers)
    daily_capacity     = min(sewing_capacity, finishing_capacity) if finishing_capacity > 0 else sewing_capacity

    # defect risk is weighted by worker count so the bigger operation matters more
    sew_defect  = estimate_defect_probability(sew_productivity, sew_targeted)
    fin_defect  = estimate_defect_probability(fin_productivity, fin_targeted)

    defect_prob = weighted_average(
        [(sew_defect, sew_workers), (fin_defect, fin_workers)],
        fallback=sew_defect,
    )
    defect_prob = round(defect_prob, 3)

    # tailor reputation starts from productivity because this stage has real productivity data
    avg_productivity = team_df["actual_productivity"].mean()
    reputation       = derive_reputation(avg_productivity)

    # raw wip makes every agent look overloaded, so use calibrated capacity fractions
    idx = agent_index - 1
    workload_fraction = AGENT_WORKLOAD_FRACTIONS[idx % len(AGENT_WORKLOAD_FRACTIONS)]
    current_workload  = int(round(daily_capacity * workload_fraction))
    cost_ratio        = AGENT_COST_RATIOS[idx % len(AGENT_COST_RATIOS)]

    product_types = AGENT_PRODUCT_SPECIALIZATIONS[idx % len(AGENT_PRODUCT_SPECIALIZATIONS)]

    return {
        "agent_id":               f"T{agent_index}",
        "source_team_id":         int(team_id),
        "stage":                  "tailoring",
        "specialty":              AGENT_SPECIALTIES[idx % len(AGENT_SPECIALTIES)],
        "scale":                  get_scale(total_workers),
        "product_types":          "|".join(sorted(product_types)),
        "daily_capacity_units":   daily_capacity,
        "current_workload_units": current_workload,
        "unit_cost_shirt":        get_unit_cost("shirt",  product_types, cost_ratio),
        "unit_cost_hoodie":       get_unit_cost("hoodie", product_types, cost_ratio),
        "unit_cost_jacket":       get_unit_cost("jacket", product_types, cost_ratio),
        "unit_cost_jeans":        get_unit_cost("jeans",  product_types, cost_ratio),
        "unit_cost_skirt":        get_unit_cost("skirt",  product_types, cost_ratio),
        "min_profit_margin":      0.10,
        "initial_markup":         MARKUPS[idx % len(MARKUPS)],
        "defect_probability":     defect_prob,
        "reputation_score":       reputation,
        "location":               LOCATIONS[idx % len(LOCATIONS)],
    }


def synthesize(input_path, output_path):
    print(f"loading data from {input_path}")

    validate_product_coverage()

    df = pd.read_csv(input_path)
    df = clean_data(df)

    print(f"loaded {len(df)} rows across {df['team'].nunique()} teams")

    agents = []
    for index, (team_id, team_df) in enumerate(df.groupby("team"), start=1):
        if pd.isna(team_id):
            continue

        agent = build_agent(team_id, team_df, index)
        agents.append(agent)
        print(f"built T{index} | team {int(team_id)} | products: {agent['product_types']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agents).to_csv(output_path, index=False)

    print(f"\ndone. {len(agents)} manufacturer agents saved to {output_path}")


if __name__ == "__main__":
    synthesize(RAW_DATA_PATH, OUTPUT_PATH)
