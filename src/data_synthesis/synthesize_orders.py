import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import random

from config import SYNTHETIC_DATA_DIR

PRODUCTS_PATH      = SYNTHETIC_DATA_DIR / "products.csv"
MANUFACTURERS_PATH = SYNTHETIC_DATA_DIR / "manufacturer_agents.csv"
MATERIALS_PATH     = SYNTHETIC_DATA_DIR / "material_agents.csv"
LOGISTICS_PATH     = SYNTHETIC_DATA_DIR / "logistics_agents.csv"
OUTPUT_PATH        = SYNTHETIC_DATA_DIR / "orders.csv"

SEED = 42
random.seed(SEED)

TOTAL_ORDERS = 50_000

ID_WIDTH = max(3, len(str(TOTAL_ORDERS)))

# creation dates spread over half a year so 50k orders do not pile
# into one month, which would look nothing like a real order book
CREATION_WINDOW_DAYS = 180

# defaults to today so deadlines never sit in the past on a live run,
# set COB_BASE_DATE=YYYY-MM-DD to regenerate an older dataset exactly
_env_date = os.getenv("COB_BASE_DATE")
if _env_date:
    try:
        BASE_DATE = datetime.strptime(_env_date, "%Y-%m-%d")
    except ValueError:
        sys.exit(f"COB_BASE_DATE must be YYYY-MM-DD, got: {_env_date}")
else:
    BASE_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# separate stream so day sampling never shifts other random fields
day_rng = random.Random(SEED + 1)
WEEKDAY_WEIGHTS = {0: 1.0, 1: 1.1, 2: 1.2, 3: 1.2, 4: 1.3, 5: 0.8, 6: 0.4}
DAY_WEIGHTS = [
    WEEKDAY_WEIGHTS[(BASE_DATE + timedelta(days=i)).weekday()]
    for i in range(CREATION_WINDOW_DAYS)
]

ALL_CITIES = ["Oulu", "Helsinki", "Tampere", "Turku", "Jyvaskyla", "Lahti", "Espoo"]

# uniform city sampling already creates enough northern deliveries
SCENARIO_MIX = {
    "random_realistic": 0.71,
    "tight_deadline":   0.06,
    "low_budget":       0.06,
    "large_quantity":   0.05,
    "high_quality":     0.05,
    "rare_combo":       0.04,
    "tight_low_budget": 0.03,
}

PRIORITY_WEIGHTS = {"urgent": 0.20, "normal": 0.60, "flexible": 0.20}

# to price the budget estimate the way agents will actually bid
PRIORITY_MARKUP = {"urgent": 0.30, "normal": 0.00, "flexible": -0.15}

QUANTITY_BANDS = {
    "small":  {"range": (100,  300),  "weight": 0.35},
    "medium": {"range": (300,  900),  "weight": 0.45},
    "large":  {"range": (900,  2000), "weight": 0.20},
}

DEADLINE_DAYS = {
    "urgent":   (7,  14),
    "normal":   (14, 30),
    "flexible": (30, 60),
}

# budget bands scale willingness to pay from chain prices after bid adjustments
BUDGET_MULTIPLIERS = {
    "urgent":     (1.00, 1.30),
    "normal":     (0.95, 1.25),
    "flexible":   (0.90, 1.15),
    "low_budget": (0.88, 0.98),
}

DEFECT_TOLERANCE = {
    "urgent":       (0.03, 0.06),
    "normal":       (0.04, 0.08),
    "flexible":     (0.05, 0.10),
    "high_quality": (0.02, 0.04),
}

REPUTATION_REQUIREMENTS = {
    "urgent":   (0.45, 0.50),
    "normal":   (0.40, 0.50),
    "flexible": (0.35, 0.48),
}

VALID_PRIORITIES = {"urgent", "normal", "flexible"}
VALID_SCENARIOS  = set(SCENARIO_MIX)

# fallback constants used when csv files are not found
FALLBACK_UNIT_COST_ESTIMATES = {
    "shirt":  4.75,
    "hoodie": 8.00,
    "jacket": 11.50,
    "jeans":  7.00,
    "skirt":  4.25,
}

# fallback derived from fred cotton anchor 2.83 eur/kg and material ratios
FALLBACK_MATERIAL_COST_PER_KG = {
    "cotton":    2.83,
    "polyester": 1.41,
    "linen":     7.07,
    "denim":     5.23,
    "leather":  28.29,
}

AVG_LOGISTICS_BASE   = 40.0
AVG_LOGISTICS_PER_KG = 0.75

# average workload markup used for budget estimates
# most orders stay below 0.50; backlog-heavy orders pay 0.05 to 0.15
AVG_WORKLOAD_MARKUP = 0.03

# fallback stage price factors, (1 + mean defect) x (1 + mean markup),
# computed from the v4 agent csvs, used only when a csv is missing
FALLBACK_STAGE_FACTORS = {
    "material":     1.32,
    "manufacturer": 1.38,
    "logistics":    1.30,
}


def load_column_means(path, columns_by_key, fallback, label):
    # load average costs for products and materials
    # use hardcoded values when files or columns are missing
    if not path.exists():
        print(f"  warning: {path} not found, using fallback {label}")
        return fallback.copy()

    try:
        df = pd.read_csv(path)
        result = {}
        for key, col in columns_by_key.items():
            if col not in df.columns:
                print(f"  warning: column {col} missing, using fallback for {key}")
                result[key] = fallback[key]
                continue
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            result[key] = fallback[key] if values.empty else round(values.mean(), 3)
        print(f"  loaded {label} from {path}")
        return result
    except Exception as e:
        print(f"  warning: could not read {path} ({e}), using fallback {label}")
        return fallback.copy()


def load_unit_cost_estimates(path):
    columns = {p: f"unit_cost_{p}" for p in FALLBACK_UNIT_COST_ESTIMATES}
    return load_column_means(path, columns, FALLBACK_UNIT_COST_ESTIMATES, "unit cost estimates")


def load_material_cost_per_kg(path):
    columns = {m: f"{m}_base_cost_eur_per_kg" for m in FALLBACK_MATERIAL_COST_PER_KG}
    return load_column_means(path, columns, FALLBACK_MATERIAL_COST_PER_KG, "material cost estimates")


def load_stage_factor(path, defect_col, markup_col, stage, extra=1.0):
    # mirror agent pricing for this stage
    # average defect and markup across agents
    fallback = FALLBACK_STAGE_FACTORS[stage]
    if not path.exists():
        print(f"  warning: {path} not found, using fallback factor for {stage}")
        return fallback

    try:
        df = pd.read_csv(path)
        defect = pd.to_numeric(df[defect_col], errors="coerce").dropna().mean()
        markup = pd.to_numeric(df[markup_col], errors="coerce").dropna().mean()
        factor = round((1 + defect) * (1 + markup) * extra, 4)
        print(f"  {stage} stage factor: {factor} (defect {defect:.3f}, markup {markup:.3f})")
        return factor
    except Exception as e:
        print(f"  warning: could not derive {stage} factor ({e}), using fallback")
        return fallback


def load_product_material_pairs(products_path):
    if not products_path.exists():
        raise FileNotFoundError(f"products file not found: {products_path}")

    df = pd.read_csv(products_path)

    required_columns = {
        "product_id",
        "material_type",
        "material_kg_per_unit",
        "finished_weight_kg_per_unit",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns in products.csv: {missing}")

    for col in ["material_kg_per_unit", "finished_weight_kg_per_unit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required_columns))

    pairs = df[list(required_columns)].to_dict("records")
    if not pairs:
        raise ValueError("products.csv has no valid rows after cleaning")

    print(f"  loaded {len(pairs)} valid product-material pairs from products.csv")
    return pairs


def load_agent_coverage(materials_path, logistics_path):
    # keep orders within agent coverage
    # drop materials and cities no agent can serve
    logistics_df = pd.read_csv(logistics_path)
    valid_cities = set()
    for regions in logistics_df["service_regions"].astype(str):
        valid_cities.update(regions.split("|"))

    materials_df = pd.read_csv(materials_path)
    valid_materials = set()
    for mat in FALLBACK_MATERIAL_COST_PER_KG:
        col = f"{mat}_available_kg"
        if col in materials_df.columns and materials_df[col].notna().any():
            valid_materials.add(mat)

    return valid_materials, valid_cities


def pick_quantity():
    bands   = list(QUANTITY_BANDS.keys())
    weights = [QUANTITY_BANDS[b]["weight"] for b in bands]
    band    = random.choices(bands, weights=weights, k=1)[0]
    lo, hi  = QUANTITY_BANDS[band]["range"]
    return random.randint(lo, hi)


def pick_priority():
    priorities = list(PRIORITY_WEIGHTS)
    weights    = list(PRIORITY_WEIGHTS.values())
    return random.choices(priorities, weights=weights, k=1)[0]


def estimate_chain_cost(product_id, material_type, quantity,
                        material_kg_per_unit, finished_weight_kg_per_unit,
                        unit_cost_estimates, material_cost_per_kg,
                        stage_factors, priority):
    # price every stage the same way agents bid
    # use expected market price instead of raw input cost
    # apply priority markup after adding the stage totals
    if material_type not in material_cost_per_kg:
        raise ValueError(f"unknown material type in cost estimate: {material_type}")
    if product_id not in unit_cost_estimates:
        raise ValueError(f"unknown product in cost estimate: {product_id}")

    material_cost      = quantity * material_kg_per_unit * material_cost_per_kg[material_type]
    manufacturing_cost = quantity * unit_cost_estimates[product_id]
    logistics_cost     = AVG_LOGISTICS_BASE + AVG_LOGISTICS_PER_KG * finished_weight_kg_per_unit * quantity

    market_price = (
        material_cost      * stage_factors["material"]
        + manufacturing_cost * stage_factors["manufacturer"]
        + logistics_cost     * stage_factors["logistics"]
    )
    return market_price * (1 + PRIORITY_MARKUP[priority])


def make_order(order_id, pair, priority, scenario_type, order_index,
               unit_cost_estimates, material_cost_per_kg, stage_factors,
               budget_override=None, quantity_override=None,
               city_override=None, defect_override=None,
               deadline_override=None, valid_cities=None):

    product_id    = pair["product_id"]
    material_type = pair["material_type"]
    mat_kg        = pair["material_kg_per_unit"]
    fin_kg        = pair["finished_weight_kg_per_unit"]

    quantity = quantity_override if quantity_override is not None else pick_quantity()

    day_index = day_rng.choices(range(CREATION_WINDOW_DAYS), weights=DAY_WEIGHTS, k=1)[0]
    order_created = BASE_DATE + timedelta(days=day_index, hours=random.randint(8, 17))

    deadline_lo, deadline_hi = deadline_override if deadline_override is not None else DEADLINE_DAYS[priority]
    deadline = order_created + timedelta(days=random.randint(deadline_lo, deadline_hi))

    city_pool  = sorted(valid_cities) if valid_cities else ALL_CITIES
    deliver_to = city_override if city_override is not None else random.choice(city_pool)

    estimated_cost = estimate_chain_cost(
        product_id, material_type, quantity, mat_kg, fin_kg,
        unit_cost_estimates, material_cost_per_kg, stage_factors, priority,
    )

    lo, hi = budget_override if budget_override is not None else BUDGET_MULTIPLIERS[priority]
    budget = round(estimated_cost * random.uniform(lo, hi), 2)

    defect_lo, defect_hi = defect_override if defect_override is not None else DEFECT_TOLERANCE[priority]
    max_defect_rate = round(random.uniform(defect_lo, defect_hi), 3)

    rep_lo, rep_hi = REPUTATION_REQUIREMENTS[priority]
    min_reputation_score = round(random.uniform(rep_lo, rep_hi), 2)

    return {
        "order_id":             order_id,
        "order_created":        order_created.strftime("%Y-%m-%dT%H:%M:%S"),
        "product_id":           product_id,
        "quantity":             quantity,
        "material_type":        material_type,
        "budget":               budget,
        "deadline":             deadline.strftime("%Y-%m-%d"),
        "deliver_to":           deliver_to,
        "max_defect_rate":      max_defect_rate,
        "min_reputation_score": min_reputation_score,
        "priority":             priority,
        "scenario_type":        scenario_type,
    }


# each scenario builder returns only its overrides
# everything else stays on realistic defaults

def build_random_realistic(pairs, ctx):
    return random.choice(pairs), pick_priority(), {}


def build_tight_deadline(pairs, ctx):
    return random.choice(pairs), "urgent", {
        "quantity_override": random.randint(400, 900),
        "deadline_override": (5, 8),
    }


def build_low_budget(pairs, ctx):
    return random.choice(pairs), "normal", {
        "budget_override": BUDGET_MULTIPLIERS["low_budget"],
    }


def build_large_quantity(pairs, ctx):
    return random.choice(pairs), "normal", {
        "quantity_override": random.randint(1200, 2000),
    }


def build_high_quality(pairs, ctx):
    return random.choice(pairs), "urgent", {
        "defect_override": DEFECT_TOLERANCE["high_quality"],
    }


def build_rare_combo(pairs, ctx):
    rare = ctx["rare_pairs"]
    pair = rare[ctx["counter"] % len(rare)] if rare else random.choice(pairs)
    return pair, "normal", {}


def build_tight_low_budget(pairs, ctx):
    # both constraints at once, budget squeezed against the urgent
    # market price but not below the cheapest realistic chain
    return random.choice(pairs), "urgent", {
        "quantity_override": random.randint(300, 800),
        "deadline_override": (6, 9),
        "budget_override":   (0.93, 1.03),
    }


SCENARIO_BUILDERS = {
    "random_realistic": build_random_realistic,
    "tight_deadline":   build_tight_deadline,
    "low_budget":       build_low_budget,
    "large_quantity":   build_large_quantity,
    "high_quality":     build_high_quality,
    "rare_combo":       build_rare_combo,
    "tight_low_budget": build_tight_low_budget,
}


def scenario_counts(total):
    # convert fractions to whole counts, remainder goes to random_realistic
    counts = {s: int(total * frac) for s, frac in SCENARIO_MIX.items()}
    counts["random_realistic"] += total - sum(counts.values())
    return counts


def generate_orders(pairs, total, unit_cost_estimates, material_cost_per_kg,
                    stage_factors, valid_cities):
    rare_pairs = [p for p in pairs if
        (p["product_id"] == "jacket" and p["material_type"] == "leather") or
        (p["product_id"] == "jeans"  and p["material_type"] == "denim")   or
        (p["product_id"] == "skirt"  and p["material_type"] == "linen")]

    counts = scenario_counts(total)
    orders = []
    idx = 0

    for scenario, count in counts.items():
        builder = SCENARIO_BUILDERS[scenario]
        for i in range(count):
            ctx = {"rare_pairs": rare_pairs, "counter": i}
            pair, priority, overrides = builder(pairs, ctx)
            order_id = f"ORD-{idx + 1:0{ID_WIDTH}d}"
            orders.append(make_order(
                order_id, pair, priority, scenario, idx,
                unit_cost_estimates, material_cost_per_kg, stage_factors,
                valid_cities=valid_cities, **overrides,
            ))
            idx += 1

    return orders


def validate_orders(orders, pairs):
    valid_combos = {(p["product_id"], p["material_type"]) for p in pairs}
    errors   = []
    seen_ids = set()

    for o in orders:
        if o["order_id"] in seen_ids:
            errors.append(f"{o['order_id']}: duplicate order_id")
        seen_ids.add(o["order_id"])

        if (o["product_id"], o["material_type"]) not in valid_combos:
            errors.append(f"{o['order_id']}: invalid product-material combo ({o['product_id']}, {o['material_type']})")

        if o["quantity"] <= 0:
            errors.append(f"{o['order_id']}: quantity must be positive")

        if o["budget"] <= 0:
            errors.append(f"{o['order_id']}: budget must be positive")

        if o["deliver_to"] not in ALL_CITIES:
            errors.append(f"{o['order_id']}: unknown delivery city {o['deliver_to']}")

        if not 0 <= o["max_defect_rate"] <= 1:
            errors.append(f"{o['order_id']}: max_defect_rate must be between 0 and 1")

        if not 0 <= o["min_reputation_score"] <= 1:
            errors.append(f"{o['order_id']}: min_reputation_score must be between 0 and 1")

        if o["priority"] not in VALID_PRIORITIES:
            errors.append(f"{o['order_id']}: invalid priority {o['priority']}")

        if o["scenario_type"] not in VALID_SCENARIOS:
            errors.append(f"{o['order_id']}: invalid scenario_type {o['scenario_type']}")

        created  = datetime.strptime(o["order_created"], "%Y-%m-%dT%H:%M:%S")
        deadline = datetime.strptime(o["deadline"], "%Y-%m-%d")
        if deadline.date() <= created.date():
            errors.append(f"{o['order_id']}: deadline must be after order_created")

    if errors:
        preview = "\n".join(errors[:50])
        raise ValueError(f"order validation failed with {len(errors)} errors, first 50:\n{preview}")

    print(f"order validation passed for {len(orders)} orders")


def print_summary(orders):
    df = pd.DataFrame(orders)
    print("\nscenario distribution:")
    print(df["scenario_type"].value_counts().to_string())
    print("\npriority distribution:")
    print(df["priority"].value_counts().to_string())
    print("\nproduct distribution:")
    print(df["product_id"].value_counts().to_string())
    print("\nquantity stats:")
    print(df["quantity"].describe()[["min", "mean", "max"]].to_string())
    print("\nbudget stats:")
    print(df["budget"].describe()[["min", "mean", "max"]].to_string())


def synthesize(products_path, manufacturers_path, materials_path, logistics_path, output_path):
    print("loading reference data:")
    pairs                = load_product_material_pairs(products_path)
    unit_cost_estimates  = load_unit_cost_estimates(manufacturers_path)
    material_cost_per_kg = load_material_cost_per_kg(materials_path)

    stage_factors = {
        "material":     load_stage_factor(materials_path, "defect_probability", "initial_markup", "material"),
        "manufacturer": load_stage_factor(manufacturers_path, "defect_probability", "initial_markup",
                                          "manufacturer", extra=1 + AVG_WORKLOAD_MARKUP),
        "logistics":    load_stage_factor(logistics_path, "damage_probability", "initial_markup", "logistics"),
    }

    valid_materials, valid_cities = load_agent_coverage(materials_path, logistics_path)
    pairs = [p for p in pairs if p["material_type"] in valid_materials]
    print(f"  valid materials: {sorted(valid_materials)}")
    print(f"  valid cities: {sorted(valid_cities)}")

    print(f"\ngenerating {TOTAL_ORDERS} orders")
    all_orders = generate_orders(
        pairs, TOTAL_ORDERS,
        unit_cost_estimates, material_cost_per_kg, stage_factors, valid_cities,
    )

    validate_orders(all_orders, pairs)
    print_summary(all_orders)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_orders).to_csv(output_path, index=False)

    # record the seed and input hashes
    # keep the base date and scenario mix with the dataset
    # pin the agent csvs used for each published run
    def file_sha256(path):
        if not path.exists():
            return "missing"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    metadata = {
        "generated_at":         datetime.now().isoformat(timespec="seconds"),
        "seed":                 SEED,
        "total_orders":         TOTAL_ORDERS,
        "base_date":            BASE_DATE.strftime("%Y-%m-%d"),
        "creation_window_days": CREATION_WINDOW_DAYS,
        "scenario_mix":         SCENARIO_MIX,
        "priority_weights":     PRIORITY_WEIGHTS,
        "budget_anchor":        "expected market price, stage factors from agent csvs plus priority markup",
        "stage_factors":        stage_factors,
        "input_sha256": {
            "products":      file_sha256(products_path),
            "manufacturers": file_sha256(manufacturers_path),
            "materials":     file_sha256(materials_path),
            "logistics":     file_sha256(logistics_path),
        },
    }
    meta_path = output_path.parent / "orders_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\ndone. {len(all_orders)} orders saved to {output_path}")
    print(f"metadata saved to {meta_path}")


if __name__ == "__main__":
    synthesize(PRODUCTS_PATH, MANUFACTURERS_PATH, MATERIALS_PATH, LOGISTICS_PATH, OUTPUT_PATH)
