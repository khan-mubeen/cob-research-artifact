import pandas as pd

from config import RAW_DATA_DIR, SYNTHETIC_DATA_DIR

RAW_COTTON_PATH = RAW_DATA_DIR / "material_cotton_FRED_PCOTTINDUSDM.csv"
OUTPUT_PATH     = SYNTHETIC_DATA_DIR / "material_agents.csv"

MIN_COVERAGE_PER_MATERIAL = 3
EXPECTED_AGENT_COUNT = 15

# fixed fx rate usd to eur for the selected anchor window
FX_USD_TO_EUR = 0.92

# anchor years used from the downloaded fred file
ANCHOR_START_YEAR = 2024
ANCHOR_END_YEAR   = 2025

# processing markup applied to raw fiber price to get fabric-ready price
# spinning, weaving, finishing typically adds 1.5x to 2x on top of raw fiber
PROCESSING_MARKUP = 1.7

# price ratios relative to cotton, kept as documented synthetic assumptions:
# polyester: calibrated ratio; bls synthetic-fibers ppi is trend context only
# linen:     calibrated ratio; indicative supplier prices support a broad premium
# denim:     calibrated ratio; fabric prices and manufacturing costs support a premium
# leather:   calibrated ratio; faostat cattle-hide producer prices are input context only
MATERIAL_RATIOS = {
    "cotton":    1.0,
    "polyester": 0.50,
    "linen":     2.50,
    "denim":     1.85,
    "leather":   10.0,
}

ALL_MATERIALS = ["cotton", "polyester", "linen", "denim", "leather"]

# base stock and restock amounts in kg used as reference before agent ratios applied
BASE_STOCK_KG = {
    "cotton":    1000,
    "polyester": 900,
    "linen":     400,
    "denim":     600,
    "leather":   100,
}

BASE_RESTOCK_KG = {
    "cotton":    1500,
    "polyester": 1200,
    "linen":     550,
    "denim":     800,
    "leather":   150,
}

# 15 agents covering 5 materials with minimum 3 agents per material
# cotton:    M01 M02 M03 M06 M10            = 5 agents
# polyester: M01 M04 M05 M06 M07 M11        = 6 agents
# linen:     M08 M09 M10 M11 M12            = 5 agents
# denim:     M03 M07 M12 M13 M14            = 5 agents
# leather:   M07 M08 M09 M13 M14 M15        = 6 agents
# specialty records each agent's main tradeoff

AGENT_SPECS = [
    {
        "agent_id":              "M01",
        "location":              "Helsinki",
        # large clean stock supports bulk orders without restocking
        "specialty":             "bulk cotton and polyester at fair price, biggest clean stock for large orders",
        "default_completion_days": 2,
        "lead_time_variance":    0.5,
        "defect_probability":    0.04,
        "initial_markup":        0.26,
        "materials": {
            "cotton":    {"stock_ratio": 2.2, "cost_ratio": 1.04, "restock_kg_ratio": 2.5, "restock_days": 3},
            "polyester": {"stock_ratio": 2.0, "cost_ratio": 1.00, "restock_kg_ratio": 2.2, "restock_days": 2},
        },
    },
    {
        "agent_id":              "M02",
        "location":              "Espoo",
        "specialty":             "premium quality cotton specialist, lowest defect rate",
        "default_completion_days": 1,
        "lead_time_variance":    0.3,
        "defect_probability":    0.01,
        "initial_markup":        0.35,
        "materials": {
            "cotton": {"stock_ratio": 0.8, "cost_ratio": 1.15, "restock_kg_ratio": 1.0, "restock_days": 2},
        },
    },
    {
        "agent_id":              "M03",
        "location":              "Tampere",
        "specialty":             "budget cotton and denim, largest stock but slower and higher defect",
        "default_completion_days": 3,
        "lead_time_variance":    1.0,
        "defect_probability":    0.06,
        "initial_markup":        0.22,
        "materials": {
            "cotton": {"stock_ratio": 2.0, "cost_ratio": 0.88, "restock_kg_ratio": 2.5, "restock_days": 4},
            "denim":  {"stock_ratio": 1.2, "cost_ratio": 0.82, "restock_kg_ratio": 1.5, "restock_days": 4},
        },
    },
    {
        "agent_id":              "M04",
        "location":              "Oulu",
        "specialty":             "cheapest polyester with fastest restock, slightly higher defect",
        "default_completion_days": 2,
        "lead_time_variance":    0.5,
        "defect_probability":    0.07,
        "initial_markup":        0.20,
        "materials": {
            "polyester": {"stock_ratio": 1.3, "cost_ratio": 0.85, "restock_kg_ratio": 1.8, "restock_days": 1},
        },
    },
    {
        "agent_id":              "M05",
        "location":              "Helsinki",
        "specialty":             "premium polyester, low defect, fast reliable delivery",
        "default_completion_days": 1,
        "lead_time_variance":    0.2,
        "defect_probability":    0.02,
        "initial_markup":        0.32,
        "materials": {
            "polyester": {"stock_ratio": 0.9, "cost_ratio": 1.18, "restock_kg_ratio": 1.0, "restock_days": 2},
        },
    },
    {
        "agent_id":              "M06",
        "location":              "Turku",
        "specialty":             "balanced cotton and polyester, mid-tier across all metrics",
        "default_completion_days": 2,
        "lead_time_variance":    0.2,
        "defect_probability":    0.04,
        "initial_markup":        0.27,
        "materials": {
            # slower cotton restock keeps fastest bids from concentrating on one supplier
            "cotton":    {"stock_ratio": 1.0, "cost_ratio": 1.0, "restock_kg_ratio": 1.2, "restock_days": 2},
            "polyester": {"stock_ratio": 1.0, "cost_ratio": 0.99, "restock_kg_ratio": 1.2, "restock_days": 2},
        },
    },
    {
        "agent_id":              "M07",
        "location":              "Jyvaskyla",
        "specialty":             "bulk polyester and denim at low margin, high volume lower quality",
        "default_completion_days": 3,
        "lead_time_variance":    1.2,
        "defect_probability":    0.06,
        "initial_markup":        0.18,
        "materials": {
            "polyester": {"stock_ratio": 1.8, "cost_ratio": 0.88, "restock_kg_ratio": 2.2, "restock_days": 2},
            "denim":     {"stock_ratio": 1.5, "cost_ratio": 0.90, "restock_kg_ratio": 2.0, "restock_days": 3},
            "leather":     {"stock_ratio": 0.9, "cost_ratio": 0.93, "restock_kg_ratio": 1.1, "restock_days": 8},
        },
    },
    {
        "agent_id":              "M08",
        "location":              "Lahti",
        "specialty":             "affordable linen with large stock, mid quality",
        "default_completion_days": 2,
        "lead_time_variance":    0.7,
        "defect_probability":    0.05,
        "initial_markup":        0.25,
        "materials": {
            # extra linen stock keeps large jacket orders feasible without making this supplier dominant
            "linen": {"stock_ratio": 2.0, "cost_ratio": 0.90, "restock_kg_ratio": 2.2, "restock_days": 5},
            "leather": { "stock_ratio": 0.8, "cost_ratio": 1.00, "restock_kg_ratio": 1.0, "restock_days": 7},
        },
    },
    {
        "agent_id":              "M09",
        "location":              "Helsinki",
        "specialty":             "fast linen and leather delivery, premium pricing and low defect",
        "default_completion_days": 1,
        "lead_time_variance":    0.3,
        "defect_probability":    0.02,
        "initial_markup":        0.33,
        "materials": {
            "linen":   {"stock_ratio": 0.7, "cost_ratio": 1.12, "restock_kg_ratio": 0.9, "restock_days": 4},
            "leather": {"stock_ratio": 0.9, "cost_ratio": 1.10, "restock_kg_ratio": 1.2, "restock_days": 7},
        },
    },
    {
        "agent_id":              "M10",
        "location":              "Tampere",
        "specialty":             "reliable cotton and linen mid-tier supplier",
        "default_completion_days": 2,
        "lead_time_variance":    0.6,
        "defect_probability":    0.04,
        "initial_markup":        0.26,
        "materials": {
            "cotton": {"stock_ratio": 1.1, "cost_ratio": 1.02, "restock_kg_ratio": 1.3, "restock_days": 3},
            "linen":  {"stock_ratio": 0.9, "cost_ratio": 0.88, "restock_kg_ratio": 1.0, "restock_days": 5},
        },
    },
    {
        "agent_id":              "M11",
        "location":              "Espoo",
        "specialty":             "polyester and linen combo, fast and reliable, slightly premium",
        "default_completion_days": 1,
        "lead_time_variance":    0.4,
        "defect_probability":    0.03,
        "initial_markup":        0.30,
        "materials": {
            "polyester": {"stock_ratio": 1.0, "cost_ratio": 1.08, "restock_kg_ratio": 1.2, "restock_days": 2},
            "linen":     {"stock_ratio": 0.8, "cost_ratio": 1.08, "restock_kg_ratio": 1.0, "restock_days": 4},
        },
    },
    {
        "agent_id":              "M12",
        "location":              "Oulu",
        "specialty":             "linen and denim specialist covering northern finland",
        "default_completion_days": 2,
        "lead_time_variance":    0.8,
        "defect_probability":    0.04,
        "initial_markup":        0.25,
        "materials": {
            "linen": {"stock_ratio": 1.1, "cost_ratio": 0.98, "restock_kg_ratio": 1.3, "restock_days": 5},
            "denim": {"stock_ratio": 1.0, "cost_ratio": 0.97, "restock_kg_ratio": 1.2, "restock_days": 4},
        },
    },
    {
        "agent_id":              "M13",
        "location":              "Turku",
        "specialty":             "premium denim and leather, lowest defect across both",
        "default_completion_days": 2,
        "lead_time_variance":    0.5,
        "defect_probability":    0.02,
        "initial_markup":        0.35,
        "materials": {
            "denim":   {"stock_ratio": 0.9, "cost_ratio": 1.12, "restock_kg_ratio": 1.0, "restock_days": 4},
            # extra leather stock let strict-quality mid-size orders clear
            "leather": {"stock_ratio": 2.0, "cost_ratio": 1.05, "restock_kg_ratio": 2.2, "restock_days": 6},
        },
    },
    {
        "agent_id":              "M14",
        "location":              "Helsinki",
        "specialty":             "largest leather stock and denim range, slower but high availability",
        "default_completion_days": 3,
        "lead_time_variance":    1.0,
        "defect_probability":    0.03,
        "initial_markup":        0.30,
        "materials": {
            # deep leather stock covers large orders without removing cost-quality tradeoffs
            "leather": {"stock_ratio": 5.0, "cost_ratio": 0.95, "restock_kg_ratio": 6.0, "restock_days": 7},
            "denim":   {"stock_ratio": 1.3, "cost_ratio": 0.94, "restock_kg_ratio": 1.5, "restock_days": 3},
        },
    },
    {
        "agent_id":              "M15",
        "location":              "Lahti",
        "specialty":             "budget leather only, cheapest price but slowest delivery and higher defect",
        "default_completion_days": 4,
        "lead_time_variance":    1.5,
        "defect_probability":    0.06,
        "initial_markup":        0.22,
        "materials": {
            # large restocks suit relaxed deadlines; low cost trades off against risk and speed
            "leather": {"stock_ratio": 3.5, "cost_ratio": 0.88, "restock_kg_ratio": 5.0, "restock_days": 7},
        },
    },
]


def load_cotton_anchor(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"raw cotton file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_columns = {"observation_date", "PCOTTINDUSDM"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns in raw cotton file: {missing}")

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["cents_per_lb"]     = pd.to_numeric(df["PCOTTINDUSDM"], errors="coerce")
    df = df.dropna(subset=["observation_date", "cents_per_lb"])

    if df.empty:
        raise ValueError("raw cotton file has no valid observations after cleaning")

    mask = (
        (df["observation_date"].dt.year >= ANCHOR_START_YEAR)
        & (df["observation_date"].dt.year <= ANCHOR_END_YEAR)
    )
    recent = df.loc[mask, "cents_per_lb"]

    if recent.empty:
        raise ValueError(
            f"no cotton price observations found for {ANCHOR_START_YEAR}-{ANCHOR_END_YEAR} in the file"
        )

    avg_cents_per_lb  = recent.mean()
    usd_per_kg_raw    = (avg_cents_per_lb / 100) * 2.20462
    eur_per_kg_raw    = usd_per_kg_raw * FX_USD_TO_EUR
    eur_per_kg_fabric = eur_per_kg_raw * PROCESSING_MARKUP

    print(f"cotton {ANCHOR_START_YEAR}-{ANCHOR_END_YEAR} avg: {avg_cents_per_lb:.2f} cents/lb ({len(recent)} observations)")
    print(f"raw cotton fiber:     EUR {eur_per_kg_raw:.3f}/kg")
    print(f"fabric-ready cotton:  EUR {eur_per_kg_fabric:.3f}/kg (after {PROCESSING_MARKUP}x processing markup)")

    return round(eur_per_kg_fabric, 3)


def derive_material_prices(cotton_anchor_eur):
    return {mat: round(cotton_anchor_eur * ratio, 3) for mat, ratio in MATERIAL_RATIOS.items()}


def validate_material_config():
    expected = set(ALL_MATERIALS)

    for name, values in {
        "MATERIAL_RATIOS": MATERIAL_RATIOS,
        "BASE_STOCK_KG": BASE_STOCK_KG,
        "BASE_RESTOCK_KG": BASE_RESTOCK_KG,
    }.items():
        missing = expected - set(values)
        extra = set(values) - expected

        if missing or extra:
            raise ValueError(
                f"{name} mismatch. missing={sorted(missing)}, extra={sorted(extra)}"
            )


def validate_agent_specs(specs):
    if len(specs) != EXPECTED_AGENT_COUNT:
        raise ValueError(f"expected {EXPECTED_AGENT_COUNT} material agents, found {len(specs)}")

    seen_ids = set()
    required_agent_keys = {
        "agent_id",
        "location",
        "specialty",
        "default_completion_days",
        "lead_time_variance",
        "defect_probability",
        "initial_markup",
        "materials",
    }

    for spec in specs:
        missing_agent_keys = required_agent_keys - set(spec)
        if missing_agent_keys:
            raise ValueError(f"agent spec missing required keys: {missing_agent_keys}")

        agent_id = spec["agent_id"]

        if agent_id in seen_ids:
            raise ValueError(f"duplicate agent_id: {agent_id}")
        seen_ids.add(agent_id)

        if not spec["materials"]:
            raise ValueError(f"{agent_id}: materials cannot be empty")

        if spec["default_completion_days"] <= 0:
            raise ValueError(f"{agent_id}: default_completion_days must be positive")

        if spec["lead_time_variance"] < 0:
            raise ValueError(f"{agent_id}: lead_time_variance cannot be negative")

        if not 0 <= spec["defect_probability"] <= 1:
            raise ValueError(f"{agent_id}: defect_probability must be between 0 and 1")

        if spec["initial_markup"] < 0.10:
            raise ValueError(f"{agent_id}: initial_markup should be at least min_profit_margin (0.10)")

        for mat, m in spec["materials"].items():
            if mat not in ALL_MATERIALS:
                raise ValueError(f"{agent_id}: unknown material type: {mat}")

            for key in ["stock_ratio", "cost_ratio", "restock_kg_ratio", "restock_days"]:
                if key not in m:
                    raise ValueError(f"{agent_id}: missing {key} for {mat}")

            if m["stock_ratio"] <= 0:
                raise ValueError(f"{agent_id}: stock_ratio for {mat} must be positive")

            if m["cost_ratio"] <= 0:
                raise ValueError(f"{agent_id}: cost_ratio for {mat} must be positive")

            if m["restock_kg_ratio"] <= 0:
                raise ValueError(f"{agent_id}: restock_kg_ratio for {mat} must be positive")

            if m["restock_days"] <= 0:
                raise ValueError(f"{agent_id}: restock_days for {mat} must be positive")

    print(f"agent spec validation passed for {len(specs)} agents")


def validate_coverage(specs):
    coverage = {mat: 0 for mat in ALL_MATERIALS}
    for spec in specs:
        for mat in spec["materials"]:
            coverage[mat] += 1

    print("material coverage:")
    all_ok = True
    for mat, count in coverage.items():
        status = "ok" if count >= MIN_COVERAGE_PER_MATERIAL else f"FAIL need at least {MIN_COVERAGE_PER_MATERIAL}"
        print(f"  {mat:12s}: {count} agents  [{status}]")
        if count < MIN_COVERAGE_PER_MATERIAL:
            all_ok = False

    if not all_ok:
        raise ValueError("coverage check failed, fix agent specs before proceeding")

    print("all materials covered by at least 3 agents")


def validate_no_dominance(specs):
    # if one agent is at least as good on price, quality, speed, restock
    # and stock for a shared material, and strictly better somewhere, the
    # weaker agent can never win that material and is dead weight
    problems = []
    for mat in ALL_MATERIALS:
        holders = [s for s in specs if mat in s["materials"]]
        for a in holders:
            for b in holders:
                if a["agent_id"] == b["agent_id"]:
                    continue
                ma, mb = a["materials"][mat], b["materials"][mat]
                as_good = (
                    ma["cost_ratio"] <= mb["cost_ratio"]
                    and a["defect_probability"] <= b["defect_probability"]
                    and a["default_completion_days"] <= b["default_completion_days"]
                    and ma["restock_days"] <= mb["restock_days"]
                    and ma["stock_ratio"] >= mb["stock_ratio"]
                )
                strictly_better = (
                    ma["cost_ratio"] < mb["cost_ratio"]
                    or a["defect_probability"] < b["defect_probability"]
                    or a["default_completion_days"] < b["default_completion_days"]
                    or ma["restock_days"] < mb["restock_days"]
                    or ma["stock_ratio"] > mb["stock_ratio"]
                )
                if as_good and strictly_better:
                    problems.append(f"{a['agent_id']} dominates {b['agent_id']} on {mat}")

    if problems:
        raise ValueError("dominance check failed:\n" + "\n".join(problems))

    print("no agent fully dominates another on any shared material")


def build_agent(spec, base_prices):
    materials = spec["materials"]

    row = {
        "agent_id":                spec["agent_id"],
        "stage":                   "material",
        "location":                spec["location"],
        "specialty":               spec["specialty"],
        "material_types":          "|".join(sorted(materials.keys())),
        "default_completion_days": spec["default_completion_days"],
        "lead_time_variance":      spec["lead_time_variance"],
        "min_profit_margin":       0.10,
        "initial_markup":          spec["initial_markup"],
        "defect_probability":      spec["defect_probability"],
        "reputation_score":        0.5,
        # fx rate kept for reproducibility and paper citation
        "fx_usd_to_eur":           FX_USD_TO_EUR,
    }

    # null means agent does not handle this material at all
    # this is semantically different from 0 (which would mean out of stock)
    for mat in ALL_MATERIALS:
        if mat in materials:
            m = materials[mat]
            row[f"{mat}_available_kg"]         = round(BASE_STOCK_KG[mat]   * m["stock_ratio"])
            row[f"{mat}_base_cost_eur_per_kg"] = round(base_prices[mat]     * m["cost_ratio"], 3)
            row[f"{mat}_restock_amount_kg"]    = round(BASE_RESTOCK_KG[mat] * m["restock_kg_ratio"])
            row[f"{mat}_restock_time_days"]    = m["restock_days"]
        else:
            row[f"{mat}_available_kg"]         = None
            row[f"{mat}_base_cost_eur_per_kg"] = None
            row[f"{mat}_restock_amount_kg"]    = None
            row[f"{mat}_restock_time_days"]    = None

    return row


def synthesize(raw_path, output_path):
    print(f"loading cotton anchor from {raw_path}\n")
    cotton_anchor_eur = load_cotton_anchor(raw_path)

    validate_material_config()

    print("\nderived base prices (eur/kg):")
    base_prices = derive_material_prices(cotton_anchor_eur)
    for mat, price in base_prices.items():
        print(f"  {mat:12s}: EUR {price:.3f}")

    print()
    validate_agent_specs(AGENT_SPECS)
    validate_coverage(AGENT_SPECS)
    validate_no_dominance(AGENT_SPECS)
    print()

    agents = []
    for spec in AGENT_SPECS:
        agent = build_agent(spec, base_prices)
        agents.append(agent)
        print(f"built {agent['agent_id']} | {agent['specialty'][:55]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agents).to_csv(output_path, index=False)
    print(f"\ndone. {len(agents)} material agents saved to {output_path}")


if __name__ == "__main__":
    synthesize(RAW_COTTON_PATH, OUTPUT_PATH)
