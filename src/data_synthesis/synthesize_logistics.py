import pandas as pd

from config import SYNTHETIC_DATA_DIR

OUTPUT_PATH = SYNTHETIC_DATA_DIR / "logistics_agents.csv"

MIN_CITY_COVERAGE = 4
MIN_PROFIT_MARGIN = 0.10

# delivery cities used by orders
# every city must be covered by multiple agents to keep bidding competitive
ALL_CITIES = ["Oulu", "Helsinki", "Tampere", "Turku", "Jyvaskyla", "Lahti", "Espoo"]

# 12 logistics agents with tradeoffs instead of obvious winners and losers
# large carriers have wider coverage and higher capacity
# regional carriers are cheaper or faster in selected cities
# premium carriers have lower damage and delay, but higher cost
# budget carriers remain useful for flexible low-budget orders
AGENT_SPECS = [
    {
        "agent_id": "L01",
        "location": "Helsinki",
        "specialty": "national carrier, balanced pricing across all cities",
        "service_regions": ["Oulu", "Helsinki", "Tampere", "Turku", "Jyvaskyla", "Lahti", "Espoo"],
        "vehicle_capacity_kg": 1800,
        "current_load_kg": 450,
        "base_delivery_cost": 42.0,
        "cost_per_kg": 0.78,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 3,
        "delay_probability": 0.06,
        "damage_probability": 0.035,
        "initial_markup": 0.25,
    },
    {
        "agent_id": "L02",
        "location": "Tampere",
        "specialty": "central finland carrier with fast 2-day intercity service",
        "service_regions": ["Helsinki", "Tampere", "Turku", "Jyvaskyla", "Lahti", "Espoo"],
        "vehicle_capacity_kg": 1200,
        "current_load_kg": 280,
        "base_delivery_cost": 38.0,
        "cost_per_kg": 0.82,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 2,
        "delay_probability": 0.04,
        "damage_probability": 0.030,
        "initial_markup": 0.27,
    },
    {
        "agent_id": "L03",
        "location": "Oulu",
        "specialty": "northern budget carrier, cheapest base cost, wins small northern orders",
        "service_regions": ["Oulu", "Jyvaskyla", "Tampere", "Lahti"],
        "vehicle_capacity_kg": 900,
        "current_load_kg": 170,
        "base_delivery_cost": 30.0,
        "cost_per_kg": 0.88,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 3,
        "delay_probability": 0.05,
        "damage_probability": 0.025,
        "initial_markup": 0.23,
    },
    {
        "agent_id": "L04",
        "location": "Turku",
        "specialty": "southwest carrier, wins small and medium orders in southern cities",
        "service_regions": ["Turku", "Helsinki", "Espoo", "Tampere", "Lahti"],
        "vehicle_capacity_kg": 1000,
        "current_load_kg": 360,
        "base_delivery_cost": 35.0,
        "cost_per_kg": 0.80,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 2,
        "delay_probability": 0.05,
        "damage_probability": 0.032,
        "initial_markup": 0.25,
    },
    {
        "agent_id": "L05",
        "location": "Espoo",
        "specialty": "premium carrier with lowest damage risk, wins large southern orders",
        "service_regions": ["Espoo", "Helsinki"],
        "vehicle_capacity_kg": 850,
        "current_load_kg": 180,
        "base_delivery_cost": 55.0,
        "cost_per_kg": 0.68,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 2,
        "delay_probability": 0.025,
        "damage_probability": 0.010,
        "initial_markup": 0.32,
    },
    {
        "agent_id": "L06",
        "location": "Lahti",
        "specialty": "northern budget carrier, lowest base cost, wins small central and northern orders",
        "service_regions": ["Lahti", "Oulu", "Tampere", "Jyvaskyla"],
        "vehicle_capacity_kg": 1600,
        "current_load_kg": 300,
        "base_delivery_cost": 27.0,
        "cost_per_kg": 0.84,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 4,
        "delay_probability": 0.08,
        "damage_probability": 0.045,
        "initial_markup": 0.20,
    },
    {
        "agent_id": "L07",
        "location": "Helsinki",
        "specialty": "express carrier, fastest at 1.5 days intercity, wins urgent small southern orders",
        "service_regions": ["Helsinki", "Espoo", "Turku", "Tampere", "Lahti"],
        "vehicle_capacity_kg": 550,
        "current_load_kg": 90,
        "base_delivery_cost": 62.0,
        "cost_per_kg": 0.76,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 1.5,
        "delay_probability": 0.020,
        "damage_probability": 0.018,
        "initial_markup": 0.36,
    },
    {
        "agent_id": "L08",
        "location": "Jyvaskyla",
        "specialty": "northern bulk carrier, high capacity, wins large central and northern orders",
        "service_regions": ["Jyvaskyla", "Tampere", "Oulu", "Lahti"],
        "vehicle_capacity_kg": 2500,
        "current_load_kg": 1100,
        "base_delivery_cost": 48.0,
        "cost_per_kg": 0.74,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 3.5,
        "delay_probability": 0.07,
        "damage_probability": 0.040,
        "initial_markup": 0.24,
    },
    {
        "agent_id": "L09",
        "location": "Oulu",
        "specialty": "reliable northern premium carrier, low damage, competes closely with L08",
        "service_regions": ["Oulu", "Jyvaskyla", "Tampere", "Helsinki"],
        "vehicle_capacity_kg": 700,
        "current_load_kg": 120,
        "base_delivery_cost": 50.0,
        "cost_per_kg": 0.74,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 3,
        "delay_probability": 0.030,
        "damage_probability": 0.015,
        "initial_markup": 0.31,
    },
    {
        "agent_id": "L10",
        "location": "Tampere",
        "specialty": "flexible low-load carrier, balanced across central cities",
        "service_regions": ["Tampere", "Helsinki", "Espoo", "Lahti", "Turku", "Jyvaskyla"],
        "vehicle_capacity_kg": 1100,
        "current_load_kg": 80,
        "base_delivery_cost": 40.0,
        "cost_per_kg": 0.79,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 2.5,
        "delay_probability": 0.045,
        "damage_probability": 0.028,
        "initial_markup": 0.26,
    },
    {
        "agent_id": "L11",
        "location": "Turku",
        "specialty": "southern specialist, wins medium orders in Turku, Helsinki, Espoo",
        "service_regions": ["Turku", "Helsinki", "Espoo"],
        "vehicle_capacity_kg": 2000,
        "current_load_kg": 1450,
        "base_delivery_cost": 42.0,
        "cost_per_kg": 0.76,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 3,
        "delay_probability": 0.075,
        "damage_probability": 0.034,
        "initial_markup": 0.22,
    },
    {
        "agent_id": "L12",
        "location": "Espoo",
        "specialty": "small careful southern carrier, second lowest damage, competitive on light loads",
        "service_regions": ["Espoo", "Helsinki", "Lahti", "Turku"],
        "vehicle_capacity_kg": 450,
        "current_load_kg": 60,
        "base_delivery_cost": 44.0,
        "cost_per_kg": 0.84,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 2,
        "delay_probability": 0.035,
        "damage_probability": 0.012,
        "initial_markup": 0.28,
    },
]


def validate_agent_specs(specs):
    expected_count = 12
    if len(specs) != expected_count:
        raise ValueError(f"expected {expected_count} logistics agents, found {len(specs)}")

    seen_ids = set()
    required_keys = {
        "agent_id",
        "location",
        "specialty",
        "service_regions",
        "vehicle_capacity_kg",
        "current_load_kg",
        "base_delivery_cost",
        "cost_per_kg",
        "default_completion_same_city_days",
        "default_completion_intercity_days",
        "delay_probability",
        "damage_probability",
        "initial_markup",
    }

    for spec in specs:
        missing = required_keys - set(spec)
        if missing:
            raise ValueError(f"agent spec missing required keys: {missing}")

        agent_id = spec["agent_id"]

        if agent_id in seen_ids:
            raise ValueError(f"duplicate agent_id: {agent_id}")
        seen_ids.add(agent_id)

        if not spec["service_regions"]:
            raise ValueError(f"{agent_id}: service_regions cannot be empty")

        unknown_cities = set(spec["service_regions"]) - set(ALL_CITIES)
        if unknown_cities:
            raise ValueError(f"{agent_id}: unknown service regions: {sorted(unknown_cities)}")

        if spec["location"] not in ALL_CITIES:
            raise ValueError(f"{agent_id}: unknown location: {spec['location']}")

        if spec["vehicle_capacity_kg"] <= 0:
            raise ValueError(f"{agent_id}: vehicle_capacity_kg must be positive")

        if spec["current_load_kg"] < 0:
            raise ValueError(f"{agent_id}: current_load_kg cannot be negative")

        if spec["current_load_kg"] > spec["vehicle_capacity_kg"]:
            raise ValueError(f"{agent_id}: current_load_kg cannot exceed vehicle_capacity_kg")

        if spec["base_delivery_cost"] <= 0:
            raise ValueError(f"{agent_id}: base_delivery_cost must be positive")

        if spec["cost_per_kg"] <= 0:
            raise ValueError(f"{agent_id}: cost_per_kg must be positive")

        if spec["default_completion_same_city_days"] <= 0:
            raise ValueError(f"{agent_id}: default_completion_same_city_days must be positive")

        if spec["default_completion_intercity_days"] <= 0:
            raise ValueError(f"{agent_id}: default_completion_intercity_days must be positive")

        if not 0 <= spec["delay_probability"] <= 1:
            raise ValueError(f"{agent_id}: delay_probability must be between 0 and 1")

        if not 0 <= spec["damage_probability"] <= 1:
            raise ValueError(f"{agent_id}: damage_probability must be between 0 and 1")

        if spec["initial_markup"] < MIN_PROFIT_MARGIN:
            raise ValueError(f"{agent_id}: initial_markup should be at least min_profit_margin ({MIN_PROFIT_MARGIN})")

    print(f"agent spec validation passed for {len(specs)} agents")


def validate_city_coverage(specs):
    coverage = {city: 0 for city in ALL_CITIES}

    for spec in specs:
        for city in spec["service_regions"]:
            coverage[city] += 1

    print("city coverage:")
    all_ok = True

    for city, count in coverage.items():
        status = "ok" if count >= MIN_CITY_COVERAGE else f"FAIL need at least {MIN_CITY_COVERAGE}"
        print(f"  {city:10s}: {count} agents  [{status}]")

        if count < MIN_CITY_COVERAGE:
            all_ok = False

    if not all_ok:
        raise ValueError("city coverage check failed, fix logistics agent specs")

    print("all cities covered by enough logistics agents")


def validate_no_dominance(specs):
    # only compare carriers when one covers every city served by the other
    # dominance then requires no worse price, safety, or speed
    # at least one edge must be strict
    problems = []
    for a in specs:
        for b in specs:
            if a["agent_id"] == b["agent_id"]:
                continue
            if not set(a["service_regions"]) >= set(b["service_regions"]):
                continue
            as_good = (
                a["base_delivery_cost"] <= b["base_delivery_cost"]
                and a["cost_per_kg"] <= b["cost_per_kg"]
                and a["damage_probability"] <= b["damage_probability"]
                and a["delay_probability"] <= b["delay_probability"]
                and a["default_completion_intercity_days"] <= b["default_completion_intercity_days"]
            )
            strictly_better = (
                a["base_delivery_cost"] < b["base_delivery_cost"]
                or a["cost_per_kg"] < b["cost_per_kg"]
                or a["damage_probability"] < b["damage_probability"]
                or a["delay_probability"] < b["delay_probability"]
                or a["default_completion_intercity_days"] < b["default_completion_intercity_days"]
            )
            if as_good and strictly_better:
                problems.append(f"{a['agent_id']} dominates {b['agent_id']}")

    if problems:
        raise ValueError("dominance check failed:\n" + "\n".join(problems))

    print("no carrier fully dominates another within shared coverage")


def build_agent(spec):
    return {
        "agent_id":                            spec["agent_id"],
        "stage":                               "logistics",
        "location":                            spec["location"],
        "specialty":                           spec["specialty"],
        "service_regions":                     "|".join(sorted(spec["service_regions"])),
        "base_delivery_cost":                  spec["base_delivery_cost"],
        "cost_per_kg":                         spec["cost_per_kg"],
        "default_completion_same_city_days":   spec["default_completion_same_city_days"],
        "default_completion_intercity_days":   spec["default_completion_intercity_days"],
        "delay_probability":                   spec["delay_probability"],
        "damage_probability":                  spec["damage_probability"],
        "min_profit_margin":                   MIN_PROFIT_MARGIN,
        "initial_markup":                      spec["initial_markup"],
        "reputation_score":                    0.5,
    }


def synthesize(output_path):
    validate_agent_specs(AGENT_SPECS)
    validate_city_coverage(AGENT_SPECS)
    validate_no_dominance(AGENT_SPECS)
    print()

    agents = []
    for spec in AGENT_SPECS:
        agent = build_agent(spec)
        agents.append(agent)
        print(
            f"built {agent['agent_id']} | regions: {agent['service_regions']} | "
            f"damage: {agent['damage_probability']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agents).to_csv(output_path, index=False)

    print(f"\ndone. {len(agents)} logistics agents saved to {output_path}")


if __name__ == "__main__":
    synthesize(OUTPUT_PATH)
