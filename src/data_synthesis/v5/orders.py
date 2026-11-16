"""v5 orders and budgets, generated inside the provisional v5 window

  PYTHONPATH=. python -m src.data_synthesis.v5.orders

the order design is v4's, unchanged: scenario mix, priorities, quantity bands,
deadlines, budget bands, defect and reputation ranges, the weekday-weighted day
sampler and the same two seeded random streams. those constants are copied from
src/data_synthesis/synthesize_orders.py, which is not imported, and tests check the
copies still match. all twelve v4 product-material pairs are generated. two things
differ, both because the prices differ:

  the window is the provisional v5 window, fixed, with no run-day fallback
  the budget anchor is priced from the v5 profiles on the order's own date

budget anchor, per order, on order_created's date, keeping v4's structure of adding
the stage prices and applying the priority markup to the chain:
  material      quantity x kg per unit x mean agent USD/kg that day x material stage factor
  manufacturer  quantity x mean agent USD unit cost x manufacturer stage factor
  logistics     mean v5 route price over the agents serving deliver_to whose weight range
                covers the shipment, at the mean distance from the manufacturers that make the
                product; the route price already carries damage risk and the priority markup.
                where no serving agent covers the weight, no price is invented: the term is
                left out and the order is counted in the metadata
  market price  (material + manufacturer) x (1 + priority markup) + logistics
"""

import random
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data_synthesis.v5 import logistics as logistics_module
from src.data_synthesis.v5 import materials as materials_module
from src.data_synthesis.v5 import products as products_module
from src.data_synthesis.v5.settings import (
    AGENT_CALIBRATION_PATH, ALL_MATERIALS, ALL_PRODUCTS, BETA_BASIS, CREATION_WINDOW_DAYS,
    DISTANCE_MATRIX_PATH, FIXED_PRICE_MATERIALS, PRIORITY_MARKUP, SEED, SHARED_FUEL_BASIS,
    SHARED_FUEL_SERIES_ID, TOTAL_ORDERS, V4_DIR, V5_DIR, WINDOW_BASIS, WINDOW_END_EXCLUSIVE, WINDOW_START,
    WINDOW_STATUS, rel, sha256_file, write_csv, write_json,
)

OUTPUT_PATH = V5_DIR / "orders.csv"
METADATA_PATH = V5_DIR / "orders_metadata.json"

# ── copied verbatim from src/data_synthesis/synthesize_orders.py ──────────────
# PRIORITY_MARKUP is copied into settings.py, because the logistics price uses it too
WEEKDAY_WEIGHTS = {0: 1.0, 1: 1.1, 2: 1.2, 3: 1.2, 4: 1.3, 5: 0.8, 6: 0.4}
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
AVG_WORKLOAD_MARKUP = 0.03
# ── end of the v4 copies ──────────────────────────────────────────────────────

RARE_PAIRS = (("jacket", "leather"), ("jeans", "denim"), ("skirt", "linen"))
TIGHT_LOW_BUDGET_BAND = (0.93, 1.03)
ID_WIDTH = max(3, len(str(TOTAL_ORDERS)))


class BudgetEstimator:
    """the expected chain market price for one order, from the v5 profiles"""

    def __init__(self, materials, reference, manufacturers, logistics_profile, route_cost):
        self.route = route_cost
        self.orders_without_weight_eligible_logistics = 0
        self.material_ratio_mean = {
            m: materials[materials[f"{m}_price_ratio_to_reference"].notna()][
                f"{m}_price_ratio_to_reference"].mean() for m in ALL_MATERIALS}
        self.reference = {(r.material, r.date): r.usd_per_kg for r in reference.itertuples()}
        self.unit_cost_mean = {p: manufacturers[f"unit_cost_usd_{p}"].dropna().mean()
                               for p in ALL_PRODUCTS}
        self.stage_factor = {
            "material": (1 + materials["defect_probability"].mean())
                        * (1 + materials["initial_markup"].mean()),
            "manufacturer": (1 + manufacturers["defect_probability"].mean())
                            * (1 + manufacturers["initial_markup"].mean()) * (1 + AVG_WORKLOAD_MARKUP),
        }
        self.agents_by_city = {}
        for row in logistics_profile.itertuples():
            for city in str(row.service_regions).split("|"):
                self.agents_by_city.setdefault(city, []).append(row.agent_id)
        self.mean_distance = {}
        makers = {p: [r.location for r in manufacturers.itertuples()
                      if p in str(r.product_types).split("|")] for p in ALL_PRODUCTS}
        cities = {d for (_, d) in route_cost.distance_km}
        for p, locations in makers.items():
            for city in cities:
                self.mean_distance[(p, city)] = sum(
                    route_cost.distance_km[(loc, city)] for loc in locations) / len(locations)

    def estimate(self, product_id, material_type, quantity, material_kg_per_unit,
                 finished_weight_kg_per_unit, deliver_to, day, priority):
        material_cost = quantity * material_kg_per_unit * self.reference[(material_type, day.isoformat())] \
            * self.material_ratio_mean[material_type]
        manufacturing_cost = quantity * self.unit_cost_mean[product_id]
        weight = quantity * finished_weight_kg_per_unit
        distance = self.mean_distance[(product_id, deliver_to)]
        agents = [a for a in self.agents_by_city[deliver_to] if self.route.eligible_weight(a, weight)]
        if agents:
            logistics = sum(self.route.price_for_distance(a, weight, distance, day, priority)
                            for a in agents) / len(agents)
        else:
            logistics = 0.0
            self.orders_without_weight_eligible_logistics += 1
        goods = (material_cost * self.stage_factor["material"]
                 + manufacturing_cost * self.stage_factor["manufacturer"]) * (1 + PRIORITY_MARKUP[priority])
        return {
            "material": material_cost * self.stage_factor["material"],
            "manufacturer": manufacturing_cost * self.stage_factor["manufacturer"],
            "logistics": logistics,
            "logistics_agents_averaged": len(agents),
            "market_price": goods + logistics,
        }


class OrderGenerator:
    """v4's generation loop, run over the v5 window and budget anchor"""

    def __init__(self, pairs, estimator, cities, seed=SEED):
        self.pairs = pairs
        self.estimator = estimator
        self.cities = sorted(cities)
        self.rng = random.Random(seed)
        self.day_rng = random.Random(seed + 1)
        self.day_weights = [WEEKDAY_WEIGHTS[(WINDOW_START + timedelta(days=i)).weekday()]
                            for i in range(CREATION_WINDOW_DAYS)]
        self.rare_pairs = [p for p in pairs if (p["product_id"], p["material_type"]) in RARE_PAIRS]

    def pick_quantity(self):
        bands = list(QUANTITY_BANDS)
        band = self.rng.choices(bands, weights=[QUANTITY_BANDS[b]["weight"] for b in bands], k=1)[0]
        lo, hi = QUANTITY_BANDS[band]["range"]
        return self.rng.randint(lo, hi)

    def pick_priority(self):
        return self.rng.choices(list(PRIORITY_WEIGHTS), weights=list(PRIORITY_WEIGHTS.values()), k=1)[0]

    def scenario(self, name, counter):
        rng = self.rng
        if name == "random_realistic":
            return rng.choice(self.pairs), self.pick_priority(), {}
        if name == "tight_deadline":
            return rng.choice(self.pairs), "urgent", {"quantity_override": rng.randint(400, 900),
                                                      "deadline_override": (5, 8)}
        if name == "low_budget":
            return rng.choice(self.pairs), "normal", {"budget_override": BUDGET_MULTIPLIERS["low_budget"]}
        if name == "large_quantity":
            return rng.choice(self.pairs), "normal", {"quantity_override": rng.randint(1200, 2000)}
        if name == "high_quality":
            return rng.choice(self.pairs), "urgent", {"defect_override": DEFECT_TOLERANCE["high_quality"]}
        if name == "rare_combo":
            pair = self.rare_pairs[counter % len(self.rare_pairs)] if self.rare_pairs else rng.choice(self.pairs)
            return pair, "normal", {}
        if name == "tight_low_budget":
            return rng.choice(self.pairs), "urgent", {"quantity_override": rng.randint(300, 800),
                                                      "deadline_override": (6, 9),
                                                      "budget_override": TIGHT_LOW_BUDGET_BAND}
        raise ValueError(f"unknown scenario {name!r}")

    def make_order(self, index, pair, priority, scenario_type, budget_override=None,
                   quantity_override=None, defect_override=None, deadline_override=None):
        rng = self.rng
        quantity = quantity_override if quantity_override is not None else self.pick_quantity()
        day_index = self.day_rng.choices(range(CREATION_WINDOW_DAYS), weights=self.day_weights, k=1)[0]
        created = datetime.combine(WINDOW_START, datetime.min.time()) + timedelta(
            days=day_index, hours=rng.randint(8, 17))
        lo, hi = deadline_override if deadline_override is not None else DEADLINE_DAYS[priority]
        deadline = created + timedelta(days=rng.randint(lo, hi))
        deliver_to = rng.choice(self.cities)
        estimate = self.estimator.estimate(
            pair["product_id"], pair["material_type"], quantity, pair["material_kg_per_unit"],
            pair["finished_weight_kg_per_unit"], deliver_to, created.date(), priority)
        band_lo, band_hi = budget_override if budget_override is not None else BUDGET_MULTIPLIERS[priority]
        budget = round(estimate["market_price"] * rng.uniform(band_lo, band_hi), 2)
        d_lo, d_hi = defect_override if defect_override is not None else DEFECT_TOLERANCE[priority]
        r_lo, r_hi = REPUTATION_REQUIREMENTS[priority]
        return {
            "order_id": f"ORD-{index + 1:0{ID_WIDTH}d}",
            "order_created": created.strftime("%Y-%m-%dT%H:%M:%S"),
            "product_id": pair["product_id"],
            "quantity": quantity,
            "material_type": pair["material_type"],
            "budget_usd": budget,
            "deadline": deadline.strftime("%Y-%m-%d"),
            "deliver_to": deliver_to,
            "max_defect_rate": round(rng.uniform(d_lo, d_hi), 3),
            "min_reputation_score": round(rng.uniform(r_lo, r_hi), 2),
            "priority": priority,
            "scenario_type": scenario_type,
            "dataset_version": "v5",
            "currency": "USD",
            "pricing_date": created.date().isoformat(),
            "unit": "per_order",
        }

    def generate(self, total=TOTAL_ORDERS):
        counts = {s: int(total * frac) for s, frac in SCENARIO_MIX.items()}
        counts["random_realistic"] += total - sum(counts.values())
        orders, index = [], 0
        for name, count in counts.items():
            for i in range(count):
                pair, priority, overrides = self.scenario(name, i)
                orders.append(self.make_order(index, pair, priority, name, **overrides))
                index += 1
        return orders


def validate(orders, pairs, cities):
    combos = {(p["product_id"], p["material_type"]) for p in pairs}
    errors, seen = [], set()
    for o in orders:
        if o["order_id"] in seen:
            errors.append(f"{o['order_id']}: duplicate order_id")
        seen.add(o["order_id"])
        if (o["product_id"], o["material_type"]) not in combos:
            errors.append(f"{o['order_id']}: pair outside the v5 product table")
        if o["deliver_to"] not in cities:
            errors.append(f"{o['order_id']}: unknown city {o['deliver_to']}")
        created = datetime.strptime(o["order_created"], "%Y-%m-%dT%H:%M:%S").date()
        if not WINDOW_START <= created < WINDOW_END_EXCLUSIVE:
            errors.append(f"{o['order_id']}: created outside the v5 window")
        if datetime.strptime(o["deadline"], "%Y-%m-%d").date() <= created:
            errors.append(f"{o['order_id']}: deadline not after order_created")
        if o["budget_usd"] <= 0 or o["quantity"] <= 0:
            errors.append(f"{o['order_id']}: non-positive budget or quantity")
    if errors:
        raise ValueError(f"v5 order validation failed with {len(errors)} errors: {errors[:10]}")


def load_inputs(logistics_profile_path=logistics_module.PROFILE_PATH):
    materials = pd.read_csv(materials_module.PROFILE_PATH)
    reference = pd.read_csv(materials_module.REFERENCE_PATH)
    manufacturers = pd.read_csv(V5_DIR / "manufacturer_agents.csv")
    logistics_profile = pd.read_csv(logistics_profile_path)
    products = pd.read_csv(products_module.OUTPUT_PATH)
    route = logistics_module.RouteCost(profile_path=logistics_profile_path)
    return materials, reference, manufacturers, logistics_profile, products, route


def generate_frame(logistics_profile_path=logistics_module.PROFILE_PATH):
    materials, reference, manufacturers, logistics_profile, products, route = load_inputs(logistics_profile_path)
    estimator = BudgetEstimator(materials, reference, manufacturers, logistics_profile, route)
    cities = set()
    for regions in logistics_profile["service_regions"]:
        cities.update(str(regions).split("|"))
    pairs = products.to_dict("records")
    orders = OrderGenerator(pairs, estimator, cities).generate()
    validate(orders, pairs, cities)
    return pd.DataFrame(orders), estimator, pairs


def weight_domains(logistics_profile):
    """every agent's evidenced weight range, only for a profile that carries minimums"""
    if "min_shipment_weight_kg" not in logistics_profile.columns:
        return {}
    return {"weight_domains": {
        r["agent_id"]: {"min_kg": float(r["min_shipment_weight_kg"]), "max_kg": float(r["max_shipment_weight_kg"]),
                        "max_inclusive": bool(r["max_weight_inclusive"]), "reading": r["calibration_reading_id"]}
        for r in logistics_profile.to_dict("records")}}


def metadata(frame, estimator, pairs, logistics_profile_path=logistics_module.PROFILE_PATH,
             calibration_path=AGENT_CALIBRATION_PATH):
    v4_products = pd.read_csv(V4_DIR / "products.csv")
    v4_orders = pd.read_csv(V4_DIR / "orders.csv", usecols=["material_type"])
    reference = pd.read_csv(materials_module.REFERENCE_PATH)
    logistics_profile = pd.read_csv(logistics_profile_path)
    created = pd.to_datetime(frame["order_created"])
    fixed = {}
    for material in FIXED_PRICE_MATERIALS:
        values = reference[reference["material"] == material]["usd_per_kg"].unique()
        if len(values) != 1:
            raise ValueError(f"{material} is declared fixed but has {len(values)} prices in the window")
        fixed[material] = float(values[0])
    inputs = [products_module.OUTPUT_PATH, materials_module.PROFILE_PATH, materials_module.REFERENCE_PATH,
              V5_DIR / "manufacturer_agents.csv", logistics_profile_path, calibration_path,
              DISTANCE_MATRIX_PATH, V5_DIR.parent.parent / "raw" / "v5" / "manifest.json",
              V5_DIR.parent.parent / "raw" / "v5" / "logistics" / "manifest.json"]
    v4_read = [V4_DIR / name for name in ("products.csv", "material_agents.csv", "manufacturer_agents.csv",
                                           "logistics_agents.csv")]
    caps = logistics_profile[(logistics_profile["calibration_corner"] == "parcel")
                             & logistics_profile["max_shipment_weight_kg"].notna()]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": "v5",
        "currency": "USD",
        "seed": SEED,
        "total_orders": len(frame),
        "base_date": WINDOW_START.isoformat(),
        "creation_window_days": CREATION_WINDOW_DAYS,
        "window": {
            "start": WINDOW_START.isoformat(),
            "end_exclusive": WINDOW_END_EXCLUSIVE.isoformat(),
            "first_order_created": created.min().isoformat(),
            "last_order_created": created.max().isoformat(),
            "status": WINDOW_STATUS,
            "basis": WINDOW_BASIS,
        },
        "material_coverage": {
            "all_v4_materials_covered": True,
            "materials": list(ALL_MATERIALS),
            "product_material_pairs": {"v4": int(len(v4_products)), "v5": len(pairs)},
            "FIXED_PRICE_NOTICE": (
                "polyester's price is FIXED at " + ", ".join(f"{v:.4f} USD/kg" for v in fixed.values())
                + " on every date in the window. unlike cotton, linen, denim and leather it does not move "
                "with time: its SunSirs temporal series is still missing, so it is priced at its Tabb "
                "Textile TWPWD finished-fabric anchor, as methodology section 5.1 specifies when no "
                "credible historical series exists"),
            "price_basis": {
                **{m: "dated: finished-material anchor moved by its temporal series on each order_created date"
                   for m in ALL_MATERIALS if m not in FIXED_PRICE_MATERIALS},
                **{m: f"fixed: {basis}" for m, basis in FIXED_PRICE_MATERIALS.items()},
            },
            "fixed_prices_usd_per_kg": fixed,
            "orders_by_material": {k: int(v) for k, v in frame["material_type"].value_counts().items()},
            "v4_orders_by_material": {k: int(v) for k, v in v4_orders["material_type"].value_counts().items()},
        },
        "scenario_mix": SCENARIO_MIX,
        "priority_weights": PRIORITY_WEIGHTS,
        "budget_anchor": ("expected chain market price on order_created's date: material and manufacturer "
                          "stage prices with stage factors from the v5 agent csvs, times the priority markup, "
                          "plus the mean v5 route price of the weight-eligible agents serving the delivery "
                          "city at the mean manufacturer distance, which carries its own damage risk and "
                          "priority markup"),
        "stage_factors": {k: round(v, 6) for k, v in estimator.stage_factor.items()},
        "logistics_price": {
            "rule": logistics_module.PRICE_RULE,
            "applies": list(logistics_module.PRICE_APPLIES),
            "excludes": list(logistics_module.PRICE_EXCLUDES),
            "status": ("provisional decision pending confirmation: F, alpha and beta are treated as "
                       "market-level prices, so initial_markup and min_profit_margin are left out"),
            "no_markup_no_floor": ("refers only to initial_markup and min_profit_margin: F, alpha and beta "
                                   "are market-level prices, and no source gives the cost basis a floor "
                                   "would need. damage-risk and priority adjustments are applied. a proper "
                                   "cost-basis methodology is future work"),
            "shared_fuel_series": {"id": SHARED_FUEL_SERIES_ID, "basis": SHARED_FUEL_BASIS},
            "parcel_weight_limits": {
                r["agent_id"]: {"max_shipment_weight_kg": r["max_shipment_weight_kg"],
                                "oversized_above_kg": None if pd.isna(r["oversized_above_kg"]) else r["oversized_above_kg"],
                                "oversized_fee_eur": None if pd.isna(r["oversized_fee_eur"]) else r["oversized_fee_eur"],
                                "reading": r["calibration_reading_id"]}
                for r in caps.to_dict("records")},
            **weight_domains(logistics_profile),
            "orders_with_no_weight_eligible_logistics_agent": estimator.orders_without_weight_eligible_logistics,
            "beta": BETA_BASIS,
        },
        "input_sha256": {rel(p): sha256_file(p) for p in inputs},
        "v4_files_read_only_sha256": {rel(p): sha256_file(p) for p in v4_read},
    }


def build(output_dir=V5_DIR, logistics_profile_path=logistics_module.PROFILE_PATH,
          calibration_path=AGENT_CALIBRATION_PATH):
    frame, estimator, pairs = generate_frame(logistics_profile_path)
    write_csv(frame, output_dir / OUTPUT_PATH.name)
    write_json(output_dir / METADATA_PATH.name,
               metadata(frame, estimator, pairs, logistics_profile_path, calibration_path))
    return frame, estimator


def main():
    frame, _ = build()
    print(f"{len(frame)} orders, {frame['order_created'].min()} to {frame['order_created'].max()}")
    print(frame["material_type"].value_counts().to_string())
    print(f"budget_usd min {frame['budget_usd'].min()} mean {frame['budget_usd'].mean():.2f} "
          f"max {frame['budget_usd'].max()}")
    print(f"wrote {rel(OUTPUT_PATH)} and {rel(METADATA_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
