"""shared pieces of the v5 generators: the context gate, the v5 corpus, dated rfqs, the split
lookup and deterministic file writing

the rfqs are the ones the supervisor publishes, built as
src/visualization/slm_competition_diagnostics.build_stage_orders builds them for the v4
generators, plus the fields a v5 price needs: the dataset context and pricing date on every
stage, the reference USD/kg on the material rfq, and the pickup location, route distance, fuel
surcharge and EUR_to_USD on the logistics rfq. each dated value is read on the order's own
pricing date from the pinned v5 files. nothing here reads the system clock

the logistics origin: a v5 logistics price depends on where the shipment leaves from. the live
supervisor opens the logistics stage only after the manufacturer stage, whose provisional
winner is supervisor.pick_winner over the manufacturer bids, so the logistics rfq picks up at
that manufacturer's location. an order with no manufacturer bid never reaches the logistics
stage at runtime, so it has no logistics rfq here either

two presentation choices, both applied before any price or prompt reads the value: required_kg
and shipment_weight_kg are rounded to 4 decimals, which removes float noise such as
44.00000000000001 without changing any real weight, and market values carry the precision
src/common/v5/pricing.py publishes
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.common.bidding_formulas import decide_manufacturer_bid
from src.common.v5 import pricing
from src.data_synthesis.v5.logistics import RouteCost
from src.data_synthesis.v5.settings import SHARED_FUEL_SERIES_ID, V5_DIR
from src.dataset import context as dataset_context
from src.supervisor.supervisor import pick_winner

FINETUNE_V5 = PROJECT_ROOT / "data" / "finetune" / "v5"
SPLIT_PATH = FINETUNE_V5 / "split" / "order_split.csv"
SPLITS = ("train", "validation", "test")
STAGES = ("material", "tailoring", "logistics")
STAGE_DEFECT_KEY = {"material": "defect_probability", "tailoring": "defect_probability",
                    "logistics": "damage_probability"}
ORDER_CREATED_FORMAT = "%Y-%m-%dT%H:%M:%S"
WEIGHT_DECIMALS = 4


def require_v5():
    """the dataset context this run resolves, which must be v5

    selection goes through the existing COB_DATASET_VERSION mechanism; config.DATA_VERSION stays v4"""
    context = dataset_context.resolve()
    if context.version != "v5":
        raise SystemExit(f"this is a v5 generator but the dataset context resolves to {context.version}; "
                         f"set {dataset_context.ENV_VAR}=v5")
    return context


def load_corpus(directory=V5_DIR):
    """the v5 agent profiles, products and orders, read the way the agents read csv files"""
    products = {}
    for row in pd.read_csv(directory / "products.csv").to_dict("records"):
        products[(row["product_id"], row["material_type"])] = {
            "material_kg_per_unit": row["material_kg_per_unit"],
            "finished_weight_kg_per_unit": row["finished_weight_kg_per_unit"]}
    manufacturers = pd.read_csv(directory / "manufacturer_agents.csv").to_dict("records")
    return {
        "material": pd.read_csv(directory / "material_agents.csv").to_dict("records"),
        "tailoring": manufacturers,
        "logistics": pd.read_csv(directory / "logistics_agents.csv").to_dict("records"),
        "products": products,
        "orders": pd.read_csv(directory / "orders.csv").to_dict("records"),
        "manufacturer_location": {m["agent_id"]: m["location"] for m in manufacturers},
        "directory": Path(directory),
    }


class Market:
    """dated market values from the pinned v5 files, at the precision the v5 prices publish"""

    def __init__(self, directory=V5_DIR):
        self.route = RouteCost(profile_path=Path(directory) / "logistics_agents.csv")
        series = {a["fuel_surcharge_series_id"] for a in self.route.agents.values()}
        if series != {SHARED_FUEL_SERIES_ID}:
            raise ValueError(f"the v5 rfq carries one shared fuel surcharge, but agents use {sorted(series)}")
        reference = pd.read_csv(Path(directory) / "material_reference_prices.csv")
        self.reference = {(r.material, r.date): r.usd_per_kg for r in reference.itertuples()}

    def reference_usd_per_kg(self, material, pricing_date):
        return round(self.reference[(material, pricing_date)], pricing.REFERENCE_DECIMALS)

    def fuel_surcharge_percent(self, day):
        return round(self.route.fuel_lookup(SHARED_FUEL_SERIES_ID, day).value, pricing.FUEL_DECIMALS)

    def eur_to_usd(self, day):
        return round(self.route.eur_to_usd(day).rate, pricing.FX_DECIMALS)

    def distance_km(self, origin, destination):
        return round(self.route.distance_km[(origin, destination)], pricing.DISTANCE_DECIMALS)


WALK_SEED = 42


def walk_orders(orders):
    """the order every v5 generator visits orders in: v4 walks pandas' seeded sample of its pool,
    and a sample the size of the whole pool is a seeded permutation of it"""
    frame = pd.DataFrame(orders)
    return frame.sample(n=len(frame), random_state=WALK_SEED).to_dict("records")


def days_until_deadline(order):
    """whole days from order_created to the deadline, at least 1, as the v4 generators count them"""
    created = datetime.strptime(order["order_created"], ORDER_CREATED_FORMAT)
    deadline = datetime.strptime(order["deadline"], "%Y-%m-%d")
    return max((deadline - created).days, 1)


def base_rfq(order, context):
    created = datetime.strptime(order["order_created"], ORDER_CREATED_FORMAT)
    if created.date().isoformat() != order["pricing_date"]:
        raise ValueError(f"{order['order_id']}: pricing_date {order['pricing_date']} is not the order_created date")
    return {
        "order_id": order["order_id"],
        "max_defect_rate": order["max_defect_rate"],
        "min_reputation_score": order["min_reputation_score"],
        "days_until_deadline": days_until_deadline(order),
        "priority": order["priority"],
        **context.as_dict(),
        "pricing_date": order["pricing_date"],
    }


def logistics_origin(tailoring_rfq, corpus):
    """the provisional manufacturer winner's location, or None when no manufacturer bids"""
    bids = [b for b in (decide_manufacturer_bid(a, tailoring_rfq) for a in corpus["tailoring"])
            if b["decision"] == "BID"]
    winner = pick_winner(bids)
    return None if winner is None else corpus["manufacturer_location"][winner["agent_id"]]


def stage_rfqs(order, corpus, market, context):
    """the three stage rfqs for one order; logistics is None when no manufacturer bids"""
    per_unit = corpus["products"].get((order["product_id"], order["material_type"]))
    if per_unit is None:
        return None
    base = base_rfq(order, context)
    day = datetime.strptime(order["pricing_date"], "%Y-%m-%d").date()
    material = {**base, "material_type": order["material_type"],
                "required_kg": round(order["quantity"] * per_unit["material_kg_per_unit"], WEIGHT_DECIMALS),
                "reference_price_usd_per_kg": market.reference_usd_per_kg(order["material_type"], order["pricing_date"])}
    tailoring = {**base, "product_id": order["product_id"], "quantity": order["quantity"]}
    origin = logistics_origin(tailoring, corpus)
    logistics = None
    if origin is not None:
        logistics = {**base, "deliver_to": order["deliver_to"],
                     "shipment_weight_kg": round(order["quantity"] * per_unit["finished_weight_kg_per_unit"],
                                                 WEIGHT_DECIMALS),
                     "pickup_from": origin,
                     "route_distance_km": market.distance_km(origin, order["deliver_to"]),
                     "fuel_surcharge_percent": market.fuel_surcharge_percent(day),
                     "eur_to_usd": market.eur_to_usd(day)}
    return {"material": material, "tailoring": tailoring, "logistics": logistics}


# ── the split ───────────────────────────────────────────────────────────────

def read_split(path=SPLIT_PATH):
    if not Path(path).exists():
        raise SystemExit(f"no split at {path}; run src.experiments.v5.split first")
    frame = pd.read_csv(path)
    return dict(zip(frame["order_id"], frame["split"]))


# ── files ───────────────────────────────────────────────────────────────────

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


class JsonlFiles:
    """one open jsonl file per key, unix newlines, one record per line, counted as written"""

    def __init__(self, paths):
        self.paths = {key: Path(p) for key, p in paths.items()}
        self.counts = {key: 0 for key in self.paths}
        self._handles = {}
        for key, path in self.paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[key] = open(path, "w", encoding="utf-8", newline="\n")

    def write(self, key, record):
        self._handles[key].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[key] += 1

    def close(self):
        for handle in self._handles.values():
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
