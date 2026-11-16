"""
where an order comes from, and whether it is usable

three sources, one shape at the end of it: an id looked up in orders.csv, a full
payload posted by a caller, or a json/jsonl file read off disk. all three produce
the plain dict the supervisor already expects on order.created, so nothing
downstream can tell them apart

orders.csv is read, never written
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.api import settings
from src.common.bidding_formulas import PRIORITY_MARKUP

# what the supervisor reads off an order. order_created drives the walking clock
# under dynamic resources, so it is required here rather than defaulted
REQUIRED_FIELDS = (
    "order_id", "order_created", "product_id", "material_type", "quantity",
    "budget", "deadline", "deliver_to", "max_defect_rate", "min_reputation_score",
    "priority",
)

NUMERIC_FIELDS = ("quantity", "budget", "max_defect_rate", "min_reputation_score")

_orders = None
_products = None


class OrderError(ValueError):
    """a bad order, reported to the caller rather than raised at the pipeline"""


def orders_frame():
    global _orders
    if _orders is None:
        _orders = pd.read_csv(settings.DATA_DIR / "orders.csv")
    return _orders


def products_frame():
    global _products
    if _products is None:
        _products = pd.read_csv(settings.DATA_DIR / "products.csv")
    return _products


def _plain(row):
    # numpy scalars are not json serializable and would break the publish
    return json.loads(pd.DataFrame([row]).to_json(orient="records"))[0]


def order_by_id(order_id):
    rows = orders_frame()
    match = rows[rows["order_id"] == order_id]
    if match.empty:
        raise OrderError(f"no order {order_id!r} in orders.csv")
    return validate(_plain(match.iloc[0]))


def validate(payload):
    """the order as the supervisor will read it, or OrderError saying why not

    every check here answers a question the supervisor would otherwise answer by
    raising inside a running order, where the caller sees nothing but a timeout"""
    if not isinstance(payload, dict):
        raise OrderError(f"an order must be a json object, got {type(payload).__name__}")

    missing = [f for f in REQUIRED_FIELDS if payload.get(f) in (None, "")]
    if missing:
        raise OrderError(f"order is missing required field(s): {', '.join(missing)}")

    order = dict(payload)

    for field in NUMERIC_FIELDS:
        try:
            order[field] = float(order[field])
        except (TypeError, ValueError):
            raise OrderError(f"{field} must be a number, got {order[field]!r}")
    order["quantity"] = int(order["quantity"])

    if order["quantity"] <= 0:
        raise OrderError(f"quantity must be positive, got {order['quantity']}")
    if order["budget"] <= 0:
        raise OrderError(f"budget must be positive, got {order['budget']}")
    if not 0 <= order["max_defect_rate"] <= 1:
        raise OrderError(f"max_defect_rate must be between 0 and 1, got {order['max_defect_rate']}")
    if not 0 <= order["min_reputation_score"] <= 1:
        raise OrderError(
            f"min_reputation_score must be between 0 and 1, got {order['min_reputation_score']}")

    if order["priority"] not in PRIORITY_MARKUP:
        raise OrderError(f"priority must be one of {sorted(PRIORITY_MARKUP)}, "
                         f"got {order['priority']!r}")

    try:
        datetime.strptime(str(order["deadline"]), "%Y-%m-%d")
    except ValueError:
        raise OrderError(f"deadline must be YYYY-MM-DD, got {order['deadline']!r}")
    try:
        datetime.fromisoformat(str(order["order_created"]))
    except ValueError:
        raise OrderError(f"order_created must be an iso timestamp, got {order['order_created']!r}")

    products = products_frame()
    pair = products[(products["product_id"] == order["product_id"])
                    & (products["material_type"] == order["material_type"])]
    if pair.empty:
        raise OrderError(f"no product/material pair {order['product_id']}/{order['material_type']} "
                         "in products.csv")

    return order


def read_file(path, limit=None):
    """orders from a json or jsonl file, in file order

    .jsonl is one order per line; .json is either a list or an object with an
    "orders" list, so a curated demo file can carry its own notes beside them"""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise OrderError(f"no order file at {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise OrderError(f"order file {path} is empty")

    if path.suffix == ".jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise OrderError(f"{path.name} line {number} is not valid json: {exc}")
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OrderError(f"{path.name} is not valid json: {exc}")
        rows = loaded.get("orders") if isinstance(loaded, dict) else loaded
        if not isinstance(rows, list):
            raise OrderError(f"{path.name} must hold a list of orders, or an object with "
                             "an \"orders\" list")

    if not rows:
        raise OrderError(f"order file {path} holds no orders")

    orders = []
    for index, row in enumerate(rows[:limit] if limit else rows):
        # a bare id in a file is looked up, the same as one posted to the api
        if isinstance(row, str):
            orders.append(order_by_id(row))
            continue
        try:
            orders.append(validate(row))
        except OrderError as exc:
            raise OrderError(f"{path.name} order {index + 1}: {exc}")
    return orders


def rosters():
    """every agent per stage, for a view that draws the whole board"""
    files = {"material": "material_agents.csv",
             "tailoring": "manufacturer_agents.csv",
             "logistics": "logistics_agents.csv"}
    out = {}
    for stage, filename in files.items():
        frame = pd.read_csv(settings.DATA_DIR / filename)
        keep = [c for c in ("agent_id", "location", "specialty", "reputation_score",
                            "defect_probability", "damage_probability", "material_types",
                            "product_types", "service_regions", "daily_capacity_units",
                            "current_workload_units") if c in frame.columns]
        out[stage] = json.loads(frame[keep].to_json(orient="records"))
    return out
