"""the one place that decides what "a v4 decision" means for the golden harness

both the fixture generator and the regression test import from here, so a test
can never drift from the fixture it checks by rebuilding rfqs a second way

everything below reads the committed v4 corpus and the live v4 formulas. nothing
here has a v5 dependency, and nothing reads the system clock: stage deadlines are
anchored on each order's own order_created through the supervisor's own helper
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_logistics_bid, decide_manufacturer_bid, decide_material_bid,
)
from src.common.bidding_modes import DEFAULT_PLATFORM_MINIMUM_DECREMENT
from src.common.reverse_auction import best_standing_bid, decide_round_bid
from src.supervisor.supervisor import days_until_deadline


V4_DIR = PROJECT_ROOT / "data" / "synthetic" / "v4"
GOLDEN_DIR = Path(__file__).resolve().parent

# the six tracked files under data/synthetic/v4
V4_FILES = (
    "logistics_agents.csv",
    "manufacturer_agents.csv",
    "material_agents.csv",
    "orders.csv",
    "orders_metadata.json",
    "products.csv",
)

STAGES = ("material", "tailoring", "logistics")

SEALED_DECIDERS = {
    "material": decide_material_bid,
    "tailoring": decide_manufacturer_bid,
    "logistics": decide_logistics_bid,
}

# every 2500th order, for a readable sample that still spans the whole corpus
SAMPLE_STRIDE = 2500

# the fixed small sample used for byte-for-byte prompt goldens
PROMPT_ORDER_IDS = ("ORD-00001", "ORD-00002", "ORD-00003", "ORD-00004", "ORD-00005")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def v4_file_hashes():
    return {name: sha256_file(V4_DIR / name) for name in V4_FILES}


def load_v4():
    """the committed v4 corpus, read exactly the way the agents read it"""
    materials = pd.read_csv(V4_DIR / "material_agents.csv").to_dict("records")
    manufacturers = pd.read_csv(V4_DIR / "manufacturer_agents.csv").to_dict("records")
    logistics = pd.read_csv(V4_DIR / "logistics_agents.csv").to_dict("records")
    products = pd.read_csv(V4_DIR / "products.csv")
    orders = pd.read_csv(V4_DIR / "orders.csv").to_dict("records")
    return {
        "material": materials,
        "tailoring": manufacturers,
        "logistics": logistics,
        "products": product_index(products),
        "orders": orders,
    }


def product_index(products):
    """(product_id, material_type) -> the two per-unit weights the rfqs need"""
    index = {}
    for row in products.to_dict("records"):
        index[(row["product_id"], row["material_type"])] = {
            "material_kg_per_unit": row["material_kg_per_unit"],
            "finished_weight_kg_per_unit": row["finished_weight_kg_per_unit"],
        }
    return index


def stage_rfqs(order, products):
    """the three rfq payloads the supervisor publishes for one order

    same fields and same arithmetic as supervisor.handle_new_order() in static
    mode, which is the mode every saved order replays in"""
    prod = products.get((order["product_id"], order["material_type"]))
    if prod is None:
        return None

    total_days = days_until_deadline(order)
    base = {
        "order_id": order["order_id"],
        "max_defect_rate": order["max_defect_rate"],
        "min_reputation_score": order["min_reputation_score"],
        "deadline": order["deadline"],
        "days_until_deadline": total_days,
        "priority": order["priority"],
    }
    return {
        "material": {**base,
                     "material_type": order["material_type"],
                     "required_kg": order["quantity"] * prod["material_kg_per_unit"]},
        "tailoring": {**base,
                      "product_id": order["product_id"],
                      "quantity": order["quantity"]},
        "logistics": {**base,
                      "deliver_to": order["deliver_to"],
                      "shipment_weight_kg": order["quantity"] * prod["finished_weight_kg_per_unit"]},
    }


def decision_record(stage, agent, rfq, bid):
    """the four things the harness freezes, plus enough to locate the case

    offer_amount is kept exactly as the decision produced it, which is already
    rounded to the cent by the decide_* functions"""
    return {
        "order_id": rfq["order_id"],
        "stage": stage,
        "agent_id": agent["agent_id"],
        "decision": bid["decision"],
        "offer_amount": bid.get("offer_amount"),
        "estimated_completion_days": bid.get("estimated_completion_days"),
        "debug_reason": bid.get("debug_reason"),
    }


def sealed_decisions(order, rfqs, corpus):
    """every agent's sealed decision on one order, in roster order"""
    records = []
    for stage in STAGES:
        decide = SEALED_DECIDERS[stage]
        rfq = rfqs[stage]
        for agent in corpus[stage]:
            records.append(decision_record(stage, agent, rfq, decide(agent, rfq)))
    return records


def auction_decisions(order, rfqs, corpus, max_rounds=3):
    """the same agents replayed through a rule-based reverse auction

    round 1 is the sealed decision, so only rounds 2 and up are recorded here.
    the announced best is the standing best after the previous round, exactly
    what the supervisor publishes, so the recorded rounds include the agents that
    fall through their own floor and go silent"""
    records = []
    decrement = DEFAULT_PLATFORM_MINIMUM_DECREMENT

    for stage in STAGES:
        base_rfq = rfqs[stage]
        standing = {}
        received = 0
        for agent in corpus[stage]:
            bid = SEALED_DECIDERS[stage](agent, {**base_rfq, "round_number": 1})
            if bid["decision"] == "BID":
                received += 1
                standing[agent["agent_id"]] = {**bid, "received_seq": received}

        best = best_standing_bid(list(standing.values()))
        announced = best["offer_amount"] if best else None

        for round_number in range(2, max_rounds + 1):
            rfq = {**base_rfq,
                   "round_number": round_number,
                   "platform_minimum_decrement": decrement,
                   "current_best_bid": announced}
            round_bids = []
            for agent in corpus[stage]:
                bid = decide_round_bid(stage, agent, rfq)
                records.append({**decision_record(stage, agent, rfq, bid),
                                "round_number": round_number,
                                "current_best_bid": announced})
                if bid["decision"] == "BID":
                    received += 1
                    round_bids.append({**bid, "agent_id": agent["agent_id"],
                                       "received_seq": received})
            best_after = best_standing_bid(round_bids)
            if best_after is None:
                break
            announced = min(announced, best_after["offer_amount"]) if announced is not None \
                else best_after["offer_amount"]
    return records


def canonical(record):
    """one decision as one stable line, for streaming into a census digest"""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def census_digest(corpus, order_limit=None, max_rounds=3):
    """sha-256 over every sealed and auction decision in the corpus

    an exhaustive fixture would be a hundred megabytes of json, so the exhaustive
    arm of the harness is a digest over the canonical decision stream plus the
    counts below. a single changed cent anywhere in 50,000 orders changes it"""
    digest = hashlib.sha256()
    counts = {"sealed_bid": 0, "sealed_no_bid": 0, "auction_bid": 0, "auction_no_bid": 0}
    reasons = {}
    orders = corpus["orders"] if order_limit is None else corpus["orders"][:order_limit]
    skipped = 0

    for order in orders:
        rfqs = stage_rfqs(order, corpus["products"])
        if rfqs is None:
            skipped += 1
            continue
        for record in sealed_decisions(order, rfqs, corpus):
            digest.update(canonical(record).encode())
            key = "sealed_bid" if record["decision"] == "BID" else "sealed_no_bid"
            counts[key] += 1
            if record["debug_reason"]:
                reasons[record["debug_reason"]] = reasons.get(record["debug_reason"], 0) + 1
        for record in auction_decisions(order, rfqs, corpus, max_rounds):
            digest.update(canonical(record).encode())
            key = "auction_bid" if record["decision"] == "BID" else "auction_no_bid"
            counts[key] += 1
            if record["debug_reason"]:
                reasons[record["debug_reason"]] = reasons.get(record["debug_reason"], 0) + 1

    return {
        "orders_covered": len(orders) - skipped,
        "orders_skipped_no_product_row": skipped,
        "decision_counts": counts,
        "reason_counts": dict(sorted(reasons.items())),
        "digest_sha256": digest.hexdigest(),
    }


def sample_orders(corpus, stride=SAMPLE_STRIDE):
    return corpus["orders"][::stride]


def sample_decisions(corpus, stride=SAMPLE_STRIDE, max_rounds=3):
    """the readable arm: full records for every stride-th order"""
    records = []
    for order in sample_orders(corpus, stride):
        rfqs = stage_rfqs(order, corpus["products"])
        if rfqs is None:
            continue
        records.extend(sealed_decisions(order, rfqs, corpus))
        records.extend(auction_decisions(order, rfqs, corpus, max_rounds))
    return records
