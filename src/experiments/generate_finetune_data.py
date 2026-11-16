"""
generate labeled prompt/completion pairs for lora fine-tuning of the bidding slm.

runs the rule-based decide_* functions across a sampled slice of orders.csv and,
for every agent-order pair that passes the shared eligibility checks, saves the
exact prompt built by slm.py's prompt builders alongside the rule-based json
output as the training label. no model, no nats, pure formula calculation.

the train/holdout split is disjoint by order id, an order's pairs land entirely
in one split so the holdout set contains only orders the model never trained on.

output: data/finetune/{version}/{stage}_{split}.jsonl, one json object per line
with system, user, completion, and metadata fields.
"""

import json
import math
import random
from collections import defaultdict

from config import DATA_VERSION, PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_logistics_bid,
    decide_manufacturer_bid,
    decide_material_bid,
    logistics_eligibility_reason,
    manufacturer_eligibility_reason,
    material_eligibility_reason,
)
from src.common.slm import (
    build_logistics_prompt,
    build_manufacturer_prompt,
    build_material_prompt,
)
from src.visualization.data_utils import load_all_data
from src.visualization.slm_competition_diagnostics import build_stage_orders


ORDER_SAMPLE_SIZE = 3000
HOLDOUT_FRACTION = 0.10
RANDOM_SEED = 42

OUT_DIR = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION


def _clean_number(value, digits=None):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    if digits is not None:
        value = round(value, digits)
    return int(value) if value == int(value) else value


def completion_json(stage, order_id, agent_id, rule, defect_probability):
    """label in exactly the shape the prompt's output schema asks for"""
    is_bid = rule["decision"] == "BID"
    payload = {
        "order_id": order_id,
        "stage": stage,
        "agent_id": agent_id,
        "decision": rule["decision"],
        "offer_amount": _clean_number(rule.get("offer_amount"), 2) if is_bid else None,
        "estimated_completion_days": _clean_number(rule.get("estimated_completion_days"), 2) if is_bid else None,
        "defect_probability": _clean_number(defect_probability),
    }
    return json.dumps(payload, indent=2)


def main():
    materials, manufacturers, logistics, products, orders = load_all_data()

    rng = random.Random(RANDOM_SEED)
    sampled = orders.sample(n=min(ORDER_SAMPLE_SIZE, len(orders)), random_state=RANDOM_SEED)

    order_ids = list(sampled["order_id"])
    rng.shuffle(order_ids)
    holdout_count = int(len(order_ids) * HOLDOUT_FRACTION)
    holdout_ids = set(order_ids[:holdout_count])

    print(f"version {DATA_VERSION}: {len(sampled)} orders sampled from {len(orders)}, "
          f"seed {RANDOM_SEED}, {len(holdout_ids)} holdout / {len(order_ids) - len(holdout_ids)} train "
          "(disjoint by order id)")

    stage_specs = [
        ("material", materials.to_dict("records"),
         material_eligibility_reason, decide_material_bid, build_material_prompt, "defect_probability"),
        ("tailoring", manufacturers.to_dict("records"),
         manufacturer_eligibility_reason, decide_manufacturer_bid, build_manufacturer_prompt, "defect_probability"),
        ("logistics", logistics.to_dict("records"),
         logistics_eligibility_reason, decide_logistics_bid, build_logistics_prompt, "damage_probability"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        (stage, split): open(OUT_DIR / f"{stage}_{split}.jsonl", "w", encoding="utf-8")
        for stage, *_ in stage_specs for split in ("train", "holdout")
    }

    counts = defaultdict(int)
    bid_counts = defaultdict(int)
    skipped_orders = 0

    try:
        for _, order in sampled.iterrows():
            stage_orders = build_stage_orders(order, products)
            if stage_orders is None:
                skipped_orders += 1
                continue
            split = "holdout" if order["order_id"] in holdout_ids else "train"

            for stage, agents, eligibility_fn, rule_fn, prompt_fn, defect_key in stage_specs:
                stage_order = stage_orders[stage]
                for agent in agents:
                    # same gate as the live agents, ineligible pairs never
                    # reach the model so they are not training material either
                    if eligibility_fn(agent, stage_order) is not None:
                        continue

                    rule = rule_fn(agent, stage_order)
                    system_prompt, user_prompt = prompt_fn(agent, stage_order)

                    record = {
                        "order_id": stage_order["order_id"],
                        "agent_id": agent["agent_id"],
                        "stage": stage,
                        "rule_decision": rule["decision"],
                        "system": system_prompt,
                        "user": user_prompt,
                        "completion": completion_json(
                            stage, stage_order["order_id"], agent["agent_id"],
                            rule, agent.get(defect_key),
                        ),
                    }
                    files[(stage, split)].write(json.dumps(record) + "\n")
                    counts[(stage, split)] += 1
                    if rule["decision"] == "BID":
                        bid_counts[(stage, split)] += 1
    finally:
        for f in files.values():
            f.close()

    print(f"\norders skipped (no product/material pair): {skipped_orders}")
    print(f"\n{'stage':12s} {'split':8s} {'examples':>9s} {'BID':>7s} {'NO_BID':>7s} {'bid %':>6s}")
    total = 0
    for stage, *_ in stage_specs:
        for split in ("train", "holdout"):
            n = counts[(stage, split)]
            b = bid_counts[(stage, split)]
            total += n
            pct = b / n * 100 if n else 0.0
            print(f"{stage:12s} {split:8s} {n:9d} {b:7d} {n - b:7d} {pct:5.1f}%")
    print(f"\ntotal examples: {total}")

    meta = {
        "data_version": DATA_VERSION,
        "order_sample_size": len(sampled),
        "holdout_fraction": HOLDOUT_FRACTION,
        "random_seed": RANDOM_SEED,
        "split_disjoint_by": "order_id",
        "counts": {f"{stage}_{split}": counts[(stage, split)]
                   for stage, *_ in stage_specs for split in ("train", "holdout")},
    }
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
