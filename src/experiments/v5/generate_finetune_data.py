"""v5 sealed-round fine-tuning data, the v4 generator's method on the full v5 order pool

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_finetune_data

src/experiments/generate_finetune_data.py is the source of truth and is not edited: for every
order, every agent-order pair that passes the shared static eligibility check gets one row
holding the exact prompt slm.py builds and the rule decision as its label, serialized by v4's
completion_json. ineligible pairs never reach a model, so they are not training material.
what differs, because v5 differs:
  scale   all 50,000 v5 orders, a deliberate v5 decision, instead of v4's 3,000-order sample
  split   the global stratified train/validation/test split by order id, read from
          data/finetune/v5/split/order_split.csv, instead of v4's 90/10 train/holdout
  rfqs    carry the v5 dataset context and dated market values (src/experiments/v5/common.py)
  prices  the dispatched v5 formulas, USD, through the same decide_* and build_*_prompt calls
          v4 uses; a logistics rfq exists only once a manufacturer has bid, as at runtime

output: data/finetune/v5/{material,tailoring,logistics}_{train,validation,test}.jsonl and
data/finetune/v5/metadata.json
"""

import sys
from collections import Counter, defaultdict

from src.common.bidding_formulas import eligibility_reason
from src.common.reverse_auction import SEALED_DECIDERS
from src.experiments.generate_finetune_data import completion_json
from src.experiments.generate_reverse_auction_data import STAGE_PROMPTS
from src.experiments.v5.common import (
    FINETUNE_V5, SPLIT_PATH, SPLITS, STAGE_DEFECT_KEY, STAGES, WALK_SEED, JsonlFiles, Market, load_corpus,
    read_split, rel, require_v5, sha256_file, stage_rfqs, walk_orders, write_json,
)

OUT_DIR = FINETUNE_V5


def sealed_record(stage, agent, rfq, split, context):
    rule = SEALED_DECIDERS[stage](agent, rfq)
    system_prompt, user_prompt = STAGE_PROMPTS[stage](agent, rfq)
    return {
        "order_id": rfq["order_id"],
        "agent_id": agent["agent_id"],
        "stage": stage,
        "dataset_version": context.version,
        "split": split,
        "rule_decision": rule["decision"],
        "system": system_prompt,
        "user": user_prompt,
        "completion": completion_json(stage, rfq["order_id"], agent["agent_id"], rule,
                                      agent.get(STAGE_DEFECT_KEY[stage])),
    }


def build_rows(context, corpus, market, split_of, walk, tally):
    """((stage, split), record) for every eligible pair of every order walked, in walk order"""
    for order in walk:
        rfqs = stage_rfqs(order, corpus, market, context)
        if rfqs is None:
            tally["orders_without_a_product_pair"] += 1
            continue
        split = split_of[order["order_id"]]
        if rfqs["logistics"] is None:
            tally["logistics_rfqs_without_a_manufacturer_bid"] += 1
        for stage in STAGES:
            rfq = rfqs[stage]
            if rfq is None:
                continue
            for agent in corpus[stage]:
                # the live gate: an ineligible pair never reaches the model
                if eligibility_reason(stage, agent, rfq) is not None:
                    continue
                yield (stage, split), sealed_record(stage, agent, rfq, split, context)


def main():
    context = require_v5()
    corpus = load_corpus()
    market = Market()
    split_of = read_split()
    walk = walk_orders(corpus["orders"])
    if set(split_of) != {o["order_id"] for o in corpus["orders"]}:
        raise SystemExit("the split does not cover exactly the v5 orders")

    counts, bids = defaultdict(int), defaultdict(int)
    tally = Counter()
    orders_with_rows = {s: set() for s in SPLITS}
    paths = {(stage, split): OUT_DIR / f"{stage}_{split}.jsonl" for stage in STAGES for split in SPLITS}
    with JsonlFiles(paths) as files:
        for key, record in build_rows(context, corpus, market, split_of, walk, tally):
            files.write(key, record)
            counts[key] += 1
            bids[key] += record["rule_decision"] == "BID"
            orders_with_rows[key[1]].add(record["order_id"])

    balance = {}
    print(f"\n{'stage':10s} {'split':11s} {'rows':>8s} {'BID':>8s} {'NO_BID':>8s} {'bid %':>6s}")
    for stage in STAGES:
        for split in SPLITS:
            n, b = counts[(stage, split)], bids[(stage, split)]
            balance[f"{stage}_{split}"] = {"rows": n, "BID": b, "NO_BID": n - b,
                                           "bid_pct": round(b / n * 100, 3) if n else None}
            print(f"{stage:10s} {split:11s} {n:8d} {b:8d} {n - b:8d} {b / n * 100 if n else 0:6.2f}")
    print(f"total rows {sum(counts.values())}, skipped {dict(tally)}")

    write_json(OUT_DIR / "metadata.json", {
        "generated_from": "rule-based decide_*_bid() labels through the v5 dispatch, no model involved",
        "data_version": context.version,
        "currency": context.currency,
        "mode": "sealed",
        "scale_decision": ("deliberate v5 scaling decision: every one of the 50,000 v5 orders is used, not v4's "
                           "~3,000-order experimental sample"),
        "source_order_count": len(corpus["orders"]),
        "orders_iterated": len(walk),
        "orders_with_rows": {s: len(v) for s, v in orders_with_rows.items()},
        "split_file": rel(SPLIT_PATH),
        "split_file_sha256": sha256_file(SPLIT_PATH),
        "split_disjoint_by": "order_id",
        "walk_order": f"pandas sample of the whole pool with random_state {WALK_SEED}, as v4 walks its sample",
        "label_rule": "src.common.bidding_formulas.decide_{material,manufacturer,logistics}_bid, v5 dispatch",
        "prompt_builder": "src.common.slm.build_{material,manufacturer,logistics}_prompt, v5 dispatch",
        "worded_formulas": True,
        "prompt_wording": ("every v5 stage prompt restates each TASK operator in words beside its symbol, the "
                           "validated reverse-auction style (reports/v5/wording/validated_mapping.json)"),
        "eligibility_gate": "src.common.bidding_formulas.eligibility_reason, ineligible pairs are not rows",
        "logistics_origin": ("the location of supervisor.pick_winner over the order's manufacturer bids; an order "
                             "with no manufacturer bid has no logistics rfq, as at runtime"),
        "skipped": {"orders_without_a_product_pair": tally["orders_without_a_product_pair"],
                    "logistics_rfqs_without_a_manufacturer_bid": tally["logistics_rfqs_without_a_manufacturer_bid"]},
        "counts": {f"{stage}_{split}": counts[(stage, split)] for stage in STAGES for split in SPLITS},
        "class_balance": balance,
    })
    print(f"written to {rel(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
