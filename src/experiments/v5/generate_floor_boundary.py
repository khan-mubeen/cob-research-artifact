"""v5 reverse-auction floor-boundary set, role diagnostic, 132 rows per case per stage

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_floor_boundary

a targeted diagnostic of the one threshold the reverse-auction prompts exist to teach: whether an
agent bids when the mandated next offer lands one percent above, exactly at, or one cent below
its own minimum_bid. it is not part of any train, validation or test count

the size is fixed at 132 (order, agent) pairs per stage, each carrying all three cases, so every
one of the nine files holds 132 rows and the set holds 1,188. it is deliberately not scaled with
the 50,000-order pool: this is a diagnostic sample size, not a proportional one. v4's
already-validated held-out floor slice holds 132 prompts per case summed over the three stages
(42 material, 45 tailoring, 45 logistics); tonight's instruction fixes 132 per case per stage,
three times that per case, and that is what is built

the floor is computed once, by src.common.reverse_auction.agent_minimum_bid. the constructed
current_best_bid of every case comes from floor_cases in the v5 reverse-auction generator, which
calls agent_minimum_bid, and the prompt shows minimum_bid through auction_fields, which calls the
same function. this module never computes a floor itself: the round() against ceil() mismatch
that once gave the one-cent-below case contradictory labels came from exactly such a second
calculation. every row is checked on the way out: the floor printed in the prompt equals
agent_minimum_bid, the case sits at its offset from it, and the label is the rule's

pairs come only from test-split orders, never trained on, and never from an (order, agent) pair
that already carries a floor block inside the reverse-auction test file, so no row here repeats
a test row. one pair per order, walked in the corpus's seeded order, as v4 spreads its block

output: data/finetune/v5/reverse_auction/floor_boundary/{stage}_{below,at,above}_floor.jsonl and
metadata.json
"""

import json
import re
import sys

from src.common.bidding_formulas import eligibility_reason
from src.common.bidding_modes import DEFAULT_PLATFORM_MINIMUM_DECREMENT
from src.common.reverse_auction import SEALED_DECIDERS, agent_minimum_bid, decide_round_bid, next_round_offer
from src.experiments.generate_reverse_auction_data import make_record, round_rfq
from src.experiments.v5.common import (
    FINETUNE_V5, STAGES, JsonlFiles, Market, load_corpus, read_jsonl, read_split, rel, require_v5, stage_rfqs,
    walk_orders, write_json,
)
from src.experiments.v5.generate_reverse_auction_data import WORDED_FORMULAS, floor_cases

OUT_DIR = FINETUNE_V5 / "reverse_auction" / "floor_boundary"
ROLE = "diagnostic"
PAIRS_PER_STAGE = 132
SPLIT = "test"
CASES = {"one_cent_below_floor": "below_floor", "at_floor": "at_floor", "above_floor": "above_floor"}
EXPECTED = {"below_floor": "NO_BID", "at_floor": "BID", "above_floor": "BID"}
FLOOR_LINE = re.compile(r"^your minimum_bid \(USD\): (\S+)$", re.M)


def corpus_floor_pairs(stage):
    """(order, agent) pairs that already carry a floor block in the reverse-auction test file"""
    path = FINETUNE_V5 / "reverse_auction" / f"{stage}_{SPLIT}.jsonl"
    return {(r["order_id"], r["agent_id"]) for r in read_jsonl(path) if r["category"].startswith("floor_boundary")}


def checked_row(stage, agent, rfq, case, current_best, context):
    """one diagnostic row, verified against the shared floor before it is written"""
    number = rfq["_round_number"]
    seen = round_rfq({k: v for k, v in rfq.items() if k != "_round_number"}, number, current_best,
                     DEFAULT_PLATFORM_MINIMUM_DECREMENT)
    decision = decide_round_bid(stage, agent, seen)
    record = make_record(stage, agent, seen, decision, f"floor_boundary:{CASES[case]}", False, WORDED_FORMULAS)
    floor = agent_minimum_bid(stage, agent, seen)
    printed = float(FLOOR_LINE.search(record["user"]).group(1))
    if printed != floor:
        raise AssertionError(f"{rfq['order_id']} {agent['agent_id']}: prompt floor {printed} != shared floor {floor}")
    offer = next_round_offer(current_best, DEFAULT_PLATFORM_MINIMUM_DECREMENT)
    expected_offer = {"below_floor": round(floor - 0.01, 2), "at_floor": floor,
                      "above_floor": round(floor * 1.01, 2)}[CASES[case]]
    if abs(offer - expected_offer) > 1e-9:
        raise AssertionError(f"{rfq['order_id']} {agent['agent_id']} {case}: next offer {offer} is not {expected_offer}")
    if decision["decision"] != EXPECTED[CASES[case]] or (decision["decision"] == "BID") != (offer >= floor):
        raise AssertionError(f"{rfq['order_id']} {agent['agent_id']} {case}: label {decision} disagrees with floor {floor}")
    return {**record, "dataset_version": context.version, "split": SPLIT, "role": ROLE, "case": CASES[case],
            "displayed_minimum_bid": floor, "mandated_next_offer": offer}


def main():
    context = require_v5()
    corpus = load_corpus()
    market = Market()
    split_of = read_split()
    walk = [o for o in walk_orders(corpus["orders"]) if split_of[o["order_id"]] == SPLIT]

    paths = {(stage, name): OUT_DIR / f"{stage}_{name}.jsonl" for stage in STAGES for name in CASES.values()}
    used, orders_used = {}, {}
    with JsonlFiles(paths) as files:
        for stage in STAGES:
            excluded = corpus_floor_pairs(stage)
            pairs = 0
            stage_orders = []
            for order in walk:
                if pairs == PAIRS_PER_STAGE:
                    break
                rfqs = stage_rfqs(order, corpus, market, context)
                rfq = rfqs[stage] if rfqs else None
                if rfq is None:
                    continue
                opener = next((a for a in corpus[stage]
                               if eligibility_reason(stage, a, rfq) is None
                               and (order["order_id"], a["agent_id"]) not in excluded
                               and SEALED_DECIDERS[stage](a, rfq)["decision"] == "BID"), None)
                if opener is None:
                    continue
                pairs += 1
                stage_orders.append(order["order_id"])
                numbered = {**rfq, "_round_number": 2 + (pairs % 2)}
                for case, current_best in floor_cases(stage, opener, rfq, DEFAULT_PLATFORM_MINIMUM_DECREMENT):
                    files.write((stage, CASES[case]), checked_row(stage, opener, numbered, case, current_best, context))
            if pairs != PAIRS_PER_STAGE:
                raise SystemExit(f"{stage}: only {pairs} test-split pairs available, {PAIRS_PER_STAGE} required")
            used[stage], orders_used[stage] = pairs, stage_orders

    counts = {f"{stage}_{name}": files.counts[(stage, name)] for stage in STAGES for name in CASES.values()}
    print(json.dumps(counts, indent=2))
    write_json(OUT_DIR / "metadata.json", {
        "role": ROLE,
        "role_statement": "a targeted floor diagnostic; not training data and not in any train, validation or test count",
        "data_version": context.version,
        "currency": context.currency,
        "pairs_per_stage": PAIRS_PER_STAGE,
        "rows_per_file": PAIRS_PER_STAGE,
        "total_rows": sum(counts.values()),
        "size_basis": ("fixed, deliberately not scaled with the 50,000-order pool. v4's validated held-out floor slice "
                       "is 132 prompts per case summed over the three stages; the requested and built size is 132 per "
                       "case per stage"),
        "source_split": SPLIT,
        "excludes": "(order, agent) pairs already carrying a floor block in reverse_auction/{stage}_test.jsonl",
        "floor_function": "src.common.reverse_auction.agent_minimum_bid, shared by the prompt and the case construction",
        "case_construction": {"above_floor": "next offer = round(floor x 1.01, 2)", "at_floor": "next offer = floor",
                              "below_floor": "next offer = floor - 0.01"},
        "platform_minimum_decrement": DEFAULT_PLATFORM_MINIMUM_DECREMENT,
        "worded_formulas": WORDED_FORMULAS,
        "expected_labels": EXPECTED,
        "label_rule": "src.common.reverse_auction.decide_round_bid",
        "counts": counts,
        "orders_used_per_stage": {stage: len(ids) for stage, ids in orders_used.items()},
    })
    print(f"written to {rel(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
