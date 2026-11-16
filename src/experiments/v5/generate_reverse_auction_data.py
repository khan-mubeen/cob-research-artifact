"""v5 reverse-auction fine-tuning data, the v4 generator's method on the full v5 order pool

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_reverse_auction_data

src/experiments/generate_reverse_auction_data.py is the source of truth and is not edited. its
replay_auction, round_rfq, make_record and DECREMENT_SCHEDULE are imported and run unchanged:
every order and stage replays a full auction with the runtime AuctionMemory and
best_standing_bid, one row per eligible pair rotates across the rounds the auction actually ran,
round 1 is asserted equal to the sealed decision, every second order runs at the default
decrement and the rest at 5/10/20% of the stage's opening best bid, and the first orders walked
carry the three-case floor_boundary block inside the corpus. the prompts are worded, as the
committed v4 set data/finetune/v4/reverse_auction/v4 is. what differs, because v5 differs:
  scale    all 50,000 v5 orders instead of v4's 3,000-order sample
  split    the global stratified train/validation/test split by order id
  prices   the dispatched v5 formulas in USD; the platform decrement's default of 5 is read in
           the order's currency, USD, since it is a platform rule, not a sourced price
  floors   the floor block is constructed from reverse_auction.agent_minimum_bid, the very
           function the prompt displays minimum_bid with, rather than recomputing it
  block    v4 gave 500 of its 3,000 orders a floor block per stage. that count was relative to
           its pool, so v5 keeps the rate, one pair per six orders walked, 8,333 per stage;
           a fixed 500 at 50,000 orders would cut floor supervision about seventeen-fold

a v5 logistics price has no margin below it, so a logistics agent's minimum_bid is its opening
offer: from round 2 no logistics agent can undercut, every logistics auction closes after round
2, and its round-2 rows are all NO_BID. that is the locked no-markup, no-floor decision showing
through, and it is reported rather than worked around

output: data/finetune/v5/reverse_auction/{stage}_{split}.jsonl and metadata.json
"""

import json
import sys
from collections import Counter, defaultdict

from src.common.bidding_formulas import eligibility_reason
from src.common.bidding_modes import DEFAULT_PLATFORM_MINIMUM_DECREMENT, auction_max_rounds
from src.common.reverse_auction import (
    SEALED_DECIDERS, agent_minimum_bid, best_standing_bid, decide_round_bid, next_round_offer,
)
from src.experiments.generate_reverse_auction_data import DECREMENT_SCHEDULE, make_record, replay_auction, round_rfq
from src.experiments.v5.common import (
    FINETUNE_V5, SPLIT_PATH, SPLITS, STAGES, JsonlFiles, Market, load_corpus, read_split, rel, require_v5,
    sha256_file, stage_rfqs, walk_orders, write_json,
)

WORDED_FORMULAS = True
V4_FLOOR_PAIRS_PER_STAGE = 500
V4_ORDER_SAMPLE_SIZE = 3000
OUT_DIR = FINETUNE_V5 / "reverse_auction"


def floor_pairs_per_stage(order_count):
    """v4's floor block rate, one (order, agent) pair per six orders walked, at this pool size"""
    return round(V4_FLOOR_PAIRS_PER_STAGE * order_count / V4_ORDER_SAMPLE_SIZE)


def floor_cases(stage, agent, rfq, decrement):
    """three constructed best bids around the agent's floor, from the function the prompt shows it with

    the mandated next offer lands one percent over the floor, exactly on it, and one cent under it"""
    floor = agent_minimum_bid(stage, agent, rfq)
    return [
        ("above_floor", round(round(floor * 1.01, 2) + decrement, 2)),
        ("at_floor", round(floor + decrement, 2)),
        ("one_cent_below_floor", round(floor - 0.01 + decrement, 2)),
    ]


def tagged(record, split, context):
    return {**record, "dataset_version": context.version, "split": split}


def check_row(row):
    """v4's own build-time assertions, applied to every row as it is produced"""
    if row["withdrawn_in_earlier_round"] and row["rule_decision"] != "NO_BID":
        raise AssertionError(f"withdrawn agent labeled {row['rule_decision']} on {row['order_id']}")
    if row["rule_decision"] == "BID" and row["current_best_bid"] is not None:
        expected = next_round_offer(row["current_best_bid"], row["platform_minimum_decrement"])
        got = json.loads(row["completion"])["offer_amount"]
        if abs(got - expected) >= 1e-9:
            raise AssertionError(f"offer {got} is not current_best_bid - decrement ({expected})")
    return row


def new_tally():
    return {"skipped": Counter(), "decrement": Counter(), "replay_lengths": defaultdict(Counter),
            "floor_pairs_used": Counter()}


def build_rows(context, corpus, market, split_of, walk, tally, floor_pairs):
    """((stage, split), row) for every auction row and floor block of every order walked"""
    pair_index = defaultdict(int)
    for order_position, order in enumerate(walk):
        rfqs = stage_rfqs(order, corpus, market, context)
        if rfqs is None:
            tally["skipped"]["orders_without_a_product_pair"] += 1
            continue
        if rfqs["logistics"] is None:
            tally["skipped"]["logistics_rfqs_without_a_manufacturer_bid"] += 1
        split = split_of[order["order_id"]]
        fraction = DECREMENT_SCHEDULE[order_position % len(DECREMENT_SCHEDULE)]

        for stage in STAGES:
            rfq = rfqs[stage]
            if rfq is None:
                continue
            agents = [a for a in corpus[stage] if eligibility_reason(stage, a, rfq) is None]
            if not agents:
                continue

            opening = best_standing_bid([
                d for d in (decide_round_bid(stage, a, round_rfq(rfq, 1, None, DEFAULT_PLATFORM_MINIMUM_DECREMENT))
                            for a in agents) if d["decision"] == "BID"
            ])
            if fraction is None or opening is None:
                decrement = DEFAULT_PLATFORM_MINIMUM_DECREMENT
            else:
                decrement = max(round(opening["offer_amount"] * fraction, 2), 0.01)
            tally["decrement"][(stage, "default" if decrement == DEFAULT_PLATFORM_MINIMUM_DECREMENT
                                else f"{fraction:.0%}_of_opening")] += 1

            replay = replay_auction(stage, agents, rfq, decrement)
            tally["replay_lengths"][stage][len(replay)] += 1

            for agent in agents:
                round_number = 1 + (pair_index[stage] % len(replay))
                pair_index[stage] += 1
                entry = replay[round_number - 1]["entries"][agent["agent_id"]]
                if round_number == 1:
                    sealed = SEALED_DECIDERS[stage](agent, rfq)
                    if (entry["decision"]["decision"] != sealed["decision"]
                            or entry["decision"].get("offer_amount") != sealed.get("offer_amount")):
                        raise AssertionError(f"round 1 differs from the sealed decision for "
                                             f"{agent['agent_id']} on {rfq['order_id']}")
                yield (stage, split), check_row(tagged(make_record(
                    stage, agent, entry["rfq"], entry["decision"], "auction_round", entry["withdrawn"],
                    WORDED_FORMULAS), split, context))

            if tally["floor_pairs_used"][stage] < floor_pairs:
                opener = next((a for a in agents
                               if replay[0]["entries"][a["agent_id"]]["decision"]["decision"] == "BID"), None)
                if opener is not None:
                    tally["floor_pairs_used"][stage] += 1
                    for case, current_best in floor_cases(stage, opener, rfq, DEFAULT_PLATFORM_MINIMUM_DECREMENT):
                        number = 2 + (tally["floor_pairs_used"][stage] % 2)
                        seen = round_rfq(rfq, number, current_best, DEFAULT_PLATFORM_MINIMUM_DECREMENT)
                        yield (stage, split), check_row(tagged(make_record(
                            stage, opener, seen, decide_round_bid(stage, opener, seen), f"floor_boundary:{case}",
                            False, WORDED_FORMULAS), split, context))


def main():
    context = require_v5()
    corpus = load_corpus()
    market = Market()
    split_of = read_split()
    walk = walk_orders(corpus["orders"])
    floor_pairs = floor_pairs_per_stage(len(walk))

    paths = {(stage, split): OUT_DIR / f"{stage}_{split}.jsonl" for stage in STAGES for split in SPLITS}
    stats = defaultdict(Counter)
    round_mix = Counter()
    orders_with_rows = {s: set() for s in SPLITS}
    tally = new_tally()
    with JsonlFiles(paths) as files:
        for key, row in build_rows(context, corpus, market, split_of, walk, tally, floor_pairs):
            files.write(key, row)
            stats[key]["rows"] += 1
            stats[key]["BID"] += row["rule_decision"] == "BID"
            stats[key]["withdrawn"] += row["withdrawn_in_earlier_round"]
            stats[key]["floor_boundary"] += row["category"].startswith("floor_boundary")
            round_mix[row["round_number"]] += 1
            orders_with_rows[key[1]].add(row["order_id"])

    balance = {}
    print(f"\n{'stage':10s} {'split':11s} {'rows':>8s} {'BID':>8s} {'NO_BID':>8s} {'withdrawn':>10s} "
          f"{'floor':>7s} {'bid %':>6s}")
    for stage in STAGES:
        for split in SPLITS:
            s = stats[(stage, split)]
            n, b = s["rows"], s["BID"]
            balance[f"{stage}_{split}"] = {"rows": n, "BID": b, "NO_BID": n - b, "withdrawn": s["withdrawn"],
                                           "floor_boundary": s["floor_boundary"],
                                           "bid_pct": round(b / n * 100, 3) if n else None}
            print(f"{stage:10s} {split:11s} {n:8d} {b:8d} {n - b:8d} {s['withdrawn']:10d} {s['floor_boundary']:7d} "
                  f"{b / n * 100 if n else 0:6.2f}")
    print(f"total rows {sum(s['rows'] for s in stats.values())}, round mix {dict(sorted(round_mix.items()))}, "
          f"skipped {dict(tally['skipped'])}")
    print(f"replay lengths {({k: dict(v) for k, v in tally['replay_lengths'].items()})}")

    write_json(OUT_DIR / "metadata.json", {
        "generated_from": "rule-based decide_round_bid() labels through the v5 dispatch, no model involved",
        "data_version": context.version,
        "currency": context.currency,
        "mode": "reverse_auction",
        "scale_decision": ("deliberate v5 scaling decision: every one of the 50,000 v5 orders is used, not v4's "
                           "~3,000-order experimental sample"),
        "source_order_count": len(corpus["orders"]),
        "orders_with_rows": {s: len(v) for s, v in orders_with_rows.items()},
        "split_file": rel(SPLIT_PATH),
        "split_file_sha256": sha256_file(SPLIT_PATH),
        "split_rule": "order-id disjoint, the global v5 split shared by every stage, mode and the supervisor",
        "rounds_covered": list(range(1, auction_max_rounds() + 1)),
        "round_mix": {f"round_{r}": n for r, n in sorted(round_mix.items())},
        "replay_lengths_by_stage": {k: {str(n): c for n, c in sorted(v.items())}
                                    for k, v in tally["replay_lengths"].items()},
        "stop_rule": "an auction closes when a round brings no valid improvement, or at the round cap",
        "tie_break": "earliest arrival wins an identical offer, roster order offline",
        "minimum_bid_in_prompt": True,
        "worded_formulas": WORDED_FORMULAS,
        "minimum_bid_rounding": "ceil_to_cents via reverse_auction.agent_minimum_bid, shared by prompt and floor block",
        "label_rule": "src.common.reverse_auction.decide_round_bid, unmodified, v5 prices by dispatch",
        "prompt_builder": "src.common.slm.build_{material,manufacturer,logistics}_prompt with auction fields, v5 dispatch",
        "auction_replay": "src.experiments.generate_reverse_auction_data.replay_auction, imported and unmodified",
        "examples_per_pair": "one, rounds rotating over the rounds the replay ran",
        "round_1_label": "asserted equal to the sealed decision at build time",
        "platform_minimum_decrement": {
            "default": DEFAULT_PLATFORM_MINIMUM_DECREMENT,
            "currency": context.currency,
            "schedule": "every second order at the default, the rest at 5/10/20% of that stage's opening best bid",
            "note": ("bidding_modes documents the default as a EUR amount; it is a platform rule rather than a "
                     "sourced price, so v5 applies the same number in the order's currency, USD"),
        },
        "categories": {
            "auction_round": "the replayed round at the price the auction produced",
            "floor_boundary": ("constructed current_best_bid putting the next offer one percent above, exactly at, "
                               "and one cent below the agent's minimum_bid, inside the main corpus as in v4"),
        },
        "floor_pairs_per_stage": floor_pairs,
        "floor_pairs_used": dict(tally["floor_pairs_used"]),
        "floor_pairs_basis": ("v4's committed set used --floor-pairs 500 over 3,000 orders, one pair per six orders; "
                              "v5 keeps that rate over its pool"),
        "logistics_floor": ("a v5 logistics price carries no margin, so minimum_bid is the opening offer and every "
                            "logistics auction closes after round 2 with all round-2 rows NO_BID"),
        "decrement_mix": {f"{stage}:{kind}": n for (stage, kind), n in sorted(tally["decrement"].items())},
        "skipped": dict(tally["skipped"]),
        "counts": {k: v["rows"] for k, v in balance.items()},
        "class_balance": balance,
    })
    print(f"written to {rel(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
