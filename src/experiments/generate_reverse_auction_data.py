"""
build reverse-auction fine-tuning data from the rule-based labels.

same shape as generate_finetune_data.py, which does this for the sealed round:
sample real orders, run the rule, use its output as the training label. no model,
no nats, pure formula calculation.

the label comes from decide_round_bid() in reverse_auction.py and the prompt from
build_*_prompt(agent, rfq, auction), the same function and the same builder the
live agents call, so training labels and runtime behavior cannot drift.

every order and stage replays a full three-round auction with the runtime pieces,
AuctionMemory for withdrawal and best_standing_bid for the announced price, so
current_best_bid and the standing bids are the values a live auction would have
produced rather than a re-derivation.

the order sample, the seed and the holdout ids are the sealed set's, so a model
trained on both sets never trains on an order held out from either.

a live auction prompts every eligible agent once in each of the three rounds, so
one row per pair rotates across rounds 1, 2 and 3 and the set matches deployment.
round 1 carries a null current_best_bid and its label is the sealed decision,
which is asserted here rather than assumed

two categories, both carried in the `category` field finetune_lora already reads:
- auction_round   one round per eligible agent-order pair, rounds 1, 2 and 3
                  rotating, at the current_best_bid the replay produced
- floor_boundary  constructed current_best_bid placing the next offer one cent
                  above, exactly at, and one cent below the agent's minimum_bid.
                  at the deployed EUR 5 decrement an on-distribution round almost
                  never reaches a floor, so the threshold this dataset exists to
                  teach would otherwise be covered only incidentally

output: data/finetune/{version}/reverse_auction/{stage}/{split}.jsonl, plus
metadata.json and dataset_hashes.json alongside.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict

from config import DATA_VERSION, PROJECT_ROOT, REVERSE_AUCTION_VERSION
from src.common.bidding_formulas import eligibility_reason, stage_price_terms
from src.common.bidding_modes import (
    DEFAULT_PLATFORM_MINIMUM_DECREMENT, auction_max_rounds,
)
from src.common.reverse_auction import (
    SEALED_DECIDERS, AuctionMemory, auction_fields, best_standing_bid, ceil_to_cents,
    decide_round_bid, next_round_offer, round_improved,
)
from src.common.slm import (
    build_logistics_prompt, build_manufacturer_prompt, build_material_prompt,
)
from src.experiments.generate_finetune_data import completion_json
from src.visualization.data_utils import load_all_data
from src.visualization.slm_competition_diagnostics import build_stage_orders


# the sealed set's sample, seed and split, deliberately identical
ORDER_SAMPLE_SIZE = 3000
HOLDOUT_FRACTION = 0.10
RANDOM_SEED = 42

# every second sampled order runs at the deployed default, the rest at a fraction
# of that stage's opening best bid so floors are reached at every price magnitude:
# a fixed eur ladder that bites at tailoring (~EUR 4000) is meaningless at
# logistics (~EUR 300). platform_minimum_decrement is a prompt field precisely
# because it is configurable, a set built only at 5 teaches the constant
DECREMENT_SCHEDULE = (None, 0.05, None, 0.10, None, 0.20)

# (order, agent) pairs per stage carrying the three-case floor block
FLOOR_PAIRS_PER_STAGE = 64

# full-set token length of the rendered training text, per model tokenizer, measured
# 2026-08-18 over all 53,758 worded rows. carried as a literal because generation
# must not depend on six tokenizers being downloadable. re-measure with
# scratchpad/measure_tokens.py whenever the prompt text changes
TOKEN_LENGTHS = {
    "qwen":      {"max": 954, "p99": 947, "median": 824, "mean": 848.2, "headroom": 70},
    "granite":   {"max": 983, "p99": 978, "median": 852, "mean": 881.0, "headroom": 41},
    "phi":       {"max": 859, "p99": 855, "median": 750, "mean": 768.1, "headroom": 165},
    "ministral": {"max": 961, "p99": 953, "median": 818, "mean": 848.2, "headroom": 63},
    "gemma":     {"max": 1016, "p99": 1008, "median": 864, "mean": 899.8, "headroom": 8},
    "smollm3":   {"max": 903, "p99": 900, "median": 794, "mean": 812.5, "headroom": 121},
}

STAGES = ("material", "tailoring", "logistics")

STAGE_PROMPTS = {
    "material": build_material_prompt,
    "tailoring": build_manufacturer_prompt,
    "logistics": build_logistics_prompt,
}

STAGE_DEFECT_KEY = {
    "material": "defect_probability",
    "tailoring": "defect_probability",
    "logistics": "damage_probability",
}

OUT_DIR = (PROJECT_ROOT / "data" / "finetune" / DATA_VERSION / "reverse_auction"
           / REVERSE_AUCTION_VERSION)
SEALED_DIR = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION


def round_rfq(rfq, round_number, current_best_bid, decrement):
    """the rfq an agent receives in one auction round"""
    return {**rfq, "round_number": round_number,
            "current_best_bid": current_best_bid,
            "platform_minimum_decrement": decrement}


def replay_auction(stage, agents, rfq, decrement):
    """replay the auction with the runtime pieces

    the same stop rule the supervisor applies: a round that brings no valid
    improvement closes the auction, so a replay can be shorter than the cap and
    the row mix follows what a live auction would really have produced

    agents are polled in roster order and stamped with that arrival order, which
    is the tie-break for the identical offers every later round produces

    return per round: the announced best bid and, per agent, the rfq it saw, the
    rule's decision, and whether it had already dropped out. withdrawn agents are
    still labeled by the rule rather than by the memory, and the two are asserted
    to agree: current_best_bid is non-increasing across rounds, so an agent whose
    next offer fell below its floor once can never be above it later"""
    order_id = rfq["order_id"]
    memory = AuctionMemory()
    standing = {}
    best = None
    rounds = []
    arrival = 0

    for round_number in range(1, auction_max_rounds() + 1):
        announced = best["offer_amount"] if best is not None else None
        entries = {}

        for agent in agents:
            agent_id = agent["agent_id"]
            seen = round_rfq(rfq, round_number, announced, decrement)
            if round_number == 1:
                memory.start_round_one(agent_id, order_id, stage)
                withdrawn = False
            else:
                withdrawn = memory.has_withdrawn(agent_id, order_id, stage)

            decision = decide_round_bid(stage, agent, seen)
            if withdrawn and decision["decision"] == "BID":
                raise AssertionError(
                    f"{agent_id} withdrew earlier yet the rule bids in round "
                    f"{round_number} on {order_id}, the monotonicity assumption is broken"
                )
            if decision["decision"] != "BID":
                memory.withdraw(agent_id, order_id, stage)
            elif not withdrawn:
                # a silent agent's earlier offer stays on the table, exactly as
                # the supervisor treats standing bids
                arrival += 1
                standing[agent_id] = {**decision, "received_seq": arrival}

            entries[agent_id] = {"rfq": seen, "decision": decision, "withdrawn": withdrawn}

        best = best_standing_bid(list(standing.values()))
        rounds.append({"round_number": round_number, "announced": announced, "entries": entries})

        if not round_improved(announced, best["offer_amount"] if best is not None else None):
            break

    return rounds


def floor_cases(stage, agent, rfq, decrement):
    """three constructed best bids straddling this agent's own floor, at cents

    the current_best_bid is chosen so the mandated next offer lands exactly on
    the floor, one cent under it and one percent over it"""
    # the same cent-ceiling the prompt displays, so construction and prompt agree
    minimum_bid, _ = stage_price_terms(stage, agent, rfq)
    floor = ceil_to_cents(minimum_bid)
    return [
        ("above_floor", round(round(floor * 1.01, 2) + decrement, 2)),
        ("at_floor", round(floor + decrement, 2)),
        ("one_cent_below_floor", round(floor - 0.01 + decrement, 2)),
    ]


def make_record(stage, agent, seen, decision, category, withdrawn, worded=False):
    """one training row, prompt and label both from the runtime path"""
    prompt_fn = STAGE_PROMPTS[stage]
    system_prompt, user_prompt = prompt_fn(agent, seen, auction_fields(stage, agent, seen),
                                           worded)
    defect_probability = agent.get(STAGE_DEFECT_KEY[stage])
    return {
        "order_id": seen["order_id"],
        "agent_id": agent["agent_id"],
        "stage": stage,
        "category": category,
        "round_number": seen["round_number"],
        "current_best_bid": seen["current_best_bid"],
        "platform_minimum_decrement": seen["platform_minimum_decrement"],
        "withdrawn_in_earlier_round": withdrawn,
        "rule_decision": decision["decision"],
        "system": system_prompt,
        "user": user_prompt,
        "completion": completion_json(
            stage, seen["order_id"], agent["agent_id"], decision, defect_probability,
        ),
    }


def sealed_split_ids():
    """order ids the sealed set put in each split, for the cross-set check

    the sealed files only hold orders that had an eligible agent, so this is a
    subset of the sampled ids, which is what the assertions below account for"""
    ids = {}
    for split in ("train", "holdout"):
        seen = set()
        for stage in STAGES:
            path = SEALED_DIR / f"{stage}_{split}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seen.add(json.loads(line)["order_id"])
        ids[split] = seen
    return ids


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# one row from each interesting case rather than the first n, which would all
# come from the same order and usually the same class
SAMPLE_PICKS = (
    ("bid at the mandated offer",
     lambda r: r["category"] == "auction_round" and r["rule_decision"] == "BID"),
    ("dropped out this round",
     lambda r: r["category"] == "auction_round" and r["rule_decision"] == "NO_BID"
     and not r["withdrawn_in_earlier_round"]),
    ("withdrawn in an earlier round",
     lambda r: r["withdrawn_in_earlier_round"]),
    ("floor boundary, one cent below",
     lambda r: r["category"] == "floor_boundary:one_cent_below_floor"),
    ("floor boundary, exactly at floor",
     lambda r: r["category"] == "floor_boundary:at_floor"),
)


def show_examples(rows, stage, limit):
    """a trimmed view of one row per case, the prompts are ~2000 characters each"""
    for shown, (title, matches) in enumerate(SAMPLE_PICKS[:limit], start=1):
        row = next((r for r in rows if matches(r)), None)
        if row is None:
            print(f"\n  [{stage} {shown}] {title}: no row of this kind")
            continue
        rfq_tail = row["user"][row["user"].index("RFQ"):row["user"].index("TASK")].strip()
        print(f"\n  [{stage} {shown}] {title}")
        print(f"    {row['order_id']} {row['agent_id']} category={row['category']} "
              f"round={row['round_number']} withdrawn={row['withdrawn_in_earlier_round']}")
        for line in rfq_tail.splitlines()[-4:]:
            print(f"    rfq | {line}")
        print(f"    label | {' '.join(row['completion'].split())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=ORDER_SAMPLE_SIZE)
    parser.add_argument("--floor-pairs", type=int, default=FLOOR_PAIRS_PER_STAGE,
                        help="(order, agent) pairs per stage carrying the three floor cases")
    parser.add_argument("--show", type=int, default=3,
                        help="sample rows to print per stage")
    parser.add_argument("--worded-formulas", action="store_true",
                        help="restate every operator in the TASK block in words beside the "
                             "symbol, testing whether bare symbols are the at_floor cause")
    parser.add_argument("--holdout-only", action="store_true",
                        help="write only the holdout files, leaving the training files exactly "
                             "as they are; used to correct evaluation data without disturbing "
                             "the data existing adapters were trained on")
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts and samples, write nothing")
    args = parser.parse_args()

    materials, manufacturers, logistics, products, orders = load_all_data()
    stage_agents = {
        "material": materials.to_dict("records"),
        "tailoring": manufacturers.to_dict("records"),
        "logistics": logistics.to_dict("records"),
    }

    # identical to generate_finetune_data.py, so both sets share a split
    rng = random.Random(RANDOM_SEED)
    sampled = orders.sample(n=min(args.sample, len(orders)), random_state=RANDOM_SEED)
    order_ids = list(sampled["order_id"])
    rng.shuffle(order_ids)
    holdout_count = int(len(order_ids) * HOLDOUT_FRACTION)
    holdout_ids = set(order_ids[:holdout_count])

    print(f"version {DATA_VERSION}: {len(sampled)} orders sampled from {len(orders)}, "
          f"seed {RANDOM_SEED}, {len(holdout_ids)} holdout / {len(order_ids) - len(holdout_ids)} train "
          "(disjoint by order id, same split as the sealed set)")

    rows = {(stage, split): [] for stage in STAGES for split in ("train", "holdout")}
    counts = defaultdict(int)
    pair_index = defaultdict(int)
    floor_pairs_used = defaultdict(int)
    skipped_orders = 0
    decrement_used = defaultdict(int)

    for order_position, (_, order) in enumerate(sampled.iterrows()):
        stage_orders = build_stage_orders(order, products)
        if stage_orders is None:
            skipped_orders += 1
            continue
        split = "holdout" if order["order_id"] in holdout_ids else "train"
        fraction = DECREMENT_SCHEDULE[order_position % len(DECREMENT_SCHEDULE)]

        for stage in STAGES:
            rfq = stage_orders[stage]
            agents = [a for a in stage_agents[stage]
                      if eligibility_reason(stage, a, rfq) is None]
            if not agents:
                continue

            # the opening best bid sets a stage-relative decrement, so the ladder
            # bites the same way at every price magnitude
            opening = best_standing_bid([
                d for d in (decide_round_bid(stage, a, round_rfq(rfq, 1, None, DEFAULT_PLATFORM_MINIMUM_DECREMENT))
                            for a in agents) if d["decision"] == "BID"
            ])
            if fraction is None or opening is None:
                decrement = DEFAULT_PLATFORM_MINIMUM_DECREMENT
            else:
                decrement = max(round(opening["offer_amount"] * fraction, 2), 0.01)
            decrement_used[(stage, "default" if decrement == DEFAULT_PLATFORM_MINIMUM_DECREMENT
                            else f"{fraction:.0%}_of_opening")] += 1

            replay = replay_auction(stage, agents, rfq, decrement)

            for agent in agents:
                agent_id = agent["agent_id"]
                # one row per pair, rotating across the rounds this auction
                # actually ran, which the stop rule can cut short
                round_number = 1 + (pair_index[stage] % len(replay))
                pair_index[stage] += 1
                entry = replay[round_number - 1]["entries"][agent_id]
                if round_number == 1:
                    # round 1 must be the sealed decision, that is what keeps the
                    # two modes comparable stage by stage
                    sealed = SEALED_DECIDERS[stage](agent, rfq)
                    if (entry["decision"]["decision"] != sealed["decision"]
                            or entry["decision"].get("offer_amount") != sealed.get("offer_amount")):
                        raise AssertionError(
                            f"round 1 differs from the sealed decision for {agent_id} "
                            f"on {rfq['order_id']}"
                        )
                rows[(stage, split)].append(make_record(
                    stage, agent, entry["rfq"], entry["decision"],
                    "auction_round", entry["withdrawn"], args.worded_formulas,
                ))

            # floor block, one agent per order so the cases spread across orders
            if floor_pairs_used[stage] < args.floor_pairs:
                opener = next((a for a in agents
                               if replay[0]["entries"][a["agent_id"]]["decision"]["decision"] == "BID"),
                              None)
                if opener is not None:
                    floor_pairs_used[stage] += 1
                    for case, current_best in floor_cases(
                            stage, opener, rfq, DEFAULT_PLATFORM_MINIMUM_DECREMENT):
                        number = 2 + (floor_pairs_used[stage] % 2)
                        seen = round_rfq(rfq, number, current_best,
                                         DEFAULT_PLATFORM_MINIMUM_DECREMENT)
                        rows[(stage, split)].append(make_record(
                            stage, opener, seen, decide_round_bid(stage, opener, seen),
                            f"floor_boundary:{case}", False, args.worded_formulas,
                        ))

    # ── validation ────────────────────────────────────────────────────────────
    train_orders = {r["order_id"] for stage in STAGES for r in rows[(stage, "train")]}
    hold_orders = {r["order_id"] for stage in STAGES for r in rows[(stage, "holdout")]}
    overlap = train_orders & hold_orders
    print(f"\ntrain orders {len(train_orders)}, holdout orders {len(hold_orders)}, "
          f"order overlap {len(overlap)}")
    assert not overlap, f"train/holdout order overlap: {sorted(overlap)[:5]}"

    # the sealed split only matches at the canonical sample size, a smaller
    # --sample draws different orders and shuffles them differently
    sealed = sealed_split_ids() if args.sample == ORDER_SAMPLE_SIZE else {"train": set(), "holdout": set()}
    if sealed["train"] or sealed["holdout"]:
        cross = (sealed["train"] & hold_orders) | (sealed["holdout"] & train_orders)
        print(f"cross-set check against the sealed split: {len(cross)} order(s) on the wrong side")
        assert not cross, f"order split disagrees with the sealed set: {sorted(cross)[:5]}"
    else:
        print(f"cross-set check skipped (sample {args.sample} is not the sealed set's {ORDER_SAMPLE_SIZE})")

    for stage in STAGES:
        for row in rows[(stage, "train")] + rows[(stage, "holdout")]:
            if row["withdrawn_in_earlier_round"]:
                assert row["rule_decision"] == "NO_BID", (
                    f"withdrawn agent labeled {row['rule_decision']} on {row['order_id']}"
                )
            if row["rule_decision"] == "BID" and row["current_best_bid"] is not None:
                expected = next_round_offer(row["current_best_bid"],
                                            row["platform_minimum_decrement"])
                got = json.loads(row["completion"])["offer_amount"]
                assert abs(got - expected) < 1e-9, (
                    f"offer {got} is not current_best_bid - decrement ({expected})"
                )

    # ── class balance ─────────────────────────────────────────────────────────
    print(f"\norders skipped (no product/material pair): {skipped_orders}")
    print(f"\n{'stage':10s} {'split':8s} {'rows':>7s} {'BID':>7s} {'NO_BID':>7s} "
          f"{'withdrawn':>10s} {'floor':>6s} {'bid %':>6s}")
    balance = {}
    total = 0
    for stage in STAGES:
        for split in ("train", "holdout"):
            items = rows[(stage, split)]
            n = len(items)
            bid = sum(1 for r in items if r["rule_decision"] == "BID")
            withdrawn = sum(1 for r in items if r["withdrawn_in_earlier_round"])
            floor = sum(1 for r in items if r["category"].startswith("floor_boundary"))
            total += n
            balance[f"{stage}_{split}"] = {
                "rows": n, "BID": bid, "NO_BID": n - bid,
                "withdrawn": withdrawn, "floor_boundary": floor,
            }
            pct = bid / n * 100 if n else 0.0
            print(f"{stage:10s} {split:8s} {n:7d} {bid:7d} {n - bid:7d} "
                  f"{withdrawn:10d} {floor:6d} {pct:5.1f}%")
    print(f"\ntotal examples: {total}   (withdrawn is a subset of NO_BID)")
    round_mix = {
        f"round_{r}": sum(1 for s in STAGES for sp in ("train", "holdout")
                          for x in rows[(s, sp)] if x["round_number"] == r)
        for r in range(1, auction_max_rounds() + 1)
    }
    print(f"round mix: {round_mix}")
    print(f"decrement mix: {dict(sorted(decrement_used.items()))}")

    if args.show:
        for stage in STAGES:
            print(f"\n--- {stage} samples ---")
            show_examples(rows[(stage, "train")], stage, args.show)

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    written = {}
    for stage in STAGES:
        stage_dir = OUT_DIR / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        splits = ("holdout",) if args.holdout_only else ("train", "holdout")
        for split in splits:
            path = stage_dir / f"{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for row in rows[(stage, split)]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written[path] = len(rows[(stage, split)])
            print(f"wrote {len(rows[(stage, split)]):6d} -> {path}")
        if args.holdout_only:
            print(f"       left {stage_dir / 'train.jsonl'} untouched")

    meta = {
        "generated_from": "rule-based decide_round_bid() labels, no model involved",
        "data_version": DATA_VERSION,
        "mode": "reverse_auction",
        "rounds_covered": list(range(1, auction_max_rounds() + 1)),
        "round_mix": round_mix,
        "stop_rule": "an auction closes when a round brings no valid improvement, "
                     "or at the round cap; replays follow the runtime rule",
        "tie_break": "earliest arrival wins an identical offer, roster order offline",
        "minimum_bid_in_prompt": True,
        "worded_formulas": args.worded_formulas,
        "minimum_bid_rounding": "ceil_to_cents, the lowest legal cent offer, so the displayed "
                                "floor is exactly the rule's decision threshold",
        "holdout_only_regeneration": args.holdout_only,
        "holdout_only_note": ("holdout regenerated after the floor-rounding fix; the training "
                              "files predate it and still show round(minimum_bid, 2), so the six "
                              "adapters trained on them are unchanged"
                              if args.holdout_only else None),
        "order_sample_size": len(sampled),
        "holdout_fraction": HOLDOUT_FRACTION,
        "random_seed": RANDOM_SEED,
        "split_rule": "order-id disjoint; identical sample, seed and holdout ids to the sealed set",
        "label_rule": "src.common.reverse_auction.decide_round_bid",
        "prompt_builder": "src.common.slm.build_{material,manufacturer,logistics}_prompt with auction fields",
        "auction_replay": "AuctionMemory + best_standing_bid, the runtime pieces",
        "examples_per_pair": "one, rounds 1/2/3 rotating, matching what a live auction prompts",
        "round_1_label": "the sealed decide_*_bid() decision in an auction-shaped prompt with a "
                         "null current_best_bid, asserted equal to the sealed decision at build time",
        "platform_minimum_decrement": {
            "default": DEFAULT_PLATFORM_MINIMUM_DECREMENT,
            "schedule": "every second order at the default, the rest at 5/10/20% of that "
                        "stage's opening best bid",
        },
        "categories": {
            "auction_round": "the replayed round at the price the auction produced, rounds 1/2/3",
            "floor_boundary": "constructed current_best_bid putting the next offer above, "
                              "at, and one cent below the agent's minimum_bid",
        },
        "withdrawn_in_earlier_round": "labeled by the rule, not by the memory; the runtime "
                                      "short-circuits these before the model, so they are "
                                      "defensive coverage",
        "counts": {key: value["rows"] for key, value in balance.items()},
        "class_balance": balance,
        "floor_pairs_per_stage": args.floor_pairs,
        "max_seq_length_required": 1024,
        "token_length_check": "measured 2026-08-18 on the full worded set, every row, with each "
                              "model's own tokenizer and chat template: no row exceeds the 1024 "
                              "window for any of the six, so nothing truncates. headroom is thin "
                              "for gemma (8 tokens) and granite (41); lengthening the prompt "
                              "again would truncate the tail, which is the output schema",
        "token_length_by_tokenizer": TOKEN_LENGTHS,
    }
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    hashes = {}
    for path in list(written) + [OUT_DIR / "metadata.json"]:
        hashes[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    with open(OUT_DIR / "dataset_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
