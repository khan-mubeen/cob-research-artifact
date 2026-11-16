"""v5 supervisor chain-selection fine-tuning data, labelled against every eligible chain

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_supervisor_data

which chain is correct and which chains the model is shown are two separate questions here:

  correct  decide_chain_selection over EVERY material x manufacturer x logistics combination that
           survives eligibility and the order's budget and deadline. no widening and no shortlist
           takes part in the label. the supervisor audit of 16 September 2026 found the earlier
           five-cheapest-plus-fastest widening and the 12-candidate shortlist could both drop the
           real optimum, with ORD-06189 and ORD-10431 as saved examples
  shown    src.experiments.v5.chain_candidates.present: the true winner placed first and always
           present, the remaining places filled with the cheapest, fastest and lowest-risk chains
           and a seeded random sample as distractors, then shuffled. the cap is still the v4
           generator's MAX_CANDIDATES, which the 2048-token supervisor window is sized for

src/experiments/generate_supervisor_data.py is the source of truth for the v4 corpus and is not
edited; MAX_CANDIDATES, MIN_CANDIDATES, RANDOM_SEED and the seeded shortlist stream are imported.
v4 has one supervisor corpus, sealed only, so v5 has one too. what differs, because v5 differs:
  scale    all 50,000 v5 orders instead of v4's 12,000-order sample
  split    the global stratified train/validation/test split by order id
  bids     each stage's bids come from the dispatched v5 decide_*_bid functions, the stage labels,
           instead of v4's diagnostic re-implementation of the formulas
  origin   a v5 logistics offer depends on where the goods leave from, so each chain's logistics
           bids are priced from that chain's own manufacturer location (methodology section 7.5,
           "the candidate manufacturer's location"); a chain never pairs a manufacturer with a
           logistics price quoted from another city
  budget   the v5 order's budget_usd, read by the chain rule through the dataset context

output: data/finetune/v5/supervisor/selection_{train,validation,test}.jsonl and metadata.json
"""

import json
import random
import statistics
import sys
from collections import Counter

from src.common.bidding_formulas import chain_from_bids, chain_priority_key, decide_chain_selection
from src.common.slm import build_chain_selection_prompt
from src.experiments.generate_supervisor_data import MAX_CANDIDATES, MIN_CANDIDATES, RANDOM_SEED
from src.experiments.v5.chain_candidates import agents_of, enumerate_chains, present
from src.experiments.v5.common import (
    FINETUNE_V5, SPLIT_PATH, SPLITS, JsonlFiles, Market, load_corpus, read_split, rel, require_v5, sha256_file,
    stage_rfqs, walk_orders, write_json,
)

OUT_DIR = FINETUNE_V5 / "supervisor"

REASONS = {
    "urgent": "fewest total completion days for an urgent order",
    "flexible": "lowest total cost for a flexible order",
    "normal": "best equal-weight balance of cost, days and compound risk",
}


def tied_with(feasible, winner_key, order):
    """how many feasible chains share the winner's key up to the agent-id tie-break"""
    return sum(1 for c in feasible if chain_priority_key(chain_from_bids(c), order)[:-1] == winner_key[:-1])


def build_rows(context, corpus, market, split_of, walk, tally):
    """(split, row) for every order walked that yields a usable selection example"""
    rng = random.Random(RANDOM_SEED + 7)
    for o in walk:
        rfqs = stage_rfqs(o, corpus, market, context)
        if rfqs is None:
            tally["orders_without_a_product_pair"] += 1
            continue
        remaining = rfqs["tailoring"]["days_until_deadline"]
        order = {"order_id": o["order_id"], "priority": o["priority"], "budget_usd": float(o["budget_usd"]),
                 "days_until_deadline": remaining, "max_defect_rate": float(o["max_defect_rate"]),
                 **context.as_dict()}

        chains, counts = enumerate_chains(o, corpus, market, rfqs)
        if not chains:
            tally["stage_had_no_bidder"] += 1
            continue

        feasible = [c for c in chains if c["total_cost"] <= order["budget_usd"] and c["total_days"] <= remaining]
        if len(feasible) < MIN_CANDIDATES:
            tally["fewer_than_min_candidates"] += 1
            continue

        # the label reads every feasible chain, never a sample of them
        selection = decide_chain_selection([chain_from_bids(c) for c in feasible], order)
        if selection["decision"] != "SELECT":
            tally["rule_declined"] += 1
            continue
        winner = (selection["material_agent_id"], selection["manufacturer_agent_id"],
                  selection["logistics_agent_id"])

        shown = present(feasible, winner, rng)
        index = next(i for i, c in enumerate(shown) if agents_of(c) == winner)
        winner_key = chain_priority_key(chain_from_bids(shown[index]), order)
        system_prompt, user_prompt = build_chain_selection_prompt(order, shown, remaining)
        split = split_of[o["order_id"]]
        yield split, {
            "order_id": o["order_id"],
            "stage": "supervisor",
            "dataset_version": context.version,
            "split": split,
            "priority": o["priority"],
            "strategy": selection["strategy"],
            "n_candidates": len(shown),
            "n_chains_enumerated": counts["combinations"],
            "n_chains_feasible": len(feasible),
            "n_tied_optima": tied_with(feasible, winner_key, order),
            "system": system_prompt,
            "user": user_prompt,
            "completion": json.dumps({"selected_candidate": index, "reason": REASONS[selection["strategy"]]}),
            "label_chain": [selection["material_agent_id"], selection["manufacturer_agent_id"],
                            selection["logistics_agent_id"]],
            "label_total_cost": selection["total_cost"],
            "label_total_days": selection["total_completion_days"],
            "label_chain_risk": selection["chain_risk"],
        }


def distribution(values):
    values = sorted(values)
    return {"min": values[0], "median": statistics.median(values), "mean": round(statistics.mean(values), 4),
            "p95": values[int(len(values) * 0.95)], "max": values[-1]} if values else None


def main():
    context = require_v5()
    corpus = load_corpus()
    market = Market()
    split_of = read_split()
    walk = walk_orders(corpus["orders"])

    tally = Counter()
    strategy_counts, counts = Counter(), Counter()
    candidates_per_example, picked = [], []
    enumerated, feasible_counts, tied = [], [], []
    paths = {split: OUT_DIR / f"selection_{split}.jsonl" for split in SPLITS}
    with JsonlFiles(paths) as files:
        for split, row in build_rows(context, corpus, market, split_of, walk, tally):
            files.write(split, row)
            counts[split] += 1
            strategy_counts[row["strategy"]] += 1
            candidates_per_example.append(row["n_candidates"])
            picked.append(json.loads(row["completion"])["selected_candidate"])
            enumerated.append(row["n_chains_enumerated"])
            feasible_counts.append(row["n_chains_feasible"])
            tied.append(row["n_tied_optima"])

    total = sum(counts.values())
    skipped = {k: tally[k] for k in ("orders_without_a_product_pair", "stage_had_no_bidder",
                                     "fewer_than_min_candidates", "rule_declined")}
    print(f"usable examples {total} {dict(counts)}, skipped {skipped}, strategy mix {dict(strategy_counts)}")
    if total:
        print(f"candidates per example mean {sum(candidates_per_example) / total:.2f}, "
              f"selected index mean {sum(picked) / total:.2f}")
        print(f"chains enumerated per order {distribution(enumerated)}")
        print(f"rows whose optimum is tied with another chain {sum(1 for t in tied if t > 1)}")
    write_json(OUT_DIR / "metadata.json", {
        "generated_from": "rule-based decide_chain_selection() labels through the v5 dispatch, no model involved",
        "data_version": context.version,
        "currency": context.currency,
        "scale_decision": ("deliberate v5 scaling decision: every one of the 50,000 v5 orders is used, not v4's "
                           "12,000-order supervisor sample"),
        "source_order_count": len(corpus["orders"]),
        "split_file": rel(SPLIT_PATH),
        "split_file_sha256": sha256_file(SPLIT_PATH),
        "split_rule": "order-id disjoint, the global v5 split; an order's example lands entirely in one split",
        "single_corpus": "v4 has no separate reverse-auction supervisor set, so v5 has none either",
        "max_candidates_per_prompt": MAX_CANDIDATES,
        "min_candidates_per_prompt": MIN_CANDIDATES,
        "candidate_source": ("every eligible material x manufacturer x logistics combination, each chain's logistics "
                             "priced from its own manufacturer location; no widening and no shortlist takes part in "
                             "deciding which chain is correct"),
        "presented_candidates": ("src.experiments.v5.chain_candidates.present, random.Random(42 + 7): the true winner "
                                 "is placed first and unconditionally, the remaining places go to the cheapest, "
                                 "fastest and lowest-risk chains and then a seeded random sample as distractors, and "
                                 "the list is shuffled"),
        "guaranteed_inclusion": ("the labelled index always points at the full-enumeration winner's position in the "
                                 "shown list"),
        "label_rule": ("src.common.bidding_formulas.decide_chain_selection over every feasible chain, reading "
                       "budget_usd through the context"),
        "tie_break": ("cost and days are compared at two decimals; an exact tie falls through to the "
                      "(material, manufacturer, logistics) agent-id tuple, ascending"),
        "prompt_builder": "src.common.slm.build_chain_selection_prompt, v5 dispatch",
        "counts": {s: counts[s] for s in SPLITS},
        "skipped": skipped,
        "strategy_mix": dict(strategy_counts),
        "candidates_per_example_mean": round(sum(candidates_per_example) / total, 4) if total else None,
        "selected_index_mean": round(sum(picked) / total, 4) if total else None,
        "chains_enumerated_per_order": distribution(enumerated),
        "chains_feasible_per_order": distribution(feasible_counts),
        "rows_with_a_tied_optimum": sum(1 for t in tied if t > 1),
    })
    print(f"written to {rel(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
