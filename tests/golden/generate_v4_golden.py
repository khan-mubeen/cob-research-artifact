"""regenerate the golden v4 fixtures from the live v4 formulas

  PYTHONPATH=. python tests/golden/generate_v4_golden.py

run this only to record a v4 change that has been agreed deliberately. the whole
point of the fixtures is that an accidental change shows up as a failing test, so
regenerating to make tests/test_v4_golden.py pass again defeats the harness

the exhaustive census walks all 50,000 orders. pass --orders N to shorten it while
iterating; the committed fixture is always the full corpus
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from src.common.bidding_modes import AUCTION_ROUNDS, DEFAULT_PLATFORM_MINIMUM_DECREMENT
from src.common.slm import (
    build_chain_selection_prompt, build_logistics_prompt,
    build_manufacturer_prompt, build_material_prompt,
)
from src.common.bidding_formulas import (
    chain_from_bids, decide_chain_selection, eligibility_reason,
)
from src.common.reverse_auction import auction_fields, decide_round_bid
from tests.golden import v4_boundaries
from tests.golden.v4_baseline import (
    GOLDEN_DIR, PROMPT_ORDER_IDS, SAMPLE_STRIDE, SEALED_DECIDERS, STAGES,
    census_digest, decision_record, load_v4, sample_decisions, sample_orders,
    stage_rfqs, v4_file_hashes,
)


def write(name, payload):
    path = GOLDEN_DIR / name
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    print(f"  wrote {path.relative_to(GOLDEN_DIR.parent.parent)} "
          f"({path.stat().st_size / 1024:.0f} kb)")


def boundary_fixture():
    records = []
    for name, stage, agent, rfq in v4_boundaries.sealed_cases():
        bid = SEALED_DECIDERS[stage](agent, rfq)
        records.append({"case": name, **decision_record(stage, agent, rfq, bid)})
    for name, stage, agent, rfq in v4_boundaries.auction_cases():
        bid = decide_round_bid(stage, agent, rfq)
        records.append({"case": name, **decision_record(stage, agent, rfq, bid),
                        "round_number": rfq["round_number"],
                        "current_best_bid": rfq["current_best_bid"]})
    return records


PROMPT_BUILDERS = {
    "material": build_material_prompt,
    "tailoring": build_manufacturer_prompt,
    "logistics": build_logistics_prompt,
}


def prompt_fixture(corpus):
    """byte-for-byte prompt text for a fixed small sample of orders

    sealed and reverse-auction wording for all three stages, and the supervisor
    chain-selection prompt with the label the rule produces for it"""
    by_id = {o["order_id"]: o for o in corpus["orders"]}
    stage_prompts = []
    chain_prompts = []

    for order_id in PROMPT_ORDER_IDS:
        order = by_id[order_id]
        rfqs = stage_rfqs(order, corpus["products"])
        candidates = []

        for stage in STAGES:
            rfq = rfqs[stage]
            # the first agent this order is actually eligible with: an ineligible
            # agent never reaches a prompt in either path, and has no floor to show
            agent = next(a for a in corpus[stage] if eligibility_reason(stage, a, rfq) is None)
            build = PROMPT_BUILDERS[stage]

            system, user = build(agent, rfq)
            stage_prompts.append({"order_id": order_id, "stage": stage,
                                  "agent_id": agent["agent_id"], "mode": "sealed",
                                  "system": system, "user": user})

            auction_rfq = {**rfq, "round_number": 2,
                           "platform_minimum_decrement": DEFAULT_PLATFORM_MINIMUM_DECREMENT,
                           "current_best_bid": 1000.0}
            system, user = build(agent, auction_rfq,
                                 auction=auction_fields(stage, agent, auction_rfq))
            stage_prompts.append({"order_id": order_id, "stage": stage,
                                  "agent_id": agent["agent_id"], "mode": "reverse_auction",
                                  "system": system, "user": user})

        # two candidate chains per order, built from the cheapest and the second
        # cheapest bidding agent at each stage, so the selection label is real
        picked = {}
        for stage in STAGES:
            bids = []
            for agent in corpus[stage]:
                bid = SEALED_DECIDERS[stage](agent, rfqs[stage])
                if bid["decision"] == "BID":
                    bids.append(bid)
            picked[stage] = sorted(bids, key=lambda b: (b["offer_amount"], b["agent_id"]))[:2]

        if all(len(picked[s]) >= 1 for s in STAGES):
            width = min(len(picked[s]) for s in STAGES)
            for i in range(width):
                bids = {s: picked[s][i] for s in STAGES}
                candidates.append({
                    "material": bids["material"],
                    "tailoring": bids["tailoring"],
                    "logistics": bids["logistics"],
                    "total_cost": sum(b["offer_amount"] for b in bids.values()),
                    "total_days": sum(b["estimated_completion_days"] for b in bids.values()),
                })

        if candidates:
            remaining_days = float(rfqs["material"]["days_until_deadline"])
            system, user = build_chain_selection_prompt(order, candidates, remaining_days)
            chains = [chain_from_bids(c) for c in candidates]
            label = decide_chain_selection(
                chains, {**order, "days_until_deadline": remaining_days})
            chain_prompts.append({"order_id": order_id, "system": system, "user": user,
                                  "label": label})

    return {"stage_prompts": stage_prompts, "chain_prompts": chain_prompts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, default=None,
                        help="limit the exhaustive census, for iteration only")
    args = parser.parse_args()

    print("loading the committed v4 corpus")
    corpus = load_v4()

    print("hashing the six tracked v4 files")
    write("v4_file_hashes.json", {
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sha256": v4_file_hashes(),
    })

    print("recording constructed boundary decisions")
    write("v4_boundary_decisions.json", boundary_fixture())

    print(f"recording sampled decisions, every {SAMPLE_STRIDE}th order")
    write("v4_sample_decisions.json", {
        "stride": SAMPLE_STRIDE,
        "auction_rounds": AUCTION_ROUNDS,
        "platform_minimum_decrement": DEFAULT_PLATFORM_MINIMUM_DECREMENT,
        "order_ids": [o["order_id"] for o in sample_orders(corpus)],
        "decisions": sample_decisions(corpus),
    })

    print("recording prompt text and chain-selection labels")
    write("v4_prompts.json", prompt_fixture(corpus))

    scope = "all orders" if args.orders is None else f"first {args.orders} orders"
    print(f"running the exhaustive decision census over {scope}, this takes a moment")
    census = census_digest(corpus, order_limit=args.orders)
    if args.orders is not None:
        print("  --orders was set, not writing the census fixture")
        print(json.dumps(census, indent=2))
        return 0
    write("v4_decision_census.json", census)
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
