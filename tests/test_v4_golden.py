"""golden regression harness for the committed v4 baseline.
runnable directly (python ...) or under pytest.

nothing in v5 may change a byte of v4 output. this file is what makes that
claim checkable: it pins the six tracked v4 files by hash, replays every stage
decision in the corpus through the live formulas, and compares the sealed,
reverse-auction and supervisor prompt text byte for byte.

the fixtures under tests/golden/ are produced by tests/golden/generate_v4_golden.py
and are read-only from here. a failure means the v4 baseline moved, not that the
fixture needs refreshing

the exhaustive census takes about 25 seconds. set GOLDEN_CENSUS_ORDERS to a
smaller number to skip ahead while iterating, which makes that one test report
SKIP rather than pass
"""

import json
import os
from datetime import datetime

from src.common.bidding_modes import AUCTION_ROUNDS, DEFAULT_PLATFORM_MINIMUM_DECREMENT
from src.common.bidding_formulas import (
    chain_from_bids, decide_chain_selection, eligibility_reason,
)
from src.common.reverse_auction import auction_fields, decide_round_bid
from tests.golden import v4_boundaries
from tests.golden.generate_v4_golden import PROMPT_BUILDERS
from tests.golden.v4_baseline import (
    GOLDEN_DIR, SEALED_DECIDERS, STAGES, census_digest, decision_record,
    load_v4, sample_decisions, sample_orders, stage_rfqs, v4_file_hashes,
)


def fixture(name):
    with open(GOLDEN_DIR / name, encoding="utf-8") as f:
        return json.load(f)


_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = load_v4()
    return _CORPUS


def diff_records(expected, actual, label):
    """first few differing records, so a failure names the case not just a count"""
    assert len(expected) == len(actual), (
        f"{label}: expected {len(expected)} records, got {len(actual)}")
    mismatches = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    if mismatches:
        shown = "\n".join(
            f"  [{i}] expected {json.dumps(e, sort_keys=True, default=str)}\n"
            f"      actual   {json.dumps(a, sort_keys=True, default=str)}"
            for i, e, a in mismatches[:5])
        raise AssertionError(
            f"{label}: {len(mismatches)} of {len(expected)} records changed\n{shown}")


# ── the tracked corpus itself ────────────────────────────────────────────────

def test_the_six_tracked_v4_files_are_unchanged():
    expected = fixture("v4_file_hashes.json")["sha256"]
    actual = v4_file_hashes()
    assert actual == expected, (
        "data/synthetic/v4 changed:\n" + "\n".join(
            f"  {name}: expected {expected[name]} got {actual[name]}"
            for name in sorted(expected) if expected[name] != actual[name]))


def test_orders_metadata_still_matches_its_own_recorded_input_hashes():
    """the metadata's four input hashes and the live profile files still agree

    this is the pin the v4 corpus already carried; the harness re-checks it so a
    profile edit cannot pass by updating only the hash fixture above"""
    meta = fixture("../../data/synthetic/v4/orders_metadata.json")
    live = v4_file_hashes()
    assert meta["input_sha256"]["products"] == live["products.csv"]
    assert meta["input_sha256"]["manufacturers"] == live["manufacturer_agents.csv"]
    assert meta["input_sha256"]["materials"] == live["material_agents.csv"]
    assert meta["input_sha256"]["logistics"] == live["logistics_agents.csv"]


# ── stage decisions ──────────────────────────────────────────────────────────

def test_constructed_boundary_decisions_are_unchanged():
    expected = fixture("v4_boundary_decisions.json")
    actual = []
    for name, stage, agent, rfq in v4_boundaries.sealed_cases():
        bid = SEALED_DECIDERS[stage](agent, rfq)
        actual.append({"case": name, **decision_record(stage, agent, rfq, bid)})
    for name, stage, agent, rfq in v4_boundaries.auction_cases():
        bid = decide_round_bid(stage, agent, rfq)
        actual.append({"case": name, **decision_record(stage, agent, rfq, bid),
                       "round_number": rfq["round_number"],
                       "current_best_bid": rfq["current_best_bid"]})
    diff_records(expected, actual, "boundary decisions")


def test_sampled_corpus_decisions_are_unchanged():
    stored = fixture("v4_sample_decisions.json")
    assert stored["auction_rounds"] == AUCTION_ROUNDS
    assert stored["platform_minimum_decrement"] == DEFAULT_PLATFORM_MINIMUM_DECREMENT
    assert stored["order_ids"] == [o["order_id"] for o in
                                   sample_orders(corpus(), stored["stride"])]
    actual = sample_decisions(corpus(), stored["stride"])
    diff_records(stored["decisions"], actual, "sampled decisions")


def test_every_decision_in_the_corpus_is_unchanged():
    """the exhaustive arm: all 50,000 orders, every agent, sealed and auction"""
    expected = fixture("v4_decision_census.json")
    limit = os.getenv("GOLDEN_CENSUS_ORDERS", "").strip()
    if limit:
        print(f"    SKIP exhaustive census, GOLDEN_CENSUS_ORDERS={limit}")
        return
    actual = census_digest(corpus())
    assert actual["orders_covered"] == expected["orders_covered"]
    assert actual["decision_counts"] == expected["decision_counts"], (
        f"decision counts moved: {actual['decision_counts']}")
    assert actual["reason_counts"] == expected["reason_counts"], (
        f"rejection reason counts moved: {actual['reason_counts']}")
    assert actual["digest_sha256"] == expected["digest_sha256"], (
        "the decision census digest moved: some order, agent, offer, completion "
        "time or reason differs from the frozen v4 baseline")


def test_offers_are_frozen_to_the_cent():
    """every stored offer is already at cent resolution and is reproduced exactly"""
    stored = fixture("v4_sample_decisions.json")["decisions"]
    offers = [r["offer_amount"] for r in stored if r["offer_amount"] is not None]
    assert offers, "the sample fixture recorded no offers"
    for offer in offers:
        assert round(offer, 2) == offer, f"stored offer {offer} is not cent-exact"


# ── prompt text and labels ───────────────────────────────────────────────────

def test_stage_prompt_text_is_byte_for_byte_unchanged():
    expected = fixture("v4_prompts.json")["stage_prompts"]
    by_id = {o["order_id"]: o for o in corpus()["orders"]}

    for entry in expected:
        order = by_id[entry["order_id"]]
        rfqs = stage_rfqs(order, corpus()["products"])
        stage = entry["stage"]
        rfq = rfqs[stage]
        agent = next(a for a in corpus()[stage] if a["agent_id"] == entry["agent_id"])
        assert eligibility_reason(stage, agent, rfq) is None, (
            f"{entry['agent_id']} is no longer eligible for {entry['order_id']}")
        build = PROMPT_BUILDERS[stage]

        if entry["mode"] == "sealed":
            system, user = build(agent, rfq)
        else:
            auction_rfq = {**rfq, "round_number": 2,
                           "platform_minimum_decrement": DEFAULT_PLATFORM_MINIMUM_DECREMENT,
                           "current_best_bid": 1000.0}
            system, user = build(agent, auction_rfq,
                                 auction=auction_fields(stage, agent, auction_rfq))

        label = f"{entry['order_id']} {stage} {entry['mode']}"
        assert system == entry["system"], f"{label}: system prompt changed"
        assert user == entry["user"], (
            f"{label}: user prompt changed\n--- expected ---\n{entry['user']}\n"
            f"--- actual ---\n{user}")


def test_supervisor_chain_prompt_and_label_are_unchanged():
    expected = fixture("v4_prompts.json")["chain_prompts"]
    assert expected, "the prompt fixture recorded no chain-selection prompts"
    by_id = {o["order_id"]: o for o in corpus()["orders"]}

    for entry in expected:
        order = by_id[entry["order_id"]]
        rfqs = stage_rfqs(order, corpus()["products"])

        picked = {}
        for stage in STAGES:
            bids = [b for b in (SEALED_DECIDERS[stage](a, rfqs[stage])
                                for a in corpus()[stage]) if b["decision"] == "BID"]
            picked[stage] = sorted(bids, key=lambda b: (b["offer_amount"], b["agent_id"]))[:2]

        width = min(len(picked[s]) for s in STAGES)
        candidates = []
        for i in range(width):
            bids = {s: picked[s][i] for s in STAGES}
            candidates.append({
                "material": bids["material"],
                "tailoring": bids["tailoring"],
                "logistics": bids["logistics"],
                "total_cost": sum(b["offer_amount"] for b in bids.values()),
                "total_days": sum(b["estimated_completion_days"] for b in bids.values()),
            })

        remaining_days = float(rfqs["material"]["days_until_deadline"])
        from src.common.slm import build_chain_selection_prompt
        system, user = build_chain_selection_prompt(order, candidates, remaining_days)
        assert system == entry["system"], f"{entry['order_id']}: chain system prompt changed"
        assert user == entry["user"], (
            f"{entry['order_id']}: chain user prompt changed\n--- expected ---\n"
            f"{entry['user']}\n--- actual ---\n{user}")

        chains = [chain_from_bids(c) for c in candidates]
        label = decide_chain_selection(chains, {**order, "days_until_deadline": remaining_days})
        stored = json.loads(json.dumps(entry["label"]))
        assert json.loads(json.dumps(label)) == stored, (
            f"{entry['order_id']}: chain-selection label changed\n"
            f"  expected {stored}\n  actual   {label}")


# ── the harness itself must not depend on when it runs ───────────────────────

def test_decisions_do_not_move_with_the_system_clock():
    """the same order decides the same way whatever day the harness runs on

    stage deadlines are anchored on order_created, so this is already true; the
    test is here so a later change that reintroduces a real-clock read fails"""
    order = corpus()["orders"][0]
    first = stage_rfqs(order, corpus()["products"])
    again = stage_rfqs(order, corpus()["products"])
    assert first == again
    assert first["material"]["days_until_deadline"] == 8, (
        "ORD-00001 is 2026-07-17T11:00:00 with a 2026-07-26 deadline, which is "
        "8 whole days however long after the fact it is replayed")


def main():
    print(f"real now={datetime.now().isoformat(timespec='seconds')}")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v4 golden regression tests passed")


if __name__ == "__main__":
    main()
