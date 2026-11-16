"""unit tests for how an auction round folds replies into the standing bids.
runnable directly (python ...) or under pytest.

the rule these pin down: an agent's standing bid is the best price it offered at
any point in the stage auction. a later round can improve it and can never make
it worse. the rule-based path cannot break that on its own, a model can, and one
real batch order was decided the wrong way before this was fixed, so the two
recorded cases are kept here as fixtures.
"""

from src.common.bidding_formulas import decide_logistics_bid
from src.common.reverse_auction import best_standing_bid, decide_round_bid
from src.supervisor.auction import merge_standing


def bid(agent_id, offer, seq, days=1.0, defect=0.02):
    return {"agent_id": agent_id, "offer_amount": offer, "estimated_completion_days": days,
            "defect_probability": defect, "received_seq": seq, "decision": "BID"}


def standing_of(*rounds):
    """fold a sequence of rounds, each {agent_id: bid}, the way a stage does"""
    standing = {}
    for received in rounds:
        merge_standing(standing, received)
    return standing


# ── what a later round may and may not do ────────────────────────────────────

def test_worse_later_bid_does_not_overwrite():
    standing = standing_of({"L09": bid("L09", 567.58, 1)}, {"L09": bid("L09", 720.00, 2)})
    assert standing["L09"]["offer_amount"] == 567.58


def test_equal_later_bid_does_not_overwrite():
    # equal is not an improvement, and dropping it keeps the earlier arrival stamp
    standing = standing_of({"T5": bid("T5", 13994.99, 1)}, {"T5": bid("T5", 13994.99, 7)})
    assert standing["T5"]["offer_amount"] == 13994.99
    assert standing["T5"]["received_seq"] == 1


def test_better_later_bid_does_overwrite():
    standing = standing_of({"M02": bid("M02", 2125.63, 1)}, {"M02": bid("M02", 2120.63, 2)})
    assert standing["M02"]["offer_amount"] == 2120.63
    assert standing["M02"]["received_seq"] == 2


def test_sub_cent_change_counts_as_equal():
    # offers are published rounded to cents, so anything under half a cent is noise
    standing = standing_of({"M02": bid("M02", 100.00, 1)}, {"M02": bid("M02", 99.998, 2)})
    assert standing["M02"]["received_seq"] == 1


def test_new_agent_in_a_later_round_is_added():
    standing = standing_of({"L03": bid("L03", 648.40, 1)}, {"L09": bid("L09", 567.58, 2)})
    assert set(standing) == {"L03", "L09"}
    assert best_standing_bid(list(standing.values()))["agent_id"] == "L09"


def test_merge_returns_the_same_dict_it_was_given():
    standing = {}
    assert merge_standing(standing, {"M01": bid("M01", 10.0, 1)}) is standing


# ── the tie-break the standing stamp exists for ──────────────────────────────

def test_equal_offers_keep_resolving_to_whoever_bid_first():
    standing = standing_of({"L01": bid("L01", 500.0, 1), "L02": bid("L02", 500.0, 2)},
                           {"L01": bid("L01", 500.0, 9), "L02": bid("L02", 500.0, 10)})
    assert best_standing_bid(list(standing.values()))["agent_id"] == "L01"


# ── the two real cases, kept as fixtures ─────────────────────────────────────

def test_granite_ord_38735_material_keeps_the_best_offer():
    # both agents jumped from about 4k to 40k in round 2. before the fix that
    # retired their round-1 prices and the stage closed at 5032.45, which put the
    # chain over budget and the order was recorded infeasible
    rounds = ({"M09": bid("M09", 5293.56, 1), "M10": bid("M10", 4015.07, 2),
               "M11": bid("M11", 5032.45, 3), "M12": bid("M12", 4438.50, 4)},
              {"M10": bid("M10", 40010.07, 5), "M12": bid("M12", 40010.07, 6)})
    best = best_standing_bid(list(standing_of(*rounds).values()))
    assert (best["agent_id"], best["offer_amount"]) == ("M10", 4015.07)


def test_phi_ord_07181_tailoring_keeps_the_best_offer():
    # T5 improved to 13994.99 and then went back up by exactly one decrement
    rounds = ({"T1": bid("T1", 15590.96, 1), "T2": bid("T2", 14913.09, 2),
               "T5": bid("T5", 13999.99, 3), "T9": bid("T9", 14829.27, 4),
               "T11": bid("T11", 16705.45, 5)},
              {"T5": bid("T5", 13994.99, 6)},
              {"T5": bid("T5", 13999.99, 7)})
    best = best_standing_bid(list(standing_of(*rounds).values()))
    assert (best["agent_id"], best["offer_amount"]) == ("T5", 13994.99)


def test_closing_price_is_never_above_the_best_offer_made():
    rounds = ({"A": bid("A", 100.0, 1), "B": bid("B", 90.0, 2)},
              {"A": bid("A", 80.0, 3), "B": bid("B", 500.0, 4)},
              {"B": bid("B", 999.0, 5)})
    offers = [b["offer_amount"] for r in rounds for b in r.values()]
    best = best_standing_bid(list(standing_of(*rounds).values()))
    assert best["offer_amount"] == min(offers)


# ── why the rule-based baseline could never trigger it ───────────────────────

def agent():
    return {"agent_id": "L01", "location": "Helsinki", "service_regions": "Helsinki|Turku",
            "default_completion_same_city_days": 1, "default_completion_intercity_days": 3,
            "base_delivery_cost": 50.0, "cost_per_kg": 0.74, "damage_probability": 0.015,
            "reputation_score": 0.9, "min_profit_margin": 0.1, "initial_markup": 0.31}


def rfq(round_number, current_best):
    return {"order_id": "O1", "deliver_to": "Turku", "shipment_weight_kg": 200.0,
            "max_defect_rate": 0.5, "min_reputation_score": 0.1, "priority": "normal",
            "days_until_deadline": 6, "round_number": round_number,
            "current_best_bid": current_best, "platform_minimum_decrement": 5.0}


def test_rule_agent_never_offers_worse_than_its_standing_bid():
    a = agent()
    opening = decide_logistics_bid(a, rfq(1, None))
    assert opening["decision"] == "BID"
    standing, best = {"L01": {**opening, "received_seq": 1}}, opening["offer_amount"]
    seq = 1
    for round_number in range(2, 12):
        nxt = decide_round_bid("logistics", a, rfq(round_number, best))
        if nxt["decision"] != "BID":
            break                       # it stays silent rather than bid worse
        assert nxt["offer_amount"] < best
        seq += 1
        merge_standing(standing, {"L01": {**nxt, "received_seq": seq}})
        best = standing["L01"]["offer_amount"]
    assert standing["L01"]["offer_amount"] == best


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} auction round unit tests passed")


if __name__ == "__main__":
    main()
