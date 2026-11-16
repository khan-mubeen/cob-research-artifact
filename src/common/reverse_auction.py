"""
reverse-auction bidding rules, agent side

up to three rounds per stage, closing early when a round brings no improvement.
round 1 is exactly the sealed decision, so both modes open at the same price.
from round 2 on, an agent offers current_best_bid - platform_minimum_decrement
and stays silent for the rest of the auction as soon as that would fall below
its own minimum_bid

that single mandated offer makes every later round a tie, so the winner is
settled by arrival order, earliest bid first

eligibility, dynamic feasibility, pricing and the published bid shape are the
sealed ones, reused unchanged from bidding_formulas. only the offer changes

the rules are stage independent, so all three stages share one round handler
and the sealed decision functions supply everything else
"""

import json
import math
import os
from collections import OrderedDict

from src.common.bid_activity import publish_no_bid
from src.common.bidding_formulas import (
    apply_dynamic_guard, decide_logistics_bid, decide_manufacturer_bid,
    decide_material_bid, dynamic_feasibility_reason, dynamic_guard_enabled,
    eligibility_reason, stage_price_terms, validate_bid_values, BidDecision,
)
from src.common.slm import (
    slm_decide_logistics_bid, slm_decide_manufacturer_bid, slm_decide_material_bid,
)


SEALED_DECIDERS = {
    "material": decide_material_bid,
    "tailoring": decide_manufacturer_bid,
    "logistics": decide_logistics_bid,
}

SLM_DECIDERS = {
    "material": slm_decide_material_bid,
    "tailoring": slm_decide_manufacturer_bid,
    "logistics": slm_decide_logistics_bid,
}

# offers are published rounded to cents, so comparisons tolerate that much
_EPSILON = 0.005


def _no_bid(reason) -> BidDecision:
    return {"decision": "NO_BID", "debug_reason": reason}


def ceil_to_cents(value):
    """round up to the cent, absorbing binary float noise first

    a bare ceil(value * 100) pushes an already-exact cent to the next one when
    the product lands a hair above the integer, so the product is rounded to
    six places before the ceiling is taken"""
    return math.ceil(round(value * 100, 6)) / 100


def agent_minimum_bid(stage, agent, rfq):
    """the agent's own floor for this rfq, or None when it is undefined

    rounded UP to the cent, not to nearest: offers are cent-denominated, so the
    lowest legal offer is ceil(minimum_bid), and at that resolution
    next_bid >= ceil_to_cents(floor) is exactly the rule's next_bid >= floor.
    rounding to nearest hid the sub-cent margin and displayed a floor the agent
    may not actually bid at

    an rfq the agent cannot serve has no floor: the manufacturer workload markup
    is only defined at or below a workload ratio of 1.00"""
    if dynamic_feasibility_reason(stage, agent, rfq) is not None:
        return None
    minimum_bid, _ = stage_price_terms(stage, agent, rfq)
    return ceil_to_cents(minimum_bid)


def auction_fields(stage, agent, rfq):
    """the reverse-auction fields an agent prompt adds to a sealed one

    minimum_bid is supplied rather than left to be derived: both trained models
    reproduced the decrement arithmetic exactly but could not compute their own
    floor, so the prompt now asks them to compare instead of derive"""
    return {
        "current_best_bid": rfq.get("current_best_bid"),
        "platform_minimum_decrement": rfq["platform_minimum_decrement"],
        "round_number": rfq["round_number"],
        "minimum_bid": agent_minimum_bid(stage, agent, rfq),
    }


def next_round_offer(current_best_bid, platform_minimum_decrement):
    """the only offer a later round allows, rounded as it will be published"""
    return round(float(current_best_bid) - float(platform_minimum_decrement), 2)


def best_standing_bid(bids):
    """lowest standing offer, earliest bid wins a tie

    from round 2 on every surviving agent is required to offer exactly
    current_best_bid - platform_minimum_decrement, so essentially every later
    round is one big tie and this tie-break decides the winner outright.
    received_seq is stamped in arrival order by whoever collected the bids;
    without it the sort falls back to agent id, which keeps the sealed path and
    any older record ordering exactly as before"""
    if not bids:
        return None
    return sorted(bids, key=lambda b: (b["offer_amount"],
                                       b.get("received_seq", 0),
                                       b["agent_id"]))[0]


def round_improved(announced_best_bid, best_after_round):
    """did this round produce a valid improvement on the announced best

    the opening round improves by definition once anyone bids, there is nothing
    to beat yet. a later round improves only if the standing best actually fell,
    which is what closes an auction before the round cap"""
    if best_after_round is None:
        return False
    if announced_best_bid is None:
        return True
    return best_after_round < announced_best_bid - _EPSILON


def decide_round_bid(stage, agent, rfq) -> BidDecision:
    """rule-based decision for one round of one stage

    round 1 delegates to the sealed decision untouched. later rounds keep that
    decision's completion time and defect probability and replace only the
    offer, which is why the two modes stay comparable stage by stage"""
    decide = SEALED_DECIDERS.get(stage)
    if decide is None:
        raise ValueError(f"decide_round_bid: unknown stage {stage!r}")

    bid = decide(agent, rfq)
    if bid["decision"] != "BID" or int(rfq["round_number"]) <= 1:
        return bid

    current_best = rfq.get("current_best_bid")
    if current_best is None:
        # no round-1 bid was announced, there is nothing to undercut
        return _no_bid("no_current_best_bid")

    minimum_bid, _ = stage_price_terms(stage, agent, rfq)
    next_bid = next_round_offer(current_best, rfq["platform_minimum_decrement"])
    if next_bid < minimum_bid:
        return _no_bid("next_bid_below_minimum_bid")

    return {**bid, "offer_amount": next_bid}


async def slm_decide_round_bid(slm, stage, agent, rfq) -> BidDecision:
    """model-backed decision for one round, same prompts plus the auction fields"""
    decide = SLM_DECIDERS.get(stage)
    if decide is None:
        raise ValueError(f"slm_decide_round_bid: unknown stage {stage!r}")
    return await decide(slm, agent, rfq, auction=auction_fields(stage, agent, rfq))


# ── floor guard ───────────────────────────────────────────────────────────────
# the rule-based path cannot break the auction rules, a model can. this guard is
# the auction counterpart of Guard B and is disabled by default for the same
# reason: an unguarded run has to record what the model actually did

def auction_floor_guard_enabled():
    return os.getenv("AUCTION_FLOOR_GUARD", "0").strip().lower() in ("1", "true", "yes", "on")


def auction_rule_violation(stage, agent, rfq, offer_amount):
    """return None when an offer respects the floor and the round decrement

    an rfq the agent cannot serve has no defined floor, that case belongs to the
    dynamic guard and is left alone here"""
    if dynamic_feasibility_reason(stage, agent, rfq) is not None:
        return None

    minimum_bid, _ = stage_price_terms(stage, agent, rfq)
    if offer_amount < minimum_bid - _EPSILON:
        return "below_minimum_bid"

    current_best = rfq.get("current_best_bid")
    if int(rfq.get("round_number", 1)) > 1 and current_best is not None:
        required = next_round_offer(current_best, rfq["platform_minimum_decrement"])
        if offer_amount > required + _EPSILON:
            return "decrement_not_met"
    return None


def apply_auction_floor_guard(stage, agent, rfq, bid):
    """suppress a BID that breaks the auction rules

    NO_BID results and rule-abiding BIDs pass through unchanged, the guard never
    rewrites an offer
    return (possibly_replaced_bid, suppression_reason_or_None)"""
    if bid.get("decision") != "BID":
        return bid, None
    reason = auction_rule_violation(stage, agent, rfq, bid["offer_amount"])
    if reason is not None:
        return _no_bid(f"auction_floor_guard:{reason}"), reason
    return bid, None


# ── per-agent auction memory ──────────────────────────────────────────────────

class AuctionMemory:
    """which auctions this process has already dropped out of

    the rules say an agent that cannot meet the next decrement stays silent for
    all remaining rounds, so round 3 must not revive it. round 1 clears the
    entry, so a replayed order id starts a clean auction"""

    def __init__(self, capacity=1024):
        self._withdrawn = OrderedDict()
        self._capacity = capacity

    @staticmethod
    def _key(agent_id, order_id, stage):
        return (agent_id, order_id, stage)

    def start_round_one(self, agent_id, order_id, stage):
        self._withdrawn.pop(self._key(agent_id, order_id, stage), None)

    def withdraw(self, agent_id, order_id, stage):
        key = self._key(agent_id, order_id, stage)
        self._withdrawn[key] = True
        self._withdrawn.move_to_end(key)
        while len(self._withdrawn) > self._capacity:
            self._withdrawn.popitem(last=False)

    def has_withdrawn(self, agent_id, order_id, stage):
        return self._key(agent_id, order_id, stage) in self._withdrawn


# ── shared agent-side round handler ───────────────────────────────────────────

async def handle_auction_rfq(nc, stage, agent, rfq, slm, memory):
    """one agent's response to one auction round

    mirrors the sealed callback: ineligible requests never reach the model, the
    silence reason never travels with a bid, and only a validated bid is
    published. the reason is also reported on the monitoring feed, which no agent
    subscribes to, see bid_activity"""
    agent_id = agent["agent_id"]
    order_id = rfq.get("order_id", "unknown")
    round_number = int(rfq["round_number"])

    async def stay_silent(reason):
        memory.withdraw(agent_id, order_id, stage)
        print(f"[{agent_id}] staying silent on {order_id} round {round_number}: {reason}")
        await publish_no_bid(nc, stage, agent_id, rfq, reason)

    if round_number <= 1:
        memory.start_round_one(agent_id, order_id, stage)
    elif memory.has_withdrawn(agent_id, order_id, stage):
        print(f"[{agent_id}] staying silent on {order_id} round {round_number}: withdrawn_in_earlier_round")
        await publish_no_bid(nc, stage, agent_id, rfq, "withdrawn_in_earlier_round")
        return

    try:
        # ineligible requests never reach the model
        ineligible = eligibility_reason(stage, agent, rfq)
        if ineligible is not None:
            result = {"decision": "NO_BID", "debug_reason": ineligible}
        elif slm is not None:
            result = await slm_decide_round_bid(slm, stage, agent, rfq)
        else:
            result = decide_round_bid(stage, agent, rfq)
    except Exception as e:
        print(f"[{agent_id}] decision failed on {order_id} round {round_number}, staying silent: {e}")
        await publish_no_bid(nc, stage, agent_id, rfq, f"decision_failed: {e}")
        return

    # both guards are disabled by default (SLM_DYNAMIC_GUARD / AUCTION_FLOOR_GUARD unset)
    if slm is not None and result["decision"] == "BID" and dynamic_guard_enabled():
        result, guard_reason = apply_dynamic_guard(stage, agent, rfq, result)
        if guard_reason is not None:
            print(f"[{agent_id}] dynamic guard suppressed BID on {order_id} "
                  f"round {round_number}: {guard_reason}")
    if slm is not None and result["decision"] == "BID" and auction_floor_guard_enabled():
        result, guard_reason = apply_auction_floor_guard(stage, agent, rfq, result)
        if guard_reason is not None:
            print(f"[{agent_id}] auction floor guard suppressed BID on {order_id} "
                  f"round {round_number}: {guard_reason}")

    if result["decision"] == "NO_BID":
        await stay_silent(result["debug_reason"])
        return

    invalid_reason = validate_bid_values(result)
    if invalid_reason:
        await stay_silent(invalid_reason)
        return

    reply_to = rfq.get("reply_to")
    if not reply_to:
        print(f"[{agent_id}] rfq for {order_id} round {round_number} missing reply_to, cannot bid")
        return

    try:
        await nc.publish(reply_to, json.dumps(result).encode())
        print(f"[{agent_id}] bid {result['offer_amount']} on {order_id} round {round_number}")
    except Exception as e:
        print(f"[{agent_id}] failed to publish bid on {order_id} round {round_number}: {e}")
