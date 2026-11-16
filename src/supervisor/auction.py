"""
reverse-auction orchestration, supervisor side

up to three rounds per stage over nats. round 1 opens the auction, the current
best bid is announced with every following round, and the auction closes as soon
as a round brings no valid improvement or the round cap is reached

every accepted reply is stamped with its arrival order, because from round 2 on
each surviving agent offers the one price the rule allows and the tie-break is
what actually picks the winner

each round is one rfq on the stage subject with its own reply subject, so a late
reply from an earlier round can never be counted in a later one

the caller gets back two things:
  final_bids  one standing bid per agent, the best price that agent offered at
              any point, in exactly the list shape the sealed stage window
              returns, so chain selection downstream is unchanged
  trace       the full per-round record, published on order.auction.trace

silence in the trace is complete without repeating itself: an agent listed as
silent in round 1 never entered the auction and stays silent by definition,
while an agent listed as silent in a later round entered and then dropped out.
the trace records that an agent went quiet, never why: the reason is the agent's
own and reaches the monitoring feed in src/common/bid_activity.py, not this trace
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from src.common.bidding_formulas import bid_reply_reason
from src.common.bidding_modes import (
    REVERSE_AUCTION, auction_max_rounds, platform_minimum_decrement,
)
from src.common.reverse_auction import best_standing_bid, round_improved


# one window per round, same default as the sealed stage window
ROUND_WINDOW_SECONDS = float(os.getenv("AUCTION_ROUND_WINDOW_SECONDS", "10"))

TRACE_SUBJECT = "order.auction.trace"

# offers are published rounded to cents
_EPSILON = 0.005


def _best_of(bids):
    """lowest standing offer, agent id breaks ties like pick_winner does"""
    return best_standing_bid(bids)


def merge_standing(standing, received):
    """fold one round's replies into the standing bids, keeping the better one

    a later offer replaces an agent's standing bid only when it actually
    undercuts it. an equal offer is not an improvement and is dropped, which
    keeps the earlier bid's arrival stamp and with it the first-in tie-break

    the rule-based path cannot produce a worse later offer: it bids
    current_best - decrement or stays silent. a model can, and without this its
    worst round would retire its own best price. the trace still records what the
    agent actually said in each round, this only decides what stands"""
    for agent_id, bid in received.items():
        previous = standing.get(agent_id)
        if previous is None or bid["offer_amount"] < previous["offer_amount"] - _EPSILON:
            standing[agent_id] = bid
    return standing


def _bid_record(bid, current_best, decrement):
    """one agent's bid as it appears in the trace

    meets_decrement records whether the offer actually undercut the announced
    best bid by the platform minimum. the rule-based path cannot fail it, a
    model can, and the supervisor does not reject the bid either way"""
    record = {
        "agent_id": bid["agent_id"],
        "offer_amount": bid["offer_amount"],
        "estimated_completion_days": bid["estimated_completion_days"],
        "defect_probability": bid["defect_probability"],
        "received_seq": bid.get("received_seq"),
        "meets_decrement": None,
    }
    if current_best is not None:
        record["meets_decrement"] = bid["offer_amount"] <= current_best - decrement + _EPSILON
    return record


async def collect_round_bids(nc, stage, order_id, rfq, reply_to, window_seconds, arrival):
    """publish one round's rfq and collect the replies inside its window

    each accepted reply is stamped with the order it arrived in, which is what
    settles the tie when every agent offers the one mandated price. the stamp is
    the supervisor's own observation, the published bid shape is untouched

    return {agent_id: bid}, so a duplicate reply from one agent in one round
    cannot be counted twice"""
    bids = {}

    async def on_reply(msg):
        try:
            bid = json.loads(msg.data.decode())
        except json.JSONDecodeError as exc:
            print(f"[supervisor] discarding malformed reply on {reply_to}: {exc}")
            return
        reason = bid_reply_reason(bid)
        if reason:
            print(f"[supervisor] discarding malformed reply on {reply_to}: {reason}")
            return
        arrival["seq"] += 1
        bid["received_seq"] = arrival["seq"]
        bids[bid["agent_id"]] = bid

    sub = await nc.subscribe(reply_to, cb=on_reply)
    await nc.publish(f"rfq.{stage}.{order_id}", json.dumps(rfq).encode())
    await asyncio.sleep(window_seconds)
    await sub.unsubscribe()

    return bids


async def run_stage_auction(nc, stage, order, extra_fields, remaining_days,
                            roster=(), window_seconds=None):
    """run all three rounds for one stage

    return (final_bids, trace)"""
    order_id = order["order_id"]
    decrement = platform_minimum_decrement()
    max_rounds = auction_max_rounds()
    window = ROUND_WINDOW_SECONDS if window_seconds is None else window_seconds

    standing = {}     # agent_id -> that agent's latest bid, still on the table
    entrants = []     # agents that bid in round 1, the only ones who can bid later
    rounds = []
    best = None
    opening_best = None
    arrival = {"seq": 0}      # one counter for the whole stage auction
    closed_reason = "max_rounds"

    opened_at = datetime.now(timezone.utc).isoformat()

    for round_number in range(1, max_rounds + 1):
        current_best = best["offer_amount"] if best is not None else None
        reply_to = f"rfq.{stage}.{order_id}.replies.r{round_number}"

        rfq = {
            "order_id": order_id,
            "max_defect_rate": order["max_defect_rate"],
            "min_reputation_score": order["min_reputation_score"],
            "deadline": order["deadline"],
            "days_until_deadline": remaining_days,
            "priority": order["priority"],
            "bid_close_at": datetime.now(timezone.utc).isoformat(),
            "reply_to": reply_to,
            "round_number": round_number,
            "auction_rounds": max_rounds,
            "current_best_bid": current_best,
            "platform_minimum_decrement": decrement,
            **extra_fields,
        }

        received = await collect_round_bids(nc, stage, order_id, rfq, reply_to,
                                            window, arrival)

        if round_number == 1:
            entrants = sorted(received)
            silent = [{"agent_id": a, "status": "never_entered"}
                      for a in sorted(set(roster) - set(received))]
            opening_best = _best_of(list(received.values()))
        else:
            silent = [{"agent_id": a, "status": "withdrawn"}
                      for a in entrants if a not in received]

        merge_standing(standing, received)
        best = _best_of(list(standing.values()))

        rounds.append({
            "round_number": round_number,
            "announced_best_bid": current_best,
            "bids": [_bid_record(received[a], current_best, decrement) for a in sorted(received)],
            "silent": silent,
            "best_bid_after_round": best["offer_amount"] if best is not None else None,
            "best_agent_id_after_round": best["agent_id"] if best is not None else None,
        })

        improved = round_improved(current_best,
                                  best["offer_amount"] if best is not None else None)
        print(f"[supervisor] order {order_id} {stage} round {round_number}/{max_rounds}: "
              f"announced best {current_best if current_best is not None else 'none'}, "
              f"decrement {decrement}, {len(received)} bid(s), {len(silent)} silent, "
              f"best {best['offer_amount'] if best is not None else 'none'}")

        # the stop rule: a round that brings no valid improvement closes the
        # auction, the round cap is only the upper bound
        if not improved:
            closed_reason = "no_improvement"
            print(f"[supervisor] order {order_id} {stage} closing after round "
                  f"{round_number}: no improvement")
            break

    # the arrival stamp travels with the bid so pick_winner breaks a tie the same
    # way the rounds did; sealed bids never carry it and keep the agent-id order
    final_bids = list(standing.values())
    opening_offer = opening_best["offer_amount"] if opening_best is not None else None
    closing_offer = best["offer_amount"] if best is not None else None

    trace = {
        "order_id": order_id,
        "stage": stage,
        "mode": REVERSE_AUCTION,
        "platform_minimum_decrement": decrement,
        "rounds_planned": max_rounds,
        "rounds_run": len(rounds),
        "closed_reason": closed_reason,
        "round_window_seconds": window,
        "opened_at": opened_at,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "roster_size": len(roster),
        "entrant_agent_ids": entrants,
        "rounds": rounds,
        "final_bids": [_bid_record(standing[a], None, decrement) for a in sorted(standing)],
        "winner_agent_id": best["agent_id"] if best is not None else None,
        "winning_offer_amount": closing_offer,
        "opening_best_bid": opening_offer,
        "closing_best_bid": closing_offer,
        "total_reduction_eur": (round(opening_offer - closing_offer, 2)
                                if opening_offer is not None and closing_offer is not None
                                else None),
    }

    return final_bids, trace


async def publish_auction_trace(nc, trace):
    """the per-round record goes to its own subject, order.logging keeps the
    same per-order shape it has in sealed mode"""
    await nc.publish(TRACE_SUBJECT, json.dumps(trace).encode())
