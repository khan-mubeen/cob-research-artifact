import asyncio
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from src.common.bidding_formulas import PRIORITY_MARKUP, bid_reply_reason
from src.common.bidding_modes import (
    REVERSE_AUCTION, auction_max_rounds, bidding_mode, platform_minimum_decrement,
)
from src.common.dynamic_state import (
    CONTROL_RESET_SUBJECT, DYNAMIC_STAGES, SimClock, dynamic_resources_enabled,
    parse_sim_now, release_subject, reserve_subject,
)
from src.common.nats_client import connect_with_retry
from src.common.slm import load_slm, slm_select_chain, warmup_slm
from src.supervisor.auction import (
    ROUND_WINDOW_SECONDS, publish_auction_trace, run_stage_auction,
)
from config import DATA_VERSION

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
DATA_DIR = Path(os.getenv("DATA_DIR", f"data/synthetic/{DATA_VERSION}"))

# resolved once at startup so a bad BIDDING_MODE fails before any order arrives
BIDDING_MODE = bidding_mode()

# leaves headroom over the measured 1-1.5 second warm-inference latency
# runtime setting rather than a protocol requirement. rule-based agents answer in
# milliseconds, so a batch run can lower it without changing any decision
BIDDING_WINDOW_SECONDS = float(os.getenv("BIDDING_WINDOW_SECONDS", "10"))

# dynamic resource state, resolved once so every order in a run agrees
DYNAMIC_RESOURCES = dynamic_resources_enabled()

# the walking clock: "now" is the order_created of the order being processed,
# not real elapsed runtime. one clock for the whole run, it only moves forward
SIM_CLOCK = SimClock()

# a reservation is answered by a local in-process lookup, this is generous
RESERVATION_TIMEOUT = float(os.getenv("RESERVATION_TIMEOUT_SECONDS", "5"))

# supervisor -> anyone listening, one message when a stage opens and one when it
# closes. reporting only: it carries no decision, it says which of the three
# stages an order is in, which is otherwise only inferable from the rfq traffic
STAGE_EVENT_SUBJECT = "order.stage"

PRODUCTS = pd.read_csv(DATA_DIR / "products.csv")

STAGE_AGENT_FILES = {
    "material": "material_agents.csv",
    "tailoring": "manufacturer_agents.csv",
    "logistics": "logistics_agents.csv",
}


def _load_reputation_scores():
    # bids omit reputation, so read it from the agent csv files
    scores = {}
    for filename in STAGE_AGENT_FILES.values():
        path = DATA_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            scores.update(zip(df["agent_id"], df["reputation_score"]))
    return scores


def _load_stage_rosters():
    # auction traces name every agent that stayed silent, which needs the full
    # roster per stage, not only the agents that replied
    rosters = {}
    for stage, filename in STAGE_AGENT_FILES.items():
        path = DATA_DIR / filename
        rosters[stage] = list(pd.read_csv(path)["agent_id"]) if path.exists() else []
    return rosters


REPUTATION_SCORES = _load_reputation_scores()
STAGE_ROSTERS = _load_stage_rosters()


def sim_now_for(order):
    """advance the walking clock to this order and return the reading

    None outside dynamic mode, which has no walking clock to advance and no
    dynamic state to carry a reading to. deadline questions there anchor on the
    order's own order_created through order_reference_now() instead, so the two
    modes still measure the same interval for the same order"""
    if not DYNAMIC_RESOURCES:
        return None
    return SIM_CLOCK.advance_to(parse_sim_now(order.get("order_created")))


def order_reference_now(order):
    """the instant this order's deadline is measured from, outside dynamic mode

    dynamic mode reads the walking clock, which is itself seeded from
    order_created. static mode has no clock, so it anchors on that same field
    directly rather than on the real date, and both modes then answer the same
    deadline question for the same saved order however long after it is replayed.

    None when the order carries no usable creation time, which leaves the caller
    on the real clock as it always was"""
    created = order.get("order_created")
    try:
        reference = parse_sim_now(created)
    except (ValueError, TypeError):
        print(f"[supervisor] order {order.get('order_id')!r} has unreadable "
              f"order_created {created!r}, falling back to the real clock")
        return None
    if reference is not None and reference.tzinfo is not None:
        # deadlines are a bare date, so keep the wall clock that was written and
        # drop the offset rather than subtracting an aware from a naive datetime
        reference = reference.replace(tzinfo=None)
    return reference


def remaining_days_now(order, now=None):
    deadline = datetime.strptime(order["deadline"], "%Y-%m-%d")
    if now is None:
        now = order_reference_now(order) or datetime.now()
    return (deadline - now).total_seconds() / 86400.0


def days_until_deadline(order, now=None):
    return max(int(remaining_days_now(order, now)), 1)


def product_row(order):
    row = PRODUCTS[
        (PRODUCTS["product_id"] == order["product_id"]) &
        (PRODUCTS["material_type"] == order["material_type"])
    ]
    if row.empty:
        raise ValueError(f"no product/material pair for {order['product_id']}/{order['material_type']}")
    return row.iloc[0]


async def collect_bids(nc, stage, order, extra_fields, remaining_days):
    order_id = order["order_id"]
    reply_to = f"rfq.{stage}.{order_id}.replies"
    bids = []

    async def on_reply(msg):
        try:
            bid = json.loads(msg.data.decode())
            invalid_reason = bid_reply_reason(bid)
            if invalid_reason:
                raise ValueError(invalid_reason)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[supervisor] discarding malformed reply on {reply_to}: {exc}")
            return
        bids.append(bid)

    sub = await nc.subscribe(reply_to, cb=on_reply)

    rfq = {
        "order_id": order_id,
        "max_defect_rate": order["max_defect_rate"],
        "min_reputation_score": order["min_reputation_score"],
        "deadline": order["deadline"],
        "days_until_deadline": remaining_days,
        "priority": order["priority"],
        "bid_close_at": datetime.now(timezone.utc).isoformat(),
        "reply_to": reply_to,
        **extra_fields,
    }
    await nc.publish(f"rfq.{stage}.{order_id}", json.dumps(rfq).encode())

    await asyncio.sleep(BIDDING_WINDOW_SECONDS)
    await sub.unsubscribe()

    return bids


async def publish_stage_event(nc, event, stage, order_id, sim_now, **fields):
    await nc.publish(STAGE_EVENT_SUBJECT, json.dumps({
        "event": event,
        "stage": stage,
        "order_id": order_id,
        "mode": BIDDING_MODE,
        "sim_now": sim_now.isoformat() if sim_now is not None else None,
        "at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }).encode())


async def open_stage(nc, stage, order, extra_fields, remaining_days):
    """collect one stage's bids in whichever mode this run uses

    both modes return the same list of per-agent bids, so everything after this
    point, chain validation, selection and verdicts, is mode independent. the
    reverse auction additionally publishes its per-round trace"""
    order_id = order["order_id"]
    sim_now = parse_sim_now(extra_fields.get("sim_now"))
    roster = STAGE_ROSTERS.get(stage, [])
    started = time.perf_counter()
    await publish_stage_event(
        nc, "stage_opened", stage, order_id, sim_now,
        roster_size=len(roster),
        window_seconds=(ROUND_WINDOW_SECONDS if BIDDING_MODE == REVERSE_AUCTION
                        else BIDDING_WINDOW_SECONDS),
        max_rounds=auction_max_rounds() if BIDDING_MODE == REVERSE_AUCTION else 1,
    )

    if BIDDING_MODE != REVERSE_AUCTION:
        bids = await collect_bids(nc, stage, order, extra_fields, remaining_days)
    else:
        bids, trace = await run_stage_auction(
            nc, stage, order, extra_fields, remaining_days, roster=roster,
        )
        await publish_auction_trace(nc, trace)

    # the cheapest standing bid, which is the stage's own answer. the winning
    # chain is decided later and can pick a different one, so this is reported as
    # the best bid rather than as a winner
    best = pick_winner(bids)
    await publish_stage_event(
        nc, "stage_closed", stage, order_id, sim_now,
        bid_count=len(bids),
        bidder_agent_ids=sorted(b["agent_id"] for b in bids),
        best_agent_id=best["agent_id"] if best else None,
        best_offer_amount=best["offer_amount"] if best else None,
        elapsed_seconds=round(time.perf_counter() - started, 2),
    )
    return bids


def pick_winner(bids):
    if not bids:
        return None
    # earliest arrival takes an equal-price tie, matching how the auction rounds
    # already resolve one. sealed bids carry no arrival stamp, so they fall back
    # to agent id and behave exactly as before
    return sorted(bids, key=lambda b: (b["offer_amount"],
                                       b.get("received_seq", 0),
                                       b["agent_id"]))[0]


# keep the shortlist small enough for the model context
# must match MAX_CANDIDATES in generate_supervisor_data.py: the supervisor SLM
# is trained on prompts holding at most this many candidates, and a longer list
# at serve time would be out of distribution and can overflow the trained
# sequence window (~105 tokens per candidate plus ~334 fixed)
MAX_CHAIN_CANDIDATES = int(os.getenv("SLM_MAX_CHAIN_CANDIDATES", "12"))

SLM_ORDER_SAMPLE_SIZE = int(os.getenv("SLM_ORDER_SAMPLE_SIZE", "200"))


def order_within_sample(orders_started, slm):
    return slm is None or orders_started < SLM_ORDER_SAMPLE_SIZE


def _chain_key(chain):
    return (chain["material"]["agent_id"], chain["tailoring"]["agent_id"], chain["logistics"]["agent_id"])


def shortlist_chains(feasible, limit=MAX_CHAIN_CANDIDATES):
    """keep a mix of the cheapest and fastest chains from a cost-sorted list
    within the prompt limit"""
    if len(feasible) <= limit:
        return feasible
    cheapest = feasible[: limit // 2]
    fastest = sorted(feasible, key=lambda c: (c["total_days"], c["total_cost"]))[: limit - limit // 2]
    merged, seen = [], set()
    for chain in cheapest + fastest:
        if _chain_key(chain) not in seen:
            seen.add(_chain_key(chain))
            merged.append(chain)
    merged.sort(key=lambda c: (c["total_cost"], c["total_days"], _chain_key(c)))
    return merged


async def send_verdict(nc, agent_id, verdict):
    await nc.publish(f"order.verdict.{agent_id}", json.dumps(verdict).encode())


async def reject_all(nc, bids, reason):
    for bid in bids:
        await send_verdict(nc, bid["agent_id"], {
            "order_id": bid["order_id"],
            "agent_id": bid["agent_id"],
            "verdict": "rejected",
            "reason": reason,
        })


async def log_chain(nc, order_id, status, chain=None, total_cost=None, total_days=None,
                    reason=None, extra=None):
    entry = {"order_id": order_id, "status": status}
    if chain is not None:
        entry["chain"] = chain
        entry["total_cost"] = total_cost
        # promised delivery time of the winning chain, the only completion figure that
        # says anything about the bidding; wall clock is just the configured window
        entry["total_days"] = total_days
    if reason is not None:
        entry["reason"] = reason
    if extra:
        entry.update(extra)
    await nc.publish("order.logging", json.dumps(entry).encode())


async def select_chain(order, bids_by_stage, remaining_at_validation, slm):
    """the winning chain from whichever bids are still on the table

    lifted out of handle_new_order unchanged so a refused reservation can drop
    the offending bid and run exactly this same selection again, rather than
    needing a second, different fallback rule

    return (selection, reason_code, loser_reason); selection is None when no
    chain passes"""
    order_id = order["order_id"]
    material_bids = bids_by_stage["material"]
    manufacturer_bids = bids_by_stage["tailoring"]
    logistics_bids = bids_by_stage["logistics"]

    for stage in ("material", "tailoring", "logistics"):
        if not bids_by_stage[stage]:
            return None, f"no_remaining_bids_at_{stage}_stage", \
                "no bid at that stage could honour its reservation"

    if slm is None:
        material_winner = pick_winner(material_bids)
        manufacturer_winner = pick_winner(manufacturer_bids)
        logistics_winner = pick_winner(logistics_bids)
        total_cost = round(
            material_winner["offer_amount"] + manufacturer_winner["offer_amount"] + logistics_winner["offer_amount"], 2
        )
        total_chain_days = (
            material_winner["estimated_completion_days"]
            + manufacturer_winner["estimated_completion_days"]
            + logistics_winner["estimated_completion_days"]
        )
        if total_cost > order["budget"]:
            print(f"[supervisor] order {order_id} infeasible: chain cost {total_cost} exceeds budget {order['budget']}")
            return None, "chain_cost_exceeds_budget", "final chain cost exceeds order budget"
        if total_chain_days > remaining_at_validation:
            print(
                f"[supervisor] order {order_id} infeasible: chain needs {total_chain_days} days, "
                f"{remaining_at_validation:.2f} remain at validation"
            )
            return None, "chain_time_exceeds_remaining_time", \
                "final chain completion time exceeds remaining time"
        loser_reason = "higher price than winning bid"
    else:
        # only send cost- and time-feasible combinations to the model
        feasible = []
        for m_bid in material_bids:
            for t_bid in manufacturer_bids:
                for l_bid in logistics_bids:
                    cost = round(m_bid["offer_amount"] + t_bid["offer_amount"] + l_bid["offer_amount"], 2)
                    days = (
                        m_bid["estimated_completion_days"]
                        + t_bid["estimated_completion_days"]
                        + l_bid["estimated_completion_days"]
                    )
                    if cost <= order["budget"] and days <= remaining_at_validation:
                        stage_bids = (m_bid, t_bid, l_bid)
                        reputations = [
                            REPUTATION_SCORES[b["agent_id"]]
                            for b in stage_bids if b["agent_id"] in REPUTATION_SCORES
                        ]
                        feasible.append({
                            "material": m_bid, "tailoring": t_bid, "logistics": l_bid,
                            "total_cost": cost, "total_days": days,
                            # show chain selection the weakest quality across all three stages
                            "max_defect_probability": max(b["defect_probability"] for b in stage_bids),
                            "min_reputation": min(reputations) if reputations else None,
                        })

        if not feasible:
            print(f"[supervisor] order {order_id} infeasible: no bid combination passed final chain validation")
            return None, "no_feasible_chain", "no combination of bids passed final chain validation"

        feasible.sort(key=lambda c: (c["total_cost"], c["total_days"], _chain_key(c)))
        presented = shortlist_chains(feasible)
        if len(presented) < len(feasible):
            print(f"[supervisor] order {order_id}: {len(feasible)} feasible chains, presenting {len(presented)} to the model")

        chosen_idx, selection_debug = await slm_select_chain(slm, order, presented, remaining_at_validation)
        print(f"[supervisor] order {order_id} chain selection debug_reason: {selection_debug}")
        if chosen_idx is None:
            chosen_idx = 0  # fall back to the cheapest chain

        chosen = presented[chosen_idx]
        material_winner = chosen["material"]
        manufacturer_winner = chosen["tailoring"]
        logistics_winner = chosen["logistics"]
        total_cost = chosen["total_cost"]
        total_chain_days = chosen["total_days"]
        loser_reason = "not selected for the winning chain"

    return {
        "winners": {
            "material": material_winner,
            "tailoring": manufacturer_winner,
            "logistics": logistics_winner,
        },
        "total_cost": total_cost,
        "total_days": total_chain_days,
        "loser_reason": loser_reason,
    }, None, None


def reservation_rfq(order, extra_fields, remaining_days):
    """the same request shape the agent already checked when it bid

    extra_fields carries sim_now, so the agent re-checks against the same
    reading of the walking clock the rfq was issued under"""
    return {
        "order_id": order["order_id"],
        "max_defect_rate": order["max_defect_rate"],
        "min_reputation_score": order["min_reputation_score"],
        "deadline": order["deadline"],
        "days_until_deadline": remaining_days,
        "priority": order["priority"],
        **extra_fields,
    }


async def acquire_reservations(nc, order, winners, stage_fields, remaining_days):
    """the reservation acknowledgment step from agent_logic_design.md

    candidate winners of the stages that hold state acknowledge before any
    accepted verdict is sent. if one refuses, everything already held for this
    chain is released, so no partial chain is ever left reserved

    logistics is never asked: it holds no stock or capacity in this schema

    return (ok, failed_stage, failed_agent_id, reason)"""
    acked = []
    for stage in DYNAMIC_STAGES:
        winner = winners[stage]
        request = {
            "agent_id": winner["agent_id"],
            "promised_completion_days": winner["estimated_completion_days"],
            "rfq": reservation_rfq(order, stage_fields[stage], remaining_days),
        }
        try:
            reply = await nc.request(reserve_subject(stage, winner["agent_id"]),
                                     json.dumps(request).encode(),
                                     timeout=RESERVATION_TIMEOUT)
            answer = json.loads(reply.data.decode())
        except Exception as exc:
            answer = {"ack": False, "reason": f"no_reservation_reply: {exc}"}

        if answer.get("ack"):
            acked.append((stage, winner))
            continue

        for held_stage, held_winner in acked:
            await nc.publish(release_subject(held_stage, held_winner["agent_id"]),
                             json.dumps({"agent_id": held_winner["agent_id"],
                                         "order_id": order["order_id"]}).encode())
        return False, stage, winner["agent_id"], answer.get("reason")

    return True, None, None, None


def _dynamic_extra(sim_now, reservation_retries):
    """the dynamic-run bookkeeping carried on the order.logging entry"""
    if not DYNAMIC_RESOURCES:
        return None
    return {
        "sim_now": sim_now.isoformat() if sim_now is not None else None,
        "reservation_retries": reservation_retries,
    }


async def handle_new_order(nc, msg, slm=None):
    order = json.loads(msg.data.decode())
    order_id = order["order_id"]
    print(f"new order received: {order_id}")

    if order["priority"] not in PRIORITY_MARKUP:
        print(f"[supervisor] order {order_id} has unknown priority {order['priority']!r}, discarding")
        return

    prod_row = product_row(order)
    # the walking clock, and every deadline question answered against it. mixing
    # a simulated arrival time with a real-clock deadline would make the two
    # halves of the same order disagree about what day it is
    sim_now = sim_now_for(order)
    total_days = days_until_deadline(order, sim_now)

    # sim_now rides on extra_fields so it reaches both the sealed rfq and the
    # auction rounds without either path knowing about it
    clock_fields = {"sim_now": sim_now.isoformat()} if sim_now is not None else {}
    stage_fields = {
        "material": {
            "material_type": order["material_type"],
            "required_kg": order["quantity"] * prod_row["material_kg_per_unit"],
            **clock_fields,
        },
        "tailoring": {
            "product_id": order["product_id"],
            "quantity": order["quantity"],
            **clock_fields,
        },
        "logistics": {
            "deliver_to": order["deliver_to"],
            "shipment_weight_kg": order["quantity"] * prod_row["finished_weight_kg_per_unit"],
            **clock_fields,
        },
    }

    material_bids = await open_stage(nc, "material", order, stage_fields["material"], total_days)

    material_winner = pick_winner(material_bids)
    if material_winner is None:
        print(f"[supervisor] order {order_id} infeasible: no material bids")
        await log_chain(nc, order_id, "infeasible", reason="no_bids_at_material_stage",
                        extra=_dynamic_extra(sim_now, 0))
        return

    manufacturer_bids = await open_stage(nc, "tailoring", order, stage_fields["tailoring"], total_days)

    manufacturer_winner = pick_winner(manufacturer_bids)
    if manufacturer_winner is None:
        print(f"[supervisor] order {order_id} infeasible: no manufacturer bids")
        await reject_all(nc, material_bids, "downstream stage had no eligible bids")
        await log_chain(nc, order_id, "infeasible", reason="no_bids_at_manufacturer_stage",
                        extra=_dynamic_extra(sim_now, 0))
        return

    logistics_bids = await open_stage(nc, "logistics", order, stage_fields["logistics"], total_days)

    logistics_winner = pick_winner(logistics_bids)
    if logistics_winner is None:
        print(f"[supervisor] order {order_id} infeasible: no logistics bids")
        await reject_all(nc, material_bids, "downstream stage had no eligible bids")
        await reject_all(nc, manufacturer_bids, "downstream stage had no eligible bids")
        await log_chain(nc, order_id, "infeasible", reason="no_bids_at_logistics_stage",
                        extra=_dynamic_extra(sim_now, 0))
        return

    # check the full chain before sending verdicts or asking the model
    remaining_at_validation = remaining_days_now(order, sim_now)
    all_bids = material_bids + manufacturer_bids + logistics_bids
    bids_by_stage = {"material": material_bids,
                     "tailoring": manufacturer_bids,
                     "logistics": logistics_bids}

    # select, then have the winners acknowledge their reservation, then accept.
    # a refusal drops that bid and re-selects from the rest, which is the design
    # doc's "evaluates the next feasible chain", using the existing selection
    # rule rather than a new one. the loop always terminates: every pass either
    # accepts or removes one bid
    reservation_retries = 0
    while True:
        selection, reason_code, loser_reason = await select_chain(
            order, bids_by_stage, remaining_at_validation, slm)
        if selection is None:
            await reject_all(nc, all_bids, loser_reason)
            await log_chain(nc, order_id, "infeasible", reason=reason_code,
                            extra=_dynamic_extra(sim_now, reservation_retries))
            return

        if not DYNAMIC_RESOURCES:
            break

        ok, failed_stage, failed_agent, refuse_reason = await acquire_reservations(
            nc, order, selection["winners"], stage_fields, total_days)
        if ok:
            break

        reservation_retries += 1
        print(f"[supervisor] order {order_id}: {failed_stage} agent {failed_agent} "
              f"could not honour its reservation ({refuse_reason}), dropping that "
              f"bid and re-selecting")
        bids_by_stage[failed_stage] = [b for b in bids_by_stage[failed_stage]
                                       if b["agent_id"] != failed_agent]

    winners = selection["winners"]
    total_cost = selection["total_cost"]
    total_chain_days = selection["total_days"]
    loser_reason = selection["loser_reason"]

    for stage in ("material", "tailoring", "logistics"):
        winner = winners[stage]
        for bid in bids_by_stage[stage]:
            if bid["agent_id"] == winner["agent_id"]:
                await send_verdict(nc, bid["agent_id"], {
                    "order_id": order_id,
                    "agent_id": bid["agent_id"],
                    "verdict": "accepted",
                    "agreed_price": bid["offer_amount"],
                    "deadline": order["deadline"],
                })
            else:
                await send_verdict(nc, bid["agent_id"], {
                    "order_id": order_id,
                    "agent_id": bid["agent_id"],
                    "verdict": "rejected",
                    "reason": loser_reason,
                })

    # bids dropped along the way still hear back, they are not left waiting
    dropped = [b for b in all_bids
               if b not in bids_by_stage["material"]
               and b not in bids_by_stage["tailoring"]
               and b not in bids_by_stage["logistics"]]
    await reject_all(nc, dropped, "could not honour its reservation")

    await log_chain(nc, order_id, "completed", chain={
        "material": winners["material"]["agent_id"],
        "tailoring": winners["tailoring"]["agent_id"],
        "logistics": winners["logistics"]["agent_id"],
    }, total_cost=total_cost, total_days=total_chain_days,
       extra=_dynamic_extra(sim_now, reservation_retries))
    print(f"[supervisor] order {order_id} completed, total cost {total_cost}")


async def run_order(nc, msg, slm=None):
    try:
        await handle_new_order(nc, msg, slm)
    except Exception:
        order_id = "<unparseable>"
        try:
            order_id = json.loads(msg.data.decode()).get("order_id", order_id)
        except Exception:
            pass
        print(f"[supervisor] order {order_id} raised an unhandled exception:")
        traceback.print_exc()


def announce_configuration():
    if BIDDING_MODE == REVERSE_AUCTION:
        print(f"[supervisor] bidding mode: {BIDDING_MODE}, up to {auction_max_rounds()} rounds per stage, "
              f"decrement EUR {platform_minimum_decrement()}, "
              f"{ROUND_WINDOW_SECONDS}s per round")
    else:
        print(f"[supervisor] bidding mode: {BIDDING_MODE}, {BIDDING_WINDOW_SECONDS}s per stage")

    if DYNAMIC_RESOURCES:
        # orders are published one at a time by the experiment runner, which waits
        # for each order.logging outcome, so exactly one order is in flight and the
        # walking clock only moves forward. no locking is needed or used
        print("[supervisor] dynamic resources on: sim clock follows order_created, "
              f"reservation ack required from {', '.join(DYNAMIC_STAGES)} before any "
              "accepted verdict")
    else:
        print("[supervisor] dynamic resources off, agents bid against the static "
              "csv snapshot as before")


async def serve(nc, slm):
    """take orders on an already connected client

    lifted out of main() unchanged so the live-demo stack can run the supervisor
    and the three stage agents in one process against one shared model, which is
    the only way three configurations fit on one card"""
    # keep task references until callbacks finish
    in_flight = set()
    counters = {"started": 0}
    cap_announced = False

    async def cb(msg):
        nonlocal cap_announced
        # enforce the slm order cap before starting any work
        if not order_within_sample(counters["started"], slm):
            if not cap_announced:
                print(f"[supervisor] SLM_ORDER_SAMPLE_SIZE={SLM_ORDER_SAMPLE_SIZE} reached, "
                      "ignoring further orders for this run")
                cap_announced = True
            return
        counters["started"] += 1
        # let the subscription accept the next order while this one runs
        task = asyncio.create_task(run_order(nc, msg, slm))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)

    async def on_reset(msg):
        nonlocal cap_announced
        # the agents reseed their own stock and workload, the supervisor holds
        # only the clock and the run counter
        SIM_CLOCK.now = None
        counters["started"] = 0
        cap_announced = False
        print("[supervisor] sim clock cleared, order counter reset")
        await msg.respond(json.dumps({"ok": True, "stage": "supervisor"}).encode())

    await nc.subscribe("order.created", cb=cb)
    await nc.subscribe(f"{CONTROL_RESET_SUBJECT}.supervisor", cb=on_reset)


async def main():
    announce_configuration()

    slm = load_slm("supervisor")
    # warm the model before connecting so live orders avoid startup latency
    await warmup_slm(slm, "supervisor")

    print(f"connecting to nats at {NATS_URL}")
    nc = await connect_with_retry(NATS_URL, "supervisor")

    await serve(nc, slm)

    print("supervisor is running and waiting for orders")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
