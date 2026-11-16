"""
which agents declined an rfq, for monitoring only

a published bid is visible on its reply subject, but an agent that stays silent
says so only in its own console log. a live view cannot tell a decline from a
reply that has not arrived yet, which is exactly what a bidding round looks like
while it is open

this is a reporting feed in the same sense as order.dynamic_state: no agent
subscribes to it and no bidding decision reads it. the reverse-auction rules keep
the silence reason off the reply path and that still holds, the reason never
travels with a bid; publishing it here is a deliberate exception for the live
demo and is recorded as one rather than left implicit
"""

import json
from datetime import datetime, timezone


# agent -> anyone listening, one message per declined rfq
BID_ACTIVITY_SUBJECT = "order.bid.activity"


async def publish_no_bid(nc, stage, agent_id, rfq, reason):
    """report one agent's silence on one rfq

    failures are swallowed: a monitoring feed must never be able to take down a
    bid path that would otherwise have worked"""
    event = {
        "event": "no_bid",
        "stage": stage,
        "agent_id": agent_id,
        "order_id": rfq.get("order_id"),
        "reason": reason,
        "round_number": rfq.get("round_number"),
        "sim_now": rfq.get("sim_now"),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await nc.publish(BID_ACTIVITY_SUBJECT, json.dumps(event).encode())
    except Exception as e:
        print(f"[{agent_id}] could not report silence on "
              f"{rfq.get('order_id', 'unknown')}: {e}")
