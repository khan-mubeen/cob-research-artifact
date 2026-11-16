import asyncio
import json
import os
from pathlib import Path
import pandas as pd
from src.common.bidding_formulas import (
    decide_manufacturer_bid, manufacturer_eligibility_reason, validate_bid_values,
    apply_dynamic_guard, dynamic_guard_enabled,
)
from src.common.bid_activity import publish_no_bid
from src.common.bidding_modes import REVERSE_AUCTION, bidding_mode, is_auction_rfq
from src.common.dynamic_state import (
    DynamicStateService, build_state, dynamic_resources_enabled,
)
from src.common.nats_client import connect_with_retry
from src.common.reverse_auction import AuctionMemory, handle_auction_rfq
from src.common.slm import load_slm, slm_decide_manufacturer_bid, warmup_slm
from config import DATA_VERSION
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
DATA_DIR = Path(os.getenv("DATA_DIR", f"data/synthetic/{DATA_VERSION}"))
ACTIVE_AGENT_IDS = os.getenv("ACTIVE_AGENT_IDS", "")
# resolved once at startup so a bad BIDDING_MODE fails before any rfq arrives
BIDDING_MODE = bidding_mode()
# one memory for the process, entries are keyed by agent, order and stage
AUCTION_MEMORY = AuctionMemory()
# manufacturer rfqs use tailoring in the subject name
RFQ_SUBJECT = "rfq.tailoring.*"
def load_active_agents():
    df = pd.read_csv(DATA_DIR / "manufacturer_agents.csv")
    requested = [a.strip() for a in ACTIVE_AGENT_IDS.split(",") if a.strip()]
    if requested:
        df = df[df["agent_id"].isin(requested)]
        missing = set(requested) - set(df["agent_id"])
        if missing:
            raise RuntimeError(
                f"ACTIVE_AGENT_IDS has unknown agent_id(s): {sorted(missing)}"
            )
    if df.empty:
        raise RuntimeError(
            "no manufacturer agents to run, check manufacturer_agents.csv and ACTIVE_AGENT_IDS"
        )
    return df.to_dict("records")
async def run_agent(nc, agent, slm, state_service=None):
    agent_id = agent["agent_id"]
    async def cb(msg):
        try:
            rfq = json.loads(msg.data.decode())
        except Exception as e:
            print(f"[{agent_id}] could not parse rfq payload, ignoring: {e}")
            return
        # walk the simulated clock to this order and refresh this agent's workload
        # view before anything is decided, so completed jobs release first
        if state_service is not None:
            await state_service.on_rfq(agent, rfq)
        # a mode mismatch between supervisor and agent must be visible, not silent
        auction_rfq = is_auction_rfq(rfq)
        if auction_rfq != (BIDDING_MODE == REVERSE_AUCTION):
            print(f"[{agent_id}] ignoring rfq for {rfq.get('order_id', 'unknown')}: this process "
                  f"runs BIDDING_MODE={BIDDING_MODE} and the rfq is "
                  f"{'reverse_auction' if auction_rfq else 'sealed'}")
            return
        if auction_rfq:
            await handle_auction_rfq(nc, "tailoring", agent, rfq, slm, AUCTION_MEMORY)
            return
        try:
            # check eligibility before building the prompt
            ineligible = manufacturer_eligibility_reason(agent, rfq)
            if ineligible is not None:
                result = {"decision": "NO_BID", "debug_reason": ineligible}
            elif slm is not None:
                result = await slm_decide_manufacturer_bid(slm, agent, rfq)
            else:
                result = decide_manufacturer_bid(agent, rfq)
        except Exception as e:
            order_id = rfq.get("order_id", "unknown")
            print(f"[{agent_id}] decision failed on {order_id}, staying silent: {e}")
            await publish_no_bid(nc, "tailoring", agent_id, rfq, f"decision_failed: {e}")
            return
        # guard disabled by default (SLM_DYNAMIC_GUARD unset)
        if slm is not None and result["decision"] == "BID" and dynamic_guard_enabled():
            result, guard_reason = apply_dynamic_guard("tailoring", agent, rfq, result)
            if guard_reason is not None:
                print(f"[{agent_id}] dynamic guard suppressed BID on "
                      f"{rfq['order_id']}: {guard_reason}")
        if result["decision"] == "NO_BID":
            print(
                f"[{agent_id}] staying silent on {rfq['order_id']}: "
                f"{result['debug_reason']}"
            )
            await publish_no_bid(nc, "tailoring", agent_id, rfq, result["debug_reason"])
            return
        invalid_reason = validate_bid_values(result)
        if invalid_reason:
            print(f"[{agent_id}] staying silent on {rfq['order_id']}: {invalid_reason}")
            await publish_no_bid(nc, "tailoring", agent_id, rfq, invalid_reason)
            return
        reply_to = rfq.get("reply_to")
        if not reply_to:
            print(
                f"[{agent_id}] rfq for {rfq.get('order_id', 'unknown')} "
                "missing reply_to, cannot bid"
            )
            return
        try:
            await nc.publish(reply_to, json.dumps(result).encode())
            print(f"[{agent_id}] bid {result['offer_amount']} on {rfq['order_id']}")
        except Exception as e:
            print(f"[{agent_id}] failed to publish bid on {rfq['order_id']}: {e}")
    await nc.subscribe(RFQ_SUBJECT, cb=cb)
    print(
        f"[{agent_id}] subscribed to {RFQ_SUBJECT}, "
        "callback registered, ready for rfqs"
    )
async def main():
    agents = load_active_agents()
    agent_ids = ", ".join(a["agent_id"] for a in agents)
    print(
        f"manufacturer stage configuration loaded, active agents: {agent_ids}"
    )
    print(f"manufacturer stage bidding mode: {BIDDING_MODE}")
    slm = load_slm("manufacturer stage")
    # warm the model before connecting so live requests avoid startup latency
    await warmup_slm(slm, "manufacturer stage")
    nc = await connect_with_retry(NATS_URL, "manufacturer stage")
    print(f"manufacturer stage connected to nats at {NATS_URL}")
    state_service = None
    if dynamic_resources_enabled():
        state_service = DynamicStateService(
            nc, "tailoring", build_state("tailoring", agents), agents)
        await state_service.start()
    else:
        print("manufacturer stage dynamic resources off, bidding against the "
              "static csv snapshot as before")
    for agent in agents:
        await run_agent(nc, agent, slm, state_service)
    print(
        f"manufacturer stage ready, "
        f"{len(agents)} agent(s) live and subscribed: {agent_ids}"
    )
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await nc.drain()
if __name__ == "__main__":
    asyncio.run(main())
