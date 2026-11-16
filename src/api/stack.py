"""
one configuration, one process: the supervisor and all three stage agents

run as: python -m src.api.stack <config-name>

the batch deployment runs these four as four processes, each loading its own
copy of the model. that costs 4 x 5.15 GB per configuration and two model-backed
configurations would not fit on this card. here they share one llama context,
which costs nothing in wall time: the three stages are sequential, agents inside
one stage already serialised through a single context in the batch layout, and
the supervisor's chain selection runs after logistics has closed

nothing about the protocol changes. every subject, payload, window and decision
is the one the batch runs used, and the pieces are imported from the same
modules rather than reimplemented here
"""

import asyncio
import os
import sys

from src.agents.logistics import agent as logistics_agent
from src.agents.manufacturer import agent as manufacturer_agent
from src.agents.material import agent as material_agent
from src.api import settings
from src.common.dynamic_state import (
    DYNAMIC_STAGES, DynamicStateService, build_state, dynamic_resources_enabled,
)
from src.common.nats_client import connect_with_retry
from src.common.slm import load_slm, warmup_slm
from src.supervisor import supervisor

# (stage name used by the state module, agent module). the manufacturer process
# is the tailoring stage, the two names are used interchangeably in this repo
STAGES = (
    ("material", material_agent),
    ("tailoring", manufacturer_agent),
    ("logistics", logistics_agent),
)


async def start_stage(nc, stage, module, slm, dynamic):
    agents = module.load_active_agents()
    state_service = None
    if dynamic and stage in DYNAMIC_STAGES:
        # logistics holds no stock or capacity in this schema and deliberately
        # gains none, so it has no state service and answers no reservation
        state_service = DynamicStateService(nc, stage, build_state(stage, agents), agents)
        await state_service.start()
    for agent in agents:
        # logistics keeps no state, so its callback never took a state service
        if stage in DYNAMIC_STAGES:
            await module.run_agent(nc, agent, slm, state_service)
        else:
            await module.run_agent(nc, agent, slm)
    return [a["agent_id"] for a in agents]


async def main(name):
    mode = settings.bidding_mode()
    model = settings.model_path(name, mode)
    print(f"[stack {name}] {settings.CONFIG_LABELS[name]}, mode {mode}, "
          f"model {model.name if model else 'none'}", flush=True)
    supervisor.announce_configuration()

    # one model for the whole stack, loaded and warmed before anything subscribes
    slm = load_slm(f"stack {name}")
    await warmup_slm(slm, f"stack {name}")

    nc = await connect_with_retry(settings.nats_url(name), f"stack {name}")
    dynamic = dynamic_resources_enabled()

    for stage, module in STAGES:
        agent_ids = await start_stage(nc, stage, module, slm, dynamic)
        print(f"[stack {name}] {stage} ready: {', '.join(agent_ids)}", flush=True)

    await supervisor.serve(nc, slm)

    # the api waits for this exact line before it will route an order here
    print(f"[stack {name}] ready", flush=True)

    try:
        await asyncio.Future()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in settings.CONFIG_NAMES:
        raise SystemExit(f"usage: python -m src.api.stack {'|'.join(settings.CONFIG_NAMES)}")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    asyncio.run(main(sys.argv[1]))
