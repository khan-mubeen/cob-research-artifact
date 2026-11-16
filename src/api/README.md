# Interactive demo backend

This folder connects the bidding system to the interactive demo frontend in `ui/frontend/`. The demo is
separate from the v5 evaluation in `reports/v5/evaluation/`.

The full demo requires local GGUF model files that are not distributed with this repository. Base configurations
use pretrained models without task-specific fine-tuning; tuned configurations use earlier fine-tuning runs on a
smaller dataset, rather than the v5 adapters used in the evaluation. The rule-based configuration needs no model files.

## Setup

Install the runtime dependencies using the [root setup instructions](../../README.md#python-setup).
Then run `python -m src.api --help` from the repository root to inspect the commands.
A rule-only session uses `python -m src.api up --configs rule`; this needs no GGUF files.
The full model-backed session requires the paths listed below and `requirements-slm.txt`.
The [frontend README](../../ui/frontend/README.md) covers the separate browser interface.

This is local development tooling, not an authenticated public service. A broker may be downloaded on
first use; starting a session also creates runtime state and logs. Use a separate run directory when
checking a new setup if an existing session's files need to be retained (`DEMO_RUN_DIR`).

## How it was run

The demo ran the bidding system in `src/supervisor`, `src/agents` and `src/common` unchanged; nothing in this
folder makes a bidding decision. It was started and controlled through make targets, each a thin wrapper around
the module form:

```bash
make demo                            # python -m src.api up: brokers, five stacks, api on :8000
make demo_reset                      # python -m src.api reset: stock and workload back to the csv baseline
make demo_mode MODE=reverse_auction  # python -m src.api mode: switch bidding mode, restarts every stack
make demo_down                       # python -m src.api down: stop a session that was left running
```

Orders were submitted by id or from a file, for example `python -m src.api submit ORD-08905 --config qwen-tuned`
and `python -m src.api submit --file data/demo/restock_sequence.json --config rule`. The `rule` configuration needs
no model and no GPU (`make demo ARGS="--configs rule"`); the four model configurations need the local GGUF files
listed below. The orders used in the demo, and what each one shows, are listed in `data/demo/DEMO_ORDERS.md`.

The tooling downloads `nats-server` v2.14.5 into `tools/` when needed, checks it against the published
`SHA256SUMS` and verifies its version. An existing copy on `PATH`, or one named by `NATS_SERVER_BIN`,
is used instead when available.

## The five configurations

| configuration | bidding logic | local model path (not included) |
|---|---|---|
| `rule` | the rule-based agents, no model, no GPU | none |
| `qwen-base` | Qwen3.5-4B, without task-specific fine-tuning | `models/Qwen3.5-4B-Q8_0.gguf` |
| `qwen-tuned` | the earlier Qwen fine-tune for the current bidding mode | `models/epoch-2/deployment/qwen-sealed/` (sealed-bid), `models/reverse_auction/epoch-1/v4/deployment/qwen-auction/` (reverse auction) |
| `granite-base` | Granite 4.1-3B, without task-specific fine-tuning | `models/feasibility/smoke-artifacts/granite41-3b/` |
| `granite-tuned` | the earlier Granite fine-tune for the current bidding mode | `models/epoch-2/deployment/granite-sealed/` (sealed-bid), `models/reverse_auction/epoch-1/v4/deployment/granite-auction/` (reverse auction) |

Every model file is a Q8_0 GGUF. All five configurations were live at once, on their own brokers (4232-4236), so
choosing one was a routing decision rather than a reload. `SECOND_MODEL` in `settings.py` decides which
architecture joins Qwen. Phi and Gemma are listed there as alternatives. Ministral and SmolLM3 have no base
GGUF configured for this demo.

## What a frontend uses

| route | purpose |
|---|---|
| `GET /ws` | submit orders and read the event stream, one connection both ways |
| `GET /api/status` | mode, configurations, model menu, stack health, queue depth |
| `GET /api/agents` | the agent roster per stage, for drawing the board |
| `GET /api/state/{config}` | live stock and workload per agent |
| `POST /api/orders` | `{config, order_id \| order \| path, wait?}` |
| `POST /api/reset` | `{config?}`, reseeds state and returns the snapshot it left behind |
| `POST /api/mode` | `{mode}`, restarts every stack in the other bidding mode |

Websocket messages in are `{"action": "submit" \| "reset" \| "mode" \| "status", ...}`
with the same fields as the post body. Every message out is one envelope:

```json
{"seq": 41, "type": "bid", "config": "qwen-tuned", "order_id": "ORD-33452",
 "at": "<utc timestamp>", "stage": "material", "data": { ... }}
```

`type` is one of `hello`, `order_queued`, `order_started`, `stage_opened`, `rfq`,
`bid`, `no_bid`, `stage_closed`, `auction_trace`, `state_change`, `verdict`,
`order_outcome`, `order_finished`, `state_reset`, `mode_switching`,
`mode_switched`, `failure`, `error`. For bidding events relayed from NATS,
`data` is the original message payload.

`GET /api/status` carries the model menu used by the client's dropdown. It lists all six
architectures; `active` means an architecture is configured and its model files are available,
not that the model has finished loading. Other entries remain visible but disabled.

The relay exposes the following fields to the frontend:

**`stage_opened` identifies the stage and its bidding window:**

```json
{"seq": 4, "type": "stage_opened", "config": "rule", "order_id": "ORD-24907",
 "at": "<utc timestamp>",
 "data": {"event": "stage_opened", "stage": "material", "mode": "sealed",
          "sim_now": "<order_created>", "roster_size": 15,
          "window_seconds": 2.0, "max_rounds": 1}}
```

`window_seconds` is the sealed-bid collection window or the duration of one reverse-auction
round. `max_rounds` is 1 in sealed mode and the auction round cap otherwise; `roster_size`
is how many agents were asked. The matching `stage_closed` carries `bid_count`, `bidder_agent_ids`,
`best_agent_id`, `best_offer_amount` and `elapsed_seconds`.

**An auction `rfq` says which round it is, on the envelope itself.**
`round_number`, `current_best_bid` (null in round 1) and
`platform_minimum_decrement` are hoisted alongside `stage`, so a client does not
have to compare consecutive bids to work out where it is:

```json
{"seq": 1152, "type": "rfq", "config": "rule", "order_id": "ORD-08905",
 "stage": "material", "round_number": 2, "current_best_bid": 127.71,
 "platform_minimum_decrement": 5.0, "data": { ... }}
```

**An order that did not complete carries a plain-language `summary`** on its
`order_finished` envelope, restating the reason the supervisor logged plus the
declines the agents published, for example *no material agent bid: 15 agents
declined, 9 do not carry the material, 6 not enough stock even with a restock*.
It is a translation of logged facts, never a guess at a cause.

## The files

| file | what it is |
|---|---|
| `settings.py` | the configurations: models, ports, windows, which second model |
| `processes.py` | brokers and stacks, readiness, crash detection, mode restart |
| `stack.py` | one configuration in one process, supervisor plus three stage agents |
| `orders.py` | order sources (id, payload, file) and validation |
| `pipeline.py` | `submit_order()` and the sequential queue |
| `infeasibility.py` | the plain-language summary for an order that did not complete |
| `relay.py` | NATS to websocket, and the control requests |
| `server.py` | the routes |
| `timing.py` | where the window numbers in `settings.py` come from |
| `static/client.html` | a throwaway test client, served at the root of the api |

## Design notes

**Separate brokers.** Each configuration gets its own
`nats-server` so five stacks can be live at once without hearing each other's
RFQs. Subjects are constants throughout the bidding code, so sharing one broker
would mean namespacing every one of them.

**Shared model context.** Each demo configuration runs the supervisor and the three stage agents
in one process, sharing a model context. This avoids loading a separate copy of the GGUF for each
service. The stages run sequentially.

**A mode switch is a restart, and it is refused while an order is running.** The
supervisor and every agent read `BIDDING_MODE` once at startup, and the
fine-tuned GGUF differs between the two modes, so a switch stops every stack and
starts it again. The restart time depends on model loading and the host machine. If an order is in flight or queued it answers
`409` and changes nothing; the caller retries when the queue is idle.

`make demo_mode MODE=...`, `POST /api/mode`, and the websocket `{"action":
"mode"}` are three ways to trigger the same thing: all three call
`Service.switch_mode`, so the refusal, the `mode_switching` and `mode_switched`
broadcasts and the pid-file rewrite are identical whichever one is used.
