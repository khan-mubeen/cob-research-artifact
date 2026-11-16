## Overview

This document presents the agent logic design. It describes the overall system architecture, the behavior and interactions of the agents, and the planned experimental methodology, including how experiments will be conducted and evaluated. The architecture of the proof of concept (PoC) is also defined in this document.

---

## Shared Agent Checks

Before calculating any bid, all three agent types run the same eligibility checks.

`days_until_deadline` is computed once, from the order's absolute deadline against current processing time, not from order creation time. The same value is used for every stage's eligibility check. It is not reduced between stages based on any specific upstream agent's completion time. Bidding windows still open and close in sequence, material, then manufacturer, then logistics, per the approved flow below, but the deadline reference each stage checks against does not shrink between them.


```
IF agent does not handle the requested product, material, or delivery region:
    stay silent, publish nothing

IF agent defect or damage probability > order max_defect_rate:
    stay silent, publish nothing

IF agent reputation_score < order min_reputation_score:
    stay silent, publish nothing

IF estimated completion time > days_until_deadline:
    stay silent, publish nothing
```

Priority markup used in all bid calculations:

```
urgent:   +0.30
normal:    0.00
flexible: -0.15
```

Bidding bounds used by all agents:

```
minimum_bid = internal_cost x (1 + min_profit_margin)
initial_bid = internal_cost x (1 + initial_markup) x extra_markup_factors
final_bid   = max(minimum_bid, initial_bid)
```

---

## Material Provider Logic

**Input received via `rfq.material`:**

```
order_id, material_type, required_kg, max_defect_rate, min_reputation_score, deadline, priority, bid_close_at, reply_to
```

**Decision flow:**

```
RECEIVE rfq.material

IF material_type NOT IN agent.material_types:
    stay silent

IF agent.defect_probability > max_defect_rate:
    stay silent

IF agent.reputation_score < min_reputation_score:
    stay silent

IF agent.available_kg >= required_kg:
    completion_days = agent.default_completion_days

ELSE IF agent.available_kg + agent.restock_amount_kg >= required_kg:
    completion_days = agent.default_completion_days + agent.restock_time_days

ELSE:
    stay silent

IF completion_days > days_until_deadline:
    stay silent

material_cost = required_kg x agent.base_cost_per_kg
internal_cost = material_cost x (1 + agent.defect_probability)
minimum_bid   = internal_cost x (1 + agent.min_profit_margin)
initial_bid   = internal_cost x (1 + agent.initial_markup) x (1 + priority_markup)

PUBLISH BID to the RFQ reply_to subject
```

**Note:** Empty material columns in the CSV mean the agent does not handle that material at all. This is different from zero stock. Zero means they carry it but are out. Empty means it is not their product.

---

## Manufacturer Logic

**Input received via `rfq.tailoring`:**

```
order_id, product_id, quantity, max_defect_rate, min_reputation_score, deadline, priority, bid_close_at, reply_to
```

**Decision flow:**

```
RECEIVE rfq.tailoring

IF product_id NOT IN agent.product_types:
    stay silent

IF agent.defect_probability > max_defect_rate:
    stay silent

IF agent.reputation_score < min_reputation_score:
    stay silent

IF agent.daily_capacity_units <= 0:
    stay silent

backlog_days         = agent.current_workload_units / agent.daily_capacity_units
production_days      = quantity / agent.daily_capacity_units
estimated_completion = backlog_days + production_days

IF estimated_completion > days_until_deadline:
    stay silent

workload_ratio = (agent.current_workload_units + quantity) / (agent.daily_capacity_units x days_until_deadline)

IF workload_ratio < 0.50:
    workload_markup = 0.00
ELSE IF workload_ratio < 0.75:
    workload_markup = 0.05
ELSE IF workload_ratio < 0.90:
    workload_markup = 0.10
ELSE IF workload_ratio <= 1.00:
    workload_markup = 0.15
ELSE:
    stay silent

base_production_cost = quantity x agent.unit_cost_for_product
internal_cost        = base_production_cost x (1 + agent.defect_probability)
minimum_bid          = internal_cost x (1 + agent.min_profit_margin)
initial_bid          = internal_cost x (1 + agent.initial_markup)
                       x (1 + workload_markup) x (1 + priority_markup)

PUBLISH BID to the RFQ reply_to subject
```

**Note on data:** Sewing and finishing are merged into one agent per team. The lower of the two department capacities is used as the effective throughput limit.

`current_workload_units` is static input unless `DYNAMIC_RESOURCES` is on, where it is runtime-managed state. See Dynamic Agent State.

---

## Logistics Logic

**Input received via `rfq.logistics`:**

```
order_id, deliver_to, shipment_weight_kg, max_defect_rate, min_reputation_score, deadline, priority, bid_close_at, reply_to
```

**Decision flow:**

```
RECEIVE rfq.logistics

IF deliver_to NOT IN agent.service_regions:
    stay silent

IF agent.reputation_score < min_reputation_score:
    stay silent

IF agent.damage_probability > max_defect_rate:
    stay silent

IF deliver_to == agent.location:
    route_days = agent.default_completion_same_city_days
ELSE:
    route_days = agent.default_completion_intercity_days

IF route_days > days_until_deadline:
    stay silent

base_logistics_cost = agent.base_delivery_cost + (agent.cost_per_kg x shipment_weight_kg)

internal_cost = base_logistics_cost x (1 + agent.damage_probability)
minimum_bid   = internal_cost x (1 + agent.min_profit_margin)
initial_bid   = internal_cost x (1 + agent.initial_markup) x (1 + priority_markup)

PUBLISH BID to the RFQ reply_to subject
```
**Note:** `delay_probability` is present in the agent profile but not used in the current bid formula, reserved for future use or descriptive reporting.

---

## Final Chain Validation

Order budget is never checked by an individual stage. Each stage only checks its own eligibility and produces its own offer. After material, manufacturer, and logistics bids are all collected, the supervisor performs one aggregate check before any winner is confirmed:

total_chain_cost = material_offer + manufacturer_offer + logistics_offer
total_chain_days = material_completion + manufacturer_completion + logistics_completion

IF total_chain_cost > order.budget:
    chain is invalid, do not accept
IF total_chain_days > remaining_time_at_validation:
    chain is invalid, do not accept

remaining_time_at_validation is recomputed against current processing time at
the moment of final validation, it is not the days_until_deadline value used
earlier for per-stage eligibility, since real time has elapsed while bids were
collected.

All bids remain provisional until this check passes for a specific combination. No accepted verdict is ever sent before final validation succeeds. If no combination of collected bids passes both checks, the order is marked infeasible, and every agent that submitted a bid receives a rejected verdict.

---

## Bidding Modes

### Reverse Auction

Agents see the current best bid and can lower their price to compete. Rounds run up to a cap of three, and the auction closes earlier if a round brings no improvement.

```
Round 1:
  all eligible agents calculate and submit initial_bid
  supervisor announces current best bid to all agents

Round 2 onward:
  next_bid = current_best_bid - platform_minimum_decrement
  IF next_bid >= minimum_bid: submit next_bid
  ELSE: stay silent for remaining rounds
  supervisor announces new best bid

After every round:
  IF the best bid did not improve: close the auction and select the winner
  ELSE IF the round cap is reached: close the auction and select the winner
  ELSE: open the next round
```

The platform minimum decrement is a fixed EUR amount. For the experiments, we will use a value of EUR 5. This parameter is configurable and can be changed for future experiments. The round cap defaults to 3 and is configurable through `AUCTION_MAX_ROUNDS`.

From round 2 on the rule allows one legal offer, so every surviving agent submits the same price and the round is a tie. Ties are settled by earliest arrival at the supervisor, both between rounds and when the winning bid is finally selected.

### What is implemented

Rounds are run by the supervisor in `src/supervisor/auction.py` against `standing`, one bid per agent. An agent's standing bid is the best price it offered at any point in that stage auction: a later round replaces it only by undercutting it, and an equal offer does not replace it, which keeps the earlier arrival stamp the tie-break depends on. Going silent is not a withdrawal, the standing bid stays on the table and can still win, which is what stops a stage closing empty once every agent has reached its floor. Chain selection downstream receives one standing bid per agent, the same list shape sealed bidding produces, so nothing after the stage is mode-specific.

During validation of the mechanism a rare case was found where that rule was not enforced: a later, worse offer overwrote an agent's earlier better one, so a stage could close above the lowest price actually offered. Triggering it requires an agent to break the round-decrement rule, which the deterministic rule-based path cannot do, since it offers `current_best_bid - platform_minimum_decrement` or stays silent, so it only ever appeared under a model. Across the completed reverse-auction runs it occurred in 2 of 3,486 stage auctions and affected the recorded outcome of 1 of 150 orders in one architecture's run: granite, ORD-38735, logged infeasible where the preserved bid would have brought the chain inside budget. The cause was traced to a single line, quantified against every recorded trace, and corrected for all runs from 29 August 2026 onward. The completed batches are reported exactly as they ran, with this behaviour noted rather than retrofitted. Regression tests covering it are in `tests/test_auction_rounds.py`.

### Sealed Bidding

Each agent submits one private offer. Nobody sees what others bid. The supervisor collects all offers and builds the best feasible chain.

```
IF agent is eligible:
    calculate and publish one final offer

ELSE:
    stay silent
```

### Stage Sequencing

RFQs are published in sequence, not concurrently:

1. Material bidding window opens, material agents submit bids.
2. Material window closes, late bids are ignored.
3. Manufacturer bidding window opens, manufacturer agents submit bids.
4. Manufacturer window closes, late bids are ignored.
5. Logistics bidding window opens, logistics agents submit bids.
6. Logistics window closes, late bids are ignored.
7. Final Chain Validation runs against all collected bids.

Each window uses the same days_until_deadline reference described in Shared Agent Checks. Only window timing is sequential, the deadline value each stage checks against is not reduced by an earlier stage's specific result.

### Chain Selection Criteria

The criterion is priority-dependent and is now confirmed. `decide_chain_selection()` in `src/common/bidding_formulas.py` implements it, and the supervisor prompt teaches it:

```
urgent   -> fewest total_completion_days, lowest total_cost breaks ties
flexible -> lowest total_cost, fewest total_completion_days breaks ties
normal   -> lowest equal-weighted score of three ratios,
            total_cost / budget, total_completion_days / days remaining,
            and compound chain risk / max acceptable defect or damage probability
```

Compound chain risk is `1 - (1 - defect_p_material) * (1 - defect_p_manufacturing) * (1 - damage_p_logistics)`. Total cost and total days are the sums of the three stage offers and completion times.

Cost and days are compared at the two decimals the corpus stores and the prompt prints, so two chains that read as identical are treated as identical. When every term above is exactly equal, the tie is settled by the `(material_agent_id, manufacturer_agent_id, logistics_agent_id)` tuple in ascending string order, lowest wins. That final tie-break applies to all three priorities. It exists so an exact tie has one answer regardless of the order candidates arrive or are shuffled in; it is reached only on exact equality and does not alter the priority objective for any order where one chain is genuinely better.

This section previously stated that the live chain-selection criterion still needed confirmation, while the code and the training prompts already implemented the priority-dependent rule. That drift is recorded here rather than quietly overwritten, following the same disclosure pattern used for the reverse-auction standing-bid correction above. Two related gaps remain open and are deliberately not closed by this section: the live rule-based supervisor in `src/supervisor/supervisor.py` picks the individually cheapest bid per stage instead of calling `decide_chain_selection()`, and the supervisor prompt shows a bounded candidate list rather than every combination.

### How Prompts Differ Between Modes

In sealed bidding the agent only receives the order and its own profile. In reverse auction three extra fields are added:

```
current_best_bid
platform_minimum_decrement
round_number
```

These fields are absent from sealed bidding prompts entirely.

---

## Internal vs NATS Output

There are two separate layers. This distinction is important for implementation.

```
Layer 1: SLM inference:
  SLM receives prompt and returns an internal decision: BID or NO_BID
  This internal decision is used by the agent wrapper

Layer 2: NATS publishing:
  IF internal decision is BID:
      publish bid message to the RFQ reply_to subject
  IF internal decision is NO_BID:
      do nothing, stay silent
```

The reason field is never sent to NATS. For debugging and manual review during experiments, the reason can be saved in local logs only as a `debug_reason` field. This helps later when SLM and rule-based decisions disagree and we need to understand why.

---

## Agent Output Format

Aligned with the NATS config provided by supervisors. Every agent that decides to bid publishes this exact format. Agents that do not bid publish nothing.

**Published bid message:**

```json
{
  "order_id": "ORD-001",
  "stage": "tailoring",
  "agent_id": "T2",
  "decision": "BID",
  "offer_amount": 2450.00,
  "estimated_completion_days": 5,
  "defect_probability": 0.03
}
```

`estimated_completion_days` is used for all three agent types including logistics. Same field name everywhere so the supervisor can parse consistently.

**Supervisor verdict to accepted agent:**

```json
{
  "order_id": "ORD-001",
  "agent_id": "T1",
  "verdict": "accepted",
  "agreed_price": 420,
  "deadline": "2026-06-22"
}
```

**Supervisor verdict to rejected agent:**

```json
{
  "order_id": "ORD-001",
  "agent_id": "T3",
  "verdict": "rejected",
  "reason": "high cost and long completion time"
}
```

Only the supervisor sends reasons. Agents never do.

---

## SLM Prompt Templates

The SLM agent receives the same structured input as the rule-based agent. Structured JSON is required so the supervisor can parse decisions automatically. The TASK section includes the calculation formulas so the SLM follows the same pricing logic as the rule-based agent and results stay comparable.

**Important note on NO_BID behavior:** When the TASK steps say "return NO_BID", this means the SLM sets `decision` to `"NO_BID"` in the JSON output and fills `offer_amount` and `estimated_completion_days` with `null`. The agent wrapper reads this internal decision and stays completely silent on NATS. Nothing gets published. Only when decision is `"BID"` does the wrapper publish the JSON to NATS. The SLM never sends a NO_BID message anywhere.

### Material Provider

**Sealed bidding:**

```
System:
You are a material provider agent in a multi-agent supply chain bidding system.
Decide whether to bid on the incoming material request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

User:
AGENT PROFILE
agent_id: {agent_id}
location: {location}
materials supplied: {material_types}
available stock for requested material (kg): {available_kg}
base cost per kg (EUR): {base_cost_per_kg}
restock amount (kg): {restock_amount_kg}
restock time (days): {restock_time_days}
default completion days: {default_completion_days}
defect probability: {defect_probability}
reputation score: {reputation_score}
min profit margin: {min_profit_margin}
initial markup: {initial_markup}

RFQ
order_id: {order_id}
material needed: {material_type}
required quantity (kg): {required_kg}
max allowed defect rate: {max_defect_rate}
min required reputation: {min_reputation_score}
days until deadline: {days_until_deadline}
priority: {priority}

TASK
1. Check if you supply this material. If not, return NO_BID.
2. Check defect_probability <= max_defect_rate. If not, return NO_BID.
3. Check reputation_score >= min_reputation_score. If not, return NO_BID.
4. Check stock:
   if available_kg >= required_kg: completion = default_completion_days
   else if available_kg + restock_amount_kg >= required_kg: completion = default_completion_days + restock_time_days
   else: return NO_BID
5. Check completion <= days_until_deadline. If not, return NO_BID.
6. Calculate:
   material_cost  = required_kg x base_cost_per_kg
   internal_cost  = material_cost x (1 + defect_probability)
   minimum_bid    = internal_cost x (1 + min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost x (1 + initial_markup) x (1 + priority_markup)
   offer_amount   = max(minimum_bid, initial_bid)
7. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "{order_id}",
  "stage": "material",
  "agent_id": "{agent_id}",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": {defect_probability}
}
```

**Extra fields for reverse auction:**
```
current_best_bid (EUR): {current_best_bid}
platform_minimum_decrement (EUR): {platform_minimum_decrement}
round_number: {round_number}
```
In rounds after round 1: if current_best_bid - platform_minimum_decrement >= minimum_bid, submit that as offer. Otherwise stay silent.

---

### Manufacturer

**Sealed bidding:**

```
System:
You are a manufacturer agent in a multi-agent garment supply chain bidding system.
Decide whether to bid on the manufacturing request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

User:
AGENT PROFILE
agent_id: {agent_id}
location: {location}
products you make: {product_types}
daily capacity (units): {daily_capacity_units}
current workload (units): {current_workload_units}
unit cost for requested product (EUR): {unit_cost_for_product}
defect probability: {defect_probability}
reputation score: {reputation_score}
min profit margin: {min_profit_margin}
initial markup: {initial_markup}

RFQ
order_id: {order_id}
product: {product_id}
quantity: {quantity}
max allowed defect rate: {max_defect_rate}
min required reputation: {min_reputation_score}
days until deadline: {days_until_deadline}
priority: {priority}

TASK
1. Check if you make this product. If not, return NO_BID.
2. Check defect_probability <= max_defect_rate. If not, return NO_BID.
3. Check reputation_score >= min_reputation_score. If not, return NO_BID.
4. Calculate:
   backlog_days         = current_workload_units / daily_capacity_units
   production_days      = quantity / daily_capacity_units
   estimated_completion = backlog_days + production_days
5. Check estimated_completion <= days_until_deadline. If not, return NO_BID.
6. Calculate workload ratio and markup:
   workload_ratio = (current_workload_units + quantity) / (daily_capacity_units x days_until_deadline)
   if workload_ratio < 0.50:    workload_markup = 0.00
   else if ratio < 0.75:        workload_markup = 0.05
   else if ratio < 0.90:        workload_markup = 0.10
   else if ratio <= 1.00:       workload_markup = 0.15
   else:                        return NO_BID
7. Calculate:
   base_cost      = quantity x unit_cost_for_product
   internal_cost  = base_cost x (1 + defect_probability)
   minimum_bid    = internal_cost x (1 + min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost x (1 + initial_markup) x (1 + workload_markup) x (1 + priority_markup)
   offer_amount   = max(minimum_bid, initial_bid)
8. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "{order_id}",
  "stage": "tailoring",
  "agent_id": "{agent_id}",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": {defect_probability}
}
```

**Extra fields for reverse auction:**
```
current_best_bid (EUR): {current_best_bid}
platform_minimum_decrement (EUR): {platform_minimum_decrement}
round_number: {round_number}
```
In rounds after round 1: if current_best_bid - platform_minimum_decrement >= minimum_bid, submit that as offer. Otherwise stay silent.

---

### Logistics

**Sealed bidding:**

```
System:
You are a logistics agent in a multi-agent supply chain bidding system.
Decide whether to bid on the delivery request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

User:
AGENT PROFILE
agent_id: {agent_id}
location: {location}
service regions: {service_regions}
base delivery cost (EUR): {base_delivery_cost}
cost per kg (EUR): {cost_per_kg}
same-city delivery days: {default_completion_same_city_days}
intercity delivery days: {default_completion_intercity_days}
damage probability: {damage_probability}
reputation score: {reputation_score}
min profit margin: {min_profit_margin}
initial markup: {initial_markup}

RFQ
order_id: {order_id}
deliver to: {deliver_to}
shipment weight (kg): {shipment_weight_kg}
max allowed damage rate: {max_defect_rate}
min required reputation: {min_reputation_score}
days until deadline: {days_until_deadline}
priority: {priority}

TASK
1. Check if deliver_to is in your service_regions. If not, return NO_BID.
2. Check reputation_score >= min_reputation_score. If not, return NO_BID.
3. Check damage_probability <= max_defect_rate. If not, return NO_BID.
4. Calculate route days:
   if deliver_to == your location: route_days = default_completion_same_city_days
   else: route_days = default_completion_intercity_days
5. Check route_days <= days_until_deadline. If not, return NO_BID.
6. Calculate:
   base_cost      = base_delivery_cost + (cost_per_kg x shipment_weight_kg)
   internal_cost  = base_cost x (1 + damage_probability)
   minimum_bid    = internal_cost x (1 + min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost x (1 + initial_markup) x (1 + priority_markup)
   offer_amount   = max(minimum_bid, initial_bid)
7. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "{order_id}",
  "stage": "logistics",
  "agent_id": "{agent_id}",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": {damage_probability}
}
```

**Extra fields for reverse auction:**
```
current_best_bid (EUR): {current_best_bid}
platform_minimum_decrement (EUR): {platform_minimum_decrement}
round_number: {round_number}
```
In rounds after round 1: if current_best_bid - platform_minimum_decrement >= minimum_bid, submit that as offer. Otherwise stay silent.

---

## Local SLM Candidates

The six candidate models are listed below. Each stage process (material, manufacturer, logistics and supervisor) loads its own model instance. Agents within a stage share that instance, with each call supplying the agent profile, current state and request.

| Model | Published parameter count | Exact checkpoint |
|---|---|---|
| Qwen3.5-4B | 4B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Granite 4.1-3B | 3B | [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b) |
| Phi-4-mini-instruct | 3.8B | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Ministral 3 3B Instruct | 3B | [mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) |
| Gemma 4 E2B IT | 2.3B effective; 5.1B including embeddings | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| SmolLM3-3B | 3B | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |

Parameter counts follow the local model cards. Gemma's E2B name refers to effective parameters; its count including embeddings is 5.1B.

### Fine-tuning Settings

All six models share the common LoRA settings below. Each model is configured for one continuous three-epoch run per task, with an adapter saved after each epoch. The linear learning-rate schedule spans the full run.

LoRA adapters are attached to attention and feed-forward projection modules. The JSON lists their exact names. Phi uses fused query/key/value and gate/up projections, so its module names differ. Sequence length varies by task, and the optimizer varies by training backend.

```json
{
  "scope": "fine-tuning configuration for all six SLM candidates",
  "common": {
    "method": "LoRA",
    "num_train_epochs": 3,
    "learning_rate": 0.0002,
    "lr_scheduler_type": "linear",
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "effective_batch_size_single_device": 16,
    "seed": 42,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "lora_bias": "none",
    "gradient_checkpointing": "unsloth",
    "save_steps": 250,
    "save_epoch_adapters": true,
    "load_in_4bit": false
  },
  "max_seq_length_by_task": {
    "sealed": 1024,
    "reverse_auction": 1280,
    "supervisor": 2048
  },
  "optimizer_by_backend": {
    "rocm": "adamw_torch",
    "cuda": "adamw_8bit"
  },
  "lora_target_modules": {
    "qwen_granite_ministral_gemma_smollm3": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    "phi": [
      "qkv_proj",
      "o_proj",
      "gate_up_proj",
      "down_proj"
    ]
  }
}
```

The configuration specifies BF16 precision. Training loss is computed on assistant responses, with prompt tokens masked. Each model uses its own chat template and response markers.

Checkpoint revisions and model-specific settings are in [`model_registry.py`](../src/experiments/v5/model_registry.py). The training implementation is in [`finetune_lora.py`](../src/experiments/finetune_lora.py), and task-specific sequence lengths are defined in [`training_data.py`](../src/experiments/v5/training_data.py).

---

## Dynamic Agent State

Implemented for the material and tailoring stages in `src/common/dynamic_state.py`, shared by both bidding modes. Logistics is unchanged: it has no stock or capacity field in the current schema. The rules below are the specification; what is built against it, and what is not yet, is recorded at the end of this section.

Submitting a bid must never change any agent's stock, workload, or capacity, since the agent may not be selected. State changes only after a final accepted verdict is sent for that specific agent.

Distinguish clearly: bid submitted, provisional candidate, final accepted winner, resource reserved, resource consumed or released.

Material: available_to_promise = current_stock - reserved_stock. Reservation happens only on acceptance. Restock is threshold and lead-time gated, not instant on low stock.

Manufacturer: workload updates only on acceptance, reflected in future bids' workload_ratio calculation. Capacity releases on completion or cancellation.

Logistics: capacity reservation and release follow the same acceptance-only rule.

Static profile fields and dynamic runtime state are kept as separate concepts. Static fields load once at agent startup and never change during that process's lifetime. Dynamic state lives only inside that same agent's own process, never shared or written by another agent or the supervisor directly.

Multi-order oversubscription: before final acceptance, the supervisor requests a reservation acknowledgment from each of the three candidate winners. Only after all three acknowledge does the supervisor send accepted verdicts. If any reservation fails, no partial chain is accepted, and the supervisor either evaluates the next feasible chain or marks the order
according to approved retry policy.

### What is implemented

State is seeded once per agent process from the agent CSV files and mutated only in memory, the CSV files are never written to. Two restock triggers are active for material: proactive, when `available_kg` falls to or below 20% of that agent's starting stock for that material, and commitment-based, when an accepted bid's completion time already included `restock_time_days`. One restock is in flight per agent per material at a time; a further trigger does not stack a second. There is no stock ceiling, a restock can push `available_kg` above its starting value. Tailoring occupies `current_workload_units` by the order quantity on acceptance and self-releases once that order's own accepted `estimated_completion_days` has elapsed.

Time is simulated rather than live: orders are processed in `order_created` sequence and "now" advances to each order's own `order_created` as it is processed, carried on the RFQ as `sim_now`. Restock arrival and workload release are evaluated against that walking clock. Because orders are processed strictly in sequence, only one order touches the state at a time and no locking is needed or used.

The reservation acknowledgment above runs on `reserve.request.<stage>.<agent_id>` as a NATS request/reply before any accepted verdict is sent; a refusal releases whatever the chain already held via `reserve.release.<stage>.<agent_id>`, drops that bid, and re-runs the same chain selection over the remaining bids. Every state change is published on `order.dynamic_state` for reporting only.

Enabled by `DYNAMIC_RESOURCES`, off by default, so an unset environment behaves exactly as the static-snapshot deployment always has.

Both bidding modes and both agent kinds have since been run under it, reading live state through the same profile view: the rule-based baseline over 300 orders in each mode, and six SLM architectures over the same 150-order subset in each mode. What the reservation path does under a model is the part that only shows up there, since a rule agent computes its promise from the state the reservation re-checks and cannot fail it, while every model records refusals, overwhelmingly `promised_completion_no_longer_met`.

---

## Evaluation Plan

SLM decisions are compared against the rule-based agent outputs. The rule-based agent is deterministic, so it serves as the reference baseline for comparison.

Planned metrics:

- **JSON validity rate** does the model return parseable JSON in the required format
- **Bid participation accuracy** did the SLM correctly decide whether to publish a bid or stay silent
- **Constraint violation rate** how often does the SLM bid despite violating a constraint
- **Price deviation** how far is the SLM offer from the rule-based calculated amount
- **Inference latency** time taken per agent decision per model
- **Chain selection impact** does the SLM system select the same final chain as the rule-based system

When SLM and rule-based decisions disagree, those cases are flagged and saved with a `debug_reason` field in local logs for manual review. This helps separate genuine SLM reasoning from constraint errors.

---