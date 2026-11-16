"""
model-backed bidding and chain selection

each request gives the model one agent profile and the current rfq, then returns
the same bid shape as the rule-based path

USE_SLM enables this path and loads one llama.cpp model per process
calls share that model, run one at a time, and keep blocking inference off the
nats event loop
"""

import asyncio
import json
import math
import os
import re
import time

from src.common.bidding_formulas import BidDecision, v5_context


DEFAULT_MODEL_PATH = "models/Qwen3.5-4B-Q8_0.gguf"


def slm_enabled():
    return os.getenv("USE_SLM", "").strip().lower() in ("1", "true", "yes")


def load_slm(service_name):
    """load one model for the current process

    return None when USE_SLM is off so callers use rule-based bidding
    raise during startup if USE_SLM is on but the model cannot load"""
    if not slm_enabled():
        return None

    model_path = os.getenv("SLM_MODEL_PATH", "").strip() or DEFAULT_MODEL_PATH
    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"USE_SLM is set but no model file exists at {model_path}, "
            "download the GGUF there or point SLM_MODEL_PATH at it"
        )

    # imported lazily so rule-based deployments do not need llama-cpp-python
    from llama_cpp import Llama

    n_threads = int(os.getenv("SLM_N_THREADS", "0"))
    llama = Llama(
        model_path=model_path,
        n_ctx=int(os.getenv("SLM_N_CTX", "4096")),
        n_threads=n_threads if n_threads > 0 else None,
        # 0 keeps the CPU-only behavior; -1 offloads every layer when a CUDA build is installed
        n_gpu_layers=int(os.getenv("SLM_N_GPU_LAYERS", "0")),
        verbose=False,
    )
    print(f"[{service_name}] SLM loaded: {model_path}")
    return SLM(llama)


# synthetic warmup ids cannot collide with agent ids such as M01 or order ids
# such as ORD-000001
_WARMUP_FAKE_AGENT = {
    "agent_id": "WARMUP-FAKE-AGENT",
    "location": "Nowhere",
    "material_types": "cotton",
    "cotton_available_kg": 1.0,
    "cotton_base_cost_eur_per_kg": 1.0,
    "cotton_restock_amount_kg": 1.0,
    "cotton_restock_time_days": 1,
    "default_completion_days": 1,
    "defect_probability": 0.0,
    "reputation_score": 1.0,
    "min_profit_margin": 0.0,
    "initial_markup": 0.0,
}

_WARMUP_FAKE_RFQ = {
    "order_id": "WARMUP-FAKE-ORDER-000",
    "material_type": "cotton",
    "required_kg": 1.0,
    "max_defect_rate": 1.0,
    "min_reputation_score": 0.0,
    "days_until_deadline": 1,
    "priority": "normal",
}


async def warmup_slm(slm, service_name):
    """warm the loaded model before subscriptions begin

    the first inference pays one-time CUDA and kernel setup costs; the synthetic
    result is discarded before the process connects to nats"""
    if slm is None:
        return
    system_prompt, user_prompt = build_material_prompt(_WARMUP_FAKE_AGENT, _WARMUP_FAKE_RFQ)
    started = time.perf_counter()
    await slm.complete(system_prompt, user_prompt)
    elapsed = time.perf_counter() - started
    print(f"[{service_name}] SLM warmup completed in {elapsed:.1f}s")


class SLM:
    def __init__(self, llama):
        self._llama = llama
        # the shared llama.cpp context handles one request at a time
        self._lock = asyncio.Lock()

    async def complete(self, system_prompt, user_prompt):
        async with self._lock:
            return await asyncio.to_thread(self._complete_sync, system_prompt, user_prompt)

    def _complete_sync(self, system_prompt, user_prompt):
        out = self._llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(os.getenv("SLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("SLM_MAX_TOKENS", "512")),
        )
        return out["choices"][0]["message"]["content"]


def _fmt(value):
    # an empty csv field means the agent does not handle that item
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "null"
    return value


def _output_schema(order_id, stage, agent_id, defect_probability):
    return (
        "Return this JSON only:\n"
        "{\n"
        f'  "order_id": "{order_id}",\n'
        f'  "stage": "{stage}",\n'
        f'  "agent_id": "{agent_id}",\n'
        '  "decision": "BID or NO_BID",\n'
        '  "offer_amount": null,\n'
        '  "estimated_completion_days": null,\n'
        f'  "defect_probability": {_fmt(defect_probability)}\n'
        "}"
    )


# ── reverse auction prompt additions ──────────────────────────────────────────
# a reverse-auction prompt is the sealed prompt plus the three documented extra
# fields and the one round rule. both are produced from the same builders with
# auction set, so the two modes cannot drift apart. auction=None returns the
# sealed text unchanged, character for character

AUCTION_ROUND_RULE = (
    "In rounds after round 1: if current_best_bid - platform_minimum_decrement "
    ">= minimum_bid, submit that as offer. Otherwise stay silent."
)

# every operator restated in words next to the symbol, tested as a possible cause
# of the at_floor failures. only the TASK block is rewritten, the profile, rfq and
# output schema are left alone
_OPERATOR_WORDS = {
    "<=": "less than or equal to (<=)",
    ">=": "greater than or equal to (>=)",
    "<": "less than (<)",
    ">": "greater than (>)",
    "x": "multiplied by (x)",
    "+": "plus (+)",
    "-": "minus (-)",
    "/": "divided by (/)",
    "max(": "the larger of the two values max(",
}

# longer operators first so <= never matches as a bare <; single characters only
# match with a space either side, which keeps negative numbers and identifiers safe
_OPERATOR_RE = re.compile(r"max\(|(?<= )(?:<=|>=|<|>|x|\+|-|/)(?= )")


def word_formulas(text):
    """restate each operator in words beside the symbol, in one pass"""
    return _OPERATOR_RE.sub(lambda m: _OPERATOR_WORDS[m.group(0)], text)


def _word_task_block(user_prompt):
    """apply the wording to the TASK block only

    the agent profile and rfq carry values, and the output schema is parsed, so
    neither is touched"""
    start = user_prompt.index("\nTASK\n")
    end = user_prompt.index("\n\nReturn this JSON only:")
    return user_prompt[:start] + word_formulas(user_prompt[start:end]) + user_prompt[end:]


def _auction_rfq_lines(auction):
    if not auction:
        return ""
    # minimum_bid is given rather than derived: the round rule is a comparison
    # against this floor, and deriving it was the step models could not do
    return (
        f"\ncurrent_best_bid (EUR): {_fmt(auction['current_best_bid'])}"
        f"\nplatform_minimum_decrement (EUR): {auction['platform_minimum_decrement']}"
        f"\nround_number: {auction['round_number']}"
        f"\nyour minimum_bid (EUR): {_fmt(auction.get('minimum_bid'))}"
    )


def _auction_task_rule(auction):
    if not auction:
        return ""
    return f"\n{AUCTION_ROUND_RULE}"


MATERIAL_SYSTEM = (
    "You are a material provider agent in a multi-agent supply chain bidding system.\n"
    "Decide whether to bid on the incoming material request.\n"
    "Check all constraints carefully. Return valid JSON only, nothing outside the JSON."
)

MANUFACTURER_SYSTEM = (
    "You are a manufacturer agent in a multi-agent garment supply chain bidding system.\n"
    "Decide whether to bid on the manufacturing request.\n"
    "Check all constraints carefully. Return valid JSON only, nothing outside the JSON."
)

LOGISTICS_SYSTEM = (
    "You are a logistics agent in a multi-agent supply chain bidding system.\n"
    "Decide whether to bid on the delivery request.\n"
    "Check all constraints carefully. Return valid JSON only, nothing outside the JSON."
)


def _material_restock_wait(agent, material_type):
    """the stock line and the wait term the agent should apply right now

    with a restock already in flight the wait is what is left of it, not a fresh
    lead time. nothing is added when nothing is pending, so a prompt built from a
    static profile stays byte-identical to the frozen protocol and every earlier
    slm result stays comparable. return (extra_profile_line, wait_term)"""
    pending_days = agent.get(f"{material_type}_restock_pending_days")
    if pending_days is None:
        return "", "restock_time_days"
    return (f"\nrestock already in flight, days until it arrives: "
            f"{round(float(pending_days), 2)}"), "days until it arrives"


def build_material_prompt(agent, order, auction=None, worded=False):
    context = v5_context(agent, order)
    if context is not None:
        from src.common.v5 import prompts
        return prompts.build_material_prompt(agent, order, context, auction, worded)
    material_type = order["material_type"]
    restock_line, wait_term = _material_restock_wait(agent, material_type)
    user = f"""AGENT PROFILE
agent_id: {agent["agent_id"]}
location: {_fmt(agent.get("location"))}
materials supplied: {_fmt(agent.get("material_types"))}
available stock for requested material (kg): {_fmt(agent.get(f"{material_type}_available_kg"))}
base cost per kg (EUR): {_fmt(agent.get(f"{material_type}_base_cost_eur_per_kg"))}
restock amount (kg): {_fmt(agent.get(f"{material_type}_restock_amount_kg"))}
restock time (days): {_fmt(agent.get(f"{material_type}_restock_time_days"))}{restock_line}
default completion days: {_fmt(agent.get("default_completion_days"))}
defect probability: {_fmt(agent.get("defect_probability"))}
reputation score: {_fmt(agent.get("reputation_score"))}
min profit margin: {_fmt(agent.get("min_profit_margin"))}
initial markup: {_fmt(agent.get("initial_markup"))}

RFQ
order_id: {order["order_id"]}
material needed: {material_type}
required quantity (kg): {order["required_kg"]}
max allowed defect rate: {order["max_defect_rate"]}
min required reputation: {order["min_reputation_score"]}
days until deadline: {order["days_until_deadline"]}
priority: {order["priority"]}{_auction_rfq_lines(auction)}

TASK
1. Check if you supply this material. If not, return NO_BID.
2. Check defect_probability <= max_defect_rate. If not, return NO_BID.
3. Check reputation_score >= min_reputation_score. If not, return NO_BID.
4. Check stock:
   if available_kg >= required_kg: completion = default_completion_days
   else if available_kg + restock_amount_kg >= required_kg: completion = default_completion_days + {wait_term}
   else: return NO_BID
5. Check completion <= days_until_deadline. If not, return NO_BID.
6. Calculate:
   material_cost  = required_kg x base_cost_per_kg
   internal_cost  = material_cost x (1 + defect_probability)
   minimum_bid    = internal_cost x (1 + min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost x (1 + initial_markup) x (1 + priority_markup)
   offer_amount   = max(minimum_bid, initial_bid)
7. Return BID with calculated values.{_auction_task_rule(auction)}

{_output_schema(order["order_id"], "material", agent["agent_id"], agent.get("defect_probability"))}"""
    if worded:
        user = _word_task_block(user)
    return MATERIAL_SYSTEM, user


def build_manufacturer_prompt(agent, order, auction=None, worded=False):
    context = v5_context(agent, order)
    if context is not None:
        from src.common.v5 import prompts
        return prompts.build_manufacturer_prompt(agent, order, context, auction, worded)
    product_id = order["product_id"]
    user = f"""AGENT PROFILE
agent_id: {agent["agent_id"]}
location: {_fmt(agent.get("location"))}
products you make: {_fmt(agent.get("product_types"))}
daily capacity (units): {_fmt(agent.get("daily_capacity_units"))}
current workload (units): {_fmt(agent.get("current_workload_units"))}
unit cost for requested product (EUR): {_fmt(agent.get(f"unit_cost_{product_id}"))}
defect probability: {_fmt(agent.get("defect_probability"))}
reputation score: {_fmt(agent.get("reputation_score"))}
min profit margin: {_fmt(agent.get("min_profit_margin"))}
initial markup: {_fmt(agent.get("initial_markup"))}

RFQ
order_id: {order["order_id"]}
product: {product_id}
quantity: {order["quantity"]}
max allowed defect rate: {order["max_defect_rate"]}
min required reputation: {order["min_reputation_score"]}
days until deadline: {order["days_until_deadline"]}
priority: {order["priority"]}{_auction_rfq_lines(auction)}

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
8. Return BID with calculated values.{_auction_task_rule(auction)}

{_output_schema(order["order_id"], "tailoring", agent["agent_id"], agent.get("defect_probability"))}"""
    if worded:
        user = _word_task_block(user)
    return MANUFACTURER_SYSTEM, user


def build_logistics_prompt(agent, order, auction=None, worded=False):
    context = v5_context(agent, order)
    if context is not None:
        from src.common.v5 import prompts
        return prompts.build_logistics_prompt(agent, order, context, auction, worded)
    user = f"""AGENT PROFILE
agent_id: {agent["agent_id"]}
location: {_fmt(agent.get("location"))}
service regions: {_fmt(agent.get("service_regions"))}
base delivery cost (EUR): {_fmt(agent.get("base_delivery_cost"))}
cost per kg (EUR): {_fmt(agent.get("cost_per_kg"))}
same-city delivery days: {_fmt(agent.get("default_completion_same_city_days"))}
intercity delivery days: {_fmt(agent.get("default_completion_intercity_days"))}
damage probability: {_fmt(agent.get("damage_probability"))}
reputation score: {_fmt(agent.get("reputation_score"))}
min profit margin: {_fmt(agent.get("min_profit_margin"))}
initial markup: {_fmt(agent.get("initial_markup"))}

RFQ
order_id: {order["order_id"]}
deliver to: {order["deliver_to"]}
shipment weight (kg): {order["shipment_weight_kg"]}
max allowed damage rate: {order["max_defect_rate"]}
min required reputation: {order["min_reputation_score"]}
days until deadline: {order["days_until_deadline"]}
priority: {order["priority"]}{_auction_rfq_lines(auction)}

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
7. Return BID with calculated values.{_auction_task_rule(auction)}

{_output_schema(order["order_id"], "logistics", agent["agent_id"], agent.get("damage_probability"))}"""
    if worded:
        user = _word_task_block(user)
    return LOGISTICS_SYSTEM, user


def _extract_json(raw):
    candidates = [raw]
    if isinstance(raw, str) and "{" in raw and "}" in raw:
        # small models sometimes wrap json in fences or prose
        candidates.append(raw[raw.find("{"): raw.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_decision(raw, order_id, stage, agent_id, defect_probability) -> BidDecision:
    parsed = _extract_json(raw)

    if not isinstance(parsed, dict):
        return {"decision": "NO_BID", "debug_reason": f"slm_invalid_json: {str(raw)[:200]!r}"}

    if parsed.get("decision") != "BID":
        return {"decision": "NO_BID", "debug_reason": "slm_no_bid"}

    try:
        offer_amount = float(parsed["offer_amount"])
        completion_days = float(parsed["estimated_completion_days"])
    except (KeyError, TypeError, ValueError):
        return {"decision": "NO_BID", "debug_reason": f"slm_bid_missing_numeric_fields: {str(raw)[:200]!r}"}

    # keep identity fields from trusted inputs, not model output
    return {
        "order_id": order_id,
        "stage": stage,
        "agent_id": agent_id,
        "decision": "BID",
        "offer_amount": round(offer_amount, 2),
        "estimated_completion_days": completion_days,
        "defect_probability": defect_probability,
    }


# auction stays None for sealed bidding, callers in reverse-auction mode pass the
# three extra fields for the current round

async def slm_decide_material_bid(slm, agent, order, auction=None) -> BidDecision:
    system_prompt, user_prompt = build_material_prompt(agent, order, auction)
    raw = await slm.complete(system_prompt, user_prompt)
    return _parse_decision(raw, order["order_id"], "material",
                           agent["agent_id"], agent["defect_probability"])


async def slm_decide_manufacturer_bid(slm, agent, order, auction=None) -> BidDecision:
    system_prompt, user_prompt = build_manufacturer_prompt(agent, order, auction)
    raw = await slm.complete(system_prompt, user_prompt)
    return _parse_decision(raw, order["order_id"], "tailoring",
                           agent["agent_id"], agent["defect_probability"])


async def slm_decide_logistics_bid(slm, agent, order, auction=None) -> BidDecision:
    system_prompt, user_prompt = build_logistics_prompt(agent, order, auction)
    raw = await slm.complete(system_prompt, user_prompt)
    return _parse_decision(raw, order["order_id"], "logistics",
                           agent["agent_id"], agent["damage_probability"])


CHAIN_SELECTION_SYSTEM = (
    "You are the supervisor of a multi-agent supply chain bidding system.\n"
    "Select the single best chain of winning bids for the order.\n"
    "Every candidate listed has already passed the budget and deadline checks.\n"
    "Return valid JSON only, nothing outside the JSON."
)


def build_chain_selection_prompt(order, candidates, remaining_days):
    """the ONE supervisor chain-selection prompt, used by both the live path and
    the training-data generator.

    it exposes each stage's cost, days and failure probability rather than a
    single worst-stage figure, because decide_chain_selection() labels on
    COMPOUND risk across the three stages and a max cannot express that -- a
    model shown only the max could never reproduce the label it is trained on.
    reputation is not shown because the rule does not use it (min_rep stays
    carried but unused, pending a separate decision).

    candidates are the runtime shape: {"material": bid, "tailoring": bid,
    "logistics": bid, "total_cost": float, "total_days": float}."""
    context = v5_context(order)
    if context is not None:
        from src.common.v5 import prompts
        return prompts.build_chain_selection_prompt(order, candidates, remaining_days, context)
    lines = "\n".join(
        f"candidate {i}: "
        f"material(cost_eur={c['material']['offer_amount']}, "
        f"days={c['material']['estimated_completion_days']}, "
        f"defect_p={_fmt(c['material']['defect_probability'])}), "
        f"manufacturing(cost_eur={c['tailoring']['offer_amount']}, "
        f"days={c['tailoring']['estimated_completion_days']}, "
        f"defect_p={_fmt(c['tailoring']['defect_probability'])}), "
        f"logistics(cost_eur={c['logistics']['offer_amount']}, "
        f"days={c['logistics']['estimated_completion_days']}, "
        f"damage_p={_fmt(c['logistics']['defect_probability'])}), "
        f"total_cost_eur={round(c['total_cost'], 2)}, "
        f"total_completion_days={round(c['total_days'], 2)}"
        for i, c in enumerate(candidates)
    )
    user = f"""ORDER
order_id: {order["order_id"]}
priority: {order["priority"]}
budget (EUR): {order["budget"]}
days remaining until deadline: {round(remaining_days, 2)}
max acceptable defect or damage probability: {_fmt(order["max_defect_rate"])}

FEASIBLE CHAIN CANDIDATES
{lines}

TASK
1. Every candidate already fits the budget and the remaining time, feasibility is settled.
2. Chain risk is compound across the three stages:
   risk = 1 - (1 - defect_p_material) * (1 - defect_p_manufacturing) * (1 - damage_p_logistics)
3. Pick the single best candidate for this order given its priority:
   urgent   -> fewest total_completion_days, lowest total_cost_eur breaks ties.
   flexible -> lowest total_cost_eur, most days remaining after completion breaks ties.
   normal   -> lowest equal-weighted score of the three ratios
               total_cost_eur / budget,
               total_completion_days / days remaining,
               chain risk / max acceptable defect or damage probability.
4. Return this JSON only:
{{"selected_candidate": <candidate index>, "reason": "<one short sentence>"}}"""
    return CHAIN_SELECTION_SYSTEM, user



async def slm_select_chain(slm, order, candidates, remaining_days):
    """choose from chains that already passed the cost and time checks

    use None as the index when the model answer cannot be used so the caller can fall back
    keep the debug reason in local logs and never send it to nats"""
    system_prompt, user_prompt = build_chain_selection_prompt(order, candidates, remaining_days)
    raw = await slm.complete(system_prompt, user_prompt)

    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return None, f"slm_invalid_selection_json: {str(raw)[:200]!r}"

    idx = parsed.get("selected_candidate")
    if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < len(candidates)):
        return None, f"slm_selection_index_out_of_range: {idx!r}"

    return idx, str(parsed.get("reason", ""))
