"""
shared eligibility checks and pricing formulas for bids

agent scripts call the three decide_* functions and each service reuses these
helpers instead of copying the checks

the bidding path can change without changing nats messages or supervisor flow

the reverse-auction mode reuses the same eligibility, dynamic and pricing
helpers, it only replaces the offer an eligible agent publishes in later rounds
"""

import math
import os
from typing import Literal, TypedDict


class BidDecision(TypedDict, total=False):
    # total=false keeps bid and no-bid results on the same wire contract
    decision: Literal["BID", "NO_BID"]
    order_id: str
    stage: Literal["material", "tailoring", "logistics"]
    agent_id: str
    offer_amount: float
    estimated_completion_days: float
    defect_probability: float
    debug_reason: str


PRIORITY_MARKUP = {
    "urgent":   0.30,
    "normal":   0.00,
    "flexible": -0.15,
}


def get_workload_markup(workload_ratio):
    if workload_ratio < 0.50:
        return 0.00
    elif workload_ratio < 0.75:
        return 0.05
    elif workload_ratio < 0.90:
        return 0.10
    elif workload_ratio <= 1.00:
        return 0.15
    return None


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _is_number(value):
    # bool is an int subclass but never a valid bid value
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def bid_reply_reason(bid):
    """reject a reply that is not a usable bid message

    shared by every stage collector so sealed windows and auction rounds accept
    exactly the same messages; return None when the reply is usable"""
    if bid.get("decision") != "BID" or "offer_amount" not in bid or "agent_id" not in bid:
        return "missing required bid fields"
    return validate_bid_values(bid)


def validate_bid_values(bid):
    """check values from either bidding path before publishing or pooling a bid

    return None for a valid bid or a short reason that callers handle as no bid"""
    offer = bid.get("offer_amount")
    if not _is_number(offer) or not math.isfinite(offer) or offer <= 0:
        return f"invalid_offer_amount: {offer!r}"

    days = bid.get("estimated_completion_days")
    if not _is_number(days) or not math.isfinite(days) or days < 0:
        return f"invalid_estimated_completion_days: {days!r}"

    defect = bid.get("defect_probability")
    if not _is_number(defect) or not (0 <= defect <= 1):
        return f"invalid_defect_probability: {defect!r}"

    return None


def _no_bid(reason) -> BidDecision:
    return {"decision": "NO_BID", "debug_reason": reason}


# ── dataset dispatch ──────────────────────────────────────────────────────────
# a v5 profile, rfq or order names its dataset in a dataset_version field and a v4 record
# carries none. only a record naming v5 leaves the v4 code below for src/common/v5, and the
# context module is imported only then, so a v4 decision runs exactly the code it always ran

def v5_context(*records):
    """the v5 DatasetContext these records name, or None when none of them names v5

    records naming different versions raise rather than pick one"""
    named = {r["dataset_version"] for r in records if r.get("dataset_version") is not None}
    if not named:
        return None
    from src.dataset.context import DatasetContextError, for_version
    if len(named) > 1:
        raise DatasetContextError(f"mixed context: one decision reads records naming {sorted(named)}")
    context = for_version(named.pop())
    return context if context.version == "v5" else None


# shared dynamic feasibility checks used by rule-based decisions and Guard B
# each returns (reason_or_None, computed_completion_or_None); static eligibility
# remains with the decide functions and agent callbacks

def material_dynamic_check(agent, order):
    material_type = order["material_type"]
    available_kg = agent.get(f"{material_type}_available_kg")
    restock_kg   = agent.get(f"{material_type}_restock_amount_kg")
    restock_days = agent.get(f"{material_type}_restock_time_days")
    # days left on a restock already in flight, written by the dynamic state
    # before each decision. absent for a static profile read straight off the csv
    pending_days = agent.get(f"{material_type}_restock_pending_days")
    required_kg  = order["required_kg"]

    if available_kg >= required_kg:
        completion_days = agent["default_completion_days"]
    elif not _missing(restock_kg) and (available_kg + restock_kg) >= required_kg:
        # a restock already on its way lands at a known eta, so the wait is what
        # is left of it. with nothing pending a fresh restock is ordered on
        # acceptance and the full lead time applies
        wait_days = restock_days if _missing(pending_days) else pending_days
        completion_days = agent["default_completion_days"] + wait_days
    else:
        return "insufficient_stock_and_restock", None

    if completion_days > order["days_until_deadline"]:
        return "completion_exceeds_deadline", completion_days
    return None, completion_days


def manufacturer_dynamic_check(agent, order):
    daily_capacity = agent["daily_capacity_units"]
    if daily_capacity <= 0:
        return "no_daily_capacity", None

    backlog_days = agent["current_workload_units"] / daily_capacity
    production_days = order["quantity"] / daily_capacity
    estimated_completion = backlog_days + production_days

    if estimated_completion > order["days_until_deadline"]:
        return "completion_exceeds_deadline", estimated_completion

    workload_ratio = (agent["current_workload_units"] + order["quantity"]) / (
        daily_capacity * order["days_until_deadline"]
    )
    if get_workload_markup(workload_ratio) is None:
        return "workload_ratio_above_one", estimated_completion
    return None, estimated_completion


def logistics_dynamic_check(agent, order):
    if order["deliver_to"] == agent.get("location"):
        route_days = agent["default_completion_same_city_days"]
    else:
        route_days = agent["default_completion_intercity_days"]

    if route_days > order["days_until_deadline"]:
        return "completion_exceeds_deadline", route_days
    return None, route_days


_STAGE_DYNAMIC_CHECK = {
    "material": material_dynamic_check,
    "tailoring": manufacturer_dynamic_check,
    "manufacturer": manufacturer_dynamic_check,
    "logistics": logistics_dynamic_check,
}


def dynamic_feasibility_reason(stage, agent, order):
    """recompute dynamic feasibility from the agent profile and RFQ

    return None when feasible or the rule reason when infeasible; static
    eligibility and pricing remain outside this check"""
    check = _STAGE_DYNAMIC_CHECK.get(stage)
    if check is None:
        raise ValueError(f"dynamic_feasibility_reason: unknown stage {stage!r}")
    reason, _ = check(agent, order)
    return reason


def dynamic_guard_enabled():
    """return whether Guard B is enabled

    disabled by default to preserve pure-SLM and unguarded production behavior"""
    return os.getenv("SLM_DYNAMIC_GUARD", "0").strip().lower() in ("1", "true", "yes", "on")


def apply_dynamic_guard(stage, agent, order, bid):
    """suppress a dynamically infeasible BID using the shared rule check

    NO_BID results and feasible BIDs pass through unchanged; the guard does not
    recover refusals, alter offers or completion times, or price bids
    return (possibly_replaced_bid, suppression_reason_or_None)"""
    if bid.get("decision") != "BID":
        return bid, None
    reason = dynamic_feasibility_reason(stage, agent, order)
    if reason is not None:
        return _no_bid(f"dynamic_guard:{reason}"), reason
    return bid, None


# shared checks keep ineligible rfqs away from the model

def material_eligibility_reason(agent, order):
    if order["material_type"] not in str(agent.get("material_types", "")).split("|"):
        return "material_not_supplied"
    if agent["defect_probability"] > order["max_defect_rate"]:
        return "defect_probability_too_high"
    if agent["reputation_score"] < order["min_reputation_score"]:
        return "reputation_too_low"
    return None


def manufacturer_eligibility_reason(agent, order):
    if order["product_id"] not in str(agent.get("product_types", "")).split("|"):
        return "product_not_made"
    if agent["defect_probability"] > order["max_defect_rate"]:
        return "defect_probability_too_high"
    if agent["reputation_score"] < order["min_reputation_score"]:
        return "reputation_too_low"
    return None


def logistics_eligibility_reason(agent, order):
    context = v5_context(agent, order)
    if context is not None:
        from src.common.v5 import pricing
        return pricing.logistics_eligibility_reason(agent, order, context)
    if order["deliver_to"] not in str(agent.get("service_regions", "")).split("|"):
        return "region_not_serviced"
    if agent["reputation_score"] < order["min_reputation_score"]:
        return "reputation_too_low"
    if agent["damage_probability"] > order["max_defect_rate"]:
        return "damage_probability_too_high"
    return None


_STAGE_ELIGIBILITY = {
    "material": material_eligibility_reason,
    "tailoring": manufacturer_eligibility_reason,
    "manufacturer": manufacturer_eligibility_reason,
    "logistics": logistics_eligibility_reason,
}


def eligibility_reason(stage, agent, order):
    """static eligibility for one stage, dispatched by stage name

    same contract as the three per-stage functions, for callers that handle
    every stage through one path"""
    check = _STAGE_ELIGIBILITY.get(stage)
    if check is None:
        raise ValueError(f"eligibility_reason: unknown stage {stage!r}")
    return check(agent, order)


# ── price terms ───────────────────────────────────────────────────────────────
# minimum_bid is the floor an agent never goes below, initial_bid is its opening
# offer. sealed bidding publishes max(minimum_bid, initial_bid) once; the reverse
# auction opens with the same value and needs minimum_bid on its own for every
# later round, so the formulas live here once rather than in both paths.
# both values are unrounded, callers round the offer they publish

def material_price_terms(agent, order):
    context = v5_context(agent, order)
    if context is not None:
        from src.common.v5 import pricing
        return pricing.material_price_terms(agent, order, context)
    material_type = order["material_type"]
    priority_markup = PRIORITY_MARKUP.get(order["priority"], 0.0)

    base_cost_per_kg = agent.get(f"{material_type}_base_cost_eur_per_kg")
    required_kg    = order["required_kg"]
    material_cost = required_kg * base_cost_per_kg
    internal_cost = material_cost * (1 + agent["defect_probability"])
    minimum_bid   = internal_cost * (1 + agent["min_profit_margin"])
    initial_bid   = internal_cost * (1 + agent["initial_markup"]) * (1 + priority_markup)
    return minimum_bid, initial_bid


def manufacturer_price_terms(agent, order):
    """call only after manufacturer_dynamic_check passes

    the workload markup is undefined above a workload ratio of 1.00, which that
    check already rejects"""
    context = v5_context(agent, order)
    if context is not None:
        from src.common.v5 import pricing
        return pricing.manufacturer_price_terms(agent, order, context)
    product_id = order["product_id"]
    quantity = order["quantity"]
    priority_markup = PRIORITY_MARKUP.get(order["priority"], 0.0)

    daily_capacity = agent["daily_capacity_units"]
    workload_ratio = (agent["current_workload_units"] + quantity) / (
        daily_capacity * order["days_until_deadline"]
    )
    workload_markup = get_workload_markup(workload_ratio)

    unit_cost = agent.get(f"unit_cost_{product_id}")
    base_production_cost = quantity * unit_cost
    internal_cost = base_production_cost * (1 + agent["defect_probability"])
    minimum_bid   = internal_cost * (1 + agent["min_profit_margin"])
    initial_bid   = internal_cost * (1 + agent["initial_markup"]) \
        * (1 + workload_markup) * (1 + priority_markup)
    return minimum_bid, initial_bid


def logistics_price_terms(agent, order):
    context = v5_context(agent, order)
    if context is not None:
        from src.common.v5 import pricing
        return pricing.logistics_price_terms(agent, order, context)
    priority_markup = PRIORITY_MARKUP.get(order["priority"], 0.0)

    base_logistics_cost = agent["base_delivery_cost"] + (agent["cost_per_kg"] * order["shipment_weight_kg"])
    internal_cost = base_logistics_cost * (1 + agent["damage_probability"])
    minimum_bid   = internal_cost * (1 + agent["min_profit_margin"])
    initial_bid   = internal_cost * (1 + agent["initial_markup"]) * (1 + priority_markup)
    return minimum_bid, initial_bid


_STAGE_PRICE_TERMS = {
    "material": material_price_terms,
    "tailoring": manufacturer_price_terms,
    "manufacturer": manufacturer_price_terms,
    "logistics": logistics_price_terms,
}


def stage_price_terms(stage, agent, order):
    """(minimum_bid, initial_bid) for one stage, dispatched by stage name"""
    terms = _STAGE_PRICE_TERMS.get(stage)
    if terms is None:
        raise ValueError(f"stage_price_terms: unknown stage {stage!r}")
    return terms(agent, order)


def decide_material_bid(agent, order) -> BidDecision:
    ineligible = material_eligibility_reason(agent, order)
    if ineligible is not None:
        return _no_bid(ineligible)

    reason, completion_days = material_dynamic_check(agent, order)
    if reason is not None:
        return _no_bid(reason)

    minimum_bid, initial_bid = material_price_terms(agent, order)
    offer_amount = max(minimum_bid, initial_bid)

    return {
        "order_id": order["order_id"],
        "stage": "material",
        "agent_id": agent["agent_id"],
        "decision": "BID",
        "offer_amount": round(offer_amount, 2),
        "estimated_completion_days": completion_days,
        "defect_probability": agent["defect_probability"],
    }


def decide_manufacturer_bid(agent, order) -> BidDecision:
    ineligible = manufacturer_eligibility_reason(agent, order)
    if ineligible is not None:
        return _no_bid(ineligible)

    reason, estimated_completion = manufacturer_dynamic_check(agent, order)
    if reason is not None:
        return _no_bid(reason)

    minimum_bid, initial_bid = manufacturer_price_terms(agent, order)
    offer_amount = max(minimum_bid, initial_bid)

    return {
        "order_id": order["order_id"],
        "stage": "tailoring",
        "agent_id": agent["agent_id"],
        "decision": "BID",
        "offer_amount": round(offer_amount, 2),
        "estimated_completion_days": round(estimated_completion, 2),
        "defect_probability": agent["defect_probability"],
    }


def decide_logistics_bid(agent, order) -> BidDecision:
    ineligible = logistics_eligibility_reason(agent, order)
    if ineligible is not None:
        return _no_bid(ineligible)

    reason, route_days = logistics_dynamic_check(agent, order)
    if reason is not None:
        return _no_bid(reason)

    minimum_bid, initial_bid = logistics_price_terms(agent, order)
    offer_amount = max(minimum_bid, initial_bid)

    return {
        "order_id": order["order_id"],
        "stage": "logistics",
        "agent_id": agent["agent_id"],
        "decision": "BID",
        "offer_amount": round(offer_amount, 2),
        "estimated_completion_days": route_days,
        "defect_probability": agent["damage_probability"],
    }


# ── chain selection (supervisor) ──────────────────────────────────────────────

CHAIN_WEIGHTS_NORMAL = {"cost": 1 / 3, "days": 1 / 3, "risk": 1 / 3}

_CHAIN_RISK_KEYS = ("material_defect_probability",
                    "manufacturer_defect_probability",
                    "logistics_damage_probability")


class ChainSelection(TypedDict, total=False):
    # total=false keeps selection and no-selection results on one wire contract,
    # matching the BidDecision convention above
    decision: Literal["SELECT", "NO_SELECTION"]
    order_id: str
    strategy: Literal["urgent", "normal", "flexible"]
    material_agent_id: str
    manufacturer_agent_id: str
    logistics_agent_id: str
    total_cost: float
    total_completion_days: float
    chain_risk: float
    score: float
    debug_reason: str


def _no_selection(order_id, reason) -> ChainSelection:
    return {"order_id": order_id, "decision": "NO_SELECTION", "debug_reason": reason}


def chain_compound_risk(chain) -> float:
    """probability that at least one stage fails: 1 - prod(1 - p)

    this replaces a worst-single-stage max(), which cannot tell a chain with
    three mediocre stages from one with two clean stages and one mediocre"""
    survives = 1.0
    for key in _CHAIN_RISK_KEYS:
        survives *= (1.0 - float(chain[key]))
    return 1.0 - survives


def chain_from_bids(candidate):
    """runtime candidate (nested stage bids) -> the flat shape the rule uses

    the supervisor assembles candidates as {"material": bid, "tailoring": bid,
    "logistics": bid, ...}. this is the single conversion point, so the live
    path and the training-data generator cannot drift apart in how a chain is
    read. note the runtime calls the manufacturing stage "tailoring"."""
    m, t, l = candidate["material"], candidate["tailoring"], candidate["logistics"]
    return {
        "material_agent_id": m["agent_id"],
        "manufacturer_agent_id": t["agent_id"],
        "logistics_agent_id": l["agent_id"],
        "material_offer": m["offer_amount"],
        "manufacturer_offer": t["offer_amount"],
        "logistics_offer": l["offer_amount"],
        "material_completion_days": m["estimated_completion_days"],
        "manufacturer_completion_days": t["estimated_completion_days"],
        "logistics_completion_days": l["estimated_completion_days"],
        "material_defect_probability": m["defect_probability"],
        "manufacturer_defect_probability": t["defect_probability"],
        "logistics_damage_probability": l["defect_probability"],
    }


def chain_totals(chain):
    """(total cost, total completion days, compound risk) for one candidate

    cost and days are sums across the three stages, risk is compounded"""
    cost = (float(chain["material_offer"])
            + float(chain["manufacturer_offer"])
            + float(chain["logistics_offer"]))
    days = (float(chain["material_completion_days"])
            + float(chain["manufacturer_completion_days"])
            + float(chain["logistics_completion_days"]))
    return cost, days, chain_compound_risk(chain)


def normal_chain_score(chain, order) -> float:
    """equal-weight blend, each term relative to the order's OWN limit

    budget-relative rather than min-max across the candidate set, so a chain's
    score does not move when an unrelated candidate appears or disappears, and
    scores are comparable between orders"""
    cost, days, risk = chain_totals(chain)
    budget = _order_budget(order)
    deadline_days = float(order["days_until_deadline"])
    defect_cap = float(order["max_defect_rate"])
    w = CHAIN_WEIGHTS_NORMAL
    return (w["cost"] * (cost / budget if budget > 0 else 1.0)
            + w["days"] * (days / deadline_days if deadline_days > 0 else 1.0)
            + w["risk"] * (risk / defect_cap if defect_cap > 0 else 1.0))


def _order_budget(order):
    """the order's budget; a v5 order stores it under the v5 budget field"""
    context = v5_context(order)
    if context is not None:
        from src.common.v5 import pricing
        return pricing.order_budget(order, context)
    return float(order["budget"])


def chain_is_feasible(chain, order) -> bool:
    cost, days, _ = chain_totals(chain)
    return (cost <= _order_budget(order)
            and days <= float(order["days_until_deadline"]))


def chain_strategy(order):
    """the priority rule this order selects under; anything but urgent or flexible is normal"""
    priority = order.get("priority", "normal")
    return priority if priority in ("urgent", "flexible") else "normal"


def chain_priority_key(chain, order):
    """the full ordering key for one chain, lowest wins

    cost and days are compared at the two decimals the corpus stores and the prompt prints, so two
    chains that read as identical to a human are identical here too. the agent-id tuple is the last
    term and is reached only when every earlier term is exactly equal: it settles a true tie without
    touching the priority objective, and makes the winner independent of the order the candidates
    happen to arrive or be shuffled in"""
    cost, days, _ = chain_totals(chain)
    cost, days = round(cost, 2), round(days, 2)
    agents = (chain["material_agent_id"], chain["manufacturer_agent_id"], chain["logistics_agent_id"])
    strategy = chain_strategy(order)
    if strategy == "urgent":
        return (days, cost, agents)
    if strategy == "flexible":
        # fewer total days orders exactly as more deadline slack does, the deadline is fixed per order
        return (cost, days, agents)
    return (normal_chain_score(chain, order), cost, days, agents)


def decide_chain_selection(chains, order) -> ChainSelection:
    """pick one chain from the candidate list according to the order's priority

    urgent   - fewest total completion days, total cost breaks ties
    normal   - equal-weight budget-relative blend of cost, days and compound risk
    flexible - lowest total cost, fewest total days breaks ties

    an exact tie on all of those falls through to the agent-id tuple, see
    chain_priority_key, so the same candidate set always yields the same winner

    reputation is deliberately not a term: min_rep is carried on candidate
    records but stays unused until that is decided separately"""
    order_id = order["order_id"]
    if not chains:
        return _no_selection(order_id, "no_candidate_chains")

    feasible = [c for c in chains if chain_is_feasible(c, order)]
    if not feasible:
        return _no_selection(order_id, "no_chain_within_budget_and_deadline")

    strategy = chain_strategy(order)
    best = min(feasible, key=lambda c: chain_priority_key(c, order))
    cost, days, risk = chain_totals(best)
    return {
        "order_id": order_id,
        "decision": "SELECT",
        "strategy": strategy,
        "material_agent_id": best["material_agent_id"],
        "manufacturer_agent_id": best["manufacturer_agent_id"],
        "logistics_agent_id": best["logistics_agent_id"],
        "total_cost": round(cost, 2),
        "total_completion_days": round(days, 2),
        "chain_risk": round(risk, 6),
        "score": round(normal_chain_score(best, order), 6),
    }
