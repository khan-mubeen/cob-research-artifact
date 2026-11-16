"""
one plain sentence for an order that did not complete

this is a translation, not a diagnosis. every sentence restates a reason the
supervisor already logged, and the agent counts come from the declines the agents
themselves published on the bid-activity feed. nothing here infers a cause the
system does not already know: an unrecognised reason code is passed through as
itself rather than guessed at
"""

# the codes supervisor.log_chain writes, in the words an operator would use
REASONS = {
    "chain_cost_exceeds_budget":
        "the cheapest chain the bids allowed still cost more than the budget",
    "chain_time_exceeds_remaining_time":
        "the fastest chain the bids allowed still finished after the deadline",
    "no_feasible_chain":
        "no combination of the bids fit both the budget and the deadline",
    "timeout":
        "no outcome arrived before the timeout",
}

# the same, for the codes that name a stage
STAGE_REASONS = {
    "no_bids": "no {stage} agent bid",
    "no_remaining_bids": "every {stage} bid was dropped after its agent could not "
                         "honour the reservation",
}

STAGE_WORDS = {"material": "material", "manufacturer": "tailoring",
               "tailoring": "tailoring", "logistics": "logistics"}

# reasons agents give for declining, as read out in the summary
DECLINE_WORDS = {
    "material_not_supplied": "do not carry the material",
    "product_not_made": "do not make the product",
    "region_not_serviced": "do not deliver to that city",
    "defect_probability_too_high": "defect rate above the limit",
    "damage_probability_too_high": "damage rate above the limit",
    "reputation_too_low": "reputation below the minimum",
    "insufficient_stock_and_restock": "not enough stock even with a restock",
    "completion_exceeds_deadline": "could not finish before the deadline",
    "workload_ratio_above_one": "already fully booked",
    "no_daily_capacity": "no daily capacity",
    "slm_no_bid": "the model declined without giving a rule reason",
    "withdrawn_in_earlier_round": "had already dropped out of the auction",
}


def _stage_of(reason):
    """(prefix, stage) for the reason codes that name a stage, else (None, None)"""
    for prefix in STAGE_REASONS:
        if reason.startswith(f"{prefix}_at_") and reason.endswith("_stage"):
            stage = reason[len(prefix) + 4:-6]
            return prefix, STAGE_WORDS.get(stage, stage)
    return None, None


def _declines(counts):
    """the decline reasons for one stage, commonest first"""
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{n} {DECLINE_WORDS.get(reason, reason)}" for reason, n in ordered)


def summarize(outcome, declines=None):
    """a one-line summary of a non-completed outcome, or None for a completed one

    declines is {stage: {reason: count}} as the agents reported them, used only
    for the reason codes that are about a stage having no bids"""
    if outcome.get("status") == "completed":
        return None
    reason = outcome.get("reason")
    if not reason:
        return f"ended as {outcome.get('status', 'unknown')} without a logged reason"

    prefix, stage = _stage_of(reason)
    if prefix:
        sentence = STAGE_REASONS[prefix].format(stage=stage)
        counts = (declines or {}).get(stage, {})
        if counts:
            sentence += f": {sum(counts.values())} agents declined, {_declines(counts)}"
        return sentence

    return REASONS.get(reason, reason)
