"""every eligible chain for one v5 order, and the bounded list its supervisor prompt shows

what is correct and what the model is shown are two different questions, and they are kept apart
here:

  enumerate   every material x manufacturer x logistics combination that survives stage eligibility,
              each chain's logistics priced from that chain's own manufacturer origin. no widening
              and no shortlist, so the label rule reads the whole set and the saved answer is the
              best chain for the order rather than the best of a sample
  present     the bounded list the prompt prints. the true winner is placed in it first and
              unconditionally, the remaining places go to the cheapest, fastest and lowest-risk
              chains and then a seeded random sample, all of them distractors now, and the list is
              shuffled so no position carries a signal

the cap stays MAX_CANDIDATES from the v4 generator, which the supervisor window is sized for
(reports/v5/phase6_token_lengths.json). the presented list is a subset of the enumerated set that
always contains its winner, so the rule applied to the printed list agrees with the rule applied to
everything
"""

from src.common.bidding_formulas import chain_from_bids, chain_totals
from src.common.reverse_auction import SEALED_DECIDERS
from src.experiments.generate_supervisor_data import MAX_CANDIDATES, _bid

ANCHOR_NAMES = ("cheapest", "fastest", "lowest_risk")


def stage_bids(stage, agents, rfq, time_key, risk_key):
    """the rule's BIDs in the diagnostics bid shape, cheapest first, roster order on a tie"""
    bids = []
    for agent in agents:
        decision = SEALED_DECIDERS[stage](agent, rfq)
        if decision["decision"] == "BID":
            bids.append({"agent_id": agent["agent_id"], "offer": decision["offer_amount"],
                         time_key: decision["estimated_completion_days"], risk_key: decision["defect_probability"],
                         "reputation": agent["reputation_score"]})
    return sorted(bids, key=lambda b: b["offer"])


def candidate(material, manufacturer, logistics):
    """one chain in the runtime shape the prompt builder and the rule both read"""
    chain = {
        "material": _bid(material["agent_id"], material["offer"], material["completion"], material["defect"]),
        "tailoring": _bid(manufacturer["agent_id"], manufacturer["offer"], manufacturer["completion"],
                          manufacturer["defect"]),
        "logistics": _bid(logistics["agent_id"], logistics["offer"], logistics["delivery"], logistics["damage"]),
    }
    cost, days, _ = chain_totals(chain_from_bids(chain))
    chain["total_cost"] = round(cost, 2)
    chain["total_days"] = round(days, 2)
    return chain


def agents_of(chain):
    return (chain["material"]["agent_id"], chain["tailoring"]["agent_id"], chain["logistics"]["agent_id"])


def enumerate_chains(order, corpus, market, rfqs):
    """(chains, counts) for one order: every eligible combination, logistics priced per origin

    a logistics price depends on where the goods leave from, so the logistics stage is bid once per
    distinct manufacturer origin and reused for every manufacturer in that city. a chain never
    carries a logistics quote priced from another city"""
    material = stage_bids("material", corpus["material"], rfqs["material"], "completion", "defect")
    manufacturer = stage_bids("tailoring", corpus["tailoring"], rfqs["tailoring"], "completion", "defect")
    counts = {"material_bids": len(material), "manufacturer_bids": len(manufacturer), "origins": 0,
              "logistics_bids_by_origin": {}, "combinations": 0}
    if not material or not manufacturer:
        return [], counts

    chains = []
    by_origin = {}
    for t in manufacturer:
        origin = corpus["manufacturer_location"][t["agent_id"]]
        if origin not in by_origin:
            rfq = {**rfqs["logistics"], "pickup_from": origin,
                   "route_distance_km": market.distance_km(origin, order["deliver_to"])}
            by_origin[origin] = stage_bids("logistics", corpus["logistics"], rfq, "delivery", "damage")
        for m in material:
            chains.extend(candidate(m, t, l) for l in by_origin[origin])
    counts["origins"] = len(by_origin)
    counts["logistics_bids_by_origin"] = {k: len(v) for k, v in by_origin.items()}
    counts["combinations"] = len(chains)
    return chains, counts


def present(feasible, winner_agents, rng, cap=MAX_CANDIDATES):
    """the bounded list the prompt shows, with the true winner always in it

    the anchors are the cheapest, fastest and lowest-risk chains of the whole feasible set. they are
    distractors here, not the mechanism that keeps the right answer in the list: that is the winner
    being placed first, before anything else is chosen"""
    winner = next(c for c in feasible if agents_of(c) == tuple(winner_agents))
    picked = [winner]
    if len(feasible) <= cap:
        picked.extend(c for c in feasible if c is not winner)
    else:
        totals = {id(c): chain_totals(chain_from_bids(c)) for c in feasible}
        for term in range(len(ANCHOR_NAMES)):
            anchor = min(feasible, key=lambda c: (totals[id(c)][term], agents_of(c)))
            if not any(anchor is p for p in picked):
                picked.append(anchor)
        rest = [c for c in feasible if not any(c is p for p in picked)]
        picked.extend(rng.sample(rest, min(cap - len(picked), len(rest))))
    rng.shuffle(picked)          # no positional bias toward candidate 0
    return picked
