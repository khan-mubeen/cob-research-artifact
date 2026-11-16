"""v5 stage and chain-selection prompts, rendered from the inputs the v5 prices read

src/common/slm.py dispatches here for any record that carries dataset_version v5. the layout,
the system prompts, the output schema, the auction round rule, the restock wording and the
worded-formula rewrite are v4's, imported from slm.py rather than copied, so the two versions
differ only where v5 differs: USD amounts, the dated material reference price, and the
parametric logistics price with its pickup location, route distance, fuel surcharge, weight
limit and exchange rate. every number a v5 price uses is printed from the same
src/common/v5/pricing.py *_inputs call the price reads. v4-only wording such as base delivery
cost, cost per kg, a EUR budget or cost_eur never appears here
"""

from src.common import slm
from src.common.v5 import pricing

_fmt = slm._fmt


def _auction_rfq_lines(auction):
    if not auction:
        return ""
    return (
        f"\ncurrent_best_bid (USD): {_fmt(auction['current_best_bid'])}"
        f"\nplatform_minimum_decrement (USD): {auction['platform_minimum_decrement']}"
        f"\nround_number: {auction['round_number']}"
        f"\nyour minimum_bid (USD): {_fmt(auction.get('minimum_bid'))}"
    )


def _finish(user, worded):
    """every v5 stage prompt is written in the one validated worded style

    the reverse-auction prompts were already worded, restating each operator in words beside its
    symbol through slm._word_task_block, and v5 harmonizes the sealed prompts onto that same rewrite
    rather than a second phrasing. the worded argument is kept for the shared builder signature and
    no longer selects a style: the live slm_decide_* path never passes it, so a v5 model is served
    exactly the wording it was trained on. v4 prompts do not come through here and are unchanged"""
    return slm._word_task_block(user)


def build_material_prompt(agent, order, context, auction=None, worded=False):
    material_type = order["material_type"]
    v = pricing.material_inputs(agent, order, context)
    restock_line, wait_term = slm._material_restock_wait(agent, material_type)
    user = f"""AGENT PROFILE
agent_id: {agent["agent_id"]}
location: {_fmt(agent.get("location"))}
materials supplied: {_fmt(agent.get("material_types"))}
available stock for requested material (kg): {_fmt(agent.get(f"{material_type}_available_kg"))}
price ratio to reference for requested material: {_fmt(v["price_ratio_to_reference"])}
restock amount (kg): {_fmt(agent.get(f"{material_type}_restock_amount_kg"))}
restock time (days): {_fmt(agent.get(f"{material_type}_restock_time_days"))}{restock_line}
default completion days: {_fmt(agent.get("default_completion_days"))}
defect probability: {_fmt(v["defect_probability"])}
reputation score: {_fmt(agent.get("reputation_score"))}
min profit margin: {_fmt(v["min_profit_margin"])}
initial markup: {_fmt(v["initial_markup"])}

RFQ
order_id: {order["order_id"]}
pricing date: {order["pricing_date"]}
material needed: {material_type}
reference price on the pricing date (USD per kg): {_fmt(v["reference_price_usd_per_kg"])}
required quantity (kg): {v["required_kg"]}
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
6. Calculate, in USD:
   material_cost  = required_kg x price_ratio_to_reference x reference_price_usd_per_kg
   internal_cost  = material_cost x (1 + defect_probability)
   minimum_bid    = internal_cost x (1 + min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost x (1 + initial_markup) x (1 + priority_markup)
   offer_amount   = max(minimum_bid, initial_bid)
7. Return BID with calculated values.{slm._auction_task_rule(auction)}

{slm._output_schema(order["order_id"], "material", agent["agent_id"], agent.get("defect_probability"))}"""
    return slm.MATERIAL_SYSTEM, _finish(user, worded)


def build_manufacturer_prompt(agent, order, context, auction=None, worded=False):
    product_id = order["product_id"]
    v = pricing.manufacturer_inputs(agent, order, context)
    user = f"""AGENT PROFILE
agent_id: {agent["agent_id"]}
location: {_fmt(agent.get("location"))}
products you make: {_fmt(agent.get("product_types"))}
daily capacity (units): {_fmt(agent.get("daily_capacity_units"))}
current workload (units): {_fmt(agent.get("current_workload_units"))}
unit cost for requested product (USD): {_fmt(v["unit_cost_usd"])}
defect probability: {_fmt(v["defect_probability"])}
reputation score: {_fmt(agent.get("reputation_score"))}
min profit margin: {_fmt(v["min_profit_margin"])}
initial markup: {_fmt(v["initial_markup"])}

RFQ
order_id: {order["order_id"]}
product: {product_id}
quantity: {v["quantity"]}
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
7. Calculate, in USD:
   base_cost      = quantity x unit_cost_for_product
   internal_cost  = base_cost x (1 + defect_probability)
   minimum_bid    = internal_cost x (1 + min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost x (1 + initial_markup) x (1 + workload_markup) x (1 + priority_markup)
   offer_amount   = max(minimum_bid, initial_bid)
8. Return BID with calculated values.{slm._auction_task_rule(auction)}

{slm._output_schema(order["order_id"], "tailoring", agent["agent_id"], agent.get("defect_probability"))}"""
    return slm.MANUFACTURER_SYSTEM, _finish(user, worded)


def weight_limit_text(v):
    low, cap, inclusive = v["min_shipment_weight_kg"], v["max_shipment_weight_kg"], v["max_weight_inclusive"]
    parts = []
    if low is not None:
        parts.append(f"at least {low}")
    if cap is not None:
        parts.append(f"at most {cap}" if inclusive else f"below {cap}")
    return " and ".join(parts) if parts else "none"


def oversized_text(v):
    if v["oversized_fee_eur"] is None:
        return "none"
    return f"{v['oversized_fee_eur']} when shipment_weight_kg > {v['oversized_above_kg']}"


def build_logistics_prompt(agent, order, context, auction=None, worded=False):
    v = pricing.logistics_inputs(agent, order, context)
    user = f"""AGENT PROFILE
agent_id: {agent["agent_id"]}
location: {_fmt(agent.get("location"))}
service regions: {_fmt(agent.get("service_regions"))}
fixed charge (EUR per shipment): {_fmt(v["fixed_charge_eur"])}
alpha (EUR per kg): {_fmt(v["alpha_eur_per_kg"])}
beta (EUR per tonne-km): {_fmt(v["beta_eur_per_tonne_km"])}
shipment weight limit (kg): {weight_limit_text(v)}
oversized fee (EUR): {oversized_text(v)}
same-city delivery days: {_fmt(agent.get("default_completion_same_city_days"))}
intercity delivery days: {_fmt(agent.get("default_completion_intercity_days"))}
damage probability: {_fmt(v["damage_probability"])}
reputation score: {_fmt(agent.get("reputation_score"))}

RFQ
order_id: {order["order_id"]}
pricing date: {order["pricing_date"]}
pickup from: {_fmt(v["pickup_from"])}
deliver to: {order["deliver_to"]}
route distance from pickup (km): {_fmt(v["route_distance_km"])}
shipment weight (kg): {v["shipment_weight_kg"]}
fuel surcharge on the pricing date (%): {_fmt(v["fuel_surcharge_percent"])}
EUR to USD on the pricing date: {_fmt(v["eur_to_usd"])}
max allowed damage rate: {order["max_defect_rate"]}
min required reputation: {order["min_reputation_score"]}
days until deadline: {order["days_until_deadline"]}
priority: {order["priority"]}{_auction_rfq_lines(auction)}

TASK
1. Check if deliver_to is in your service_regions. If not, return NO_BID.
2. Check reputation_score >= min_reputation_score. If not, return NO_BID.
3. Check damage_probability <= max_defect_rate. If not, return NO_BID.
4. Check shipment_weight_kg is inside your shipment weight limit. If not, return NO_BID.
5. Calculate route days:
   if deliver_to == your location: route_days = default_completion_same_city_days
   else: route_days = default_completion_intercity_days
6. Check route_days <= days_until_deadline. If not, return NO_BID.
7. Calculate, in EUR:
   freight_eur    = fixed_charge + alpha x shipment_weight_kg + beta x shipment_weight_kg / 1000 x route_distance_km
   with_fuel_eur  = freight_eur x (1 + fuel_surcharge_percent / 100)
   oversized_eur  = the oversized fee if it applies to this shipment, else 0
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   price_eur      = (with_fuel_eur + oversized_eur) x (1 + damage_probability) x (1 + priority_markup)
8. Convert to USD, the currency of every offer:
   offer_amount   = price_eur x eur_to_usd
   minimum_bid    = offer_amount, a market price with no markup on top and no margin below
9. Return BID with calculated values.{slm._auction_task_rule(auction)}

{slm._output_schema(order["order_id"], "logistics", agent["agent_id"], agent.get("damage_probability"))}"""
    return slm.LOGISTICS_SYSTEM, _finish(user, worded)


def build_chain_selection_prompt(order, candidates, remaining_days, context):
    """v4's chain-selection prompt in USD; the rule it teaches is v4's, reading the v5 budget field"""
    lines = "\n".join(
        f"candidate {i}: "
        f"material(cost_usd={c['material']['offer_amount']}, "
        f"days={c['material']['estimated_completion_days']}, "
        f"defect_p={_fmt(c['material']['defect_probability'])}), "
        f"manufacturing(cost_usd={c['tailoring']['offer_amount']}, "
        f"days={c['tailoring']['estimated_completion_days']}, "
        f"defect_p={_fmt(c['tailoring']['defect_probability'])}), "
        f"logistics(cost_usd={c['logistics']['offer_amount']}, "
        f"days={c['logistics']['estimated_completion_days']}, "
        f"damage_p={_fmt(c['logistics']['defect_probability'])}), "
        f"total_cost_usd={round(c['total_cost'], 2)}, "
        f"total_completion_days={round(c['total_days'], 2)}"
        for i, c in enumerate(candidates)
    )
    user = f"""ORDER
order_id: {order["order_id"]}
priority: {order["priority"]}
budget (USD): {pricing.order_budget(order, context)}
days remaining until deadline: {round(remaining_days, 2)}
max acceptable defect or damage probability: {_fmt(order["max_defect_rate"])}

FEASIBLE CHAIN CANDIDATES
{lines}

TASK
1. Every candidate already fits the budget and the remaining time, feasibility is settled.
2. Chain risk is compound across the three stages:
   risk = 1 - (1 - defect_p_material) * (1 - defect_p_manufacturing) * (1 - damage_p_logistics)
3. Pick the single best candidate for this order given its priority:
   urgent   -> fewest total_completion_days, lowest total_cost_usd breaks ties.
   flexible -> lowest total_cost_usd, most days remaining after completion breaks ties.
   normal   -> lowest equal-weighted score of the three ratios
               total_cost_usd / budget,
               total_completion_days / days remaining,
               chain risk / max acceptable defect or damage probability.
4. Return this JSON only:
{{"selected_candidate": <candidate index>, "reason": "<one short sentence>"}}"""
    return slm.CHAIN_SELECTION_SYSTEM, user
