"""v5 stage prices, v5 logistics eligibility and the budget the chain rule reads

bidding_formulas.py dispatches here for any record that carries dataset_version v5.
src/common/v5/prompts.py renders its prompts from the same *_inputs functions the prices
below read, so a v5 prompt can never show a number its label was not computed from. nothing
here reads a file or a clock: every dated value rides on the rfq, resolved once per order on
its pricing date by src/experiments/v5/rfqs.py

  material      required_kg x price_ratio_to_reference x reference USD/kg on the pricing date,
                then v4's defect, margin, markup and priority terms, unchanged
  manufacturer  v4's formula on the USD unit cost, which v5 relabels at one fixed ECB rate
  logistics     ((F + alpha x kg + beta x kg / 1000 x km) x (1 + fuel %) + oversized fee)
                x (1 + damage) x (1 + priority) x EUR_to_USD, methodology section 7.5.
                F, alpha and beta are market-level prices, so initial_markup and
                min_profit_margin are not applied and no floor below the price exists:
                minimum_bid and initial_bid are both the price itself

published precision, set once here and read by the price and the prompt alike:
  reference USD/kg 4 decimals   F 4 decimals        alpha 6 decimals   beta 5 decimals
  distance km 3 decimals        fuel % 2 decimals   EUR_to_USD 4 decimals, as the ECB publishes
values stored at or below that precision are unchanged by it
"""

import math

from src.common.bidding_formulas import PRIORITY_MARKUP, get_workload_markup

REFERENCE_DECIMALS = 4
FIXED_CHARGE_DECIMALS = 4
ALPHA_DECIMALS = 6
BETA_DECIMALS = 5
DISTANCE_DECIMALS = 3
FUEL_DECIMALS = 2
FX_DECIMALS = 4

LOGISTICS_PRICE_EXCLUDES = ("initial_markup", "min_profit_margin")


class V5InputError(ValueError):
    """a v5 record is missing a value its v5 price needs, or asks for a price outside its evidence"""


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _optional(value, decimals=None):
    if _missing(value):
        return None
    value = float(value)
    return round(value, decimals) if decimals is not None else value


def check_rfq(order, context):
    """a v5 rfq names its pricing date and, where it names a currency, the v5 one"""
    if context.requires_pricing_date and _missing(order.get("pricing_date")):
        raise V5InputError(f"v5 rfq {order.get('order_id')!r} carries no pricing_date")
    currency = order.get("currency")
    if currency is not None:
        context.require_currency(currency)


def _require(values, keys, what):
    absent = [k for k in keys if values[k] is None]
    if absent:
        raise V5InputError(f"{what} is missing {absent}, which its v5 price needs")


# ── material ────────────────────────────────────────────────────────────────

def material_inputs(agent, order, context):
    check_rfq(order, context)
    material = order["material_type"]
    return {
        "required_kg": order["required_kg"],
        "price_ratio_to_reference": _optional(agent.get(f"{material}_price_ratio_to_reference")),
        "reference_price_usd_per_kg": _optional(order.get("reference_price_usd_per_kg"), REFERENCE_DECIMALS),
        "defect_probability": agent["defect_probability"],
        "min_profit_margin": agent["min_profit_margin"],
        "initial_markup": agent["initial_markup"],
        "priority_markup": PRIORITY_MARKUP.get(order["priority"], 0.0),
    }


def material_price_terms(agent, order, context):
    v = material_inputs(agent, order, context)
    _require(v, ("price_ratio_to_reference", "reference_price_usd_per_kg"),
             f"material agent {agent['agent_id']} on {order['order_id']}")
    material_cost = v["required_kg"] * v["price_ratio_to_reference"] * v["reference_price_usd_per_kg"]
    internal_cost = material_cost * (1 + v["defect_probability"])
    minimum_bid = internal_cost * (1 + v["min_profit_margin"])
    initial_bid = internal_cost * (1 + v["initial_markup"]) * (1 + v["priority_markup"])
    return minimum_bid, initial_bid


# ── manufacturer ────────────────────────────────────────────────────────────

def manufacturer_inputs(agent, order, context):
    check_rfq(order, context)
    return {
        "quantity": order["quantity"],
        "unit_cost_usd": _optional(agent.get(context.manufacturer_unit_cost_column(order["product_id"]))),
        "defect_probability": agent["defect_probability"],
        "min_profit_margin": agent["min_profit_margin"],
        "initial_markup": agent["initial_markup"],
        "priority_markup": PRIORITY_MARKUP.get(order["priority"], 0.0),
    }


def manufacturer_price_terms(agent, order, context):
    """call only after manufacturer_dynamic_check passes, exactly as in v4"""
    v = manufacturer_inputs(agent, order, context)
    _require(v, ("unit_cost_usd",), f"manufacturer {agent['agent_id']} on {order['order_id']}")
    quantity = v["quantity"]
    daily_capacity = agent["daily_capacity_units"]
    workload_ratio = (agent["current_workload_units"] + quantity) / (daily_capacity * order["days_until_deadline"])
    workload_markup = get_workload_markup(workload_ratio)

    base_production_cost = quantity * v["unit_cost_usd"]
    internal_cost = base_production_cost * (1 + v["defect_probability"])
    minimum_bid = internal_cost * (1 + v["min_profit_margin"])
    initial_bid = internal_cost * (1 + v["initial_markup"]) * (1 + workload_markup) * (1 + v["priority_markup"])
    return minimum_bid, initial_bid


# ── logistics ───────────────────────────────────────────────────────────────

def weight_domain(agent):
    """(min kg or None, max kg or None, whether max is inclusive), the v5 profile's evidenced range"""
    inclusive = agent.get("max_weight_inclusive")
    return (_optional(agent.get("min_shipment_weight_kg")), _optional(agent.get("max_shipment_weight_kg")),
            True if _missing(inclusive) else bool(inclusive))


def weight_inside_domain(agent, weight_kg):
    low, cap, inclusive = weight_domain(agent)
    if low is not None and weight_kg < low:
        return False
    if cap is None:
        return True
    return weight_kg <= cap if inclusive else weight_kg < cap


def logistics_eligibility_reason(agent, order, context):
    """v4's three static checks in v4's order, then the weight range the agent's reading covers"""
    if order["deliver_to"] not in str(agent.get("service_regions", "")).split("|"):
        return "region_not_serviced"
    if agent["reputation_score"] < order["min_reputation_score"]:
        return "reputation_too_low"
    if agent["damage_probability"] > order["max_defect_rate"]:
        return "damage_probability_too_high"
    if not weight_inside_domain(agent, order["shipment_weight_kg"]):
        return "shipment_weight_outside_domain"
    return None


def logistics_inputs(agent, order, context):
    check_rfq(order, context)
    low, cap, inclusive = weight_domain(agent)
    return {
        "shipment_weight_kg": order["shipment_weight_kg"],
        "pickup_from": order.get("pickup_from"),
        "route_distance_km": _optional(order.get("route_distance_km"), DISTANCE_DECIMALS),
        "fuel_surcharge_percent": _optional(order.get("fuel_surcharge_percent"), FUEL_DECIMALS),
        "eur_to_usd": _optional(order.get("eur_to_usd"), FX_DECIMALS),
        "fixed_charge_eur": _optional(agent.get("fixed_charge_eur"), FIXED_CHARGE_DECIMALS),
        "alpha_eur_per_kg": _optional(agent.get("alpha_eur_per_kg"), ALPHA_DECIMALS),
        "beta_eur_per_tonne_km": _optional(agent.get("beta_eur_per_tonne_km"), BETA_DECIMALS),
        "min_shipment_weight_kg": low,
        "max_shipment_weight_kg": cap,
        "max_weight_inclusive": inclusive,
        "oversized_above_kg": _optional(agent.get("oversized_above_kg")),
        "oversized_fee_eur": _optional(agent.get("oversized_fee_eur")),
        "damage_probability": agent["damage_probability"],
        "priority_markup": PRIORITY_MARKUP.get(order["priority"], 0.0),
    }


def logistics_price_usd(v):
    """the price from logistics_inputs, in the component order src/data_synthesis/v5/logistics.py uses"""
    weight = v["shipment_weight_kg"]
    freight = v["fixed_charge_eur"] + v["alpha_eur_per_kg"] * weight \
        + v["beta_eur_per_tonne_km"] * (weight / 1000.0) * v["route_distance_km"]
    with_fuel = freight * (1.0 + v["fuel_surcharge_percent"] / 100.0)
    oversized = (v["oversized_fee_eur"] if v["oversized_fee_eur"] is not None
                 and weight > v["oversized_above_kg"] else 0.0)
    eur = (with_fuel + oversized) * (1.0 + v["damage_probability"]) * (1.0 + v["priority_markup"])
    return eur * v["eur_to_usd"]


def logistics_price_terms(agent, order, context):
    v = logistics_inputs(agent, order, context)
    _require(v, ("route_distance_km", "fuel_surcharge_percent", "eur_to_usd", "fixed_charge_eur",
                 "alpha_eur_per_kg", "beta_eur_per_tonne_km"),
             f"logistics agent {agent['agent_id']} on {order['order_id']}")
    if not weight_inside_domain(agent, v["shipment_weight_kg"]):
        raise V5InputError(f"{agent['agent_id']} is not evidenced for {v['shipment_weight_kg']} kg, "
                           "a v5 logistics price is never quoted outside its weight domain")
    price = logistics_price_usd(v)
    return price, price


# ── chain selection ─────────────────────────────────────────────────────────

def order_budget(order, context):
    return float(order[context.order_budget_field])
