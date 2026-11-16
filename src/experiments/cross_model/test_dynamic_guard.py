"""unit tests for Guard B (the hybrid dynamic-feasibility guard) and its shared
feasibility helpers. runnable directly (python ...) or under pytest.

covers every dynamic rejection reason and the guard behavior contract:
suppress infeasible BID, never touch a feasible BID, never recover a NO_BID,
disabled by default. the exhaustive rule-decision regression lives separately
in decision_census.py (329,313 real+synthetic pairs, byte-identical hash).
"""

import os
from datetime import datetime, timedelta

from src.common.bidding_formulas import (
    apply_dynamic_guard,
    decide_manufacturer_bid,
    dynamic_feasibility_reason,
    dynamic_guard_enabled,
    logistics_dynamic_check,
    manufacturer_dynamic_check,
    material_dynamic_check,
)
from src.common.dynamic_state import MaterialResourceState, SimClock


def mfr(**over):
    a = {"agent_id": "T1", "product_types": "shirt", "daily_capacity_units": 100,
         "current_workload_units": 0, "defect_probability": 0.01, "reputation_score": 0.9,
         "min_profit_margin": 0.1, "initial_markup": 0.2, "unit_cost_shirt": 5.0}
    a.update(over)
    return a


def order(**over):
    o = {"order_id": "O1", "product_id": "shirt", "quantity": 100, "max_defect_rate": 0.5,
         "min_reputation_score": 0.1, "priority": "normal", "days_until_deadline": 2}
    o.update(over)
    return o


def log_agent(**over):
    a = {"agent_id": "L1", "location": "Helsinki", "service_regions": "Helsinki|Turku",
         "default_completion_same_city_days": 1, "default_completion_intercity_days": 5,
         "base_delivery_cost": 50, "cost_per_kg": 0.5, "damage_probability": 0.02,
         "reputation_score": 0.9, "min_profit_margin": 0.1, "initial_markup": 0.2}
    a.update(over)
    return a


def log_order(**over):
    o = {"order_id": "LO1", "deliver_to": "Turku", "shipment_weight_kg": 200.0,
         "max_defect_rate": 0.5, "min_reputation_score": 0.1, "priority": "normal",
         "days_until_deadline": 6}
    o.update(over)
    return o


def mat_agent(**over):
    a = {"agent_id": "M1", "material_types": "cotton", "default_completion_days": 2,
         "cotton_available_kg": 100.0, "cotton_restock_amount_kg": 50.0,
         "cotton_restock_time_days": 3, "cotton_base_cost_eur_per_kg": 2.0,
         "defect_probability": 0.02, "reputation_score": 0.9,
         "min_profit_margin": 0.1, "initial_markup": 0.2}
    a.update(over)
    return a


def mat_order(**over):
    o = {"order_id": "MO1", "material_type": "cotton", "required_kg": 50.0,
         "max_defect_rate": 0.5, "min_reputation_score": 0.0, "priority": "normal",
         "days_until_deadline": 10}
    o.update(over)
    return o


# --- every dynamic rejection reason ---

def test_manufacturer_completion_exceeds_deadline():
    r, comp = manufacturer_dynamic_check(mfr(), order(quantity=300, days_until_deadline=2))
    assert r == "completion_exceeds_deadline" and comp == 3.0


def test_manufacturer_no_daily_capacity():
    r, comp = manufacturer_dynamic_check(mfr(daily_capacity_units=0), order())
    assert r == "no_daily_capacity" and comp is None


def test_manufacturer_feasible():
    r, comp = manufacturer_dynamic_check(mfr(), order(quantity=100, days_until_deadline=2))
    assert r is None and comp == 1.0


def test_manufacturer_exact_boundary_feasible():
    # completion exactly equals deadline -> feasible (ratio 1.00, markup 0.15)
    r, comp = manufacturer_dynamic_check(mfr(), order(quantity=200, days_until_deadline=2))
    assert r is None and comp == 2.0


def test_logistics_completion_exceeds_deadline():
    r, rd = logistics_dynamic_check(log_agent(), log_order(days_until_deadline=2))
    assert r == "completion_exceeds_deadline" and rd == 5


def test_logistics_feasible():
    r, rd = logistics_dynamic_check(log_agent(), log_order(days_until_deadline=6))
    assert r is None and rd == 5


def test_material_insufficient_stock_and_restock():
    r, comp = material_dynamic_check(mat_agent(), mat_order(required_kg=1000.0))
    assert r == "insufficient_stock_and_restock" and comp is None


def test_material_completion_exceeds_deadline():
    # needs restock (avail 100 < 120 <= 150) -> completion 2+3=5 > deadline 4
    r, comp = material_dynamic_check(mat_agent(), mat_order(required_kg=120.0, days_until_deadline=4))
    assert r == "completion_exceeds_deadline" and comp == 5


def test_material_feasible_from_stock():
    r, comp = material_dynamic_check(mat_agent(), mat_order(required_kg=50.0, days_until_deadline=10))
    assert r is None and comp == 2


# --- an order arriving while a restock is already in flight ---
# the wait is what is left of that restock's known eta, not a fresh lead time.
# state is built through MaterialResourceState so the profile is written by the
# same sync_profile the live agents use, not hand-stubbed

def pending_state(available_kg=10.0, elapsed_days=1.0):
    """M1 cotton with a restock in flight, triggered elapsed_days ago

    lead time is 3 days, so the remaining wait is 3 - elapsed_days"""
    state = MaterialResourceState([{
        "agent_id": "M1", "material_types": "cotton", "cotton_available_kg": 100.0,
        "cotton_restock_amount_kg": 50.0, "cotton_restock_time_days": 3,
    }], clock=SimClock())
    slot = state.slot("M1", "cotton")
    slot.available_kg = available_kg
    triggered = datetime(2026, 9, 1, 12, 0, 0)
    state.clock.advance_to(triggered + timedelta(days=elapsed_days))
    slot.pending = {"eta": triggered + timedelta(days=slot.restock_time_days),
                    "amount": slot.restock_amount_kg, "trigger": "commitment",
                    "order_id": "MO0", "triggered_at": triggered}
    return state, slot


def test_material_pending_restock_uses_remaining_time():
    state, _ = pending_state(elapsed_days=1.0)
    agent = mat_agent()
    state.sync_profile(agent)
    assert agent["cotton_restock_pending_days"] == 2.0
    # 2 days default + 2 days left on the pending restock, not the full 3
    r, comp = material_dynamic_check(agent, mat_order(required_kg=50.0))
    assert r is None and comp == 4.0


def test_material_no_pending_restock_keeps_full_lead_time():
    state = MaterialResourceState([{
        "agent_id": "M1", "material_types": "cotton", "cotton_available_kg": 10.0,
        "cotton_restock_amount_kg": 50.0, "cotton_restock_time_days": 3,
    }], clock=SimClock())
    state.clock.advance_to(datetime(2026, 9, 1, 12, 0, 0))
    agent = mat_agent(cotton_available_kg=10.0)
    state.sync_profile(agent)
    assert agent["cotton_restock_pending_days"] is None
    r, comp = material_dynamic_check(agent, mat_order(required_kg=50.0))
    assert r is None and comp == 5


def test_material_pending_restock_absent_from_a_static_profile():
    # a profile read straight off the csv carries no pending key at all
    r, comp = material_dynamic_check(mat_agent(cotton_available_kg=10.0),
                                     mat_order(required_kg=50.0))
    assert r is None and comp == 5


def test_material_pending_restock_cleared_once_it_lands():
    state, slot = pending_state(elapsed_days=1.0)
    agent = mat_agent()
    state.sync_profile(agent)
    assert agent["cotton_restock_pending_days"] == 2.0
    state.advance_to(slot.pending["eta"], order_id="MO1")
    state.sync_profile(agent)
    # no stale remaining time survives the landing
    assert agent["cotton_restock_pending_days"] is None
    assert slot.pending is None


def test_material_pending_restock_never_negative():
    # clock past the eta without a landing still means no further wait
    state, _ = pending_state(elapsed_days=5.0)
    agent = mat_agent()
    state.sync_profile(agent)
    assert agent["cotton_restock_pending_days"] == 0.0
    r, comp = material_dynamic_check(agent, mat_order(required_kg=50.0))
    assert r is None and comp == 2


def test_material_pending_restock_rescues_a_feasible_bid():
    # the bug's real cost: deadline sits between the two figures, so the full
    # constant rejected an order the pending eta comfortably meets
    state, _ = pending_state(elapsed_days=2.0)
    agent = mat_agent()
    state.sync_profile(agent)
    order = mat_order(required_kg=50.0, days_until_deadline=4)
    r, comp = material_dynamic_check(agent, order)
    assert r is None and comp == 3.0
    # the same order against the full lead time would have been turned away
    stale = mat_agent(cotton_available_kg=10.0)
    assert material_dynamic_check(stale, order)[0] == "completion_exceeds_deadline"


def test_material_reserve_uses_pending_remaining_time():
    # reserve re-runs the check itself, so it has to pick the fix up too
    state, _ = pending_state(elapsed_days=1.0)
    agent = mat_agent()
    rfq = {"order_id": "MO1", "material_type": "cotton", "required_kg": 50.0,
           "days_until_deadline": 10, "sim_now": state.clock.now.isoformat()}
    ack, events = state.reserve(agent, rfq, promised_completion_days=4.0)
    reserved = next(e for e in events if e["event"] == "stock_reserved")
    assert ack is True and reserved["completion_days"] == 4.0


def test_material_reserve_refuses_a_promise_built_on_the_full_constant():
    # a bid promising the old 5.0 is not broken by a recomputed 4.0, a promise
    # can always be beaten. the reverse, a promise the state can no longer meet,
    # is what reserve exists to catch
    state, slot = pending_state(elapsed_days=1.0)
    agent = mat_agent()
    rfq = {"order_id": "MO1", "material_type": "cotton", "required_kg": 50.0,
           "days_until_deadline": 10, "sim_now": state.clock.now.isoformat()}
    assert state.reserve(agent, rfq, promised_completion_days=5.0)[0] is True
    slot.available_kg = -500.0
    rfq2 = dict(rfq, order_id="MO2")
    ack, events = state.reserve(agent, rfq2, promised_completion_days=4.0)
    assert ack is False
    assert events[0]["reason"] == "insufficient_stock_and_restock"


# --- guard behavior contract ---

def test_guard_suppresses_infeasible_bid():
    o = order(quantity=300, days_until_deadline=2)
    fake_bid = {"decision": "BID", "offer_amount": 999.0, "estimated_completion_days": 1.0,
                "defect_probability": 0.01}
    guarded, reason = apply_dynamic_guard("tailoring", mfr(), o, fake_bid)
    assert reason == "completion_exceeds_deadline"
    assert guarded["decision"] == "NO_BID"
    assert guarded["debug_reason"] == "dynamic_guard:completion_exceeds_deadline"


def test_guard_leaves_feasible_bid_byte_identical():
    o = order(quantity=100, days_until_deadline=2)
    bid = {"decision": "BID", "offer_amount": 660.0, "estimated_completion_days": 1.0,
           "defect_probability": 0.01}
    guarded, reason = apply_dynamic_guard("tailoring", mfr(), o, dict(bid))
    assert reason is None and guarded == bid  # no field touched, no reprice


def test_guard_never_recovers_no_bid():
    o = order(quantity=100, days_until_deadline=2)
    nb = {"decision": "NO_BID", "debug_reason": "model_said_no"}
    guarded, reason = apply_dynamic_guard("tailoring", mfr(), o, dict(nb))
    assert reason is None and guarded == nb  # NO_BID stays NO_BID, unrecoverable


def test_guard_matches_rule_on_infeasible():
    # the guard's verdict agrees with the authoritative rule decision
    o = order(quantity=300, days_until_deadline=2)
    assert decide_manufacturer_bid(mfr(), o)["decision"] == "NO_BID"
    assert dynamic_feasibility_reason("tailoring", mfr(), o) == "completion_exceeds_deadline"


def test_guard_disabled_by_default():
    os.environ.pop("SLM_DYNAMIC_GUARD", None)
    assert dynamic_guard_enabled() is False
    for v in ("1", "true", "YES", "on"):
        os.environ["SLM_DYNAMIC_GUARD"] = v
        assert dynamic_guard_enabled() is True
    os.environ["SLM_DYNAMIC_GUARD"] = "0"
    assert dynamic_guard_enabled() is False
    os.environ.pop("SLM_DYNAMIC_GUARD", None)


def test_guard_stage_aliases():
    o = order(quantity=300, days_until_deadline=2)
    assert dynamic_feasibility_reason("manufacturer", mfr(), o) == "completion_exceeds_deadline"
    assert dynamic_feasibility_reason("tailoring", mfr(), o) == "completion_exceeds_deadline"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} guard unit tests passed")


if __name__ == "__main__":
    main()
