"""constructed v4 boundary cases, one per rule edge

the 50,000-order corpus exercises the common paths but does not land exactly on
a threshold often enough to prove which side of it the code sits on. each case
below is built to sit exactly on an edge, or one representable step off it, so a
later refactor that turns a >= into a > shows up as a changed decision

these are inputs only. the expected outputs live in the generated fixture, which
is what the regression test compares against
"""

from src.common.bidding_modes import DEFAULT_PLATFORM_MINIMUM_DECREMENT


def material_agent(**overrides):
    agent = {
        "agent_id": "BM01",
        "stage": "material",
        "location": "Helsinki",
        "material_types": "cotton|linen",
        "default_completion_days": 3,
        "min_profit_margin": 0.10,
        "initial_markup": 0.25,
        "defect_probability": 0.04,
        "reputation_score": 0.60,
        "cotton_available_kg": 1000.0,
        "cotton_base_cost_eur_per_kg": 3.0,
        "cotton_restock_amount_kg": 500.0,
        "cotton_restock_time_days": 4,
    }
    agent.update(overrides)
    return agent


def material_rfq(**overrides):
    rfq = {
        "order_id": "BND-MAT",
        "material_type": "cotton",
        "required_kg": 100.0,
        "max_defect_rate": 0.05,
        "min_reputation_score": 0.50,
        "days_until_deadline": 10,
        "priority": "normal",
    }
    rfq.update(overrides)
    return rfq


def manufacturer_agent(**overrides):
    agent = {
        "agent_id": "BT01",
        "stage": "tailoring",
        "location": "Tampere",
        "product_types": "shirt|hoodie",
        "daily_capacity_units": 100,
        "current_workload_units": 0,
        "unit_cost_shirt": 5.0,
        "unit_cost_hoodie": 8.0,
        "min_profit_margin": 0.10,
        "initial_markup": 0.28,
        "defect_probability": 0.02,
        "reputation_score": 0.80,
    }
    agent.update(overrides)
    return agent


def manufacturer_rfq(**overrides):
    rfq = {
        "order_id": "BND-MAN",
        "product_id": "shirt",
        "quantity": 100,
        "max_defect_rate": 0.05,
        "min_reputation_score": 0.50,
        "days_until_deadline": 10,
        "priority": "normal",
    }
    rfq.update(overrides)
    return rfq


def logistics_agent(**overrides):
    agent = {
        "agent_id": "BL01",
        "stage": "logistics",
        "location": "Helsinki",
        "service_regions": "Helsinki|Tampere|Turku",
        "base_delivery_cost": 40.0,
        "cost_per_kg": 0.80,
        "default_completion_same_city_days": 1,
        "default_completion_intercity_days": 3.0,
        "damage_probability": 0.03,
        "min_profit_margin": 0.10,
        "initial_markup": 0.25,
        "reputation_score": 0.60,
    }
    agent.update(overrides)
    return agent


def logistics_rfq(**overrides):
    rfq = {
        "order_id": "BND-LOG",
        "deliver_to": "Tampere",
        "shipment_weight_kg": 150.0,
        "max_defect_rate": 0.05,
        "min_reputation_score": 0.50,
        "days_until_deadline": 10,
        "priority": "normal",
    }
    rfq.update(overrides)
    return rfq


def sealed_cases():
    """(case name, stage, agent, rfq) for every constructed sealed edge"""
    cases = []

    # ── material: static eligibility edges ───────────────────────────────────
    cases.append(("material_defect_exactly_at_cap", "material",
                  material_agent(defect_probability=0.05), material_rfq()))
    cases.append(("material_defect_one_step_over_cap", "material",
                  material_agent(defect_probability=0.050001), material_rfq()))
    cases.append(("material_reputation_exactly_at_floor", "material",
                  material_agent(reputation_score=0.50), material_rfq()))
    cases.append(("material_reputation_one_step_under_floor", "material",
                  material_agent(reputation_score=0.499999), material_rfq()))
    cases.append(("material_not_supplied", "material",
                  material_agent(), material_rfq(material_type="denim")))

    # ── material: stock and restock edges ────────────────────────────────────
    cases.append(("material_stock_exactly_equals_required", "material",
                  material_agent(cotton_available_kg=100.0), material_rfq()))
    cases.append(("material_stock_one_step_short_restock_covers", "material",
                  material_agent(cotton_available_kg=99.999999), material_rfq()))
    cases.append(("material_stock_plus_restock_exactly_equals_required", "material",
                  material_agent(cotton_available_kg=60.0, cotton_restock_amount_kg=40.0),
                  material_rfq()))
    cases.append(("material_stock_plus_restock_one_step_short", "material",
                  material_agent(cotton_available_kg=60.0, cotton_restock_amount_kg=39.999999),
                  material_rfq()))
    cases.append(("material_restock_already_in_flight_shortens_wait", "material",
                  material_agent(cotton_available_kg=60.0, cotton_restock_pending_days=1.0),
                  material_rfq()))

    # ── material: deadline edges ────────────────────────────────────────────
    cases.append(("material_completion_exactly_at_deadline", "material",
                  material_agent(default_completion_days=10), material_rfq()))
    cases.append(("material_completion_one_day_past_deadline", "material",
                  material_agent(default_completion_days=11), material_rfq()))
    cases.append(("material_restock_wait_exactly_at_deadline", "material",
                  material_agent(cotton_available_kg=60.0, default_completion_days=6,
                                 cotton_restock_time_days=4), material_rfq()))
    cases.append(("material_restock_wait_one_day_past_deadline", "material",
                  material_agent(cotton_available_kg=60.0, default_completion_days=7,
                                 cotton_restock_time_days=4), material_rfq()))

    # ── material: price edges ───────────────────────────────────────────────
    # flexible priority is the only markup that can push the opening offer under
    # the floor, which is where max(minimum_bid, initial_bid) actually matters
    cases.append(("material_flexible_floor_beats_opening", "material",
                  material_agent(initial_markup=0.25, min_profit_margin=0.10),
                  material_rfq(priority="flexible")))
    cases.append(("material_urgent_opening_beats_floor", "material",
                  material_agent(), material_rfq(priority="urgent")))
    cases.append(("material_opening_exactly_equals_floor", "material",
                  # (1 + 0.294117647...) x (1 - 0.15) == 1.10 to float precision
                  material_agent(initial_markup=(1.10 / 0.85) - 1.0, min_profit_margin=0.10),
                  material_rfq(priority="flexible")))
    cases.append(("material_unknown_priority_gets_no_markup", "material",
                  material_agent(), material_rfq(priority="whenever")))

    # ── manufacturer: workload band edges ───────────────────────────────────
    # workload_ratio = (workload + quantity) / (capacity x days). with capacity
    # 100 and 10 days the denominator is 1000, so the numerator is the band
    for name, workload in (("below_first_band", 399),
                           ("exactly_at_0_50", 400),
                           ("exactly_at_0_75", 650),
                           ("exactly_at_0_90", 800),
                           ("exactly_at_1_00", 900),
                           ("one_unit_above_1_00", 901)):
        cases.append((f"manufacturer_workload_{name}", "tailoring",
                      manufacturer_agent(current_workload_units=workload),
                      manufacturer_rfq()))

    # ── manufacturer: capacity and deadline edges ───────────────────────────
    cases.append(("manufacturer_zero_daily_capacity", "tailoring",
                  manufacturer_agent(daily_capacity_units=0), manufacturer_rfq()))
    cases.append(("manufacturer_completion_exactly_at_deadline", "tailoring",
                  manufacturer_agent(current_workload_units=900), manufacturer_rfq()))
    cases.append(("manufacturer_completion_one_unit_past_deadline", "tailoring",
                  manufacturer_agent(current_workload_units=901), manufacturer_rfq()))
    cases.append(("manufacturer_product_not_made", "tailoring",
                  manufacturer_agent(), manufacturer_rfq(product_id="jeans")))
    cases.append(("manufacturer_defect_exactly_at_cap", "tailoring",
                  manufacturer_agent(defect_probability=0.05), manufacturer_rfq()))
    cases.append(("manufacturer_defect_one_step_over_cap", "tailoring",
                  manufacturer_agent(defect_probability=0.050001), manufacturer_rfq()))
    cases.append(("manufacturer_reputation_exactly_at_floor", "tailoring",
                  manufacturer_agent(reputation_score=0.50), manufacturer_rfq()))
    cases.append(("manufacturer_reputation_one_step_under_floor", "tailoring",
                  manufacturer_agent(reputation_score=0.499999), manufacturer_rfq()))
    cases.append(("manufacturer_flexible_floor_beats_opening", "tailoring",
                  manufacturer_agent(), manufacturer_rfq(priority="flexible")))
    cases.append(("manufacturer_urgent_opening_beats_floor", "tailoring",
                  manufacturer_agent(), manufacturer_rfq(priority="urgent")))

    # ── logistics: coverage, route and deadline edges ───────────────────────
    cases.append(("logistics_region_not_serviced", "logistics",
                  logistics_agent(), logistics_rfq(deliver_to="Oulu")))
    cases.append(("logistics_same_city_route", "logistics",
                  logistics_agent(), logistics_rfq(deliver_to="Helsinki")))
    cases.append(("logistics_route_days_exactly_at_deadline", "logistics",
                  logistics_agent(default_completion_intercity_days=10.0),
                  logistics_rfq()))
    cases.append(("logistics_route_days_one_day_past_deadline", "logistics",
                  logistics_agent(default_completion_intercity_days=11.0),
                  logistics_rfq()))
    cases.append(("logistics_damage_exactly_at_cap", "logistics",
                  logistics_agent(damage_probability=0.05), logistics_rfq()))
    cases.append(("logistics_damage_one_step_over_cap", "logistics",
                  logistics_agent(damage_probability=0.050001), logistics_rfq()))
    cases.append(("logistics_reputation_exactly_at_floor", "logistics",
                  logistics_agent(reputation_score=0.50), logistics_rfq()))
    cases.append(("logistics_reputation_one_step_under_floor", "logistics",
                  logistics_agent(reputation_score=0.499999), logistics_rfq()))
    cases.append(("logistics_flexible_floor_beats_opening", "logistics",
                  logistics_agent(), logistics_rfq(priority="flexible")))
    cases.append(("logistics_urgent_opening_beats_floor", "logistics",
                  logistics_agent(), logistics_rfq(priority="urgent")))
    cases.append(("logistics_zero_weight_shipment", "logistics",
                  logistics_agent(), logistics_rfq(shipment_weight_kg=0.0)))

    return cases


def auction_cases():
    """(case name, stage, agent, rfq) for the reverse-auction round edges

    the interesting edge is the floor: an agent bids the mandated decrement only
    while the result stays at or above its own minimum_bid. the floors below are
    the exact minimum_bid of the matching sealed case, so current_best_bid is
    placed one decrement above, exactly on, and one cent under it
    """
    cases = []
    decrement = DEFAULT_PLATFORM_MINIMUM_DECREMENT

    # material floor: 100 kg x 3.00 x 1.04 x 1.10 = 343.20 exactly
    material_floor = 343.20
    for label, best in (("next_bid_one_cent_above_floor", material_floor + decrement + 0.01),
                        ("next_bid_exactly_at_floor", material_floor + decrement),
                        ("next_bid_one_cent_below_floor", material_floor + decrement - 0.01)):
        cases.append((f"auction_material_{label}", "material", material_agent(),
                      material_rfq(round_number=2, current_best_bid=best,
                                   platform_minimum_decrement=decrement)))

    # manufacturer floor: 100 x 5.00 x 1.02 x 1.10 = 561.00 exactly
    manufacturer_floor = 561.00
    for label, best in (("next_bid_one_cent_above_floor", manufacturer_floor + decrement + 0.01),
                        ("next_bid_exactly_at_floor", manufacturer_floor + decrement),
                        ("next_bid_one_cent_below_floor", manufacturer_floor + decrement - 0.01)):
        cases.append((f"auction_manufacturer_{label}", "tailoring", manufacturer_agent(),
                      manufacturer_rfq(round_number=2, current_best_bid=best,
                                       platform_minimum_decrement=decrement)))

    # logistics floor: (40 + 0.80 x 150) x 1.03 x 1.10 = 181.28 exactly
    logistics_floor = 181.28
    for label, best in (("next_bid_one_cent_above_floor", logistics_floor + decrement + 0.01),
                        ("next_bid_exactly_at_floor", logistics_floor + decrement),
                        ("next_bid_one_cent_below_floor", logistics_floor + decrement - 0.01)):
        cases.append((f"auction_logistics_{label}", "logistics", logistics_agent(),
                      logistics_rfq(round_number=2, current_best_bid=best,
                                    platform_minimum_decrement=decrement)))

    # round 1 is the sealed decision whatever the auction fields say
    cases.append(("auction_round_one_is_the_sealed_decision", "material", material_agent(),
                  material_rfq(round_number=1, current_best_bid=None,
                               platform_minimum_decrement=decrement)))
    # nothing to undercut: no round-1 bid was announced
    cases.append(("auction_no_current_best_bid", "material", material_agent(),
                  material_rfq(round_number=2, current_best_bid=None,
                               platform_minimum_decrement=decrement)))
    # an ineligible agent is refused before the round rule is reached
    cases.append(("auction_ineligible_agent_stays_refused", "material",
                  material_agent(defect_probability=0.09),
                  material_rfq(round_number=2, current_best_bid=1000.0,
                               platform_minimum_decrement=decrement)))
    return cases
