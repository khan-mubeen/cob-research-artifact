"""unit tests for the v5 dispatch in bidding_formulas.py and slm.py.
runnable directly (python ...) or under pytest.

the v4 half of the contract, identical v4 decisions, prices and prompts, is proven by the golden
harness and by src/experiments/v5/v4_regression_snapshot.py. these tests prove the v5 half and
the seam between them: v4 records never reach v5 code, v5 records reach it for every wired
function, a mixed record raises, every v5 price is the v5 formula, and every v5 prompt shows
the numbers its label uses and no v4-only wording
"""

import ast
import hashlib
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config
from src.common import bidding_formulas, slm
from src.common.reverse_auction import agent_minimum_bid, auction_fields, ceil_to_cents, decide_round_bid
from src.common.v5 import pricing, prompts
from src.data_synthesis.v5.logistics import RouteCost
from src.dataset.context import DatasetContextError, ENV_VAR, for_version

ROOT = Path(config.PROJECT_ROOT)
V5 = for_version("v5")
_CACHE = {}

V4_ONLY_WORDING = ("base cost per kg (EUR)", "unit cost for requested product (EUR)", "base delivery cost",
                   "cost per kg (EUR)", "current_best_bid (EUR)", "platform_minimum_decrement (EUR)",
                   "your minimum_bid (EUR)", "budget (EUR)", "cost_eur=", "total_cost_eur")
V5_ONLY_WORDING = ("USD", "pricing date", "reference price", "fixed charge", "fuel surcharge", "EUR to USD",
                   "pickup from", "route distance")


def v5_world():
    if "world" not in _CACHE:
        from src.experiments.v5.common import Market, load_corpus, stage_rfqs
        corpus, market = load_corpus(), Market()
        rfqs = {}
        for order in corpus["orders"][:40]:
            rfqs[order["order_id"]] = stage_rfqs(order, corpus, market, V5)
        _CACHE["world"] = (corpus, market, rfqs)
    return _CACHE["world"]


def eligible_pairs(stage, limit=None):
    corpus, _, rfqs = v5_world()
    pairs = []
    for stage_rfqs in rfqs.values():
        rfq = stage_rfqs[stage]
        if rfq is None:
            continue
        pairs += [(a, rfq) for a in corpus[stage] if bidding_formulas.eligibility_reason(stage, a, rfq) is None]
    return pairs[:limit] if limit else pairs


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


# ── the seam ────────────────────────────────────────────────────────────────

def test_v4_records_never_import_the_v5_package():
    code = (
        "import sys\n"
        "from src.common import bidding_formulas as bf, slm\n"
        "from src.common.reverse_auction import decide_round_bid, auction_fields\n"
        "from tests.golden.v4_baseline import load_v4, stage_rfqs, SEALED_DECIDERS\n"
        "corpus = load_v4()\n"
        "for order in corpus['orders'][:30]:\n"
        "    rfqs = stage_rfqs(order, corpus['products'])\n"
        "    for stage, build in (('material', slm.build_material_prompt), ('tailoring', slm.build_manufacturer_prompt),"
        " ('logistics', slm.build_logistics_prompt)):\n"
        "        for agent in corpus[stage]:\n"
        "            SEALED_DECIDERS[stage](agent, rfqs[stage])\n"
        "            if bf.eligibility_reason(stage, agent, rfqs[stage]) is None:\n"
        "                build(agent, rfqs[stage])\n"
        "                r = {**rfqs[stage], 'round_number': 2, 'platform_minimum_decrement': 5.0, 'current_best_bid': 900.0}\n"
        "                decide_round_bid(stage, agent, r); build(agent, r, auction=auction_fields(stage, agent, r))\n"
        "bf.decide_chain_selection([], {'order_id': 'x'})\n"
        "print(sorted(m for m in sys.modules if m.startswith('src.common.v5') or m == 'src.dataset.context'))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout


def test_a_record_without_a_version_takes_the_v4_path():
    assert bidding_formulas.v5_context({"agent_id": "L01"}, {"order_id": "o"}) is None
    assert bidding_formulas.v5_context({"dataset_version": "v4"}, {}) is None
    assert bidding_formulas.v5_context({}, {"dataset_version": "v5"}).version == "v5"


def test_records_naming_different_versions_raise():
    raises(lambda: bidding_formulas.v5_context({"dataset_version": "v5"}, {"dataset_version": "v4"}),
           DatasetContextError, "mixed context")
    raises(lambda: bidding_formulas.v5_context({"dataset_version": "v9"}, {}), DatasetContextError, "unknown")


def test_a_v5_rfq_with_a_v4_profile_does_not_price_off_v4_columns():
    from tests.golden.v4_baseline import load_v4
    v4_agent = load_v4()["material"][0]
    _, _, rfqs = v5_world()
    rfq = next(r["material"] for r in rfqs.values() if r["material"]["material_type"] in v4_agent["material_types"])
    raises(lambda: bidding_formulas.material_price_terms(v4_agent, rfq), pricing.V5InputError, "price_ratio")


def test_every_wired_function_dispatches_a_v5_record():
    corpus, _, rfqs = v5_world()
    for stage, v4_fn, v5_fn in (("material", bidding_formulas.material_price_terms, pricing.material_price_terms),
                                ("tailoring", bidding_formulas.manufacturer_price_terms, pricing.manufacturer_price_terms),
                                ("logistics", bidding_formulas.logistics_price_terms, pricing.logistics_price_terms)):
        for agent, rfq in eligible_pairs(stage, 25):
            if stage == "tailoring" and bidding_formulas.manufacturer_dynamic_check(agent, rfq)[0] is not None:
                continue
            assert v4_fn(agent, rfq) == v5_fn(agent, rfq, V5)
    for stage, build, v5_build in (("material", slm.build_material_prompt, prompts.build_material_prompt),
                                   ("tailoring", slm.build_manufacturer_prompt, prompts.build_manufacturer_prompt),
                                   ("logistics", slm.build_logistics_prompt, prompts.build_logistics_prompt)):
        for agent, rfq in eligible_pairs(stage, 10):
            assert build(agent, rfq) == v5_build(agent, rfq, V5)
            assert build(agent, rfq, worded=True) == v5_build(agent, rfq, V5, worded=True)
    agent, rfq = eligible_pairs("logistics", 1)[0]
    assert bidding_formulas.logistics_eligibility_reason(agent, {**rfq, "shipment_weight_kg": 10 ** 6}) in (
        None, "shipment_weight_outside_domain")


# ── v5 prices ───────────────────────────────────────────────────────────────

def test_the_material_price_is_the_dated_reference_formula_by_hand():
    for agent, rfq in eligible_pairs("material", 20):
        material = rfq["material_type"]
        cost = rfq["required_kg"] * agent[f"{material}_price_ratio_to_reference"] * round(rfq["reference_price_usd_per_kg"], 4)
        internal = cost * (1 + agent["defect_probability"])
        markup = bidding_formulas.PRIORITY_MARKUP[rfq["priority"]]
        assert bidding_formulas.material_price_terms(agent, rfq) == (
            internal * (1 + agent["min_profit_margin"]), internal * (1 + agent["initial_markup"]) * (1 + markup))


def test_the_manufacturer_price_is_v4s_formula_on_the_usd_unit_cost():
    for agent, rfq in eligible_pairs("tailoring", 40):
        if bidding_formulas.manufacturer_dynamic_check(agent, rfq)[0] is not None:
            continue
        v4_shaped = {k: v for k, v in agent.items() if k != "dataset_version"}
        v4_shaped[f"unit_cost_{rfq['product_id']}"] = agent[f"unit_cost_usd_{rfq['product_id']}"]
        v4_rfq = {k: v for k, v in rfq.items() if k not in ("dataset_version", "currency", "requires_pricing_date")}
        assert bidding_formulas.manufacturer_price_terms(agent, rfq) == bidding_formulas.manufacturer_price_terms(
            v4_shaped, v4_rfq)


def test_the_logistics_price_matches_the_v5_route_cost_at_published_precision():
    corpus, market, rfqs = v5_world()
    route = RouteCost()
    worst = 0.0
    for agent, rfq in eligible_pairs("logistics"):
        day = datetime.strptime(rfq["pricing_date"], "%Y-%m-%d").date()
        exact = route.price_for_distance(agent["agent_id"], rfq["shipment_weight_kg"],
                                         route.distance_km[(rfq["pickup_from"], rfq["deliver_to"])], day, rfq["priority"])
        minimum, initial = bidding_formulas.logistics_price_terms(agent, rfq)
        assert minimum == initial, "a v5 logistics price has no floor below it"
        worst = max(worst, abs(minimum - exact))
    assert worst < 0.01, f"published precision moves a logistics price by {worst} USD"


def test_the_logistics_price_by_hand():
    for agent, rfq in eligible_pairs("logistics", 30):
        w = rfq["shipment_weight_kg"]
        freight = round(agent["fixed_charge_eur"], 4) + round(agent["alpha_eur_per_kg"], 6) * w \
            + round(agent["beta_eur_per_tonne_km"], 5) * (w / 1000.0) * rfq["route_distance_km"]
        fee = agent["oversized_fee_eur"]
        oversized = fee if not (fee is None or math.isnan(fee)) and w > agent["oversized_above_kg"] else 0.0
        eur = (freight * (1 + rfq["fuel_surcharge_percent"] / 100) + oversized) * (1 + agent["damage_probability"]) \
            * (1 + bidding_formulas.PRIORITY_MARKUP[rfq["priority"]])
        assert bidding_formulas.logistics_price_terms(agent, rfq)[0] == eur * rfq["eur_to_usd"]


def test_weight_outside_a_v5_agents_domain_is_ineligible_and_never_priced():
    corpus, _, rfqs = v5_world()
    parcel = next(a for a in corpus["logistics"] if a["agent_id"] == "L03")
    rfq = next(r["logistics"] for r in rfqs.values() if r["logistics"] and r["logistics"]["deliver_to"] in parcel["service_regions"])
    heavy = {**rfq, "shipment_weight_kg": 30.0, "max_defect_rate": 1.0, "min_reputation_score": 0.0}
    assert bidding_formulas.logistics_eligibility_reason(parcel, heavy) == "shipment_weight_outside_domain"
    raises(lambda: bidding_formulas.logistics_price_terms(parcel, heavy), pricing.V5InputError, "weight domain")
    light = {**heavy, "shipment_weight_kg": 25.0}
    assert bidding_formulas.logistics_eligibility_reason(parcel, light) is None


def test_v5_static_eligibility_is_v4s_three_checks_plus_the_weight_domain():
    corpus, _, rfqs = v5_world()
    for stage_rfqs in rfqs.values():
        rfq = stage_rfqs["logistics"]
        if rfq is None:
            continue
        for agent in corpus["logistics"]:
            v4_agent = {k: v for k, v in agent.items() if k != "dataset_version"}
            v4_rfq = {k: v for k, v in rfq.items() if k not in ("dataset_version", "currency", "requires_pricing_date")}
            v4 = bidding_formulas.logistics_eligibility_reason(v4_agent, v4_rfq)
            v5 = bidding_formulas.logistics_eligibility_reason(agent, rfq)
            inside = pricing.weight_inside_domain(agent, rfq["shipment_weight_kg"])
            assert v5 == (v4 if v4 is not None or inside else "shipment_weight_outside_domain")


def test_the_chain_rule_reads_the_v5_budget_field():
    chain = {"material_agent_id": "M01", "manufacturer_agent_id": "T1", "logistics_agent_id": "L01",
             "material_offer": 100.0, "manufacturer_offer": 100.0, "logistics_offer": 100.0,
             "material_completion_days": 1, "manufacturer_completion_days": 1, "logistics_completion_days": 1,
             "material_defect_probability": 0.01, "manufacturer_defect_probability": 0.01,
             "logistics_damage_probability": 0.01}
    order = {"order_id": "o", "priority": "normal", "budget_usd": 400.0, "days_until_deadline": 10,
             "max_defect_rate": 0.05, "dataset_version": "v5"}
    assert bidding_formulas.decide_chain_selection([chain], order)["decision"] == "SELECT"
    assert bidding_formulas.decide_chain_selection([chain], {**order, "budget_usd": 250.0})["decision"] == "NO_SELECTION"
    raises(lambda: bidding_formulas.decide_chain_selection([chain], {**order, "budget": 400.0, "budget_usd": None}),
           TypeError)


# ── v5 prompts ──────────────────────────────────────────────────────────────

def test_v5_prompts_carry_usd_and_no_v4_only_wording():
    for stage, build in (("material", slm.build_material_prompt), ("tailoring", slm.build_manufacturer_prompt),
                         ("logistics", slm.build_logistics_prompt)):
        for agent, rfq in eligible_pairs(stage, 10):
            auction_rfq = {**rfq, "round_number": 2, "platform_minimum_decrement": 5.0, "current_best_bid": 900.0}
            for _, user in (build(agent, rfq), build(agent, auction_rfq, auction=auction_fields(stage, agent, auction_rfq))):
                assert "USD" in user
                leaked = [w for w in V4_ONLY_WORDING if w in user]
                assert not leaked, f"{stage} v5 prompt carries v4-only wording {leaked}"


def test_v4_prompts_carry_no_v5_wording():
    from tests.golden.v4_baseline import load_v4, stage_rfqs
    corpus = load_v4()
    for order in corpus["orders"][:5]:
        rfqs = stage_rfqs(order, corpus["products"])
        for stage, build in (("material", slm.build_material_prompt), ("tailoring", slm.build_manufacturer_prompt),
                             ("logistics", slm.build_logistics_prompt)):
            agent = next(a for a in corpus[stage] if bidding_formulas.eligibility_reason(stage, a, rfqs[stage]) is None)
            _, user = build(agent, rfqs[stage])
            leaked = [w for w in V5_ONLY_WORDING if w in user]
            assert not leaked, f"{stage} v4 prompt carries v5 wording {leaked}"


def test_every_number_a_v5_price_reads_is_the_number_its_prompt_prints():
    def printed(user, label):
        return re.search(rf"^{re.escape(label)}: (.+)$", user, re.M).group(1)

    for agent, rfq in eligible_pairs("logistics", 30):
        v = pricing.logistics_inputs(agent, rfq, V5)
        _, user = slm.build_logistics_prompt(agent, rfq)
        for label, key in (("fixed charge (EUR per shipment)", "fixed_charge_eur"), ("alpha (EUR per kg)", "alpha_eur_per_kg"),
                           ("beta (EUR per tonne-km)", "beta_eur_per_tonne_km"),
                           ("route distance from pickup (km)", "route_distance_km"),
                           ("shipment weight (kg)", "shipment_weight_kg"),
                           ("fuel surcharge on the pricing date (%)", "fuel_surcharge_percent"),
                           ("EUR to USD on the pricing date", "eur_to_usd"), ("damage probability", "damage_probability")):
            assert float(printed(user, label)) == v[key], (label, printed(user, label), v[key])
    for agent, rfq in eligible_pairs("material", 20):
        v = pricing.material_inputs(agent, rfq, V5)
        _, user = slm.build_material_prompt(agent, rfq)
        assert float(printed(user, "reference price on the pricing date (USD per kg)")) == v["reference_price_usd_per_kg"]
        assert float(printed(user, "price ratio to reference for requested material")) == v["price_ratio_to_reference"]
        assert float(printed(user, "required quantity (kg)")) == v["required_kg"]


def test_the_v5_chain_prompt_is_in_usd():
    order = {"order_id": "o", "priority": "urgent", "budget_usd": 1234.5, "max_defect_rate": 0.05, "dataset_version": "v5"}
    bid = {"agent_id": "X", "offer_amount": 10.0, "estimated_completion_days": 1, "defect_probability": 0.01}
    candidate = {"material": bid, "tailoring": bid, "logistics": bid, "total_cost": 30.0, "total_days": 3}
    _, user = slm.build_chain_selection_prompt(order, [candidate, candidate], 5)
    assert "budget (USD): 1234.5" in user and "total_cost_usd=30.0" in user
    assert not [w for w in V4_ONLY_WORDING if w in user]


# ── the parts that must not change ──────────────────────────────────────────

def test_reverse_auction_and_bidding_modes_are_byte_identical_to_phase_1():
    recorded = (ROOT / "reports" / "v5" / "wiring" / "phase1_git_and_hashes.txt").read_text(encoding="utf-8")
    for name in ("src/common/reverse_auction.py", "src/common/bidding_modes.py", "config.py"):
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert f"{digest} *{name}" in recorded, f"{name} changed since phase 1"


def test_reverse_auction_runs_v5_rounds_without_a_change():
    for stage in ("material", "tailoring", "logistics"):
        for agent, rfq in eligible_pairs(stage, 8):
            if bidding_formulas.dynamic_feasibility_reason(stage, agent, rfq) is not None:
                continue
            minimum, _ = bidding_formulas.stage_price_terms(stage, agent, rfq)
            assert agent_minimum_bid(stage, agent, rfq) == ceil_to_cents(minimum)
            at = {**rfq, "round_number": 2, "platform_minimum_decrement": 5.0,
                  "current_best_bid": round(ceil_to_cents(minimum) + 5.0, 2)}
            below = {**at, "current_best_bid": round(ceil_to_cents(minimum) - 0.01 + 5.0, 2)}
            assert decide_round_bid(stage, agent, at)["decision"] == "BID"
            assert decide_round_bid(stage, agent, below)["decision"] == "NO_BID"


def test_the_data_version_default_is_still_v4_and_v5_generators_need_the_environment():
    assert config.DATA_VERSION == "v4"
    from src.experiments.v5.common import require_v5
    saved = os.environ.pop(ENV_VAR, None)
    try:
        raises(require_v5, SystemExit, ENV_VAR)
        os.environ[ENV_VAR] = "v5"
        assert require_v5().version == "v5"
    finally:
        os.environ.pop(ENV_VAR, None)
        if saved is not None:
            os.environ[ENV_VAR] = saved


def test_no_v5_pricing_or_generation_module_reads_the_clock():
    banned = {("datetime", "now"), ("datetime", "utcnow"), ("datetime", "today"), ("date", "today"),
              ("time", "time"), ("time", "time_ns"), ("time", "localtime")}
    folders = (ROOT / "src" / "common" / "v5", ROOT / "src" / "experiments" / "v5")
    for path in (p for folder in folders for p in folder.glob("*.py")):
        # run tooling that stamps its own record with wall-clock time; none of it prices or
        # generates corpus data, and none of those stamps is ever read back as an input
        if path.name in ("v4_regression_snapshot.py", "backend_smoke.py", "evaluate.py",
                         "results_store.py", "freeze_check.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert (node.value.id, node.attr) not in banned, f"{path.name} reads the clock: {node.value.id}.{node.attr}"


def test_the_logistics_rfq_picks_up_at_the_provisional_manufacturer_winner():
    from src.supervisor.supervisor import pick_winner
    corpus, _, rfqs = v5_world()
    for stage_rfqs in rfqs.values():
        bids = [b for b in (bidding_formulas.decide_manufacturer_bid(a, stage_rfqs["tailoring"]) for a in corpus["tailoring"])
                if b["decision"] == "BID"]
        winner = pick_winner(bids)
        if winner is None:
            assert stage_rfqs["logistics"] is None
        else:
            assert stage_rfqs["logistics"]["pickup_from"] == corpus["manufacturer_location"][winner["agent_id"]]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 wiring unit tests passed")


if __name__ == "__main__":
    main()
