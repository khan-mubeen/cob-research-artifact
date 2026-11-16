"""unit tests for phase 0, the L09 counterfactual: the stored decision is what its own metrics
imply, the guards are strict, the control reproduced production, and the counting is right.
runnable directly (python ...) or under pytest.
"""

import json
from fractions import Fraction
from types import SimpleNamespace

import pandas as pd

from src.data_synthesis.v5 import freight_candidates
from src.data_synthesis.v5 import l09_counterfactual as cf
from src.data_synthesis.v5.settings import AGENT_CALIBRATION_PATH

_CACHE = {}


def decision():
    if "decision" not in _CACHE:
        _CACHE["decision"] = json.loads((cf.REPORT_DIR / "decision.json").read_text(encoding="utf-8"))
    return _CACHE["decision"]


def restore(metrics):
    """a stored metric block back into the shapes the guards read"""
    out = dict(metrics)
    for key in ("lowest_cost_logistics_win_share", "l09_win_share_by_city"):
        out[key] = {k: Fraction(v["exact"]) for k, v in metrics[key].items()}
    out["tied_price_group_sets"] = {tuple(g) for g in metrics["tied_price_group_sets"]}
    out["domination_pairs"] = {tuple(p) for p in metrics["domination_pairs"]}
    return out


def test_the_stored_decision_follows_from_the_stored_metrics():
    stored = decision()
    production = restore(stored["production"])
    primary = {int(n): restore(m) for n, m in stored["primary"].items()}
    rows, winner = cf.decide(production, primary)
    by_number = {r["candidate"]: r for r in stored["candidates"]}
    for row in rows:
        kept = by_number[row["candidate"]]
        assert row["checks"] == kept["checks"], row["candidate"]
        assert row["removes_l05_l09_tie"] == kept["removes_l05_l09_tie"]
        assert round(float(row["total_abs_win_share_change"]), 6) == kept["total_abs_win_share_change"]["value"]
    assert (winner["candidate"] if winner else None) == stored["decision"]["winner"]


def test_every_other_candidate_was_tried_and_the_held_one_was_not():
    stored = decision()
    assert stored["l09_production_candidate"] == 7
    assert stored["candidates_tried"] == [1, 2, 3, 4, 5, 6]
    assert stored["candidate_verification"]["discrepancies"] == 0
    assert "fixed before any counterfactual was run" in stored["convention"]


def test_the_control_rebuild_reproduced_production_byte_for_byte():
    assert all(decision()["control_rebuild"].values()), decision()["control_rebuild"]
    assert decision()["production_files_unchanged"] is True


def test_production_still_holds_what_the_decision_says():
    stored = decision()
    production = pd.read_csv(AGENT_CALIBRATION_PATH).set_index("agent_id")
    if stored["decision"]["winner"] is None:
        assert stored["decision"]["outcome"] == "keep production unchanged"
        assert production.loc[cf.AGENT, "calibration_reading_id"] == freight_candidates.CANDIDATES[7][0]
    else:
        number = stored["decision"]["winner"]
        assert production.loc[cf.AGENT, "calibration_reading_id"] == freight_candidates.CANDIDATES[number][0]


def test_shares_divide_by_lowest_cost_wins_not_by_all_orders():
    production = decision()["production"]
    wins = production["lowest_cost_logistics_wins"]
    assert sum(wins.values()) == production["feasible_orders"]
    for agent, n in wins.items():
        assert Fraction(production["lowest_cost_logistics_win_share"][agent]["exact"]) == Fraction(n, sum(wins.values()))
    for city, n in production["l09_wins_by_city"].items():
        total = production["lowest_cost_wins_by_city"][city]
        assert Fraction(production["l09_win_share_by_city"][city]["exact"]) == Fraction(n, total)


def fake_dataset():
    """two agents in one city; A is dearer than B at every weight, B serves no slower"""
    prices = {"A": 20.0, "B": 10.0}
    agents = [{"agent_id": a, "location": "X", "regions": {"X"}, "service_regions": "X",
               "default_completion_same_city_days": 1, "default_completion_intercity_days": 2} for a in "AB"]
    route = SimpleNamespace(
        agents={a: {"fixed_charge_eur": prices[a], "alpha_eur_per_kg": 0.1, "beta_eur_per_tonne_km": 0.07,
                    "fuel_surcharge_series_id": "s", "oversized_above_kg": None, "oversized_fee_eur": None,
                    "min_shipment_weight_kg": None, "max_shipment_weight_kg": None, "max_weight_inclusive": True}
                for a in "AB"},
        eligible_weight=lambda agent, weight: True,
        price_for_distance=lambda agent, weight, km, day, priority: prices[agent] + weight)
    return SimpleNamespace(logistics=agents, route=route)


def test_an_infeasible_orders_nan_winner_is_never_counted():
    data = fake_dataset()
    facts = [{"order_id": f"O{i}", "city": "X", "weight": 5.0, "band": "under 25 kg", "scenario": "random_realistic",
              "day": None, "priority": "normal", "km_range": (1.0, 2.0)} for i in range(3)]
    records = pd.DataFrame({"logistics_winner": ["B", None, "B"], "feasible": [True, False, True],
                            "logistics_eligible": [("A", "B"), ("A", "B"), ("A", "B")]})
    assert records["logistics_winner"].isna().sum() == 1
    metrics = cf.measure(data, {"records": records}, facts)
    assert metrics["lowest_cost_logistics_win_share"]["B"] == Fraction(1)
    assert metrics["lowest_cost_wins_by_city"] == {"X": 2}
    assert metrics["domination_pairs"] == {("X", "A", "B")}


def test_the_guards_pass_an_identical_variant_and_fail_any_single_regression():
    production = restore(decision()["production"])
    checks, groups, pairs = cf.guards(production, production)
    assert all(checks.values()) and not groups and not pairs
    worse = dict(production, feasible_orders=production["feasible_orders"] - 1)
    assert cf.guards(production, worse)[0]["feasible_orders_not_fewer"] is False
    city = cf.NICHE_CITIES[1]
    shares = dict(production["l09_win_share_by_city"])
    shares[city] -= Fraction(1, 10 ** 9)
    assert cf.guards(production, dict(production, l09_win_share_by_city=shares))[0][f"l09_share_not_lower_in_{city}"] is False
    extra = dict(production, domination_pairs=production["domination_pairs"] | {("Oulu", "L09", "L08")})
    assert cf.guards(production, extra)[0]["no_new_structural_domination_pair"] is False


def test_the_tie_is_only_removed_when_production_had_it_and_the_variant_does_not():
    production = {"tied_price_group_sets": {("L05", "L09")}}
    assert cf.tie_removed(production, {"tied_price_group_sets": set()})
    assert not cf.tie_removed(production, {"tied_price_group_sets": {("L05", "L08", "L09")}})
    assert not cf.tie_removed({"tied_price_group_sets": set()}, {"tied_price_group_sets": set()})


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} L09 counterfactual unit tests passed")


if __name__ == "__main__":
    main()
