"""unit tests for proposal v3: candidate reuse, calibration and evaluation separation,
the isolated regeneration, and the comparison it reports.
runnable directly (python ...) or under pytest.
"""

import builtins
import copy
import io
import json
import pathlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.data_synthesis.v5 import calibration_proposal_v3 as v3
from src.data_synthesis.v5 import diagnostics, freight_assignment, freight_candidates, proposal_validation, settings

_CACHE = {}


def cached(key, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def proposal():
    return cached("proposal", lambda: json.loads(v3.PROPOSAL_PATH.read_text(encoding="utf-8")))


def comparison():
    return cached("comparison", lambda: json.loads((v3.REPORT_DIR / "comparison.json").read_text(encoding="utf-8")))


def report(label):
    return cached(label, lambda: json.loads(
        (v3.REPORT_DIR / label / f"{diagnostics.REPORT_STEM}.json").read_text(encoding="utf-8")))


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


# ── part 1 ──────────────────────────────────────────────────────────────────

def test_the_seven_candidates_are_reused_exactly_from_the_prior_output():
    prior = json.loads(freight_candidates.PRIOR_PROPOSAL_PATH.read_text(encoding="utf-8"))["readings"]
    candidates = freight_candidates.load_candidates()
    assert sorted(candidates) == [1, 2, 3, 4, 5, 6, 7]
    for number, (reading_id, _) in freight_candidates.CANDIDATES.items():
        for field in freight_candidates.REUSED_FIELDS:
            assert candidates[number][field] == prior[reading_id][field], (number, field)
    assert freight_candidates.verify_candidates(candidates)["discrepancies"] == 0
    stored = {c["candidate"]: c for c in proposal()["part_1_candidates"]["candidates"]}
    for number, c in candidates.items():
        assert stored[number]["fixed_charge_eur"] == c["fixed_charge_eur"]
        assert stored[number]["alpha_eur_per_kg"] == c["alpha_eur_per_kg"]


def test_a_candidate_that_drifts_from_its_source_stops_the_run():
    candidates = copy.deepcopy(freight_candidates.load_candidates())
    candidates[3]["alpha_eur_per_kg"] += 1e-9
    raises(lambda: freight_candidates.verify_candidates(candidates), freight_candidates.CandidateDiscrepancy, "alpha")


def test_candidate_1_lower_bound_is_decided_with_both_readings_quoted():
    decision = proposal()["part_1_candidates"]["candidate_1_lower_bound"]
    assert decision["used_kg"] == 0.0
    assert "yli 35 kg tai isokokoisille" in decision["reading_35_kg"]
    assert "0-79 kg" in decision["reading_0_kg"] and "no minimum-weight column" in decision["reading_0_kg"]
    assert decision["reason"].count(".") <= 1
    assert freight_candidates.load_candidates()[1]["min_shipment_weight_kg"] == 0.0


# ── parts 2 to 4: separation from the evaluation set ────────────────────────

def test_the_calibration_seed_is_not_the_production_seed():
    assert v3.CALIBRATION_SEED != settings.SEED and v3.CALIBRATION_SEED + 1 != settings.SEED
    assert "arbitrary" in proposal()["part_2_calibration_sample"]["seed_note"]
    raises(lambda: v3.calibration_sample(["Espoo"], v3.product_pairs(), seed=settings.SEED), ValueError, "production seed")


def test_parts_2_to_4_never_read_an_order_file_or_the_specialty_column():
    opened, csv_calls = [], []
    real_open, real_io_open, real_path_open, real_read_csv = builtins.open, io.open, pathlib.Path.open, pd.read_csv

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def spy_path_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    def spy_read_csv(path, *args, **kwargs):
        csv_calls.append((str(path), kwargs.get("usecols")))
        return real_read_csv(path, *args, **kwargs)

    builtins.open, io.open, pathlib.Path.open, pd.read_csv = spy_open, spy_open, spy_path_open, spy_read_csv
    try:
        result = v3.assign_from_calibration()
    finally:
        builtins.open, io.open, pathlib.Path.open, pd.read_csv = real_open, real_io_open, real_path_open, real_read_csv
    touched = [Path(p).name for p in opened] + [Path(p).name for p, _ in csv_calls]
    assert "orders.csv" not in touched, touched
    v4_logistics = [cols for p, cols in csv_calls if Path(p).resolve() == (settings.V4_DIR / "logistics_agents.csv").resolve()]
    assert v4_logistics and all(cols is not None and "specialty" not in cols for cols in v4_logistics)
    assert result["sample_orders"] == 50000


def test_the_calibration_sample_is_not_the_production_order_set():
    pairs = v3.product_pairs()
    sample = pd.DataFrame(v3.calibration_sample(v3.fleet()[2], pairs))
    production = pd.read_csv(settings.V5_DIR / "orders.csv")
    columns = ["product_id", "material_type", "quantity", "deliver_to"]
    same = (sample[columns].reset_index(drop=True) == production[columns].reset_index(drop=True)).all(axis=1).mean()
    assert same < 0.05, same
    assert list(sample["scenario_type"].value_counts().sort_index()) == list(production["scenario_type"].value_counts().sort_index())


def test_the_stored_assignment_is_what_a_fresh_solve_returns():
    result = cached("fresh", v3.assign_from_calibration)
    winner = {agent: int(c) for agent, c in proposal()["part_4_assignment"]["winner"]["candidates_by_agent"].items()}
    assert result["ranking"][0]["assignment"] == winner
    ranking = pd.read_csv(v3.REPORT_DIR / "assignment_ranking.csv")
    agents = list(v3.FREIGHT_AGENTS)
    assert len(ranking) == 5040
    sequences = {tuple(row) for row in ranking[agents].itertuples(index=False)}
    assert len(sequences) == 5040 and all(sorted(s) == [1, 2, 3, 4, 5, 6, 7] for s in sequences)
    assert {a: int(ranking.iloc[0][a]) for a in agents} == winner
    assert (ranking["covered_once"].diff().dropna() <= 0).all()


def test_meaningful_cells_and_rare_residuals_follow_the_one_percent_rule():
    part3 = proposal()["part_3_demand"]
    residuals = {(r["city"], r["candidate"]) for r in part3["rare_residuals"]}
    for city, row in part3["demand_by_city_and_candidate"].items():
        for candidate, cell in row["by_candidate"].items():
            assert cell["meaningful"] is (cell["orders"] * 100 >= row["orders"])
            assert cell["meaningful"] is ((city, int(candidate)) not in residuals)
    assert all(r["share_pct"] < 1 for r in part3["rare_residuals"])
    assert "not a mathematical derivation" in part3["threshold"] and "3%" in part3["threshold"]


# ── part 5 ──────────────────────────────────────────────────────────────────

def test_the_assignment_is_a_bijection_and_the_parcel_corner_is_production():
    agents = {row["agent_id"]: row for row in proposal()["agents"]}
    freight = [agents[a]["calibration_reading_id"] for a in v3.FREIGHT_AGENTS]
    assert len(set(freight)) == 7 and set(freight) == {rid for rid, _ in freight_candidates.CANDIDATES.values()}
    production = pd.read_csv(settings.AGENT_CALIBRATION_PATH).set_index("agent_id")
    for agent_id in v3.PARCEL_AGENTS:
        row, prod = agents[agent_id], production.loc[agent_id]
        assert row["calibration_reading_id"] == prod["calibration_reading_id"]
        assert (row["fixed_charge_eur"], row["alpha_eur_per_kg"], row["max_shipment_weight_kg"]) == (
            prod["fixed_charge_eur"], prod["alpha_eur_per_kg"], prod["max_shipment_weight_kg"])


def test_no_agent_wins_outside_its_assigned_domain():
    result = report("v3")
    assert result["weight_domain_violations"] == 0
    for agent_id, heaviest in result["logistics_heaviest_win_kg"].items():
        domain = result["logistics_weight_domains_kg"][agent_id]
        assert result["logistics_lightest_win_kg"][agent_id] >= domain["min"]
        assert heaviest <= domain["max"] if domain["max_inclusive"] else heaviest < domain["max"]


def test_the_comparison_is_recomputed_from_the_reports():
    agent_ids = sorted(v3.FREIGHT_AGENTS + v3.PARCEL_AGENTS)
    stored = comparison()["comparisons"]
    assert stored["v3_against_production"] == json.loads(json.dumps(
        proposal_validation.compare(report("production"), report("v3"), agent_ids)))
    assert stored["v3_against_specialty_proposal"] == json.loads(json.dumps(
        proposal_validation.compare(report("specialty_proposal_on_file"), report("v3"), agent_ids)))
    assert comparison()["specialty_proposal_rediagnosis_matches_file"] is True


def test_production_is_untouched_and_v3_orders_differ_only_in_budget():
    assert comparison()["production_files_unchanged"] is True
    assert proposal()["production_files_sha256_at_proposal_time"] == proposal_validation.production_hashes()
    assert "min_shipment_weight_kg" not in pd.read_csv(settings.AGENT_CALIBRATION_PATH).columns
    production = pd.read_csv(settings.V5_DIR / "orders.csv")
    variant = pd.read_csv(v3.DATA_DIR / "orders.csv")
    assert variant.drop(columns=["budget_usd"]).equals(production.drop(columns=["budget_usd"]))
    assert comparison()["evaluation_set"] == "data/synthetic/v5/orders.csv"


def test_tied_price_groups_split_contested_orders_by_winner():
    agents = {a: {"fixed_charge_eur": f, "alpha_eur_per_kg": 0.3, "beta_eur_per_tonne_km": 0.07,
                  "fuel_surcharge_series_id": "s", "oversized_above_kg": None, "oversized_fee_eur": None,
                  "min_shipment_weight_kg": 0.0, "max_shipment_weight_kg": 25.0, "max_weight_inclusive": True}
              for a, f in (("A", 5.9), ("B", 5.9), ("C", 8.9))}
    records = pd.DataFrame({"logistics_winner": ["A", "A", "B", "C", None],
                            "logistics_eligible": [("A", "B"), ("A", "B", "C"), ("B",), ("A", "B", "C"), ("A", "B")]})
    groups = diagnostics.tied_price_groups(SimpleNamespace(route=SimpleNamespace(agents=agents)), {"records": records})
    assert groups == [{"agents": ["A", "B"], "contested_orders": 2, "wins": {"A": 2, "B": 0},
                       "one_agent_wins_every_contested_order": True}]
    v3_groups = comparison()["tied_price_groups"]["v3"]
    assert all(set(g["agents"]).isdisjoint(v3.FREIGHT_AGENTS) for g in v3_groups)


def test_the_proposal_states_it_awaits_review():
    assert "awaiting review" in proposal()["status"] and "awaiting review" in comparison()["status"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} calibration proposal v3 unit tests passed")


if __name__ == "__main__":
    main()
