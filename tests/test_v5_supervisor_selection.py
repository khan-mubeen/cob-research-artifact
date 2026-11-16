"""unit tests for v5 supervisor chain selection: full enumeration, guaranteed inclusion, the
deterministic tie-break and the evaluator's metric set.
runnable directly (python ...) or under pytest.

the selection rule is restated here from the design document rather than imported, so the label
cannot validate itself: a test that called chain_priority_key would agree with the generator even
if both were wrong. what is imported is the candidate construction, which is what these tests are
checking the labels against
"""

import json
import random
import re
from pathlib import Path

import config
from src.common.bidding_formulas import chain_from_bids, chain_priority_key, decide_chain_selection
from src.experiments.v5 import evaluate
from src.experiments.v5.chain_candidates import agents_of, enumerate_chains, present
from src.experiments.v5.common import FINETUNE_V5, Market, load_corpus, stage_rfqs
from src.dataset.context import for_version

ROOT = Path(config.PROJECT_ROOT)
V5 = for_version("v5")
SPLITS = ("train", "validation", "test")
CANDIDATE_LINE = re.compile(r"candidate (\d+): .*total_cost_usd=([\d.]+), total_completion_days=([\d.]+)")
SAMPLE_PER_SPLIT = 40
SAMPLE_SEED = 20260917
_CACHE = {}


def world():
    if "world" not in _CACHE:
        _CACHE["world"] = (load_corpus(), Market())
    return _CACHE["world"]


def orders_by_id():
    if "orders" not in _CACHE:
        _CACHE["orders"] = {o["order_id"]: o for o in world()[0]["orders"]}
    return _CACHE["orders"]


def rows_of(split):
    key = f"rows:{split}"
    if key not in _CACHE:
        with open(FINETUNE_V5 / "supervisor" / f"selection_{split}.jsonl", encoding="utf-8") as f:
            _CACHE[key] = [json.loads(line) for line in f]
    return _CACHE[key]


def sample(split, n=SAMPLE_PER_SPLIT):
    rows = rows_of(split)
    return random.Random(SAMPLE_SEED).sample(rows, min(n, len(rows)))


def restated_key(chain, order):
    """the selection rule as docs/agent_logic_design.md states it, written out here

    urgent minimises days then cost, flexible cost then days, normal the equal-weighted blend of
    cost/budget, days/deadline and compound risk/defect cap, then cost, then days. cost and days
    are compared at two decimals, and an exact tie falls to the agent-id tuple"""
    cost = chain["material_offer"] + chain["manufacturer_offer"] + chain["logistics_offer"]
    days = (chain["material_completion_days"] + chain["manufacturer_completion_days"]
            + chain["logistics_completion_days"])
    survives = 1.0
    for key in ("material_defect_probability", "manufacturer_defect_probability", "logistics_damage_probability"):
        survives *= 1.0 - float(chain[key])
    risk = 1.0 - survives
    agents = (chain["material_agent_id"], chain["manufacturer_agent_id"], chain["logistics_agent_id"])
    rounded = (round(cost, 2), round(days, 2))
    if order["priority"] == "urgent":
        return (rounded[1], rounded[0], agents)
    if order["priority"] == "flexible":
        return (rounded[0], rounded[1], agents)
    budget, deadline, cap = order["budget_usd"], order["days_until_deadline"], order["max_defect_rate"]
    score = ((cost / budget if budget > 0 else 1.0) / 3 + (days / deadline if deadline > 0 else 1.0) / 3
             + (risk / cap if cap > 0 else 1.0) / 3)
    return (score, rounded[0], rounded[1], agents)


def true_winner(order_row):
    """the best chain over every eligible combination, found without the generator's help"""
    corpus, market = world()
    rfqs = stage_rfqs(order_row, corpus, market, V5)
    if rfqs is None:
        return None, [], None
    remaining = rfqs["tailoring"]["days_until_deadline"]
    order = {"order_id": order_row["order_id"], "priority": order_row["priority"],
             "budget_usd": float(order_row["budget_usd"]), "days_until_deadline": remaining,
             "max_defect_rate": float(order_row["max_defect_rate"]), **V5.as_dict()}
    chains, _ = enumerate_chains(order_row, corpus, market, rfqs)
    feasible = [c for c in chains if c["total_cost"] <= order["budget_usd"] and c["total_days"] <= remaining]
    if not feasible:
        return None, feasible, order
    best = min(feasible, key=lambda c: restated_key(chain_from_bids(c), order))
    return best, feasible, order


def test_full_enumeration_picks_the_chain_the_audit_found_and_the_corpus_agrees():
    # the two saved examples from prompt-reports/supervisor_chain_selection_audit_20260916_171444.md
    expected = {"ORD-06189": ["M11", "T1", "L02"], "ORD-10431": ["M05", "T4", "L02"]}
    saved = {r["order_id"]: r for split in SPLITS for r in rows_of(split) if r["order_id"] in expected}
    for order_id, chain in expected.items():
        best, _, _ = true_winner(orders_by_id()[order_id])
        assert list(agents_of(best)) == chain, order_id
        assert saved[order_id]["label_chain"] == chain, order_id


def test_every_sampled_row_is_labelled_with_the_full_enumeration_winner():
    checked = 0
    for split in SPLITS:
        for row in sample(split):
            best, feasible, _ = true_winner(orders_by_id()[row["order_id"]])
            assert best is not None, row["order_id"]
            assert list(agents_of(best)) == row["label_chain"], row["order_id"]
            assert row["n_chains_feasible"] == len(feasible), row["order_id"]
            checked += 1
    assert checked >= 3 * SAMPLE_PER_SPLIT - 3


def test_the_labelled_index_points_at_the_winner_in_every_row_of_every_split():
    for split in SPLITS:
        for row in rows_of(split):
            shown = CANDIDATE_LINE.findall(row["user"])
            index = json.loads(row["completion"])["selected_candidate"]
            assert len(shown) == row["n_candidates"], row["order_id"]
            assert 0 <= index < len(shown), row["order_id"]
            position, cost, days = shown[index]
            assert int(position) == index, row["order_id"]
            assert float(cost) == row["label_total_cost"], row["order_id"]
            assert float(days) == row["label_total_days"], row["order_id"]


def test_the_winner_is_never_missing_from_the_shown_list_even_when_the_cap_binds():
    capped = [row for split in SPLITS for row in sample(split) if row["n_chains_feasible"] > row["n_candidates"]]
    assert capped, "no sampled row had more feasible chains than shown places"
    for row in capped:
        best, feasible, order = true_winner(orders_by_id()[row["order_id"]])
        shown = [(float(c), float(d)) for _, c, d in CANDIDATE_LINE.findall(row["user"])]
        assert (row["label_total_cost"], row["label_total_days"]) in shown, row["order_id"]
        assert list(agents_of(best)) == row["label_chain"], row["order_id"]


def test_the_tie_break_is_invariant_to_the_order_the_candidates_are_shuffled_in():
    order = {"order_id": "ORD-TIE", "priority": "urgent", "budget_usd": 10000.0, "days_until_deadline": 10.0,
             "max_defect_rate": 0.05, **V5.as_dict()}

    def chain(material, manufacturer, logistics, cost, days):
        return {"material_agent_id": material, "manufacturer_agent_id": manufacturer,
                "logistics_agent_id": logistics, "material_offer": cost / 3, "manufacturer_offer": cost / 3,
                "logistics_offer": cost / 3, "material_completion_days": days / 3,
                "manufacturer_completion_days": days / 3, "logistics_completion_days": days / 3,
                "material_defect_probability": 0.01, "manufacturer_defect_probability": 0.01,
                "logistics_damage_probability": 0.01}

    tied = [chain("M03", "T2", "L04", 900.0, 6.0), chain("M01", "T9", "L01", 900.0, 6.0),
            chain("M07", "T1", "L02", 900.0, 6.0), chain("M02", "T2", "L03", 1500.0, 8.0)]
    winners, keys = set(), set()
    rng = random.Random(7)
    for _ in range(25):
        shuffled = tied[:]
        rng.shuffle(shuffled)
        selection = decide_chain_selection(shuffled, order)
        winners.add((selection["material_agent_id"], selection["manufacturer_agent_id"],
                     selection["logistics_agent_id"]))
        keys.add(chain_priority_key(min(shuffled, key=lambda c: chain_priority_key(c, order)), order))
    assert winners == {("M01", "T9", "L01")}, winners     # lowest agent-id tuple of the three tied chains
    assert len(keys) == 1


def test_a_real_order_gives_the_same_winner_however_its_candidates_are_ordered():
    row = next(r for split in SPLITS for r in sample(split))
    _, feasible, order = true_winner(orders_by_id()[row["order_id"]])
    chains = [chain_from_bids(c) for c in feasible]
    rng = random.Random(11)
    picks = set()
    for _ in range(10):
        shuffled = chains[:]
        rng.shuffle(shuffled)
        selection = decide_chain_selection(shuffled, order)
        picks.add((selection["material_agent_id"], selection["manufacturer_agent_id"],
                   selection["logistics_agent_id"]))
    assert picks == {tuple(row["label_chain"])}


def test_the_tie_break_never_overrides_a_genuinely_better_chain():
    order = {"order_id": "ORD-NOTIE", "priority": "flexible", "budget_usd": 10000.0, "days_until_deadline": 10.0,
             "max_defect_rate": 0.05, **V5.as_dict()}
    cheap = {"material_agent_id": "M99", "manufacturer_agent_id": "T99", "logistics_agent_id": "L99",
             "material_offer": 100.0, "manufacturer_offer": 100.0, "logistics_offer": 100.0,
             "material_completion_days": 1.0, "manufacturer_completion_days": 1.0, "logistics_completion_days": 1.0,
             "material_defect_probability": 0.01, "manufacturer_defect_probability": 0.01,
             "logistics_damage_probability": 0.01}
    dear = {**cheap, "material_agent_id": "M01", "manufacturer_agent_id": "T01", "logistics_agent_id": "L01",
            "material_offer": 500.0}
    selection = decide_chain_selection([dear, cheap], order)
    # the lowest agent-id tuple loses because it costs more, the tie-break is not reached
    assert selection["material_agent_id"] == "M99"


def test_presentation_places_the_winner_first_before_any_distractor_is_chosen():
    row = next(r for split in SPLITS for r in sample(split) if r["n_chains_feasible"] > r["n_candidates"])
    best, feasible, _ = true_winner(orders_by_id()[row["order_id"]])
    winner = agents_of(best)
    for seed in range(5):
        shown = present(feasible, winner, random.Random(seed))
        assert len(shown) == min(row["n_candidates"], len(feasible))
        assert winner in [agents_of(c) for c in shown], seed


def test_the_evaluator_reports_parsing_and_a_usable_index_separately():
    rows, prompt = evaluator_fixture()
    summary, records = evaluate.score_supervisor(rows, ["{\"selected_candidate\": 0}", "not json at all",
                                                        "{\"selected_candidate\": 99}"])
    assert summary["json_parse_pct"] == 66.67 and summary["json_valid_pct"] == 33.33
    assert summary["usable_index_pct"] == summary["json_valid_pct"]
    assert summary["invalid_or_out_of_range_pct"] == 66.67
    assert records[2]["json_parsed"] is True and records[2]["valid_json"] is False
    assert prompt in rows[0]["user"]


def test_the_evaluator_separates_a_secondary_rule_error_from_a_primary_miss():
    rows, _ = evaluator_fixture()
    # candidate 0 is the label: 6.0 days at 900 usd. candidate 1 ties on days and costs more,
    # candidate 2 takes longer
    summary, records = evaluate.score_supervisor(rows[:1] * 2, ["{\"selected_candidate\": 1}",
                                                               "{\"selected_candidate\": 2}"])
    assert [r["miss_kind"] for r in records] == ["secondary_error", "primary_miss"]
    assert summary["secondary_error_pct"] == 50.0 and summary["primary_miss_pct"] == 50.0
    assert summary["by_strategy"]["urgent"]["secondary_error"] == 1


def test_the_evaluator_reports_deviations_feasibility_and_an_unresolvable_tie():
    rows, _ = evaluator_fixture()
    summary, records = evaluate.score_supervisor(rows[:1] * 2, ["{\"selected_candidate\": 3}",
                                                               "{\"selected_candidate\": 1}"])
    assert records[0]["miss_kind"] == "indistinguishable_tie"     # same printed cost and days as the label
    assert summary["indistinguishable_tie_pct"] == 50.0
    assert records[1]["cost_dev_pct"] > 0 and records[1]["days_dev_pct"] == 0.0
    assert summary["feasibility_violation_pct"] == 0.0
    assert summary["mean_abs_cost_dev_pct"] > 0


def test_the_selection_key_still_reads_the_supervisor_summary():
    from src.experiments.v5 import epoch_selection
    rows, _ = evaluator_fixture()
    summary, _ = evaluate.score_supervisor(rows[:1], ["{\"selected_candidate\": 0}"])
    metrics = epoch_selection.selection_metrics("supervisor", summary)
    assert metrics["key"] == [100.0, 100.0, -0.0] or metrics["key"][0] == 100.0


def test_the_metadata_records_full_enumeration_and_guaranteed_inclusion():
    meta = json.loads((FINETUNE_V5 / "supervisor" / "metadata.json").read_text(encoding="utf-8"))
    assert "no widening and no shortlist" in meta["candidate_source"]
    assert "placed first and unconditionally" in meta["presented_candidates"]
    assert "agent-id tuple" in meta["tie_break"]
    assert meta["chains_enumerated_per_order"]["max"] >= meta["max_candidates_per_prompt"]
    assert sum(meta["counts"].values()) == sum(len(rows_of(s)) for s in SPLITS)


def evaluator_fixture():
    """one hand-built supervisor row whose four candidates cover every outcome the metrics name"""
    prompt = (
        "ORDER\norder_id: ORD-FIXTURE\npriority: urgent\nbudget (USD): 5000.0\n"
        "days remaining until deadline: 10.0\nmax acceptable defect or damage probability: 0.05\n\n"
        "FEASIBLE CHAIN CANDIDATES\n"
        "candidate 0: material(cost_usd=300.0, days=2.0, defect_p=0.01), "
        "manufacturing(cost_usd=300.0, days=2.0, defect_p=0.01), "
        "logistics(cost_usd=300.0, days=2.0, damage_p=0.01), "
        "total_cost_usd=900.0, total_completion_days=6.0\n"
        "candidate 1: material(cost_usd=500.0, days=2.0, defect_p=0.01), "
        "manufacturing(cost_usd=300.0, days=2.0, defect_p=0.01), "
        "logistics(cost_usd=300.0, days=2.0, damage_p=0.01), "
        "total_cost_usd=1100.0, total_completion_days=6.0\n"
        "candidate 2: material(cost_usd=300.0, days=4.0, defect_p=0.01), "
        "manufacturing(cost_usd=300.0, days=2.0, defect_p=0.01), "
        "logistics(cost_usd=300.0, days=2.0, damage_p=0.01), "
        "total_cost_usd=900.0, total_completion_days=8.0\n"
        "candidate 3: material(cost_usd=300.0, days=2.0, defect_p=0.01), "
        "manufacturing(cost_usd=300.0, days=2.0, defect_p=0.01), "
        "logistics(cost_usd=300.0, days=2.0, damage_p=0.01), "
        "total_cost_usd=900.0, total_completion_days=6.0\n")
    row = {"order_id": "ORD-FIXTURE", "strategy": "urgent", "user": prompt,
           "completion": json.dumps({"selected_candidate": 0, "reason": "x"})}
    return [row, dict(row), dict(row)], "candidate 3"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 supervisor selection unit tests passed")


if __name__ == "__main__":
    main()
