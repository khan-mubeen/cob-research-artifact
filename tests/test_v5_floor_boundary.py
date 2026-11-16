"""unit tests for the v5 reverse-auction floor-boundary diagnostic set.
runnable directly (python ...) or under pytest.

the round() against ceil() mismatch found once in this project gave the one-cent-below case
contradictory labels because the prompt's floor and the label's floor were two calculations.
these tests hold the v5 set to one: every row's printed floor, displayed floor and case offset
equal reverse_auction.agent_minimum_bid recomputed on the row's own rfq, the label is the rule's,
and neither generator computes a floor any other way
"""

import ast
import json
import re
from pathlib import Path

import config
from src.common.bidding_formulas import eligibility_reason
from src.common.reverse_auction import agent_minimum_bid, decide_round_bid, next_round_offer
from src.dataset.context import for_version
from src.experiments.v5 import generate_floor_boundary as floor
from src.experiments.v5.common import Market, load_corpus, read_jsonl, stage_rfqs

ROOT = Path(config.PROJECT_ROOT)
FLOOR_LINE = re.compile(r"^your minimum_bid \(USD\): (\S+)$", re.M)
BEST_LINE = re.compile(r"^current_best_bid \(USD\): (\S+)$", re.M)
_CACHE = {}


def rows():
    if "rows" not in _CACHE:
        _CACHE["rows"] = {(stage, case): list(read_jsonl(floor.OUT_DIR / f"{stage}_{case}.jsonl"))
                          for stage in ("material", "tailoring", "logistics") for case in floor.CASES.values()}
    return _CACHE["rows"]


def test_nine_files_of_132_rows_1188_in_all():
    counts = {key: len(items) for key, items in rows().items()}
    assert len(counts) == 9 and set(counts.values()) == {132}, counts
    meta = json.loads((floor.OUT_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert meta["total_rows"] == 1188 and meta["role"] == "diagnostic"


def test_every_prompt_floor_is_the_shared_floor_on_the_rows_own_rfq():
    corpus, market, context = load_corpus(), Market(), for_version("v5")
    orders = {o["order_id"]: o for o in corpus["orders"]}
    agents = {stage: {a["agent_id"]: a for a in corpus[stage]} for stage in ("material", "tailoring", "logistics")}
    for (stage, case), items in rows().items():
        for row in items:
            rfq = stage_rfqs(orders[row["order_id"]], corpus, market, context)[stage]
            agent = agents[stage][row["agent_id"]]
            assert eligibility_reason(stage, agent, rfq) is None
            shared = agent_minimum_bid(stage, agent, rfq)
            printed = float(FLOOR_LINE.search(row["user"]).group(1))
            assert printed == shared == row["displayed_minimum_bid"], (row["order_id"], printed, shared)
            best = float(BEST_LINE.search(row["user"]).group(1))
            offer = next_round_offer(best, row["platform_minimum_decrement"])
            assert offer == row["mandated_next_offer"]
            expected = {"below_floor": round(shared - 0.01, 2), "at_floor": shared,
                        "above_floor": round(shared * 1.01, 2)}[case]
            assert abs(offer - expected) < 1e-9, (row["order_id"], case, offer, expected)
            seen = {**rfq, "round_number": row["round_number"], "current_best_bid": best,
                    "platform_minimum_decrement": row["platform_minimum_decrement"]}
            decision = decide_round_bid(stage, agent, seen)
            assert json.loads(row["completion"])["decision"] == decision["decision"] == floor.EXPECTED[case]
            assert (decision["decision"] == "BID") == (offer >= shared)


def _called_names(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))}


def _function_source(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_neither_generator_computes_a_floor_except_through_agent_minimum_bid():
    diagnostic = ROOT / "src" / "experiments" / "v5" / "generate_floor_boundary.py"
    corpus_generator = ROOT / "src" / "experiments" / "v5" / "generate_reverse_auction_data.py"
    for path in (diagnostic, corpus_generator):
        names = _called_names(path)
        assert "ceil_to_cents" not in names and "stage_price_terms" not in names, path.name
    cases = _function_source(corpus_generator, "floor_cases")
    called = {n.func.id for n in ast.walk(cases) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "agent_minimum_bid" in called
    assert "floor_cases" in _called_names(diagnostic) and "agent_minimum_bid" in _called_names(diagnostic)


def test_the_diagnostic_rows_come_from_test_orders_and_repeat_no_test_row():
    from src.experiments.v5.common import read_split
    split = read_split()
    for (stage, _), items in rows().items():
        corpus_pairs = floor.corpus_floor_pairs(stage)
        for row in items:
            assert split[row["order_id"]] == "test"
            assert (row["order_id"], row["agent_id"]) not in corpus_pairs


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 floor-boundary unit tests passed")


if __name__ == "__main__":
    main()
