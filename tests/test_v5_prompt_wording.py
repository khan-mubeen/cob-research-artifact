"""unit tests for the v5 prompt wording harmonization.
runnable directly (python ...) or under pytest.

the harmonized style is the reverse-auction builder's own rewrite, extracted literally. these tests
hold the pieces to that: the recorded mapping is the builder's, every v5 stage prompt is worded
whatever the flag says, reverse-auction prompts still rebuild to the accepted rows, v4 prompts stay
symbolic, the independent rewording rule behaves on hand-made text, and the corpus verification
and promotion records say what they must
"""

import json
from pathlib import Path

import config
from src.common import bidding_formulas, slm
from src.common.reverse_auction import auction_fields
from src.dataset.context import for_version
from src.experiments.v5 import verify_prompt_wording as verify

ROOT = Path(config.PROJECT_ROOT)
V5 = for_version("v5")
BUILDERS = {"material": slm.build_material_prompt, "tailoring": slm.build_manufacturer_prompt,
            "logistics": slm.build_logistics_prompt}
_CACHE = {}


def wording():
    return verify.Wording(verify.load_mapping())


def task(user):
    s, e = verify.task_bounds(user)
    return user[s:e]


def v5_world():
    if "world" not in _CACHE:
        from src.experiments.v5.common import Market, load_corpus
        _CACHE["world"] = (load_corpus(), Market())
    return _CACHE["world"]


def test_the_recorded_mapping_is_the_reverse_auction_builders_literal_rewrite():
    recorded = json.loads(verify.MAPPING_PATH.read_text(encoding="utf-8"))
    assert recorded["mapping"] == slm._OPERATOR_WORDS
    assert recorded["rule_as_written"] == slm._OPERATOR_RE.pattern
    assert verify.Wording(recorded["mapping"]).pattern.pattern.replace("max\\(|", "") \
        .replace("(?<= )(?:", "").rstrip(")").replace("(?= ", "") != ""


def test_the_independent_rule_rewords_exactly_as_the_builder_does_on_hand_made_text():
    text = ("\nTASK\n1. Check a <= b and c >= d, e < f, g > h.\n   y = p x (1 + q) - r / s\n"
            "   offer = max(a, b)\n   priority_markup: urgent=0.30, flexible=-0.15\n   if a == b: z\n")
    w = wording()
    assert w.reword(text) == slm.word_formulas(text)
    assert not w.bare_operators(w.reword(text))
    assert w.bare_operators(text) and w.phrases(text) == 0
    assert "urgent=0.30, flexible=-0.15" in w.reword(text) and "a == b" in w.reword(text)


def test_every_v5_stage_prompt_is_worded_whatever_the_flag_says():
    corpus, market = v5_world()
    from src.experiments.v5.common import stage_rfqs
    w = wording()
    checked = 0
    for order in corpus["orders"][:12]:
        rfqs = stage_rfqs(order, corpus, market, V5)
        for stage, build in BUILDERS.items():
            for agent in corpus[stage]:
                if bidding_formulas.eligibility_reason(stage, agent, rfqs[stage]) is not None:
                    continue
                plain, flagged = build(agent, rfqs[stage]), build(agent, rfqs[stage], worded=True)
                assert plain == flagged
                assert not w.bare_operators(task(plain[1])) and w.phrases(task(plain[1])) > 0
                checked += 1
    assert checked > 50


def test_reverse_auction_prompts_rebuild_to_the_accepted_rows():
    corpus, market = v5_world()
    from src.experiments.v5.common import stage_rfqs
    agents = {stage: {a["agent_id"]: a for a in corpus[stage]} for stage in BUILDERS}
    orders = {o["order_id"]: o for o in corpus["orders"]}
    for stage in BUILDERS:
        with open(ROOT / "data" / "finetune" / "v5" / "reverse_auction" / f"{stage}_validation.jsonl", encoding="utf-8") as f:
            for _, line in zip(range(40), f):
                row = json.loads(line)
                rfq = stage_rfqs(orders[row["order_id"]], corpus, market, V5)[stage]
                seen = {**rfq, "round_number": row["round_number"], "current_best_bid": row["current_best_bid"],
                        "platform_minimum_decrement": row["platform_minimum_decrement"]}
                agent = agents[stage][row["agent_id"]]
                assert BUILDERS[stage](agent, seen, auction=auction_fields(stage, agent, seen), worded=True) == \
                    (row["system"], row["user"])


def test_v4_prompts_are_still_symbolic():
    from tests.golden.v4_baseline import load_v4, stage_rfqs
    v4 = load_v4()
    w = wording()
    order = v4["orders"][0]
    rfqs = stage_rfqs(order, v4["products"])
    for stage, build in BUILDERS.items():
        agent = next(a for a in v4[stage] if bidding_formulas.eligibility_reason(stage, a, rfqs[stage]) is None)
        user = build(agent, rfqs[stage])[1]
        assert w.bare_operators(task(user)) and w.phrases(task(user)) == 0


def test_the_accepted_sealed_and_edge_files_are_worded():
    w = wording()
    for relative in verify.CHANGED:
        with open(ROOT / "data" / "finetune" / "v5" / relative, encoding="utf-8") as f:
            for _, line in zip(range(150), f):
                assert not w.bare_operators(task(json.loads(line)["user"])), relative


def test_the_tier_1_verification_passed_and_the_promotion_is_recorded():
    report = json.loads(verify.REPORT_PATH.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["totals"]["non_prompt_field_mismatches"] == 0
    assert report["totals"]["changed_rows_compared"] == report["totals"]["changed_rows_reworded_exactly"] > 0
    for relative in verify.UNCHANGED_WORDED:
        assert report["files"][relative]["candidate_byte_identical"] is True
    promotion = json.loads((ROOT / "reports" / "v5" / "wording" / "promotion.json").read_text(encoding="utf-8"))
    assert promotion["manifest_verified"] is True
    for relative, digest in promotion["promoted"].items():
        from src.experiments.v5.common import sha256_file
        assert sha256_file(ROOT / "data" / "finetune" / "v5" / relative) == digest
    assert not (ROOT / "data" / "finetune" / ".v5_wording_scratch").exists()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 prompt wording unit tests passed")


if __name__ == "__main__":
    main()
