"""
synthetic edge examples for tailoring and logistics fine-tuning coverage.

the v4 census showed zero dynamic NO_BIDs across all 50,000 orders, so this
builds the missing coverage as a fine-tuning data artifact under
data/finetune/{version}/, orders.csv and the agent csvs stay untouched.

two categories per stage, built from real agent profiles:
- infeasible: completion meaningfully exceeds the deadline, labeled NO_BID
- near_boundary: completion just under (or exactly at) the deadline, labeled
  BID with the honestly computed completion value. this category exists
  because the baseline showed the base model misreporting its completion
  number toward eligibility, refusal-only training would teach overcaution
  instead of honest reporting.

every example is verified against the real rule-based decide functions before
labeling, the same standard as the six fixed baseline cases, and checked for
zero overlap with those cases and within itself.

output: data/finetune/{version}/edge_{stage}_{train,holdout}.jsonl
"""

import json
import math
import random
from collections import defaultdict

from config import DATA_VERSION, PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_logistics_bid,
    decide_manufacturer_bid,
    logistics_eligibility_reason,
    manufacturer_eligibility_reason,
)
from src.common.slm import build_logistics_prompt, build_manufacturer_prompt
from src.experiments.generate_finetune_data import completion_json
from src.experiments.nobid_edge_baseline import CASES as BASELINE_CASES
from src.visualization.data_utils import load_all_data


RANDOM_SEED = 42
HOLDOUT_PER_BUCKET = 2  # per (stage, category)

OUT_DIR = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION

# permissive static fields so the dynamic condition is the only variable
ORDER_BASE = {
    "max_defect_rate": 0.5,
    "min_reputation_score": 0.1,
    "priority": "normal",
}


# deterministic ratio targets, cycled per agent so every manufacturer gets
# NO_BID supervision just above 1.00 and BID supervision spread below it,
# round one trained nothing in (1.00, 1.315) and missed exactly there
BOUNDARY_INFEASIBLE_RATIOS = (1.03, 1.06, 1.10, 1.15)
NEAR_BOUNDARY_RATIOS = (0.90, 0.95, 0.98)


def solve_quantity(cap, workload, days, target_ratio):
    """quantity whose completion/deadline ratio lands on target_ratio,
    exact at 1.00, within rounding of it elsewhere"""
    if target_ratio == 1.0:
        return int(round(cap * days - workload))
    return max(1, int(round(cap * days * target_ratio - workload)))


def tailoring_cases(manufacturers):
    """per agent: far-infeasible (unchanged from round one), two deterministic
    boundary-infeasible ratios, one exact at-boundary bid, one near-boundary
    bid with per-agent varied ratio"""
    cases = []
    rng = random.Random(RANDOM_SEED)
    for idx, agent in enumerate(manufacturers.to_dict("records")):
        cap = agent["daily_capacity_units"]
        workload = agent["current_workload_units"]
        products = str(agent["product_types"]).split("|")
        days = 3 + (idx % 4)  # deadlines 3..6 for variety

        # far infeasible: est completion around 1.3-1.9x the deadline,
        # kept as in round one, realized 1.32-1.84 there with no issues
        factor = 1.3 + rng.random() * 0.6
        qty_inf = max(1, math.ceil(cap * days * factor - workload))

        wanted = [("infeasible", qty_inf, products[idx % len(products)], "")]

        ratio_a = BOUNDARY_INFEASIBLE_RATIOS[idx % 4]
        ratio_b = BOUNDARY_INFEASIBLE_RATIOS[(idx + 2) % 4]
        for tag, ratio in (("A", ratio_a), ("B", ratio_b)):
            qty = solve_quantity(cap, workload, days, ratio)
            # rounding must not drop the realized ratio to or below 1.00
            while (workload + qty) / cap <= days:
                qty += 1
            wanted.append(("boundary_infeasible", qty, products[idx % len(products)], tag))

        qty_at = solve_quantity(cap, workload, days, 1.0)
        wanted.append(("at_boundary", qty_at, products[(idx + 1) % len(products)], ""))

        nb_ratio = NEAR_BOUNDARY_RATIOS[idx % 3]
        qty_nb = solve_quantity(cap, workload, days, nb_ratio)
        while qty_nb > 1 and (workload + qty_nb) / cap >= days:
            qty_nb -= 1
        wanted.append(("near_boundary", qty_nb, products[(idx + 1) % len(products)], ""))

        for category, qty, product, tag in wanted:
            if qty < 1:
                continue
            suffix = f"-{tag}" if tag else ""
            cases.append({
                "stage": "tailoring",
                "category": category,
                "agent": agent,
                "order": {
                    **ORDER_BASE,
                    "order_id": f"SYNEDGE-T-{category.upper()[:3]}{suffix}-{agent['agent_id']}",
                    "product_id": product,
                    "quantity": qty,
                    "days_until_deadline": days,
                },
            })
    return cases


def logistics_cases(logistics, real_keys=frozenset()):
    """per agent: one infeasible and one near-boundary intercity delivery,
    deadline derived from the agent's own intercity route days

    deliver_to prefers a region whose (agent, region, days) tuple does not
    appear among the real training examples, any intercity region gives the
    same route days so the ratio is unaffected"""
    cases = []
    rng = random.Random(RANDOM_SEED + 1)
    for agent in logistics.to_dict("records"):
        regions = str(agent["service_regions"]).split("|")
        intercity = [r for r in regions if r != agent["location"]]
        if not intercity:
            continue
        route = float(agent["default_completion_intercity_days"])
        preferred = intercity[rng.randrange(len(intercity))]
        weight = round(200 + rng.random() * 1300, 1)

        def pick_region(days, first_choice=preferred):
            options = [first_choice] + [r for r in intercity if r != first_choice]
            for region in options:
                if ("logistics", agent["agent_id"], region, days) not in real_keys:
                    return region
            return None  # every region already appears in real data at this deadline

        # infeasible: the tightest deadline the route still cannot meet
        days_inf = max(1, math.ceil(route) - (1 if route == int(route) else 0) - 0)
        if not route > days_inf:
            days_inf = max(1, int(route) - 1)
        # near boundary: smallest whole-day deadline the route just fits
        days_nb = math.ceil(route)

        # boundary infeasible: only half-day intercity routes can realize a
        # ratio below 1.333 at all (3.5/3=1.167, 2.5/2=1.25), round one
        # trained neither and the learned threshold sat right beside 1.167
        wanted = [("infeasible", days_inf), ("near_boundary", days_nb)]
        if route != int(route) and route > 1.5:
            wanted.append(("boundary_infeasible", int(route)))

        for category, days in wanted:
            if category in ("infeasible", "boundary_infeasible") and not route > days:
                continue  # agent too fast to ever miss a >=1 day deadline
            region = pick_region(days)
            if region is None:
                # only feasible boundary cases can hit this, and there real
                # data already contains identically-parameterized BID
                # examples, a synthetic copy would add nothing but a
                # values-level collision
                continue
            cases.append({
                "stage": "logistics",
                "category": category,
                "agent": agent,
                "order": {
                    **ORDER_BASE,
                    "order_id": f"SYNEDGE-L-{category.upper()[:3]}-{agent['agent_id']}",
                    "deliver_to": region,
                    "shipment_weight_kg": weight,
                    "days_until_deadline": days,
                },
            })
    return cases


def load_real_param_keys():
    """parameter tuples of every real train and holdout example, edge cases
    must never coincide with any of them at the values level"""
    import re
    keys = set()
    for stage in ("tailoring", "logistics"):
        for split in ("train", "holdout"):
            path = OUT_DIR / f"{stage}_{split}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    user = r["user"]
                    days = int(re.search(r"^days until deadline: (\d+)", user, re.M).group(1))
                    if stage == "tailoring":
                        keys.add((stage, r["agent_id"],
                                  re.search(r"^product: (\S+)", user, re.M).group(1),
                                  int(re.search(r"^quantity: (\d+)", user, re.M).group(1)),
                                  days))
                    else:
                        keys.add((stage, r["agent_id"],
                                  re.search(r"^deliver to: (\S+)", user, re.M).group(1),
                                  days))
    return keys


def _param_key(stage, agent_id, order):
    if stage == "tailoring":
        return (stage, agent_id, order["product_id"], order["quantity"], order["days_until_deadline"])
    return (stage, agent_id, order["deliver_to"], order["days_until_deadline"])


def main():
    materials, manufacturers, logistics, products, orders = load_all_data()

    real_keys = load_real_param_keys()
    cases = tailoring_cases(manufacturers) + logistics_cases(logistics, real_keys)

    baseline_keys = {_param_key(c["stage"], c["agent_id"], c["order"]) for c in BASELINE_CASES}
    seen_keys = set()
    records = []

    for case in cases:
        agent, order, stage, category = case["agent"], case["order"], case["stage"], case["category"]

        if stage == "tailoring":
            eligibility_fn, rule_fn, prompt_fn, defect_key = (
                manufacturer_eligibility_reason, decide_manufacturer_bid,
                build_manufacturer_prompt, "defect_probability")
        else:
            eligibility_fn, rule_fn, prompt_fn, defect_key = (
                logistics_eligibility_reason, decide_logistics_bid,
                build_logistics_prompt, "damage_probability")

        # verified against the real rule path before labeling, same standard
        # as the fixed baseline cases
        assert eligibility_fn(agent, order) is None, f"{order['order_id']}: statically ineligible"
        rule = rule_fn(agent, order)
        if category in ("infeasible", "boundary_infeasible"):
            assert rule["decision"] == "NO_BID" and rule["debug_reason"] == "completion_exceeds_deadline", \
                f"{order['order_id']}: expected dynamic NO_BID, got {rule}"
        else:  # at_boundary and near_boundary must both be feasible bids
            assert rule["decision"] == "BID", f"{order['order_id']}: expected BID, got {rule}"

        key = _param_key(stage, agent["agent_id"], order)
        assert key not in baseline_keys, f"{order['order_id']}: overlaps a fixed baseline case"
        assert key not in seen_keys, f"{order['order_id']}: duplicate within the new set"
        assert key not in real_keys, f"{order['order_id']}: coincides with a real train/holdout example"
        seen_keys.add(key)

        system_prompt, user_prompt = prompt_fn(agent, order)
        records.append({
            "order_id": order["order_id"],
            "agent_id": agent["agent_id"],
            "stage": stage,
            "category": category,
            "rule_decision": rule["decision"],
            "system": system_prompt,
            "user": user_prompt,
            "completion": completion_json(stage, order["order_id"], agent["agent_id"],
                                          rule, agent.get(defect_key)),
        })

    # seeded holdout pick per (stage, category), disjoint from train by id
    rng = random.Random(RANDOM_SEED)
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[(r["stage"], r["category"])].append(r)
    holdout_ids = set()
    for bucket, rows in sorted(by_bucket.items()):
        # tiny buckets stay fully in training, the boundary stress set covers
        # their evaluation instead of starving the training signal
        n_hold = HOLDOUT_PER_BUCKET if len(rows) >= 6 else (1 if len(rows) >= 3 else 0)
        picked = rng.sample(rows, n_hold)
        holdout_ids.update(r["order_id"] for r in picked)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    handles = {}
    try:
        for stage in ("tailoring", "logistics"):
            for split in ("train", "holdout"):
                handles[(stage, split)] = open(OUT_DIR / f"edge_{stage}_{split}.jsonl",
                                               "w", encoding="utf-8")
        for r in records:
            split = "holdout" if r["order_id"] in holdout_ids else "train"
            handles[(r["stage"], split)].write(json.dumps(r) + "\n")
            counts[(r["stage"], r["category"], split)] += 1
    finally:
        for f in handles.values():
            f.close()

    print(f"{len(records)} verified edge examples, zero overlap with the "
          f"{len(BASELINE_CASES)} fixed baseline cases and no internal duplicates")
    print(f"\n{'stage':10s} {'category':20s} {'train':>6s} {'holdout':>8s}")
    categories = ("infeasible", "boundary_infeasible", "at_boundary", "near_boundary")
    for stage in ("tailoring", "logistics"):
        for category in categories:
            n_train = counts[(stage, category, "train")]
            n_hold = counts[(stage, category, "holdout")]
            if n_train or n_hold:
                print(f"{stage:10s} {category:20s} {n_train:6d} {n_hold:8d}")
    print(f"\nwritten to {OUT_DIR} as edge_{{stage}}_{{split}}.jsonl")


if __name__ == "__main__":
    main()
