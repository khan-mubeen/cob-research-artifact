"""Protocol v2 edge training set + locked robustness holdout.

Design rationale (full document: reports/epoch-1/production/protocol/protocol_v2_design.md):
- round one's tailoring edge set had ZERO infeasible coverage in (1.00, 1.315)
  and a feasible-side monoculture (all ten feasible examples at exactly 0.97,
  verified from the preserved files) -> qwen r1 and granite v1 both stayed
  permissive just above the boundary
- round two concentrated 22 of 32 infeasible rows at 1.03-1.15 with x5
  repetition against a 0.90-1.00 feasible side -> over-conservative
- v2 therefore uses a symmetric LADDER of unique examples, both sides of
  1.00, many distances, every manufacturer x every ratio, repetition 1

Tailoring train: per manufacturer, all 15 target ratios
  feasible   0.70 0.80 0.88 0.94 0.97 0.99 1.00
  infeasible 1.01 1.03 1.07 1.12 1.20 1.35 1.60 1.85
deadlines cycle 7..10 days - a range no previous fixed set uses (r2 edges
3..6, stress 3..6, final holdout 3..6), which prevents parameter collisions
by construction and adds novel deadline values.

Logistics train: same categories as round two (evidence: both architectures
already pass logistics at 100%), fresh weights/regions, tuples checked
against real data and every evaluation set.

Locked robustness holdout (NEW, tier 3): 12 manufacturers x 3 classes
(below_feasible, exact_feasible, above_infeasible). Construction deliberately
differs from training: ratios are DRAWN from bands (below [0.90, 0.995],
above [1.005, 1.08], seeded rng), not ladder points; deadlines 11-12 days,
used nowhere else. Frozen and hashed before any v2 training.

Every label is verified through the production rule functions; every
completion is serialized through the production output contract
(completion_json). Zero-overlap is asserted at the parameter-tuple level and
re-checked at the exact-prompt level against every evaluation set.

output: data/finetune/v4/protocol_v2/{edge_tailoring_train.jsonl,
        edge_logistics_train.jsonl, robustness_holdout.jsonl}
report: reports/epoch-1/production/protocol/protocol_v2_data_report.json
"""

import hashlib
import sys
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime

from config import DATA_VERSION, PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_logistics_bid,
    decide_manufacturer_bid,
    logistics_eligibility_reason,
    manufacturer_eligibility_reason,
)
from src.common.slm import build_logistics_prompt, build_manufacturer_prompt
from src.experiments.generate_edge_examples import (
    ORDER_BASE,
    _param_key,
    load_real_param_keys,
)
from src.experiments.generate_finetune_data import completion_json
from src.experiments.nobid_edge_baseline import CASES as BASELINE_CASES
from src.visualization.data_utils import load_all_data

DATA_DIR = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION
OUT_DIR = DATA_DIR / "protocol_v2"
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
from report_paths import report_dir  # noqa: E402
REPORT = report_dir("production", "protocol") / "protocol_v2_data_report.json"

FEASIBLE_RATIOS = (0.70, 0.80, 0.88, 0.94, 0.97, 0.99, 1.00)
INFEASIBLE_RATIOS = (1.01, 1.03, 1.07, 1.12, 1.20, 1.35, 1.60, 1.85)
TRAIN_DAYS = (7, 8, 9, 10)          # cycles per (agent, ratio index)
ROBUST_DAYS = (11, 12)              # used nowhere else in the project
ROBUST_SEED = 4242
LOGISTICS_SEED = 4243
ROBUST_BELOW_BAND = (0.90, 0.995)
ROBUST_ABOVE_BAND = (1.005, 1.08)


def solve_qty(cap, workload, days, target):
    if target == 1.0:
        return int(round(cap * days - workload))
    return max(1, int(round(cap * days * target - workload)))


def eval_set_param_keys():
    """parameter tuples of every fixed evaluation set on disk"""
    keys = set()
    files = ["edge_tailoring_holdout.jsonl", "edge_logistics_holdout.jsonl",
             "boundary_stress.jsonl", "final_boundary_holdout.jsonl"]
    for name in files:
        path = DATA_DIR / name
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                u = r["user"]
                days = int(re.search(r"^days until deadline: (\d+)", u, re.M).group(1))
                if r["stage"] == "logistics":
                    keys.add(("logistics", r["agent_id"],
                              re.search(r"^deliver to: (\S+)", u, re.M).group(1), days))
                else:
                    keys.add(("tailoring", r["agent_id"],
                              re.search(r"^product: (\S+)", u, re.M).group(1),
                              int(re.search(r"^quantity: (\d+)", u, re.M).group(1)), days))
    return keys


def make_tailoring_record(agent, order, category, extra):
    assert manufacturer_eligibility_reason(agent, order) is None, \
        f"{order['order_id']}: statically ineligible"
    rule = decide_manufacturer_bid(agent, order)
    system_prompt, user_prompt = build_manufacturer_prompt(agent, order)
    return rule, {
        "order_id": order["order_id"], "agent_id": agent["agent_id"],
        "stage": "tailoring", "category": category,
        "rule_decision": rule["decision"], **extra,
        "system": system_prompt, "user": user_prompt,
        "completion": completion_json("tailoring", order["order_id"], agent["agent_id"],
                                      rule, agent.get("defect_probability")),
    }


def tailoring_train(manufacturers, forbidden, seen):
    records = []
    all_ratios = [(r, "feasible_ladder") for r in FEASIBLE_RATIOS] + \
                 [(r, "infeasible_ladder") for r in INFEASIBLE_RATIOS]
    for idx, agent in enumerate(manufacturers.to_dict("records")):
        cap, wl = agent["daily_capacity_units"], agent["current_workload_units"]
        products = str(agent["product_types"]).split("|")
        for k, (target, category) in enumerate(all_ratios):
            days = TRAIN_DAYS[(idx + k) % 4]
            product = products[(idx + k) % len(products)]
            qty = solve_qty(cap, wl, days, target)
            if target <= 1.0:
                while qty > 1 and (wl + qty) / cap > days:
                    qty -= 1
            else:
                while (wl + qty) / cap <= days:
                    qty += 1
            if qty < 1:
                continue
            order = {**ORDER_BASE,
                     "order_id": f"EDGEV2-T-{'F' if target <= 1.0 else 'I'}"
                                 f"{str(target).replace('.', '')}-{agent['agent_id']}",
                     "product_id": product, "quantity": qty,
                     "days_until_deadline": days}
            realized = (wl + qty) / cap / days
            rule, rec = make_tailoring_record(
                agent, order, category,
                {"target_ratio": target, "realized_ratio": round(realized, 6)})
            expected = "BID" if target <= 1.0 else "NO_BID"
            assert rule["decision"] == expected, \
                f"{order['order_id']}: rule {rule['decision']}, expected {expected} " \
                f"(realized {realized:.6f})"
            key = _param_key("tailoring", agent["agent_id"], order)
            assert key not in forbidden, f"{order['order_id']}: collides with a protected set"
            assert key not in seen, f"{order['order_id']}: duplicate"
            seen.add(key)
            records.append(rec)
    return records


def logistics_train(logistics, forbidden, seen):
    import math
    records = []
    rng = random.Random(LOGISTICS_SEED)
    for agent in logistics.to_dict("records"):
        regions = str(agent["service_regions"]).split("|")
        intercity = [r for r in regions if r != agent["location"]]
        if not intercity:
            continue
        route = float(agent["default_completion_intercity_days"])
        weight = round(150 + rng.random() * 1400, 1)
        wanted = []
        days_inf = max(1, math.ceil(route) - (1 if route == int(route) else 0))
        if not route > days_inf:
            days_inf = max(1, int(route) - 1)
        if route > days_inf:
            wanted.append(("infeasible", days_inf))
        wanted.append(("near_boundary", math.ceil(route)))
        if route != int(route) and route > 1.5 and route > int(route):
            wanted.append(("boundary_infeasible", int(route)))
        for category, days in wanted:
            region = next((r for r in [intercity[rng.randrange(len(intercity))]] + intercity
                           if ("logistics", agent["agent_id"], r, days) not in forbidden
                           and ("logistics", agent["agent_id"], r, days) not in seen), None)
            if region is None:
                continue
            order = {**ORDER_BASE,
                     "order_id": f"EDGEV2-L-{category.upper()[:3]}-{agent['agent_id']}",
                     "deliver_to": region, "shipment_weight_kg": weight,
                     "days_until_deadline": days}
            assert logistics_eligibility_reason(agent, order) is None
            rule = decide_logistics_bid(agent, order)
            expected = "NO_BID" if category in ("infeasible", "boundary_infeasible") else "BID"
            assert rule["decision"] == expected, f"{order['order_id']}: {rule}"
            key = _param_key("logistics", agent["agent_id"], order)
            seen.add(key)
            system_prompt, user_prompt = build_logistics_prompt(agent, order)
            records.append({
                "order_id": order["order_id"], "agent_id": agent["agent_id"],
                "stage": "logistics", "category": category,
                "rule_decision": rule["decision"],
                "realized_ratio": round(route / days, 6),
                "system": system_prompt, "user": user_prompt,
                "completion": completion_json("logistics", order["order_id"],
                                              agent["agent_id"], rule,
                                              agent.get("damage_probability")),
            })
    return records


def robustness_holdout(manufacturers, forbidden, seen):
    records = []
    rng = random.Random(ROBUST_SEED)
    for idx, agent in enumerate(manufacturers.to_dict("records")):
        cap, wl = agent["daily_capacity_units"], agent["current_workload_units"]
        products = str(agent["product_types"]).split("|")
        days = ROBUST_DAYS[idx % 2]
        product = products[(idx + 2) % len(products)]
        targets = [
            (rng.uniform(*ROBUST_BELOW_BAND), "below_feasible"),
            (1.00, "exact_feasible"),
            (rng.uniform(*ROBUST_ABOVE_BAND), "above_infeasible"),
        ]
        for target, klass in targets:
            qty = solve_qty(cap, wl, days, target)
            if klass != "above_infeasible":
                while qty > 1 and (wl + qty) / cap > days:
                    qty -= 1
            else:
                while (wl + qty) / cap <= days:
                    qty += 1
            order = {**ORDER_BASE,
                     "order_id": f"ROBUSTB-T-{klass.upper()[:5]}-{agent['agent_id']}",
                     "product_id": product, "quantity": qty,
                     "days_until_deadline": days}
            realized = (wl + qty) / cap / days
            rule, rec = make_tailoring_record(
                agent, order, klass,
                {"target_ratio": round(target, 6), "realized_ratio": round(realized, 6)})
            expected = "BID" if klass != "above_infeasible" else "NO_BID"
            assert rule["decision"] == expected, \
                f"{order['order_id']}: rule {rule['decision']} expected {expected}"
            key = _param_key("tailoring", agent["agent_id"], order)
            assert key not in forbidden and key not in seen, f"{order['order_id']}: collision"
            seen.add(key)
            records.append(rec)
    return records


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    materials, manufacturers, logistics, products, orders = load_all_data()

    forbidden = load_real_param_keys() | eval_set_param_keys()
    forbidden |= {_param_key(c["stage"], c["agent_id"], c["order"]) for c in BASELINE_CASES}
    seen = set()

    t_train = tailoring_train(manufacturers, forbidden, seen)
    l_train = logistics_train(logistics, forbidden, seen)
    robust = robustness_holdout(manufacturers, forbidden, seen)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {"edge_tailoring_train.jsonl": t_train,
             "edge_logistics_train.jsonl": l_train,
             "robustness_holdout.jsonl": robust}
    for name, recs in files.items():
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    # prompt-level (exact user string) leakage re-check against eval sets
    eval_prompts = set()
    for name in ("edge_tailoring_holdout.jsonl", "edge_logistics_holdout.jsonl",
                 "boundary_stress.jsonl", "final_boundary_holdout.jsonl",
                 "material_holdout.jsonl", "tailoring_holdout.jsonl",
                 "logistics_holdout.jsonl"):
        with open(DATA_DIR / name, encoding="utf-8") as f:
            eval_prompts.update(json.loads(l)["user"] for l in f)
    train_prompts = {r["user"] for r in t_train + l_train}
    robust_prompts = {r["user"] for r in robust}
    leaks = {
        "v2_train_vs_eval_sets": len(train_prompts & eval_prompts),
        "v2_train_vs_robustness_holdout": len(train_prompts & robust_prompts),
        "robustness_holdout_vs_eval_sets": len(robust_prompts & eval_prompts),
    }
    assert all(v == 0 for v in leaks.values()), f"PROMPT LEAKAGE: {leaks}"

    def dist(recs):
        by = defaultdict(list)
        for r in recs:
            by[r["category"]].append(r["realized_ratio"])
        return {c: {"n": len(v), "min": min(v), "max": max(v),
                    "values": sorted(round(x, 4) for x in v)} for c, v in sorted(by.items())}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {k: len(v) for k, v in files.items()},
        "tailoring_train_label_balance": dict(Counter(r["rule_decision"] for r in t_train)),
        "logistics_train_label_balance": dict(Counter(r["rule_decision"] for r in l_train)),
        "robustness_holdout_classes": dict(Counter(r["category"] for r in robust)),
        "ratio_distributions": {
            "tailoring_train": dist(t_train),
            "logistics_train": dist(l_train),
            "robustness_holdout": dist(robust),
        },
        "leak_checks_prompt_level": leaks,
        "param_key_disjointness": "asserted against real train+holdout, all fixed eval sets, baseline six, and internally",
        "label_verification": "every record asserted through decide_manufacturer_bid / decide_logistics_bid, serialized via completion_json",
        "hashes": {name: sha256(OUT_DIR / name) for name in files},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "ratio_distributions"}, indent=2))
    for name, d in report["ratio_distributions"].items():
        print(f"\n== {name} ==")
        for cat, s in d.items():
            print(f"  {cat:20s} n={s['n']:3d}  [{s['min']:.4f} .. {s['max']:.4f}]")
    print(f"\nwritten to {OUT_DIR}; report {REPORT}")


if __name__ == "__main__":
    main()
