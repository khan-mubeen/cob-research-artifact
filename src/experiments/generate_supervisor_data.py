"""
build supervisor chain-selection fine-tuning data from the rule-based labels.

same shape as generate_finetune_data.py, which does this for the three stage
agents: sample real orders, run the rule, use its output as the training label.
no model, no nats, pure formula calculation.

candidate generation is imported from bid_competition_diagnostics rather than
reimplemented, so the chains a supervisor trains on are exactly the chains the
diagnostics pipeline builds. the label comes from decide_chain_selection() in
bidding_formulas.py, the same function the runtime supervisor will call.

the train/holdout split is disjoint by order id, so the holdout set contains
only orders the model never trained on.

output: data/finetune/{version}/supervisor/selection_{split}.jsonl, one json
object per line, plus metadata.json alongside.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from config import DATA_VERSION, PROJECT_ROOT
from src.common.bidding_formulas import chain_from_bids, chain_totals, decide_chain_selection
from src.common.slm import build_chain_selection_prompt
from src.visualization.bid_competition_diagnostics import (
    build_chains,
    simulate_logistics_bids,
    simulate_manufacturer_bids,
    simulate_material_bids,
    widen_candidates,
)
from src.visualization.data_utils import load_all_data

ORDER_SAMPLE_SIZE = 12000
HOLDOUT_FRACTION = 0.10
RANDOM_SEED = 42
MAX_CANDIDATES = 12          # ~105 tokens each + ~334 fixed, so 12 needs a 2048 window;
                             # must match MAX_CHAIN_CANDIDATES in supervisor.py
MIN_CANDIDATES = 2           # an order with one chain teaches nothing

OUT_DIR = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION / "supervisor"


def days_until_deadline(order):
    try:
        created = pd.to_datetime(order["order_created"])
        deadline = pd.to_datetime(order["deadline"])
        return max((deadline - created).days, 1)
    except Exception:
        return 14


def _bid(agent_id, offer, days, prob):
    """a stage bid in the runtime wire shape the supervisor publishes"""
    return {"agent_id": agent_id, "offer_amount": offer,
            "estimated_completion_days": days, "defect_probability": prob}


def to_runtime_candidates(raw_chains, mat_bids, mfr_bids, log_bids):
    """diagnostics chain records -> the EXACT candidate shape the live supervisor
    assembles and passes to build_chain_selection_prompt

    building the runtime shape here rather than a generator-local one is what
    makes the parity check meaningful: the training prompt is produced by the
    same builder, from the same structure, as the served prompt"""
    md = {b["agent_id"]: b for b in mat_bids}
    td = {b["agent_id"]: b for b in mfr_bids}
    ld = {b["agent_id"]: b for b in log_bids}
    out = []
    for c in raw_chains:
        m, t, l = md[c["material"]], td[c["manufacturer"]], ld[c["logistics"]]
        cand = {
            "material":  _bid(c["material"], m["offer"], m["completion"], m["defect"]),
            "tailoring": _bid(c["manufacturer"], t["offer"], t["completion"], t["defect"]),
            "logistics": _bid(c["logistics"], l["offer"], l["delivery"], l.get("damage", 0.0)),
        }
        cost, days, _ = chain_totals(chain_from_bids(cand))
        cand["total_cost"] = round(cost, 2)
        cand["total_days"] = round(days, 2)
        out.append(cand)
    return out


def shortlist(candidates, remaining_days, rng):
    """cap the candidate list without biasing it toward any one strategy

    take the cheapest, the fastest and the lowest-risk explicitly so the correct
    answer for every priority is always present, then fill the rest at random"""
    if len(candidates) <= MAX_CANDIDATES:
        picked = list(candidates)
    else:
        totals = {id(c): chain_totals(chain_from_bids(c)) for c in candidates}
        anchors = [min(candidates, key=lambda c: totals[id(c)][i]) for i in (0, 1, 2)]
        picked = []
        for a in anchors:
            if not any(a is p for p in picked):
                picked.append(a)
        rest = [c for c in candidates if not any(c is p for p in picked)]
        picked.extend(rng.sample(rest, min(MAX_CANDIDATES - len(picked), len(rest))))
    rng.shuffle(picked)          # no positional bias toward candidate 0
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=ORDER_SAMPLE_SIZE)
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts and one example, write nothing")
    args = parser.parse_args()

    materials, manufacturers, logistics, products, orders = load_all_data()
    sampled = orders.sample(n=min(args.sample, len(orders)), random_state=RANDOM_SEED)

    order_ids = sorted(sampled["order_id"].tolist())
    random.Random(RANDOM_SEED).shuffle(order_ids)
    holdout_count = int(len(order_ids) * HOLDOUT_FRACTION)
    holdout_ids = set(order_ids[:holdout_count])
    print(f"sampled {len(sampled)} orders, seed {RANDOM_SEED}, "
          f"{len(holdout_ids)} holdout / {len(order_ids) - len(holdout_ids)} train order ids")

    rng = random.Random(RANDOM_SEED + 7)
    rows = {"train": [], "holdout": []}
    skipped = {"stage_had_no_bidder": 0, "fewer_than_min_candidates": 0, "rule_declined": 0}
    strategy_counts = {"urgent": 0, "normal": 0, "flexible": 0}

    for _, o in sampled.iterrows():
        mat = simulate_material_bids(o, materials, products)
        mfr = simulate_manufacturer_bids(o, manufacturers)
        log = simulate_logistics_bids(o, logistics, products)
        if not (mat and mfr and log):
            skipped["stage_had_no_bidder"] += 1
            continue

        remaining = days_until_deadline(o)
        order = {
            "order_id": o["order_id"],
            "priority": o["priority"],
            "budget": float(o["budget"]),
            "days_until_deadline": remaining,
            "max_defect_rate": float(o["max_defect_rate"]),
        }

        raw = build_chains(widen_candidates(mat, "completion"),
                           widen_candidates(mfr, "completion"),
                           widen_candidates(log, "delivery"), top_n=6)
        chains = to_runtime_candidates(raw, mat, mfr, log)
        feasible = [c for c in chains
                    if c["total_cost"] <= order["budget"]
                    and c["total_days"] <= remaining]
        if len(feasible) < MIN_CANDIDATES:
            skipped["fewer_than_min_candidates"] += 1
            continue

        candidates = shortlist(feasible, remaining, rng)
        # the rule reads the flat shape, the prompt reads the runtime shape;
        # chain_from_bids is the single conversion between them
        selection = decide_chain_selection([chain_from_bids(c) for c in candidates], order)
        if selection["decision"] != "SELECT":
            skipped["rule_declined"] += 1
            continue

        index = next(i for i, c in enumerate(candidates)
                     if (c["material"]["agent_id"], c["tailoring"]["agent_id"],
                         c["logistics"]["agent_id"])
                     == (selection["material_agent_id"], selection["manufacturer_agent_id"],
                         selection["logistics_agent_id"]))
        system_prompt, user_prompt = build_chain_selection_prompt(order, candidates, remaining)
        reason = {
            "urgent": "fewest total completion days for an urgent order",
            "flexible": "lowest total cost for a flexible order",
            "normal": "best equal-weight balance of cost, days and compound risk",
        }[selection["strategy"]]

        split = "holdout" if o["order_id"] in holdout_ids else "train"
        strategy_counts[selection["strategy"]] += 1
        rows[split].append({
            "order_id": o["order_id"],
            "stage": "supervisor",
            "priority": o["priority"],
            "strategy": selection["strategy"],
            "n_candidates": len(candidates),
            "system": system_prompt,
            "user": user_prompt,
            "completion": json.dumps({"selected_candidate": index, "reason": reason}),
            "label_chain": [selection["material_agent_id"],
                            selection["manufacturer_agent_id"],
                            selection["logistics_agent_id"]],
            "label_total_cost": selection["total_cost"],
            "label_total_days": selection["total_completion_days"],
            "label_chain_risk": selection["chain_risk"],
        })

    total = len(rows["train"]) + len(rows["holdout"])
    print(f"\nusable examples: {total}  (train {len(rows['train'])}, holdout {len(rows['holdout'])})")
    print(f"skipped: {skipped}")
    print(f"strategy mix: {strategy_counts}")
    if total:
        ncand = [r["n_candidates"] for r in rows["train"] + rows["holdout"]]
        print(f"candidates per example: mean {sum(ncand)/len(ncand):.1f}, "
              f"min {min(ncand)}, max {max(ncand)}")
        picked = [json.loads(r["completion"])["selected_candidate"]
                  for r in rows["train"] + rows["holdout"]]
        print(f"selected index distribution (should be flat, no positional bias): "
              f"min {min(picked)}, max {max(picked)}, mean {sum(picked)/len(picked):.2f}")

    if args.dry_run:
        print("\n--- dry run, nothing written. example: ---")
        if rows["train"]:
            ex = rows["train"][0]
            print(ex["system"][:200], "\n...\n", ex["user"][:900], "\n->", ex["completion"])
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        path = OUT_DIR / f"selection_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(items):6d} -> {path}")

    meta = {
        "generated_from": "rule-based decide_chain_selection() labels, no model involved",
        "data_version": DATA_VERSION,
        "order_sample_size": len(sampled),
        "holdout_fraction": HOLDOUT_FRACTION,
        "random_seed": RANDOM_SEED,
        "max_candidates_per_prompt": MAX_CANDIDATES,
        "min_candidates_per_prompt": MIN_CANDIDATES,
        "split_rule": "order-id disjoint; an order's example lands entirely in one split",
        "candidate_source": "src/visualization/bid_competition_diagnostics.build_chains, top_n=6",
        "prompt_builder": "src.common.slm.build_chain_selection_prompt (the same builder the live supervisor calls)",
        "max_seq_length_required": 2048,
        "label_rule": "src.common.bidding_formulas.decide_chain_selection",
        "rule_summary": {
            "risk": "compound: 1 - (1-d_mat)(1-d_mfr)(1-d_log)",
            "normalization": "budget-relative (budget, days_until_deadline, max_defect_rate)",
            "normal_weights": "1/3 cost, 1/3 days, 1/3 risk",
            "urgent": "min total days, cost breaks ties",
            "flexible": "min total cost, deadline slack breaks ties",
            "reputation": "deliberately excluded, min_rep left unused",
        },
        "counts": {split: len(items) for split, items in rows.items()},
        "skipped": skipped,
        "strategy_mix": strategy_counts,
    }
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
