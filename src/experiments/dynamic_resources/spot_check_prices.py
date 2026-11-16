"""compare a model's offers against the rule-based figure on real holdout orders

case 3 of the 300-order set shipped a model that priced two orders of magnitude
low, because the reverse-auction adapter was used for sealed bidding and sealed
requires the model to derive a price it was never trained to derive. that only
surfaced after a 2.5 hour batch. this runs a handful of orders first.

pass when the median ratio sits inside the tolerance band and no sampled offer is
off by more than an order of magnitude

usage:
  python -m src.experiments.dynamic_resources.spot_check_prices <gguf> [--mode sealed|auction] [--n 6]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from statistics import median

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
# llama_cpp finds the cuda runtime through PATH, not add_dll_directory
os.environ["PATH"] = (str(REPO / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64")
                      + ";" + os.environ.get("PATH", ""))

from config import DATA_VERSION  # noqa: E402
from src.common.bidding_formulas import (  # noqa: E402
    decide_material_bid, decide_manufacturer_bid, eligibility_reason,
)

DATA_DIR = REPO / "data" / "synthetic" / DATA_VERSION
SUBSET = REPO / "reports" / "bidding" / "holdout_150_subset.json"

# an offer this far from the rule figure is a different quantity, not a pricing style
HARD_LIMIT = 10.0
# the trained models price within a couple of percent; allow far more than that
# before calling it a pass, the point is catching 100x, not grading accuracy
SOFT_BAND = (0.25, 4.0)


def orders_sample(n):
    ids = json.load(SUBSET.open(encoding="utf-8"))["order_ids"]
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    orders = orders[orders["order_id"].isin(set(ids))]
    orders = orders.sort_values(["order_created", "order_id"])
    step = max(1, len(orders) // n)
    return orders.iloc[::step].head(n).to_dict("records")


PRODUCTS = pd.read_csv(DATA_DIR / "products.csv")


def as_rfq(order):
    """the rfq shape the supervisor publishes, derived the same way

    required_kg and shipment_weight_kg come from the product row, not the order,
    and days_until_deadline is computed against the order's own creation time"""
    row = PRODUCTS[(PRODUCTS["product_id"] == order["product_id"])
                   & (PRODUCTS["material_type"] == order["material_type"])].iloc[0]
    created = pd.Timestamp(order["order_created"])
    deadline = pd.Timestamp(order["deadline"])
    rfq = dict(order)
    rfq["required_kg"] = order["quantity"] * row["material_kg_per_unit"]
    rfq["shipment_weight_kg"] = order["quantity"] * row["finished_weight_kg_per_unit"]
    rfq["days_until_deadline"] = max(int((deadline - created).total_seconds() / 86400), 1)
    return rfq


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--mode", choices=("sealed", "auction"), default="sealed")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    os.environ["USE_SLM"] = "1"
    os.environ["SLM_MODEL_PATH"] = args.gguf
    os.environ.setdefault("SLM_N_GPU_LAYERS", "-1")
    from src.common.slm import load_slm, slm_decide_material_bid, slm_decide_manufacturer_bid
    from src.common.reverse_auction import slm_decide_round_bid

    slm = load_slm("spot-check")
    materials = pd.read_csv(DATA_DIR / "material_agents.csv").to_dict("records")
    makers = pd.read_csv(DATA_DIR / "manufacturer_agents.csv").to_dict("records")

    rows = []
    for order in orders_sample(args.n):
        rfq = as_rfq(order)
        for stage, agents, rule_fn, slm_fn in (
                ("material", materials, decide_material_bid, slm_decide_material_bid),
                ("tailoring", makers, decide_manufacturer_bid, slm_decide_manufacturer_bid)):
            agent = next((a for a in agents if eligibility_reason(stage, a, rfq) is None), None)
            if agent is None:
                continue
            rule = rule_fn(agent, rfq)
            if rule["decision"] != "BID":
                continue
            if args.mode == "auction":
                auction = {"current_best_bid": rule["offer_amount"] * 1.05,
                           "platform_minimum_decrement": 5.0, "round_number": 1,
                           "minimum_bid": rule["offer_amount"] * 0.9}
                got = await slm_decide_round_bid(slm, stage, agent, {**rfq, **auction})
            else:
                got = await slm_fn(slm, agent, rfq)
            if got.get("decision") != "BID":
                rows.append((order["order_id"], stage, agent["agent_id"],
                             rule["offer_amount"], None, None))
                continue
            ratio = got["offer_amount"] / rule["offer_amount"]
            rows.append((order["order_id"], stage, agent["agent_id"],
                         rule["offer_amount"], got["offer_amount"], ratio))

    print(f"\n{'order':<12}{'stage':<11}{'agent':<6}{'rule':>12}{'model':>12}{'ratio':>8}")
    for oid, stage, aid, rule, got, ratio in rows:
        got_s = f"{got:>12.2f}" if got is not None else f"{'NO_BID':>12}"
        ratio_s = f"{ratio:>8.3f}" if ratio is not None else f"{'-':>8}"
        print(f"{oid:<12}{stage:<11}{aid:<6}{rule:>12.2f}{got_s}{ratio_s}")

    ratios = [r for *_, r in rows if r is not None]
    if not ratios:
        print("\nFAIL: model produced no comparable bid")
        return 1
    med = median(ratios)
    worst = max(max(ratios), 1 / min(ratios))
    print(f"\nsampled {len(ratios)} offers, median ratio {med:.3f}, worst deviation {worst:.1f}x")
    if worst > HARD_LIMIT:
        print(f"FAIL: an offer is more than {HARD_LIMIT:.0f}x off, this is the case-3 signature")
        return 1
    if not (SOFT_BAND[0] <= med <= SOFT_BAND[1]):
        print(f"FAIL: median ratio outside {SOFT_BAND}")
        return 1
    print("PASS: offers are the right order of magnitude")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
