"""compare a sealed batch against a reverse-auction batch over the same orders

both checkpoints come from run_bidding_experiment.py, so the records have the
same shape. orders are paired on order_id and only the orders present in both
files are compared, which keeps every figure like for like

the cost and delivery-time comparison uses the orders both modes completed. the
fill comparison uses all paired orders, because an order one mode filled and the
other did not is itself a result

writes a json summary next to the auction checkpoint
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DATA_VERSION, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / DATA_VERSION
REPORT_ROOT = PROJECT_ROOT / "reports" / "bidding"

# a cost difference below this is a tie, not a win either way
TIE_EUR = 0.005


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sealed", required=True, help="sealed jsonl checkpoint")
    p.add_argument("--auction", required=True, help="reverse-auction jsonl checkpoint")
    p.add_argument("--output", default=None, help="where to write the json summary")
    return p.parse_args()


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load(path):
    """order_id -> record, last record wins if an order was somehow run twice"""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["order_id"]] = record
    return records


def _stats(values):
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "total": round(sum(ordered), 2),
        "mean": round(sum(ordered) / n, 4),
        "median": round(ordered[n // 2] if n % 2 else
                        (ordered[n // 2 - 1] + ordered[n // 2]) / 2, 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def fill_comparison(sealed, auction, order_ids):
    """which orders each mode could fill, and what stopped the other one"""
    both, sealed_only, auction_only, neither = [], [], [], []
    for oid in order_ids:
        s_ok = sealed[oid]["outcome"]["status"] == "completed"
        a_ok = auction[oid]["outcome"]["status"] == "completed"
        if s_ok and a_ok:
            both.append(oid)
        elif s_ok:
            sealed_only.append(oid)
        elif a_ok:
            auction_only.append(oid)
        else:
            neither.append(oid)

    def reasons(ids, records):
        return dict(Counter(records[oid]["outcome"].get("reason", "unknown") for oid in ids))

    return {
        "completed_in_both": len(both),
        "sealed_only": len(sealed_only),
        "auction_only": len(auction_only),
        "completed_in_neither": len(neither),
        "sealed_only_order_ids": sealed_only,
        "auction_only_order_ids": auction_only,
        # why the mode that failed, failed
        "auction_failed_on_sealed_only": reasons(sealed_only, auction),
        "sealed_failed_on_auction_only": reasons(auction_only, sealed),
        "failed_in_neither_sealed_reason": reasons(neither, sealed),
        "failed_in_neither_auction_reason": reasons(neither, auction),
    }, both


def head_to_head(sealed, auction, paired_ids, orders):
    """per-order cost and delivery-time difference on the orders both filled"""
    rows = []
    for oid in paired_ids:
        s, a = sealed[oid]["outcome"], auction[oid]["outcome"]
        meta = orders.get(oid, {})
        saving = s["total_cost"] - a["total_cost"]
        rows.append({
            "order_id": oid,
            "priority": sealed[oid]["priority"],
            "scenario_type": meta.get("scenario_type"),
            "budget": sealed[oid]["budget"],
            "sealed_cost": s["total_cost"],
            "auction_cost": a["total_cost"],
            "cost_saving_eur": round(saving, 2),
            "cost_saving_pct": round(saving / s["total_cost"] * 100, 4),
            "sealed_days": s.get("total_days"),
            "auction_days": a.get("total_days"),
            "days_diff": (None if s.get("total_days") is None or a.get("total_days") is None
                          else round(a["total_days"] - s["total_days"], 4)),
            "same_chain": s["chain"] == a["chain"],
            "sealed_elapsed_seconds": sealed[oid]["elapsed_seconds"],
            "auction_elapsed_seconds": auction[oid]["elapsed_seconds"],
        })
    return rows


def win_rates(rows):
    """how often each mode came out ahead, on price and on delivery time"""
    cost = Counter()
    for r in rows:
        if r["cost_saving_eur"] > TIE_EUR:
            cost["auction_cheaper"] += 1
        elif r["cost_saving_eur"] < -TIE_EUR:
            cost["sealed_cheaper"] += 1
        else:
            cost["tie"] += 1

    days = Counter()
    for r in rows:
        if r["days_diff"] is None:
            days["unknown"] += 1
        elif r["days_diff"] < -0.0005:
            days["auction_faster"] += 1
        elif r["days_diff"] > 0.0005:
            days["sealed_faster"] += 1
        else:
            days["tie"] += 1

    return {"cost": dict(cost), "delivery_days": dict(days),
            "same_winning_chain": sum(1 for r in rows if r["same_chain"])}


def by_group(rows, key):
    """the same head-to-head numbers, split by one order attribute"""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key] or "unknown"].append(r)
    out = {}
    for name, group in sorted(groups.items()):
        savings = [r["cost_saving_eur"] for r in group]
        out[name] = {
            "n": len(group),
            "sealed_total_cost": round(sum(r["sealed_cost"] for r in group), 2),
            "auction_total_cost": round(sum(r["auction_cost"] for r in group), 2),
            "total_saving_eur": round(sum(savings), 2),
            "mean_saving_eur": round(sum(savings) / len(group), 2),
            "mean_saving_pct": round(sum(r["cost_saving_pct"] for r in group) / len(group), 4),
            "auction_cheaper": sum(1 for s in savings if s > TIE_EUR),
            "sealed_cheaper": sum(1 for s in savings if s < -TIE_EUR),
            "tie": sum(1 for s in savings if abs(s) <= TIE_EUR),
        }
    return out


def auction_mechanics(auction, paired_ids):
    """what the auction itself did: rounds run, bidders, price reduction per stage"""
    reductions, rounds_run, entrants, opened_at_floor = [], [], [], 0
    decrements, per_stage = set(), defaultdict(list)
    for oid in paired_ids:
        for trace in auction[oid]["stage_traces"]:
            decrements.add(trace["platform_minimum_decrement"])
            rounds_run.append(len(trace["rounds"]))
            entrants.append(len(trace["entrant_agent_ids"]))
            reduction = trace["total_reduction_eur"]
            if reduction is not None:
                reductions.append(reduction)
                per_stage[trace["stage"]].append(reduction)
                if reduction == 0:
                    opened_at_floor += 1
    return {
        "stage_auctions": len(rounds_run),
        "decrement_eur": sorted(decrements)[0] if len(decrements) == 1 else sorted(decrements),
        "rounds_run": dict(Counter(rounds_run)),
        "mean_entrants": round(sum(entrants) / len(entrants), 2) if entrants else None,
        "stage_reduction_eur": _stats(reductions),
        "stages_with_no_reduction": opened_at_floor,
        "stage_reduction_by_stage": {k: _stats(v) for k, v in sorted(per_stage.items())},
    }


def compare(sealed_path, auction_path):
    sealed, auction = load(sealed_path), load(auction_path)
    paired = sorted(set(sealed) & set(auction))
    if not paired:
        raise RuntimeError("the two checkpoints share no order ids")

    frame = pd.read_csv(DATA_DIR / "orders.csv")
    orders = frame.set_index("order_id").to_dict(orient="index")

    fills, both = fill_comparison(sealed, auction, paired)
    rows = head_to_head(sealed, auction, both, orders)
    savings = [r["cost_saving_eur"] for r in rows]

    sealed_total = round(sum(r["sealed_cost"] for r in rows), 2)
    auction_total = round(sum(r["auction_cost"] for r in rows), 2)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "data_version": DATA_VERSION,
        "sealed_checkpoint": str(sealed_path),
        "auction_checkpoint": str(auction_path),
        "orders_in_sealed": len(sealed),
        "orders_in_auction": len(auction),
        "orders_paired": len(paired),
        "fill": fills,
        "cost": {
            "sealed_total_eur": sealed_total,
            "auction_total_eur": auction_total,
            "total_saving_eur": round(sealed_total - auction_total, 2),
            "total_saving_pct": round((sealed_total - auction_total) / sealed_total * 100, 4),
            "per_order_saving_eur": _stats(savings),
            "per_order_saving_pct": _stats([r["cost_saving_pct"] for r in rows]),
        },
        "delivery_days": {
            "sealed": _stats([r["sealed_days"] for r in rows if r["sealed_days"] is not None]),
            "auction": _stats([r["auction_days"] for r in rows if r["auction_days"] is not None]),
            "difference": _stats([r["days_diff"] for r in rows if r["days_diff"] is not None]),
        },
        "wall_clock_seconds": {
            # set by the configured bidding window, not by agent behaviour
            "sealed": _stats([r["sealed_elapsed_seconds"] for r in rows]),
            "auction": _stats([r["auction_elapsed_seconds"] for r in rows]),
        },
        "win_rate": win_rates(rows),
        "by_priority": by_group(rows, "priority"),
        "by_scenario_type": by_group(rows, "scenario_type"),
        "auction_mechanics": auction_mechanics(auction, both),
        "per_order": rows,
    }


def print_report(summary):
    fill, cost, win = summary["fill"], summary["cost"], summary["win_rate"]
    print(f"paired orders: {summary['orders_paired']}")
    print(f"  completed in both      {fill['completed_in_both']}")
    print(f"  sealed only            {fill['sealed_only']}")
    print(f"  auction only           {fill['auction_only']}")
    print(f"  completed in neither   {fill['completed_in_neither']}")
    print(f"\ncost over the {cost['per_order_saving_eur']['n']} orders both filled")
    print(f"  sealed  EUR {cost['sealed_total_eur']:>14,.2f}")
    print(f"  auction EUR {cost['auction_total_eur']:>14,.2f}")
    print(f"  saved   EUR {cost['total_saving_eur']:>14,.2f}  "
          f"({cost['total_saving_pct']:.3f}%)")
    print(f"  per order: mean EUR {cost['per_order_saving_eur']['mean']:,.2f}, "
          f"median EUR {cost['per_order_saving_eur']['median']:,.2f}, "
          f"range EUR {cost['per_order_saving_eur']['min']:,.2f} to "
          f"{cost['per_order_saving_eur']['max']:,.2f}")
    print(f"\nwin rate on price: {win['cost']}")
    print(f"win rate on delivery time: {win['delivery_days']}")
    print(f"same winning chain in both modes: {win['same_winning_chain']}")
    days = summary["delivery_days"]
    if days["sealed"]:
        print(f"\ndelivery days: sealed mean {days['sealed']['mean']:.3f}, "
              f"auction mean {days['auction']['mean']:.3f}, "
              f"difference mean {days['difference']['mean']:+.3f}")


if __name__ == "__main__":
    args = parse_args()
    sealed_path, auction_path = _project_path(args.sealed), _project_path(args.auction)
    result = compare(sealed_path, auction_path)
    out = (_project_path(args.output) if args.output
           else auction_path.with_name(auction_path.stem + "_vs_sealed.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print_report(result)
    print(f"\nwritten: {out}")
