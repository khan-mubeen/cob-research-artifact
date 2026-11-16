"""
drive one bidding experiment end to end over nats, in either mode

the broker, the supervisor and the three stage agents must already be running
with the same BIDDING_MODE as this script. that env var is the single switch:
unset means sealed, exactly as before

orders are published one at a time and the script waits for that order's
order.logging outcome before starting the next, so a reverse-auction trace can
never be attributed to the wrong order

every finished order is appended to a jsonl checkpoint and flushed immediately.
rerunning with --resume on the same file skips the order ids already recorded,
so an interrupted run continues where it stopped instead of starting over

with DYNAMIC_RESOURCES set, agents mutate stock and workload as orders are
accepted, the batch is sequenced by order_created, and the results nest one
level deeper under dynamic_resources/

results: reports/bidding/<mode>/run_<timestamp>.jsonl plus a _summary.json
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DATA_VERSION, PROJECT_ROOT, REVERSE_AUCTION_VERSION
from src.common.bidding_modes import REVERSE_AUCTION, bidding_mode
from src.common.dynamic_state import STATE_EVENT_SUBJECT, dynamic_resources_enabled
from src.common.nats_client import connect_with_retry
from src.common.slm import slm_enabled
from src.supervisor.auction import TRACE_SUBJECT


NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / DATA_VERSION
REPORT_ROOT = PROJECT_ROOT / "reports" / "bidding"

# the order-disjoint slm holdout, the same set every scenario must run against
HOLDOUT_DIR = (PROJECT_ROOT / "data" / "finetune" / DATA_VERSION /
               "reverse_auction" / REVERSE_AUCTION_VERSION)
HOLDOUT_STAGES = ("material", "tailoring", "logistics")

# a dynamic-state event is published by an agent, not by the supervisor, so it
# can still be in flight when the order's outcome arrives. this short drain is
# for the report only, nothing in the protocol depends on it
DYNAMIC_FLUSH_SECONDS = float(os.getenv("DYNAMIC_FLUSH_SECONDS", "0.3"))

# generous, the wait ends as soon as the order's logging message arrives
DEFAULT_ORDER_TIMEOUT = 300.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orders", type=int, default=1,
                   help="how many orders to run, taken from the top of orders.csv")
    p.add_argument("--sample", type=int, default=0,
                   help="run a seeded sample of N orders instead of the top N; the top of "
                        "orders.csv is all one scenario type, a sample follows the population "
                        "mix. same seed as the dataset generators, so the set is reproducible")
    p.add_argument("--order-id", action="append", default=[],
                   help="run a specific order id, repeatable, overrides --orders")
    p.add_argument("--timeout", type=float, default=DEFAULT_ORDER_TIMEOUT,
                   help="seconds to wait for one order to finish")
    p.add_argument("--output", default=None, help="jsonl checkpoint path")
    p.add_argument("--resume", default=None,
                   help="continue an existing jsonl checkpoint, skipping recorded order ids")
    p.add_argument("--quiet", action="store_true", help="do not print the per-round trace")
    p.add_argument("--holdout", action="store_true",
                   help="run the order-disjoint slm holdout set instead of orders.csv, "
                        "the same 300 orders every scenario must be compared on. "
                        "--orders then caps how many of them run")
    p.add_argument("--chronological", action="store_true",
                   help="process orders in order_created sequence. implied by "
                        "DYNAMIC_RESOURCES, whose walking clock needs it")
    return p.parse_args()


def holdout_order_ids():
    """the holdout order ids, derived from the holdout files themselves

    the three per-stage files must name the same orders, and the count must be
    what the split metadata says it is. both are asserted rather than assumed,
    a silently short holdout would make two scenarios incomparable"""
    per_stage = {}
    for stage in HOLDOUT_STAGES:
        path = HOLDOUT_DIR / stage / "holdout.jsonl"
        ids = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["order_id"])
        per_stage[stage] = ids

    union = set().union(*per_stage.values())
    for stage, ids in per_stage.items():
        if ids != union:
            raise RuntimeError(
                f"holdout stage {stage} names {len(ids)} order ids, the union is "
                f"{len(union)}; the three stage holdouts must cover the same orders")

    with open(HOLDOUT_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    expected = round(meta["order_sample_size"] * meta["holdout_fraction"])
    if len(union) != expected:
        raise RuntimeError(f"holdout has {len(union)} order ids, metadata implies {expected}")

    print(f"holdout verified: {len(union)} unique order ids, identical across "
          f"{', '.join(HOLDOUT_STAGES)}, from {HOLDOUT_DIR}")
    return union


def _rows(df):
    # native python types, json.dumps cannot serialize numpy scalars
    return json.loads(df.to_json(orient="records"))


# same seed the dataset generators sample with, so a batch is reproducible
SAMPLE_SEED = 42


def select_orders(orders, args, chronological=False):
    if args.holdout:
        ids = holdout_order_ids()
        rows = orders[orders["order_id"].isin(ids)]
        missing = ids - set(rows["order_id"])
        if missing:
            raise RuntimeError(f"holdout order id(s) not in orders.csv: {sorted(missing)}")
        # always chronological, then capped: the first N of the holdout is a
        # prefix of the same sequence the full 300-order run walks, so a slice
        # stays consistent when it is later extended
        rows = rows.sort_values(["order_created", "order_id"]).head(args.orders)
        return _rows(rows)
    if args.order_id:
        rows = orders[orders["order_id"].isin(args.order_id)]
        missing = set(args.order_id) - set(rows["order_id"])
        if missing:
            raise RuntimeError(f"unknown order id(s): {sorted(missing)}")
        selected = rows
    elif args.sample:
        selected = orders.sample(n=min(args.sample, len(orders)), random_state=SAMPLE_SEED)
    else:
        selected = orders.head(args.orders)
    if chronological:
        selected = selected.sort_values(["order_created", "order_id"])
    return _rows(selected)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_records(path):
    """records already in a checkpoint

    a half-written last line from a crash is dropped, that order is simply rerun"""
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def completed_ids(path):
    """order ids already in a checkpoint, so a resumed run skips them"""
    return {r["order_id"] for r in load_records(path) if "order_id" in r}


def print_trace(trace):
    stage = trace["stage"]
    print(f"  {stage}: roster {trace['roster_size']}, entrants {len(trace['entrant_agent_ids'])}, "
          f"decrement EUR {trace['platform_minimum_decrement']}")
    for rnd in trace["rounds"]:
        announced = rnd["announced_best_bid"]
        bids = ", ".join(f"{b['agent_id']} {b['offer_amount']}" for b in rnd["bids"]) or "none"
        print(f"    round {rnd['round_number']}  announced best: "
              f"{announced if announced is not None else '-'}")
        print(f"      bids   : {bids}")
        if rnd["silent"]:
            by_status = defaultdict(list)
            for entry in rnd["silent"]:
                by_status[entry["status"]].append(entry["agent_id"])
            for status, agents in sorted(by_status.items()):
                print(f"      silent ({status}): {', '.join(agents)}")
        print(f"      best after round: {rnd['best_bid_after_round']} "
              f"({rnd['best_agent_id_after_round']})")
    print(f"    winner {trace['winner_agent_id']} at {trace['winning_offer_amount']}, "
          f"opened at {trace['opening_best_bid']}, reduction {trace['total_reduction_eur']}")


async def run(args):
    mode = bidding_mode()
    dynamic = dynamic_resources_enabled()
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    # the walking clock only moves forward, so a dynamic run must be sequenced
    selected = select_orders(orders, args, chronological=args.chronological or dynamic)

    if args.resume or args.output:
        out_path = _project_path(args.resume or args.output)
    else:
        # same run_<timestamp>.jsonl convention as every other file under
        # reports/bidding, nested one level for the dynamic-resources runs.
        # the mode is already the directory, so the agent kind goes on the front
        # of the name and the four dynamic cases stay apart without opening one
        report_dir = REPORT_ROOT / mode / "dynamic_resources" if dynamic else REPORT_ROOT / mode
        # SLM_MODEL_SLUG folds the architecture into the name when several models
        # are run against the same mode, so the files group by agent kind then model
        slug = os.getenv("SLM_MODEL_SLUG", "").strip()
        kind = ("slm_" if slm_enabled() else "rule_") if dynamic else ""
        prefix = f"{kind}{slug}_" if kind and slug else kind
        out_path = report_dir / f"{prefix}run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already = completed_ids(out_path) if args.resume else set()
    pending = [o for o in selected if o["order_id"] not in already]
    if already:
        print(f"resuming {out_path.name}, {len(already)} order(s) already recorded, "
              f"{len(pending)} to go")

    print(f"bidding mode: {mode}, dynamic resources "
          f"{'on' if dynamic else 'off'}, {len(pending)} order(s), checkpoint {out_path}")
    if dynamic:
        first, last = pending[0], pending[-1]
        print(f"simulated clock walks {first['order_created']} -> {last['order_created']}")

    nc = await connect_with_retry(NATS_URL, "experiment runner")

    outcomes = {}
    traces = defaultdict(list)
    waiters = {}

    async def on_logging(msg):
        entry = json.loads(msg.data.decode())
        outcomes[entry["order_id"]] = entry
        waiter = waiters.get(entry["order_id"])
        if waiter is not None and not waiter.done():
            waiter.set_result(True)

    async def on_trace(msg):
        trace = json.loads(msg.data.decode())
        traces[trace["order_id"]].append(trace)

    # every restock, reservation and release the agents perform, keyed by the
    # order that was being processed when it happened
    dynamic_events = defaultdict(list)

    async def on_dynamic_event(msg):
        event = json.loads(msg.data.decode())
        dynamic_events[event.get("observed_on_order_id") or event["order_id"]].append(event)

    log_sub = await nc.subscribe("order.logging", cb=on_logging)
    trace_sub = await nc.subscribe(TRACE_SUBJECT, cb=on_trace)
    state_sub = await nc.subscribe(STATE_EVENT_SUBJECT, cb=on_dynamic_event)

    records = []
    with open(out_path, "a", encoding="utf-8") as sink:
        for order in pending:
            order_id = order["order_id"]
            waiters[order_id] = asyncio.get_running_loop().create_future()
            started = datetime.now()
            print(f"\npublishing {order_id} ({order['priority']}, budget {order['budget']})")
            await nc.publish("order.created", json.dumps(order).encode())

            try:
                await asyncio.wait_for(waiters[order_id], timeout=args.timeout)
                outcome = outcomes[order_id]
            except asyncio.TimeoutError:
                outcome = {"order_id": order_id, "status": "timeout",
                           "reason": f"no outcome within {args.timeout}s"}

            if dynamic:
                # let any state event published alongside the outcome land first
                await asyncio.sleep(DYNAMIC_FLUSH_SECONDS)

            record = {
                "order_id": order_id,
                "mode": mode,
                "priority": order["priority"],
                "budget": order["budget"],
                "outcome": outcome,
                "stage_traces": traces.get(order_id, []),
                "dynamic_events": dynamic_events.get(order_id, []),
                "elapsed_seconds": round((datetime.now() - started).total_seconds(), 2),
                "recorded_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            records.append(record)

            print(f"{order_id}: {outcome['status']}"
                  + (f", total cost {outcome['total_cost']}" if "total_cost" in outcome else "")
                  + (f", reason {outcome['reason']}" if "reason" in outcome else ""))
            if mode == REVERSE_AUCTION and not args.quiet:
                for trace in record["stage_traces"]:
                    print_trace(trace)

    await log_sub.unsubscribe()
    await trace_sub.unsubscribe()
    await state_sub.unsubscribe()
    await nc.drain()

    # summarize the whole checkpoint, not only this invocation, so a resumed run
    # still reports the complete experiment
    summary = summarize(mode, load_records(out_path), out_path, dynamic)
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary: {json.dumps(summary['totals'])}")
    print(f"written: {out_path}\n         {summary_path}")


def summarize_dynamic(records):
    """what the resource state actually did across the batch

    counted straight off the recorded events, so the summary cannot drift from
    the per-order records it is derived from"""
    counts = defaultdict(int)
    triggers = defaultdict(int)
    restocks_by_agent = defaultdict(int)
    reservation_retries = 0
    order_created = []
    for record in records:
        reservation_retries += record["outcome"].get("reservation_retries", 0) or 0
        sim_now = record["outcome"].get("sim_now")
        if sim_now:
            order_created.append(sim_now)
        for event in record.get("dynamic_events", []):
            counts[event["event"]] += 1
            if event["event"] == "restock_triggered":
                triggers[event["trigger"]] += 1
                restocks_by_agent[f"{event['agent_id']}:{event['material']}"] += 1
    return {
        "events_by_type": dict(sorted(counts.items())),
        "restock_triggers_by_rule": dict(sorted(triggers.items())),
        "restocks_triggered_by_agent_material": dict(sorted(restocks_by_agent.items())),
        "reservation_retries": reservation_retries,
        "sim_clock_from": min(order_created) if order_created else None,
        "sim_clock_to": max(order_created) if order_created else None,
    }


def summarize(mode, records, out_path, dynamic=False):
    statuses = defaultdict(int)
    reductions = []
    for record in records:
        statuses[record["outcome"]["status"]] += 1
        for trace in record["stage_traces"]:
            if trace["total_reduction_eur"] is not None:
                reductions.append(trace["total_reduction_eur"])
    totals = {
        "orders_recorded": len(records),
        "by_status": dict(statuses),
    }
    if reductions:
        totals["stage_auctions"] = len(reductions)
        totals["mean_stage_reduction_eur"] = round(sum(reductions) / len(reductions), 2)
        totals["max_stage_reduction_eur"] = max(reductions)
    summary = {
        "mode": mode,
        "dynamic_resources": dynamic,
        "data_version": DATA_VERSION,
        "checkpoint": str(out_path),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": totals,
    }
    if dynamic:
        summary["order_ids"] = [r["order_id"] for r in records]
        summary["dynamic_state"] = summarize_dynamic(records)
    return summary


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
