"""
slm vs rule-based bid comparison for chain-of-bid synthetic data.

runs slm_decide_* from src/common/slm.py against the rule-based decide_*
functions on the same orders and agents. the GGUF loads once through load_slm;
the comparison runs in one process without agents, nats, or docker.

rfq dictionaries and eligibility gates match the live supervisor and agents.

verification checks:
- print the first order's raw model response verbatim
- measure generated tokens per second at the llama boundary
- flag empty responses and duplicate text across orders
"""

import asyncio
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# configure the model runtime before load_slm imports llama_cpp
# enable SLM, offload every layer, and expose the bundled CUDA DLLs through PATH
os.environ.setdefault("USE_SLM", "true")
os.environ.setdefault("SLM_N_GPU_LAYERS", "-1")
_nvidia_bin = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
if _nvidia_bin.is_dir():
    os.environ["PATH"] = str(_nvidia_bin) + os.pathsep + os.environ["PATH"]

from config import DATA_VERSION, PROJECT_ROOT
from src.common.bidding_formulas import (
    decide_logistics_bid,
    decide_manufacturer_bid,
    decide_material_bid,
    logistics_eligibility_reason,
    manufacturer_eligibility_reason,
    material_eligibility_reason,
)
from src.common.slm import (
    _extract_json,
    load_slm,
    slm_decide_logistics_bid,
    slm_decide_manufacturer_bid,
    slm_decide_material_bid,
    warmup_slm,
)
from src.visualization.data_utils import load_all_data
from src.visualization import version_guard


# single point of change for experiment size
SLM_CHECK_ORDER_COUNT = 50

REPORT_DIR = PROJECT_ROOT / "reports" / "epoch-1" / "production" / "evaluation" / "raw"


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


class GenerationRecorder:
    """record raw text and token usage at the llama boundary

    wraps create_chat_completion on the loaded model without changing the
    decision path, recording the same calls used by slm_decide_*"""

    def __init__(self, llama):
        self.calls = []
        self._orig = llama.create_chat_completion
        llama.create_chat_completion = self._record

    def _record(self, *args, **kwargs):
        started = time.perf_counter()
        out = self._orig(*args, **kwargs)
        elapsed = time.perf_counter() - started
        usage = out.get("usage", {})
        self.calls.append({
            "raw": out["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "seconds": elapsed,
        })
        return out


def build_stage_orders(order, products):
    """one rfq dict per stage, mirroring the supervisor's collect_bids payloads"""
    prod_row = products[
        (products["product_id"] == order["product_id"]) &
        (products["material_type"] == order["material_type"])
    ]
    if prod_row.empty:
        return None

    prod = prod_row.iloc[0]
    try:
        created = pd.to_datetime(order["order_created"])
        deadline = pd.to_datetime(order["deadline"])
        days_until_deadline = max((deadline - created).days, 1)
    except Exception:
        days_until_deadline = 14

    base = {
        "order_id": order["order_id"],
        "max_defect_rate": order["max_defect_rate"],
        "min_reputation_score": order["min_reputation_score"],
        "days_until_deadline": days_until_deadline,
        "priority": order["priority"],
    }
    return {
        "material": {
            **base,
            "material_type": order["material_type"],
            "required_kg": order["quantity"] * prod["material_kg_per_unit"],
        },
        "tailoring": {
            **base,
            "product_id": order["product_id"],
            "quantity": order["quantity"],
        },
        "logistics": {
            **base,
            "deliver_to": order["deliver_to"],
            "shipment_weight_kg": order["quantity"] * prod["finished_weight_kg_per_unit"],
        },
    }


def compare_row(stage, order_id, agent_id, rule, slm_dec, call):
    raw = call["raw"]
    both_bid = rule["decision"] == "BID" and slm_dec["decision"] == "BID"

    offer_pct_deviation = None
    completion_days_deviation = None
    if both_bid and rule["offer_amount"] > 0:
        offer_pct_deviation = round(
            (slm_dec["offer_amount"] - rule["offer_amount"]) / rule["offer_amount"] * 100, 2
        )
        completion_days_deviation = round(
            slm_dec["estimated_completion_days"] - rule["estimated_completion_days"], 2
        )

    return {
        "order_id": order_id,
        "stage": stage,
        "agent_id": agent_id,
        "rule_decision": rule["decision"],
        "slm_decision": slm_dec["decision"],
        "decision_matched": rule["decision"] == slm_dec["decision"],
        "rule_offer": rule.get("offer_amount"),
        "slm_offer": slm_dec.get("offer_amount"),
        "offer_pct_deviation": offer_pct_deviation,
        "rule_completion_days": rule.get("estimated_completion_days"),
        "slm_completion_days": slm_dec.get("estimated_completion_days"),
        "completion_days_deviation": completion_days_deviation,
        "json_valid": isinstance(_extract_json(raw), dict),
        "slm_debug_reason": slm_dec.get("debug_reason", ""),
        "gen_tokens": call["completion_tokens"],
        "gen_seconds": round(call["seconds"], 3),
        "raw_response": raw,
    }


async def run():
    # a v5 request must not be answered with a v4 report. there is no v5 equivalent of this
    # diagnostic (it drives a live slm against v4 data), so this refuses and names the v5
    # prompt-inspection command rather than routing somewhere that answers a different question
    pair = version_guard.mismatch()
    if pair:
        version_guard.route_or_refuse(
            "make slm_check", pair,
            equivalent="PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.sample_check")

    materials, manufacturers, logistics, products, orders = load_all_data()
    orders = orders.head(SLM_CHECK_ORDER_COUNT)

    section(f"SLM COMPETITION DIAGNOSTICS - version {DATA_VERSION}")
    print(f"comparing rule-based vs slm bidding on the first {len(orders)} orders")
    print(f"model: {os.getenv('SLM_MODEL_PATH', '') or 'default from src/common/slm.py'}")

    load_started = time.perf_counter()
    slm = load_slm("slm_check")
    print(f"model loaded in {time.perf_counter() - load_started:.1f}s")

    # keep the one-time cuda warmup out of the recorded batch statistics
    await warmup_slm(slm, "slm_check")

    recorder = GenerationRecorder(slm._llama)

    stage_specs = [
        ("material", materials.to_dict("records"),
         material_eligibility_reason, decide_material_bid, slm_decide_material_bid),
        ("tailoring", manufacturers.to_dict("records"),
         manufacturer_eligibility_reason, decide_manufacturer_bid, slm_decide_manufacturer_bid),
        ("logistics", logistics.to_dict("records"),
         logistics_eligibility_reason, decide_logistics_bid, slm_decide_logistics_bid),
    ]

    rows = []
    skipped_orders = 0
    first_raw_shown = False

    for order_index, (_, order) in enumerate(orders.iterrows(), start=1):
        stage_orders = build_stage_orders(order, products)
        if stage_orders is None:
            skipped_orders += 1
            print(f"[{order_index}/{len(orders)}] {order['order_id']}: no product/material pair, skipped")
            continue

        for stage, agents, eligibility_fn, rule_fn, slm_fn in stage_specs:
            stage_order = stage_orders[stage]
            for agent in agents:
                # the live agents never send ineligible rfqs to the model,
                # mirror that gate so the comparison covers the same calls
                if eligibility_fn(agent, stage_order) is not None:
                    continue

                rule = rule_fn(agent, stage_order)

                calls_before = len(recorder.calls)
                slm_dec = await slm_fn(slm, agent, stage_order)
                if len(recorder.calls) != calls_before + 1:
                    raise RuntimeError(
                        "slm decision produced no recorded model call, "
                        "the model path is not being exercised"
                    )
                call = recorder.calls[-1]

                if not first_raw_shown:
                    first_raw_shown = True
                    section(f"SAFEGUARD - FULL RAW MODEL RESPONSE "
                            f"(order {stage_order['order_id']}, stage {stage}, agent {agent['agent_id']})")
                    print(call["raw"])
                    section("END RAW MODEL RESPONSE")

                rows.append(compare_row(
                    stage, stage_order["order_id"], agent["agent_id"], rule, slm_dec, call,
                ))

        total_tokens = sum(c["completion_tokens"] for c in recorder.calls)
        total_seconds = sum(c["seconds"] for c in recorder.calls)
        tps = total_tokens / total_seconds if total_seconds > 0 else 0.0
        print(f"[{order_index}/{len(orders)}] {order['order_id']}: "
              f"{len(rows)} comparisons so far, cumulative {tps:.1f} gen tok/s")

    if not rows:
        raise SystemExit("no comparisons produced, check the order slice and agent data")

    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORT_DIR / f"slm_comparison_{len(orders)}orders_{stamp}.csv"
    # raw text stays out of the csv so rows keep one line per comparison,
    # a truncated preview is enough to eyeball response variety
    df["raw_response"] = df["raw_response"].str.replace(r"\s+", " ", regex=True).str.slice(0, 200)
    df.to_csv(csv_path, index=False)

    section("GENERATION EVIDENCE (anti-stub safeguards)")
    total_tokens = sum(c["completion_tokens"] for c in recorder.calls)
    total_seconds = sum(c["seconds"] for c in recorder.calls)
    print(f"model calls recorded:        {len(recorder.calls)}")
    print(f"total generated tokens:      {total_tokens}")
    print(f"total inference wall time:   {total_seconds:.1f}s")
    if total_seconds > 0:
        print(f"average generated tok/s:     {total_tokens / total_seconds:.1f}")

    flags = []
    empty_calls = sum(1 for c in recorder.calls if not c["raw"].strip())
    if empty_calls:
        flags.append(f"{empty_calls} model call(s) returned empty text")

    raw_to_orders = defaultdict(set)
    for row in rows:
        raw_to_orders[row["raw_response"]].add(row["order_id"])
    cross_order_dupes = {
        raw: oids for raw, oids in raw_to_orders.items() if len(oids) > 1
    }
    if cross_order_dupes:
        flags.append(
            f"{len(cross_order_dupes)} raw response text(s) repeated verbatim across different "
            f"orders, e.g. across {sorted(next(iter(cross_order_dupes.values())))[:4]}"
        )

    if flags:
        for flag in flags:
            print(f"  FLAG: {flag} - verify that each row used fresh model output")
    else:
        print(f"distinct raw responses:      {len(raw_to_orders)} across {len(rows)} rows, "
              "no empty or cross-order duplicate generations")

    section("DECISION AGREEMENT")
    total = len(df)
    matched = int(df["decision_matched"].sum())
    print(f"rows compared:            {total}  (orders skipped: {skipped_orders})")
    print(f"decision matched:         {matched} / {total} ({matched / total * 100:.1f}%)")
    print(f"json valid:               {int(df['json_valid'].sum())} / {total} "
          f"({df['json_valid'].mean() * 100:.1f}%)")
    print("\nper stage:")
    for stage, group in df.groupby("stage"):
        print(f"  {stage:10s} rows {len(group):4d}  "
              f"decision match {group['decision_matched'].mean() * 100:5.1f}%  "
              f"json valid {group['json_valid'].mean() * 100:5.1f}%")

    section("OFFER DEVIATION (rows where both paths bid)")
    dev = df["offer_pct_deviation"].dropna()
    if dev.empty:
        print("no rows where both paths returned BID")
    else:
        print(f"rows with both BID:       {len(dev)}")
        print(f"mean abs deviation:       {dev.abs().mean():.1f}%")
        print(f"median abs deviation:     {dev.abs().median():.1f}%")
        print(f"p90 abs deviation:        {dev.abs().quantile(0.9):.1f}%")
        exact = (dev.abs() < 0.5).sum()
        print(f"within 0.5% of rule:      {exact} / {len(dev)} ({exact / len(dev) * 100:.1f}%)")
        cdev = df["completion_days_deviation"].dropna()
        print(f"completion days, mean abs deviation: {cdev.abs().mean():.2f} days, "
              f"exact match {(cdev == 0).sum()} / {len(cdev)}")

    print(f"\ncsv written to {csv_path}")


if __name__ == "__main__":
    asyncio.run(run())
