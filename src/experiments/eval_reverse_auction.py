"""
evaluate a reverse-auction checkpoint against the rule's own decision, and report
pass/fail against the launch gates.

the round rule is narrow: offer exactly current_best_bid - platform_minimum_
decrement, or stay silent when that falls below the agent's minimum_bid. so a
pooled accuracy number is close to meaningless here, and this script never
reports one on its own. three breakdowns exist to stop a model hiding in the
aggregate:

  by priority   the training mix is 88.7% BID on urgent and 4.5% on flexible,
                because the flexible -15% markup opens most agents AT their
                floor. a model that learns "flexible -> NO_BID" and nothing else
                scores well pooled, so BID recall and NO_BID recall are reported
                per priority, not just accuracy
  by category   floor_boundary rows sit one cent either side of the floor and
                are 100%/0% BID by construction. a model gaming the auction_round
                distribution cannot also pass these
  withdrawn     rows where the agent had already dropped out; the label is still
                the rule's, since current_best_bid is non-increasing

offer accuracy is reported as an exact match to the cent, not only as a percent
deviation: the rule mandates one number, and "close" is wrong.

gates, first version: floor accuracy >= 90% on each of the three cases, model BID
rate within 10 points of the label rate for every priority and every stage, JSON
validity >= 99%, and no days_exact_match regression against the reference arm.

usage:
    python src/experiments/eval_reverse_auction.py --tag base
    python src/experiments/eval_reverse_auction.py --tag ra-smoke-1ep --adapter <dir>
"""

import argparse
import json
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime

from unsloth import FastLanguageModel  # must import before transformers

import torch

from config import DATA_VERSION, PROJECT_ROOT, REVERSE_AUCTION_VERSION
from src.common.reverse_auction import decide_round_bid
from src.common.slm import _extract_json

BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_SEQ_LENGTH = 1024
RANDOM_SEED = 42
HOLDOUT_PER_STAGE = 300
GEN_BATCH = 16
MAX_NEW_TOKENS = 200

# pass/fail bars, first version
GATE_FLOOR_MIN_ACCURACY = 90.0
GATE_BID_RATE_TOLERANCE = 10.0
GATE_JSON_VALID_MIN = 99.0

DATA_DIR = (PROJECT_ROOT / "data" / "finetune" / DATA_VERSION / "reverse_auction"
            / REVERSE_AUCTION_VERSION)
STAGES = ("material", "tailoring", "logistics")
PRIORITIES = ("urgent", "normal", "flexible")

# the priority the model was actually shown, read back from the prompt rather
# than re-joined from orders.csv, so the grouping cannot disagree with the input
PRIORITY_RE = re.compile(r"^priority: (\w+)$", re.M)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prompt_priority(user):
    match = PRIORITY_RE.search(user)
    if match is None or match.group(1) not in PRIORITIES:
        raise ValueError(f"no usable priority line in prompt: {user[:120]!r}")
    return match.group(1)


def load_holdout(per_stage, holdout_path=None):
    """the fixed eval rows, either one carved file or a seeded per-stage sample

    every floor_boundary row is taken, never sampled: they are 1.6% of the
    holdout, so a uniform sample leaves single digits and the floor slice this
    script exists to report cannot be read. the auction_round rows are sampled
    per stage as usual and the two are reported separately"""
    if holdout_path:
        rows = read_jsonl(holdout_path)
    else:
        rng = random.Random(RANDOM_SEED)
        rows = []
        for stage in STAGES:
            stage_rows = read_jsonl(DATA_DIR / stage / "holdout.jsonl")
            floor = [r for r in stage_rows if r["category"].startswith("floor_boundary")]
            auction = [r for r in stage_rows if not r["category"].startswith("floor_boundary")]
            rows.extend(rng.sample(auction, min(per_stage, len(auction))))
            rows.extend(floor)
    for row in rows:
        row["priority"] = prompt_priority(row["user"])
    return rows


def verify_labels(rows, products, agents_by_stage, order_rows, build_stage_orders):
    """re-derive every stored label from decide_round_bid()

    the eval measures agreement with the rule, so the stored labels have to be
    the rule's. a stale or hand-edited dataset fails here instead of silently
    scoring the model against the wrong truth"""
    mismatches = []
    for row in rows:
        stage = row["stage"]
        rfq = build_stage_orders(order_rows[row["order_id"]], products)[stage]
        seen = {**rfq,
                "round_number": row["round_number"],
                "current_best_bid": row["current_best_bid"],
                "platform_minimum_decrement": row["platform_minimum_decrement"]}
        fresh = decide_round_bid(stage, agents_by_stage[stage][row["agent_id"]], seen)
        stored = json.loads(row["completion"])
        if fresh["decision"] != stored["decision"]:
            mismatches.append(row["order_id"] + "/" + row["agent_id"])
        elif fresh["decision"] == "BID" and round(fresh["offer_amount"], 2) != stored["offer_amount"]:
            mismatches.append(row["order_id"] + "/" + row["agent_id"])
    return mismatches


def generate_all(model, tokenizer, rows, stop_ids=None):
    # stop_ids: extra stop ids from the v5 stop rule (src/experiments/v5/stop_rule.py). v4 never
    # passes them, and None passes nothing to generate(), so the default call is unchanged
    stop_kw = {"eos_token_id": list(stop_ids)} if stop_ids else {}
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text_tok.padding_side = "left"
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    outputs = []
    started = time.perf_counter()
    for i in range(0, len(rows), GEN_BATCH):
        batch = rows[i:i + GEN_BATCH]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system"]},
                 {"role": "user", "content": r["user"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for r in batch
        ]
        enc = text_tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=text_tok.pad_token_id, **stop_kw)
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            outputs.append(text_tok.decode(gen, skip_special_tokens=True))
        done = i + len(batch)
        print(f"  generated {done}/{len(rows)} "
              f"({done / (time.perf_counter() - started):.2f} ex/s)", flush=True)
    return outputs


def score(rows, outputs):
    scored = []
    for row, raw in zip(rows, outputs):
        parsed = _extract_json(raw)
        json_valid = isinstance(parsed, dict)
        decision = parsed.get("decision") if json_valid else None
        label = json.loads(row["completion"])
        entry = {
            "id": row["order_id"] + "/" + row["agent_id"],
            "stage": row["stage"],
            "priority": row["priority"],
            "category": row["category"],
            "round_number": row["round_number"],
            "withdrawn_in_earlier_round": row["withdrawn_in_earlier_round"],
            "current_best_bid": row["current_best_bid"],
            "platform_minimum_decrement": row["platform_minimum_decrement"],
            "json_valid": json_valid,
            "label_decision": label["decision"],
            "model_decision": decision,
            "decision_matched": decision == label["decision"],
            "offer_exact": None,
            "offer_pct_dev": None,
            "days_matched": None,
            "days_abs_dev": None,
            "days_pct_dev": None,
            "raw": raw[:400],
        }
        if json_valid and decision == "BID" and label["decision"] == "BID":
            try:
                model_offer = round(float(parsed.get("offer_amount")), 2)
                label_offer = float(label["offer_amount"])
                entry["offer_exact"] = model_offer == label_offer
                entry["offer_pct_dev"] = abs(model_offer - label_offer) / label_offer * 100
            except (TypeError, ValueError, ZeroDivisionError):
                entry["offer_exact"] = False
            try:
                predicted_days = float(parsed.get("estimated_completion_days"))
                expected_days = float(label["estimated_completion_days"])
                entry["days_matched"] = predicted_days == expected_days
                entry["days_abs_dev"] = abs(predicted_days - expected_days)
                # against the rule's own expected duration, never deadline slack
                entry["days_pct_dev"] = (entry["days_abs_dev"] / expected_days * 100) if expected_days > 0 else None
            except (TypeError, ValueError):
                entry["days_matched"] = False
        scored.append(entry)
    return scored


def _pct(part, whole):
    return round(part / whole * 100, 2) if whole else None


def group_stats(rows):
    """one group's numbers, recall per class so a majority-class guesser shows"""
    n = len(rows)
    label_bid = [r for r in rows if r["label_decision"] == "BID"]
    label_nobid = [r for r in rows if r["label_decision"] != "BID"]
    exact = [r["offer_exact"] for r in rows if r["offer_exact"] is not None]
    devs = [r["offer_pct_dev"] for r in rows if r["offer_pct_dev"] is not None]
    days = [r["days_matched"] for r in rows if r["days_matched"] is not None]
    abs_days = [r["days_abs_dev"] for r in rows if r.get("days_abs_dev") is not None]
    pct_days = [r["days_pct_dev"] for r in rows if r.get("days_pct_dev") is not None]
    excluded = sum(1 for r in rows if r.get("days_abs_dev") is not None and r.get("days_pct_dev") is None)
    return {
        "n": n,
        "accuracy_pct": _pct(sum(r["decision_matched"] for r in rows), n),
        "json_valid_pct": _pct(sum(r["json_valid"] for r in rows), n),
        "label_bid_rate_pct": _pct(len(label_bid), n),
        "model_bid_rate_pct": _pct(sum(r["model_decision"] == "BID" for r in rows), n),
        "bid_recall_pct": _pct(sum(r["decision_matched"] for r in label_bid), len(label_bid)),
        "nobid_recall_pct": _pct(sum(r["decision_matched"] for r in label_nobid), len(label_nobid)),
        "offer_exact_match_pct": _pct(sum(exact), len(exact)),
        "mean_abs_offer_dev_pct": round(statistics.mean(devs), 4) if devs else None,
        "median_abs_offer_dev_pct": round(statistics.median(devs), 4) if devs else None,
        "days_exact_match_pct": _pct(sum(days), len(days)),
        # primary completion-time metric, magnitude rather than exact match
        "mean_abs_days_dev": round(statistics.mean(abs_days), 3) if abs_days else None,
        "days_dev_rows": len(abs_days),
        # secondary, against the rule's expected duration; rows with expected 0 are excluded
        "mean_days_dev_pct": round(statistics.mean(pct_days), 2) if pct_days else None,
        "days_dev_pct_rows_included": len(pct_days),
        "days_dev_pct_rows_excluded_expected_days_zero": excluded,
    }


def by(rows, key):
    buckets = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return {str(k): group_stats(v) for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def summarize(tag, model_name, rows):
    overall = group_stats(rows)
    summary = {
        "tag": tag,
        "model": model_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "by_priority": by(rows, lambda r: r["priority"]),
        "by_category": by(rows, lambda r: r["category"].split(":")[0]),
        "by_floor_case": by([r for r in rows if r["category"].startswith("floor_boundary")],
                            lambda r: r["category"]),
        "by_stage": by(rows, lambda r: r["stage"]),
        "by_round": by(rows, lambda r: f"round_{r['round_number']}"),
        "withdrawn": group_stats([r for r in rows if r["withdrawn_in_earlier_round"]]),
        "not_withdrawn": group_stats([r for r in rows if not r["withdrawn_in_earlier_round"]]),
        "confusion": {
            "label_BID_model_BID": sum(1 for r in rows if r["label_decision"] == "BID" and r["model_decision"] == "BID"),
            "label_BID_model_NO_BID": sum(1 for r in rows if r["label_decision"] == "BID" and r["model_decision"] == "NO_BID"),
            "label_NO_BID_model_BID": sum(1 for r in rows if r["label_decision"] == "NO_BID" and r["model_decision"] == "BID"),
            "label_NO_BID_model_NO_BID": sum(1 for r in rows if r["label_decision"] == "NO_BID" and r["model_decision"] == "NO_BID"),
            "unparseable": sum(1 for r in rows if r["model_decision"] is None),
        },
    }
    return summary


def evaluate_gates(summary, days_baseline=None):
    """pass/fail against the bars rather than numbers alone

    the bid-rate bars compare the model's BID RATE to the label BID rate, not its
    recall to the label rate: those are different quantities, and the literal
    recall-vs-prevalence reading would let a model that refuses every flexible
    row pass, which is the exact shortcut these bars exist to catch. recall is
    carried in the evidence"""
    gates = {}

    for index, case in enumerate(("one_cent_below_floor", "at_floor", "above_floor"), start=1):
        stats = summary["by_floor_case"].get(f"floor_boundary:{case}")
        bar = f">= {GATE_FLOOR_MIN_ACCURACY}% accuracy"
        if stats is None:
            gates[f"{index}_floor_{case}"] = {
                "status": "NOT_EVALUATED", "bar": bar, "measured": None,
                "evidence": "no rows of this case in the eval set",
            }
            continue
        measured = stats["accuracy_pct"]
        gates[f"{index}_floor_{case}"] = {
            "status": "PASS" if measured >= GATE_FLOOR_MIN_ACCURACY else "FAIL",
            "bar": bar, "measured": measured, "n": stats["n"],
            "evidence": f"{stats['n']} rows, accuracy {measured}%, "
                        f"model bid rate {stats['model_bid_rate_pct']}% "
                        f"against a label rate of {stats['label_bid_rate_pct']}%",
        }

    for index, (section, label) in enumerate((("by_priority", "priority"), ("by_stage", "stage")), start=4):
        for group, stats in summary[section].items():
            gap = round(abs(stats["model_bid_rate_pct"] - stats["label_bid_rate_pct"]), 2)
            gates[f"{index}_bid_rate_{label}_{group}"] = {
                "status": "PASS" if gap <= GATE_BID_RATE_TOLERANCE else "FAIL",
                "bar": f"model BID rate within {GATE_BID_RATE_TOLERANCE} points of the label rate",
                "measured": gap, "n": stats["n"],
                "evidence": f"model {stats['model_bid_rate_pct']}% vs label "
                            f"{stats['label_bid_rate_pct']}%, gap {gap} points, "
                            f"BID recall {stats['bid_recall_pct']}%, "
                            f"NO_BID recall {stats['nobid_recall_pct']}%",
            }

    json_valid = summary["overall"]["json_valid_pct"]
    gates["6_json_validity"] = {
        "status": "PASS" if json_valid >= GATE_JSON_VALID_MIN else "FAIL",
        "bar": f">= {GATE_JSON_VALID_MIN}%", "measured": json_valid,
        "evidence": f"{summary['overall']['n']} rows scored",
    }

    days = summary["overall"]["days_exact_match_pct"]
    if days_baseline is None:
        gates["7_days_no_regression"] = {
            "status": "NOT_EVALUATED", "bar": ">= the reference arm", "measured": days,
            "evidence": "pass --days-baseline with the reference arm's days_exact_match_pct",
        }
    else:
        gates["7_days_no_regression"] = {
            "status": "PASS" if days is not None and days >= days_baseline else "FAIL",
            "bar": f">= {days_baseline}% (reference arm)", "measured": days,
            "evidence": "estimated_completion_days is carried through from the round-1 "
                        "decision and must not degrade",
        }

    statuses = [g["status"] for g in gates.values()]
    failed = statuses.count("FAIL")
    decision = "PASS" if failed == 0 else f"FAIL ({failed} gate(s))"
    return {"gates": gates, "decision": decision,
            "counts": {s: statuses.count(s) for s in ("PASS", "FAIL", "NOT_EVALUATED")}}


def print_gates(result):
    print(f"\n== gates: {result['decision']} ==")
    print(f"  {'gate':44s} {'status':14s} {'measured':>9s}  bar")
    for name, gate in result["gates"].items():
        print(f"  {name:44s} {gate['status']:14s} {str(gate['measured']):>9s}  {gate['bar']}")


def print_summary(summary):
    o = summary["overall"]
    print(f"\n== {summary['tag']} on {o['n']} rows ==")
    print(f"  decisions {o['accuracy_pct']}%   json {o['json_valid_pct']}%   "
          f"offer exact {o['offer_exact_match_pct']}%   days exact {o['days_exact_match_pct']}%")
    for name in ("by_priority", "by_category", "by_floor_case", "by_stage", "by_round"):
        print(f"\n  {name}")
        print(f"    {'group':28s} {'n':>5s} {'acc':>7s} {'BIDrec':>7s} {'NOrec':>7s} "
              f"{'lblBID':>7s} {'mdlBID':>7s} {'offer=':>7s}")
        for group, s in summary[name].items():
            print(f"    {group:28s} {s['n']:5d} {str(s['accuracy_pct']):>7s} "
                  f"{str(s['bid_recall_pct']):>7s} {str(s['nobid_recall_pct']):>7s} "
                  f"{str(s['label_bid_rate_pct']):>7s} {str(s['model_bid_rate_pct']):>7s} "
                  f"{str(s['offer_exact_match_pct']):>7s}")
    w = summary["withdrawn"]
    print(f"\n  withdrawn rows n={w['n']} accuracy {w['accuracy_pct']}%  "
          f"(model bid rate {w['model_bid_rate_pct']}%, should be 0)")
    print(f"  confusion {summary['confusion']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="", help="lora adapter dir, empty = base model")
    ap.add_argument("--tag", required=True, help="label for the output json")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION,
                    help="exact hf revision, used only when no --adapter is given")
    ap.add_argument("--holdout", default="",
                    help="one carved jsonl instead of the seeded per-stage sample")
    ap.add_argument("--per-stage", type=int, default=HOLDOUT_PER_STAGE,
                    help="rows sampled per stage from the holdout files")
    ap.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument("--out-dir", default="reports/reverse_auction/smoke",
                    help="where the result json lands; real runs pass a version folder, "
                         "e.g. reports/reverse_auction/epoch-1/v3/evaluation/raw")
    ap.add_argument("--gates-dir", default="reports/reverse_auction/smoke",
                    help="where the pass/fail gates json lands, e.g. "
                         "reports/reverse_auction/epoch-1/v3/gates")
    ap.add_argument("--days-baseline", type=float, default=None,
                    help="reference arm's days_exact_match_pct for the no-regression gate")
    ap.add_argument("--no-verify-labels", action="store_true",
                    help="skip re-deriving the stored labels from decide_round_bid()")
    ap.add_argument("--dry-run", action="store_true", help="load and verify only, no model")
    args = ap.parse_args()

    rows = load_holdout(args.per_stage, args.holdout or None)
    counts = defaultdict(int)
    for row in rows:
        counts[(row["stage"], row["priority"])] += 1
    print(f"{len(rows)} eval rows")
    for (stage, priority), n in sorted(counts.items()):
        print(f"  {stage:10s} {priority:9s} {n:5d}")

    if not args.no_verify_labels:
        from src.visualization.data_utils import load_all_data
        from src.visualization.slm_competition_diagnostics import build_stage_orders
        materials, manufacturers, logistics, products, orders = load_all_data()
        agents_by_stage = {
            "material": {a["agent_id"]: a for a in materials.to_dict("records")},
            "tailoring": {a["agent_id"]: a for a in manufacturers.to_dict("records")},
            "logistics": {a["agent_id"]: a for a in logistics.to_dict("records")},
        }
        order_rows = {o["order_id"]: o for _, o in orders.iterrows()}
        mismatches = verify_labels(rows, products, agents_by_stage, order_rows, build_stage_orders)
        print(f"label check against decide_round_bid(): {len(mismatches)} mismatch(es)")
        if mismatches:
            raise SystemExit(f"stored labels disagree with the rule, e.g. {mismatches[:5]}")

    if args.dry_run:
        return

    model_name = args.adapter or args.base_model
    print(f"\nloading {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, revision=None if args.adapter else args.revision,
        max_seq_length=args.max_seq_length, load_in_4bit=False, dtype=None,
    )
    FastLanguageModel.for_inference(model)

    outputs = generate_all(model, tokenizer, rows)
    scored = score(rows, outputs)
    summary = summarize(args.tag, model_name, scored)
    print_summary(summary)

    gate_result = evaluate_gates(summary, args.days_baseline)
    print_gates(gate_result)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{args.tag}_{stamp}"
    path = out_dir / f"{stem}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "gates": gate_result, "rows": scored}, f, indent=2)

    # the gates file is the eval stem plus _gates, so one can be found from the
    # other by string match alone
    gates_dir = PROJECT_ROOT / args.gates_dir
    gates_dir.mkdir(parents=True, exist_ok=True)
    gates_path = gates_dir / f"{stem}_gates.json"
    with open(gates_path, "w", encoding="utf-8") as f:
        json.dump({
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "family": "reverse-auction rounds 1/2/3",
            "model": model_name,
            "revision": None if args.adapter else args.revision,
            "tag": args.tag,
            "eval_rows": summary["overall"]["n"],
            "eval_result": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            **gate_result,
        }, f, indent=2)
    print(f"\nwritten to {path}\n         {gates_path}")


if __name__ == "__main__":
    main()
