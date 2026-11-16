"""
evaluate a bidding model (base or base+lora adapter) on the fixed eval sets.

runs the same hf runtime for whichever checkpoint is given so base-vs-finetuned
comparisons are apples to apples, quantization effects are a separate later
step. prompts are rendered with enable_thinking=False, which produces exactly
the assistant prefix the deployed gguf template uses.

eval sets, fixed in advance:
- the six baseline cases from nobid_edge_baseline.py (expected NO_BID)
- the 8 edge holdout examples (4 NO_BID, 4 near-boundary BID)
- a seeded 500-per-stage subsample of the main holdout jsonl files

usage:
    python src/experiments/eval_finetune.py --tag base
    python src/experiments/eval_finetune.py --tag finetuned --adapter models/epoch-1/production/qwen3.5-4b-bidding-lora-r1/final

with COB_DATASET_VERSION=v5 the v5 evaluation runs instead, on the full validation or test split:
see src/experiments/v5/evaluate.py. without it, everything below is unchanged
"""

import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime

from unsloth import FastLanguageModel  # must import before transformers

import torch

from config import DATA_VERSION, PROJECT_ROOT
from src.common.slm import (
    _extract_json,
    build_logistics_prompt,
    build_manufacturer_prompt,
    build_material_prompt,
)
from src.experiments.nobid_edge_baseline import CASES as BASELINE_CASES, load_agent
from src.experiments.report_paths import report_dir

# baseline-six prompts are rebuilt live, so the stage -> builder mapping must be
# explicit: an unrecognised stage has to fail loudly rather than silently render
# through whichever builder a default happened to point at.
STAGE_PROMPT_BUILDERS = {
    "material": build_material_prompt,
    "tailoring": build_manufacturer_prompt,
    "logistics": build_logistics_prompt,
}

BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 1024
RANDOM_SEED = 42
HOLDOUT_PER_STAGE = 500
GEN_BATCH = 16
MAX_NEW_TOKENS = 200

DATA_DIR = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION
DEFAULT_RESULT_EPOCH = 1

STAGES = ("material", "tailoring", "logistics")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_eval_sets():
    sets = {}

    baseline = []
    for case in BASELINE_CASES:
        agent = load_agent(case["stage"], case["agent_id"])
        try:
            prompt_fn = STAGE_PROMPT_BUILDERS[case["stage"]]
        except KeyError:
            raise ValueError(
                f"baseline case {case['case_id']!r} has unknown stage {case['stage']!r}; "
                f"expected one of {sorted(STAGE_PROMPT_BUILDERS)}") from None
        system_prompt, user_prompt = prompt_fn(agent, case["order"])
        baseline.append({
            "id": case["case_id"], "stage": case["stage"], "system": system_prompt,
            "user": user_prompt, "label": {"decision": "NO_BID"},
        })
    sets["baseline_six"] = baseline

    edge = []
    for stage in ("tailoring", "logistics"):
        for r in read_jsonl(DATA_DIR / f"edge_{stage}_holdout.jsonl"):
            edge.append({
                "id": r["order_id"] + "/" + r["agent_id"], "stage": r["stage"],
                "system": r["system"], "user": r["user"],
                "label": json.loads(r["completion"]),
            })
    sets["edge_holdout"] = edge

    rng = random.Random(RANDOM_SEED)
    holdout = []
    for stage in STAGES:
        rows = read_jsonl(DATA_DIR / f"{stage}_holdout.jsonl")
        for r in rng.sample(rows, min(HOLDOUT_PER_STAGE, len(rows))):
            holdout.append({
                "id": r["order_id"] + "/" + r["agent_id"], "stage": r["stage"],
                "system": r["system"], "user": r["user"],
                "label": json.loads(r["completion"]),
            })
    sets["main_holdout"] = holdout

    # boundary_stress: development stress set, context only (its results fed
    # round-three design). final_boundary_holdout: untouched independent
    # evidence, required quality condition from round three on.
    # protocol_v2/robustness_holdout: tier-3 locked v2 gate (protocol_v2.json)
    for name, path in (
            ("boundary_stress", DATA_DIR / "boundary_stress.jsonl"),
            ("final_boundary_holdout", DATA_DIR / "final_boundary_holdout.jsonl"),
            ("robustness_holdout_v2", DATA_DIR / "protocol_v2" / "robustness_holdout.jsonl")):
        if path.exists():
            sets[name] = [{
                "id": r["order_id"], "stage": r["category"],
                "system": r["system"], "user": r["user"],
                "label": json.loads(r["completion"]),
            } for r in read_jsonl(path)]

    # explanatory diagnostics (section 3/4): scale groups by counterfactual
    # family, json-format groups by offer-magnitude band. label key differs
    # per generator, so accept either "completion" or "completion_json".
    for name, path, group, label_key in (
            ("scale_diagnostic", DATA_DIR / "protocol_v2" / "scale_diagnostic.jsonl", "family", "completion_json"),
            ("json_format_stress", DATA_DIR / "protocol_v2" / "json_format_stress.jsonl", "band", "completion"),
            ("magnitude_diagnostic", DATA_DIR / "protocol_v2" / "magnitude_diagnostic.jsonl", "family", "completion_json")):
        if path.exists():
            sets[name] = [{
                "id": r["order_id"], "stage": r[group],
                "system": r["system"], "user": r["user"],
                "label": json.loads(r[label_key]),
            } for r in read_jsonl(path)]

    return sets


def generate_all(model, tokenizer, examples, stop_ids=None):
    # stop_ids: extra stop ids from the v5 stop rule (src/experiments/v5/stop_rule.py). v4 never
    # passes them, and None passes nothing to generate(), so the default call is unchanged
    stop_kw = {"eos_token_id": list(stop_ids)} if stop_ids else {}
    # this checkpoint is multimodal so unsloth returns a processor, text-only
    # encode/decode must go through the inner tokenizer or positional text
    # lands in the image pathway
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text_tok.padding_side = "left"
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    outputs = []
    started = time.perf_counter()
    for i in range(0, len(examples), GEN_BATCH):
        batch = examples[i:i + GEN_BATCH]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": e["system"]},
                 {"role": "user", "content": e["user"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for e in batch
        ]
        enc = text_tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=text_tok.pad_token_id, **stop_kw,
            )
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            outputs.append(text_tok.decode(gen, skip_special_tokens=True))
        done = i + len(batch)
        print(f"  generated {done}/{len(examples)} "
              f"({done / (time.perf_counter() - started):.1f} ex/s)", flush=True)
    return outputs


def score(examples, outputs):
    rows = []
    for e, raw in zip(examples, outputs):
        parsed = _extract_json(raw)
        json_valid = isinstance(parsed, dict)
        decision = parsed.get("decision") if json_valid else None
        label = e["label"]
        row = {
            "id": e["id"], "stage": e["stage"], "json_valid": json_valid,
            "label_decision": label["decision"], "model_decision": decision,
            "decision_matched": decision == label["decision"],
            "raw": raw[:400],
        }
        if json_valid and decision == "BID" and label["decision"] == "BID":
            try:
                model_offer = float(parsed.get("offer_amount"))
                label_offer = float(label.get("offer_amount"))
                row["offer_pct_dev"] = abs(model_offer - label_offer) / label_offer * 100
            except (TypeError, ValueError, ZeroDivisionError):
                row["offer_pct_dev"] = None
            try:
                predicted_days = float(parsed.get("estimated_completion_days"))
                expected_days = float(label.get("estimated_completion_days"))
                row["days_matched"] = predicted_days == expected_days
                row["days_abs_dev"] = abs(predicted_days - expected_days)
                # the denominator is the rule's own expected duration, never days_until_deadline:
                # that would measure error against deadline slack, not completion-time prediction
                row["days_pct_dev"] = (row["days_abs_dev"] / expected_days * 100) if expected_days > 0 else None
            except (TypeError, ValueError):
                row["days_matched"] = False
                row["days_abs_dev"] = None
                row["days_pct_dev"] = None
        rows.append(row)
    return rows


def summarize(name, rows):
    print(f"\n== {name} ({len(rows)} examples) ==")
    stages = sorted({r["stage"] for r in rows})
    summary = {}
    for stage in stages:
        sr = [r for r in rows if r["stage"] == stage]
        match = sum(r["decision_matched"] for r in sr)
        jv = sum(r["json_valid"] for r in sr)
        devs = [r["offer_pct_dev"] for r in sr if r.get("offer_pct_dev") is not None]
        days = [r for r in sr if "days_matched" in r]
        abs_days = [r["days_abs_dev"] for r in sr if r.get("days_abs_dev") is not None]
        pct_days = [r["days_pct_dev"] for r in sr if r.get("days_pct_dev") is not None]
        # a row counts toward the percentage only when the rule's expected duration is above zero
        excluded = sum(1 for r in sr if r.get("days_abs_dev") is not None and r.get("days_pct_dev") is None)
        summary[stage] = {
            "n": len(sr),
            "decision_agreement_pct": round(match / len(sr) * 100, 2),
            "json_valid_pct": round(jv / len(sr) * 100, 2),
            "mean_abs_offer_dev_pct": round(statistics.mean(devs), 2) if devs else None,
            "median_abs_offer_dev_pct": round(statistics.median(devs), 2) if devs else None,
            "days_exact_match_pct": round(
                sum(r["days_matched"] for r in days) / len(days) * 100, 2) if days else None,
            # the primary completion-time metric: a one-day miss and a thirty-day miss differ here
            "mean_abs_days_dev": round(statistics.mean(abs_days), 3) if abs_days else None,
            "days_dev_rows": len(abs_days),
            # secondary: a short true duration turns a small absolute miss into a large percentage
            "mean_days_dev_pct": round(statistics.mean(pct_days), 2) if pct_days else None,
            "days_dev_pct_rows_included": len(pct_days),
            "days_dev_pct_rows_excluded_expected_days_zero": excluded,
        }
        s = summary[stage]
        print(f"  {stage:10s} n={s['n']:4d}  decisions {s['decision_agreement_pct']:6.2f}%  "
              f"json {s['json_valid_pct']:6.2f}%  "
              f"offer dev mean {s['mean_abs_offer_dev_pct']}%  "
              f"days exact {s['days_exact_match_pct']}%  "
              f"days dev {s['mean_abs_days_dev']}d")
    # one overall figure, the equal mean of the stage agreements. this is deliberately the SAME
    # quantity src/experiments/v5/epoch_selection.py already ranks checkpoints by, so the reported
    # headline and the selection metric cannot drift apart. it is a macro mean over stages, not a
    # pooled row-level accuracy, and the name says so
    agreements = [summary[stage]["decision_agreement_pct"] for stage in stages]
    summary["macro"] = {
        "stages": list(stages),
        "macro_decision_agreement_pct": round(statistics.mean(agreements), 2) if agreements else None,
        "definition": "equal mean of the per-stage decision agreement, the v5 epoch-selection key",
    }
    print(f"  {'macro':10s} decision agreement {summary['macro']['macro_decision_agreement_pct']}% "
          f"over {len(stages)} stage(s)")
    return summary


def main():
    # a v5 dataset context takes the v5 evaluation before any v4 argument is parsed
    if os.getenv("COB_DATASET_VERSION", "").strip():
        from src.dataset import context as dataset_context
        if dataset_context.resolve().version == "v5":
            from src.experiments.v5 import evaluate
            return evaluate.main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="", help="path to lora adapter dir, empty = base model")
    parser.add_argument("--tag", required=True, help="label for the output json, e.g. base / finetuned")
    parser.add_argument("--dry-run", action="store_true", help="load eval sets only, no model")
    # cross-model protocol args, defaults reproduce the qwen evals exactly
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="hf checkpoint to evaluate when no --adapter is given")
    parser.add_argument("--revision", default=None, help="exact hf revision of --base-model")
    parser.add_argument("--only-set", default="",
                        help="evaluate a single named set (e.g. robustness_holdout_v2), "
                             "for cheap re-evals of already-scored checkpoints on a new set")
    parser.add_argument("--result-epoch", type=int, default=DEFAULT_RESULT_EPOCH,
                        help="which epoch-N report tree to write raw results into; the "
                             "evaluation sets themselves are frozen and identical across epochs")
    args = parser.parse_args()
    report_root = report_dir("production", "evaluation/raw", args.result_epoch)

    sets = load_eval_sets()
    if args.only_set:
        names = [s.strip() for s in args.only_set.split(",") if s.strip()]
        sets = {n: sets[n] for n in names}
    for name, examples in sets.items():
        print(f"{name}: {len(examples)} examples")
    if args.dry_run:
        return

    model_name = args.adapter or args.base_model
    print(f"\nloading {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, revision=args.revision, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False, dtype=None,
    )
    FastLanguageModel.for_inference(model)

    results = {"tag": args.tag, "model": model_name,
               "generated_at": datetime.now().isoformat(timespec="seconds")}
    for name, examples in sets.items():
        print(f"\ngenerating for {name}...")
        outputs = generate_all(model, tokenizer, examples)
        rows = score(examples, outputs)
        results[name] = {"summary": summarize(name, rows), "rows": rows}

    if "baseline_six" in results:
        baseline_rows = results["baseline_six"]["rows"]
        correct = sum(r["model_decision"] == "NO_BID" for r in baseline_rows)
        results["baseline_six"]["no_bid_correct"] = correct
        print(f"\nbaseline six: {correct}/6 NO_BID correct")

    report_root.mkdir(parents=True, exist_ok=True)
    out_path = report_root / f"eval_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
