"""
evaluate a supervisor chain-selection checkpoint against the rule's own choice.

two metrics, both computed against decide_chain_selection()'s known-correct pick:

  selection accuracy - fraction where the model returns the rule's exact index
  regret             - how much worse the model's pick is on the strategy's OWN
                       objective, as a fraction of the rule's value. urgent is
                       judged on total days, flexible on total cost, normal on
                       the equal-weight budget-relative score. the rule takes the
                       minimum, so regret is >= 0 and 0 means an equally good
                       pick even when the index differs

accuracy alone is misleading here: many candidates are near-duplicates, so
picking "wrong" can cost nothing. regret is what says whether a miss matters.

usage:
    python src/experiments/eval_supervisor.py --adapter <dir> --holdout <jsonl> --tag <name>
"""

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from unsloth import FastLanguageModel  # must import before transformers

import torch

from config import PROJECT_ROOT
from src.common.bidding_formulas import chain_compound_risk
from src.common.slm import _extract_json

GEN_BATCH = 4
MAX_NEW_TOKENS = 64          # the answer is one short json object

CAND_RE = re.compile(
    r"candidate (\d+): "
    r"material\(cost_eur=([\d.]+), days=([\d.]+), defect_p=([\d.]+)\), "
    r"manufacturing\(cost_eur=([\d.]+), days=([\d.]+), defect_p=([\d.]+)\), "
    r"logistics\(cost_eur=([\d.]+), days=([\d.]+), damage_p=([\d.]+)\), "
    r"total_cost_eur=([\d.]+), total_completion_days=([\d.]+)")
BUDGET_RE = re.compile(r"budget \(EUR\): ([\d.]+)")
DAYS_RE = re.compile(r"days remaining until deadline: ([\d.]+)")
CAP_RE = re.compile(r"max acceptable defect or damage probability: ([\d.]+)")


def parse_prompt(user):
    """recover the candidate table from the prompt the model was actually shown,
    so the objective is computed on exactly that information"""
    cands = []
    for m in CAND_RE.finditer(user):
        g = m.groups()
        cands.append({
            "index": int(g[0]),
            "material_defect_probability": float(g[3]),
            "manufacturer_defect_probability": float(g[6]),
            "logistics_damage_probability": float(g[9]),
            "total_cost": float(g[10]),
            "total_days": float(g[11]),
        })
    order = {
        "budget": float(BUDGET_RE.search(user).group(1)),
        "days_until_deadline": float(DAYS_RE.search(user).group(1)),
        "max_defect_rate": float(CAP_RE.search(user).group(1)),
    }
    return cands, order


def objective(cand, order, strategy):
    """lower is better, matching the rule's min() for that strategy"""
    if strategy == "urgent":
        return cand["total_days"]
    if strategy == "flexible":
        return cand["total_cost"]
    risk = chain_compound_risk(cand)
    return ((cand["total_cost"] / order["budget"]) / 3
            + (cand["total_days"] / order["days_until_deadline"]) / 3
            + (risk / order["max_defect_rate"]) / 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="lora adapter dir, empty = base model")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--out-dir", default="reports/supervisor/smoke")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")]
    print(f"{len(rows)} holdout examples from {args.holdout}")

    name = args.adapter or args.base_model
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name, revision=None if args.adapter else args.revision,
        max_seq_length=args.max_seq_length, load_in_4bit=False, dtype=None)
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tok, "tokenizer", tok)
    text_tok.padding_side = "left"
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    outputs = []
    started = time.perf_counter()
    for i in range(0, len(rows), GEN_BATCH):
        batch = rows[i:i + GEN_BATCH]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
        enc = text_tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=text_tok.pad_token_id)
        for j in range(len(batch)):
            outputs.append(text_tok.decode(out[j][enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True))
        done = i + len(batch)
        print(f"  {done}/{len(rows)} ({done/(time.perf_counter()-started):.2f} ex/s)", flush=True)

    per_strategy = {}
    records = []
    n_valid = n_correct = 0
    regrets = []
    for r, raw in zip(rows, outputs):
        strategy = r["strategy"]
        truth = json.loads(r["completion"])["selected_candidate"]
        cands, order = parse_prompt(r["user"])
        parsed = _extract_json(raw)
        idx = parsed.get("selected_candidate") if isinstance(parsed, dict) else None
        valid = isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(cands)
        if valid:
            n_valid += 1
            best = objective(cands[truth], order, strategy)
            got = objective(cands[idx], order, strategy)
            reg = (got - best) / best if best > 0 else 0.0
        else:
            reg = None
        correct = bool(valid and idx == truth)
        n_correct += correct
        if reg is not None:
            regrets.append(reg)
        s = per_strategy.setdefault(strategy, {"n": 0, "correct": 0, "valid": 0, "regret": []})
        s["n"] += 1; s["correct"] += correct; s["valid"] += valid
        if reg is not None:
            s["regret"].append(reg)
        records.append({"order_id": r["order_id"], "strategy": strategy, "truth": truth,
                        "predicted": idx, "valid_json": valid, "correct": correct,
                        "regret": reg, "raw": raw[:200]})

    n = len(rows)
    summary = {
        "tag": args.tag, "model": name, "n": n,
        "json_valid_pct": round(n_valid / n * 100, 2),
        "selection_accuracy_pct": round(n_correct / n * 100, 2),
        "mean_regret_pct": round(statistics.mean(regrets) * 100, 3) if regrets else None,
        "median_regret_pct": round(statistics.median(regrets) * 100, 3) if regrets else None,
        "zero_regret_pct": round(sum(1 for x in regrets if x <= 1e-9) / len(regrets) * 100, 2)
                           if regrets else None,
        "p90_regret_pct": round(sorted(regrets)[int(0.9 * len(regrets))] * 100, 3) if regrets else None,
        "by_strategy": {k: {
            "n": v["n"],
            "accuracy_pct": round(v["correct"] / v["n"] * 100, 2),
            "json_valid_pct": round(v["valid"] / v["n"] * 100, 2),
            "mean_regret_pct": round(statistics.mean(v["regret"]) * 100, 3) if v["regret"] else None,
        } for k, v in sorted(per_strategy.items())},
    }
    print("\n" + json.dumps(summary, indent=2))

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": records}, f, indent=2)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
