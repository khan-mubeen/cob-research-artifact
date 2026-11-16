"""latency diagnostic: base, adapter attached and adapter merged, on validation rows, one request at a time

for one cell (mode, model) at its epoch selected on validation, three conditions are kept resident on one
gcd and run on the same seeded validation rows, interleaved in blocks with a rotating order:
  base      the base checkpoint, loaded as the official profile loads it
  attached  the base checkpoint with the lora adapter attached, as the official profile loads it
  merged    a fresh 16-bit copy with the adapter merged in memory (never written to disk)
per request: input and output tokens (stop token included), time to first token, total latency, time per
output token after the first, and the generated text. resumable: records are appended per request and a
rerun skips what is done (after fresh warm-up requests, logged separately).

  PYTHONDONTWRITEBYTECODE=1 COB_DATASET_VERSION=v5 python run_diagnostic.py --mode sealed --model gemma
  ... --dry-run          plan only, no model is loaded
  ... --rows 30 --hook-check 20 --out pilot     the pilot
"""
import argparse
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "config.py").exists())
sys.path.insert(0, str(REPO))

from src.experiments.v5 import model_registry, stop_rule  # noqa: E402  (read only)
from src.experiments.v5.evaluate import load_rows  # noqa: E402  (validation rows only)

CONDITIONS = ("base", "attached", "merged")
SEED, WARMUP, MAX_NEW_TOKENS = 7, 3, 200


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("sealed", "reverse_auction", "supervisor"))
    ap.add_argument("--model", required=True, choices=sorted(model_registry.MODELS))
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--block", type=int, default=10, help="rows per block before the condition order rotates")
    ap.add_argument("--hook-check", type=int, default=0, help="rows timed with and without the first-token hook")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def plan(a):
    sel = json.loads((REPO / "reports/v5/selection" / a.mode / a.model / "selection.json").read_text())
    epoch = sel["chosen"]["epoch"]
    run = f"{a.model}-{a.mode}-v5-r01"
    adapter = REPO / "models/training" / a.mode / a.model / run / f"epoch-{epoch}"
    base = REPO / "models/base" / a.model
    rows = load_rows(a.mode, "validation")
    picked = random.Random(SEED).sample(range(len(rows)), a.rows + WARMUP)
    return {"epoch": epoch, "adapter": adapter, "base": base, "rows": rows, "warmup": picked[:WARMUP],
            "measure": picked[WARMUP:], "window": model_registry.MODES[a.mode]["window"]}


def gpu_state():
    import torch
    free, total = torch.cuda.mem_get_info()
    out = {"free_gib": round(free / 2 ** 30, 2), "total_gib": round(total / 2 ** 30, 2)}
    try:
        out["rocm_smi"] = subprocess.run(["rocm-smi", "--showuse", "--showmemuse"], capture_output=True, text=True,
                                         timeout=30).stdout[-1500:]
    except Exception as e:
        out["rocm_smi"] = f"unavailable: {type(e).__name__}"
    return out


def load(path, window, merge=False):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(path), max_seq_length=window,
                                                         load_in_4bit=False, dtype=None)
    if merge:
        model = model.merge_and_unload()     # in memory only
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token
    return model, tokenizer, text_tok


def make_hook():
    import torch
    from transformers import StoppingCriteria

    class FirstToken(StoppingCriteria):
        """records the time of the first generated token; never stops generation"""
        def __init__(self):
            self.t, self.calls = None, 0

        def __call__(self, input_ids, scores, **kw):
            self.calls += 1
            if self.t is None:
                torch.cuda.synchronize()
                self.t = time.perf_counter()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    return FirstToken()


def time_one(c, row, hook=True):
    """the official profile's timed region (generate between two synchronisations), plus a first-token hook"""
    import torch
    from transformers import StoppingCriteriaList
    model, tokenizer, text_tok, stops = c["model"], c["tokenizer"], c["text_tok"], c["stops"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = text_tok(prompt, return_tensors="pt").to(model.device)
    n_in = int(encoded["input_ids"].shape[1])
    h = make_hook() if hook else None
    extra = {"stopping_criteria": StoppingCriteriaList([h])} if hook else {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=text_tok.pad_token_id,
                             **({"eos_token_id": list(stops)} if stops else {}), **extra)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    ids = [int(t) for t in out[0][n_in:].tolist()]
    n_out = len(ids)
    ttft = (h.t - t0) if hook and h.t else None
    return {"input_tokens": n_in, "output_tokens": n_out, "total_ms": round(total * 1000, 3),
            "ttft_ms": round(ttft * 1000, 3) if ttft is not None else None,
            "tpot_ms": round((total - ttft) * 1000 / (n_out - 1), 4) if ttft is not None and n_out > 1 else None,
            "hook_calls": h.calls if hook else None, "hit_cap": n_out >= MAX_NEW_TOKENS,
            "text": text_tok.decode(out[0][n_in:], skip_special_tokens=True)}


def main(argv=None):
    a = parse_args(argv)
    p = plan(a)
    cell_dir = HERE / a.out / a.mode / a.model
    rec_path, done_path = cell_dir / "records.jsonl", cell_dir / "cell_done.json"
    print(f"{a.mode} {a.model}: epoch {p['epoch']}, {len(p['measure'])} validation rows (seed {SEED}) of "
          f"{len(p['rows'])}, window {p['window']}, conditions {CONDITIONS}, out {cell_dir}", flush=True)
    if a.dry_run:
        print(f"  base {p['base']} exists {p['base'].exists()}; adapter {p['adapter']} exists {p['adapter'].exists()}")
        print(f"  first rows {p['measure'][:5]}, warm-up rows {p['warmup']}")
        return 0
    if done_path.exists():
        print("  cell already complete")
        return 0
    import torch
    cell_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "measure":
                done.add((r["condition"], r["row"]))
    meta = {"mode": a.mode, "model": a.model, "epoch": p["epoch"], "rows": len(p["measure"]), "seed": SEED,
            "split": "validation", "node": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(0), "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "gpu_at_start": gpu_state(), "started": time.strftime("%H:%M:%S"), "python": platform.python_version()}
    import peft
    import transformers
    import unsloth
    meta["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                        "unsloth": getattr(unsloth, "__version__", "?")}
    conds, t_load = {}, {}
    for name in CONDITIONS:
        t = time.perf_counter()
        model, tok, text_tok = load(p["adapter"] if name != "base" else p["base"], p["window"], merge=name == "merged")
        stops, info = stop_rule.stop_ids(model, tok, a.model, a.mode)
        conds[name] = {"model": model, "tokenizer": tok, "text_tok": text_tok, "stops": stops, "stop_rule": info,
                       "attn": getattr(model.config, "_attn_implementation", None),
                       "peft": type(model).__name__}
        t_load[name] = round(time.perf_counter() - t, 1)
        print(f"  loaded {name} in {t_load[name]} s: {conds[name]['peft']}, attention {conds[name]['attn']}, "
              f"stop ids {stops}", flush=True)
    meta["load_seconds"] = t_load
    meta["conditions"] = {n: {"class": c["peft"], "attn": c["attn"], "stop_rule": c["stop_rule"]} for n, c in conds.items()}
    meta["gpu_after_load"] = gpu_state()

    def write(rec):
        with open(rec_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    kind = "warmup_resume" if done else "warmup"
    for name in CONDITIONS:
        for i in p["warmup"]:
            write({"kind": kind, "condition": name, "row": i, **time_one(conds[name], p["rows"][i])})
    if a.hook_check:
        for i in p["measure"][:a.hook_check]:
            for hook in (True, False):
                r = time_one(conds["attached"], p["rows"][i], hook=hook)
                write({"kind": "hook_check", "condition": "attached", "row": i, "hook": hook, **r})
    t_measure = time.perf_counter()
    blocks = [p["measure"][k:k + a.block] for k in range(0, len(p["measure"]), a.block)]
    for b, rows in enumerate(blocks):
        order = CONDITIONS[b % 3:] + CONDITIONS[:b % 3]
        for name in order:
            for i in rows:
                if (name, i) in done:
                    continue
                write({"kind": "measure", "condition": name, "row": i, "block": b, **time_one(conds[name], p["rows"][i])})
        print(f"  block {b + 1}/{len(blocks)} done, {time.perf_counter() - t_measure:.0f} s", flush=True)
    meta["measure_seconds"] = round(time.perf_counter() - t_measure, 1)
    meta["finished"] = time.strftime("%H:%M:%S")
    done_path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(f"  cell complete in {meta['measure_seconds']} s of measurement", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
