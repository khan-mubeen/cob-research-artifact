"""
per-model feasibility probe, stages B-D of the cross-model matrix.

runs in the isolated .venv-feas environment, separate from the qwen training
venv. each step records its status and exact error; independent later steps
still run after a failure.

steps:
  B1 tokenizer/processor load, chat-template renders saved verbatim
     (system, user turn, generation prefix, full supervised example),
     assistant-boundary consistency check, thinking-control detection by
     comparing rendered output with and without candidate kwargs; ignored
     template kwargs are not reported as supported controls
  B2 model load (bf16, cuda), load time, load vram, trust_remote_code
     tried False first then True, whichever worked is recorded
  B3 module inventory: linear layer name suffixes with counts
  B4 json inference smoke on real bidding prompts, greedy, parse check
  C1 lora r=16 alpha=16 on the qwen target set intersected with available
     modules, trainable counts, excluded-module report
  C2 gradient smoke: 8 optimizer steps at the exact qwen config
     (seq 1024, batch 8, grad-accum 2, bf16, checkpointing, adamw 8bit),
     loss series, grad-norm nonzero check, median step time, peak vram;
     falls back to a reduced batch on OOM and records the fallback in the result
  C3 adapter save -> reload -> merge_and_unload -> save merged (optional)

usage:
  python src/experiments/feasibility/probe_model.py --model <hf-id> --revision <sha> --tag <name> [--merge-dir <path>] [--skip-train]
"""

import argparse
import json
import re
import statistics
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "reports" / "feasibility"
DATA_DIR = REPO / "data" / "finetune" / "v4"

QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
THINKING_KWARGS = [{"enable_thinking": False}, {"thinking": False}, {"reasoning_effort": "none"},
                   {"add_reasoning_prompt": False}]

SMOKE_EXAMPLES = 48
SMOKE_STEPS = 8
SEQ_LEN = 1024
BATCH = 8
GRAD_ACCUM = 2


def gib(n):
    return round(n / 1024**3, 2)


class Probe:
    def __init__(self, tag):
        self.tag = tag
        self.result = {"tag": tag, "steps": {}}

    def step(self, name):
        probe = self

        class _Ctx:
            def __enter__(self):
                probe.result["steps"][name] = {"status": "running"}
                self.t0 = time.perf_counter()
                return probe.result["steps"][name]

            def __exit__(self, exc_type, exc, tb):
                entry = probe.result["steps"][name]
                entry["seconds"] = round(time.perf_counter() - self.t0, 1)
                if exc is None:
                    entry["status"] = "pass"
                else:
                    entry["status"] = "fail"
                    entry["error"] = f"{exc_type.__name__}: {str(exc)[:500]}"
                    entry["trace_tail"] = traceback.format_exc().splitlines()[-3:]
                    print(f"[{probe.tag}] {name} FAILED: {entry['error']}", flush=True)
                return True  # continue to next step

        return _Ctx()

    def save(self):
        path = OUT_DIR / f"probe_{self.tag}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.result, f, indent=2)
        print(f"[{self.tag}] probe result -> {path}", flush=True)


def load_bid_examples(n):
    records = []
    for stage in ("material", "tailoring", "logistics"):
        with open(DATA_DIR / f"{stage}_train.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n // 3 + 1:
                    break
                records.append(json.loads(line))
    return records[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--merge-dir", default="")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = Probe(args.tag)
    probe.result["model"] = args.model
    probe.result["revision"] = args.revision

    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    tok = None
    text_tok = None
    with probe.step("B1_tokenizer_and_template") as entry:
        try:
            tok = AutoProcessor.from_pretrained(args.model, revision=args.revision)
            entry["loader"] = "AutoProcessor"
        except Exception:
            tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
            entry["loader"] = "AutoTokenizer"
        text_tok = getattr(tok, "tokenizer", tok)

        rec = load_bid_examples(3)[0]
        msgs = [{"role": "system", "content": rec["system"]},
                {"role": "user", "content": rec["user"]}]
        gen_default = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        controls = {}
        for kw in THINKING_KWARGS:
            key = ",".join(f"{k}={v}" for k, v in kw.items())
            try:
                alt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
                controls[key] = "changes_output" if alt != gen_default else "accepted_but_ignored"
                if alt != gen_default:
                    entry.setdefault("thinking_off_render_tail", repr(alt[-120:]))
            except Exception as e:
                controls[key] = f"rejected: {type(e).__name__}"
        entry["thinking_controls"] = controls
        supported = [k for k, v in controls.items() if v == "changes_output"]
        gen_kwargs = {}
        if supported:
            k, v = supported[0].split("=")
            gen_kwargs = {k: (False if v == "False" else v)}
        probe.result["thinking_off_kwargs"] = gen_kwargs

        full = tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": rec["completion"]}],
            tokenize=False, add_generation_prompt=False, **gen_kwargs)
        gen_prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **gen_kwargs)
        entry["gen_prefix_is_training_prefix"] = full.startswith(gen_prefix)
        entry["default_gen_tail"] = repr(gen_default[-120:])

        tpl_path = OUT_DIR / f"templates_{args.tag}.txt"
        with open(tpl_path, "w", encoding="utf-8") as f:
            f.write("=== generation prefix (default kwargs) ===\n" + gen_default +
                    "\n\n=== generation prefix (thinking-off kwargs: " + json.dumps(gen_kwargs) + ") ===\n" +
                    gen_prefix + "\n\n=== full supervised example ===\n" + full)
        entry["templates_file"] = str(tpl_path)

    model = None
    with probe.step("B2_model_load") as entry:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, revision=args.revision, dtype=torch.bfloat16).cuda()
            entry["trust_remote_code"] = False
        except Exception as first:
            entry["without_trust_remote_code"] = f"{type(first).__name__}: {str(first)[:200]}"
            model = AutoModelForCausalLM.from_pretrained(
                args.model, revision=args.revision, dtype=torch.bfloat16,
                trust_remote_code=True).cuda()
            entry["trust_remote_code"] = True
        entry["load_seconds"] = round(time.perf_counter() - t0, 1)
        entry["load_vram_gib"] = gib(torch.cuda.memory_allocated())
        entry["param_count"] = sum(p.numel() for p in model.parameters())
        entry["config_architectures"] = list(getattr(model.config, "architectures", []) or [])

    with probe.step("B3_module_inventory") as entry:
        from collections import Counter
        suffixes = Counter()
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Linear" or "Linear" in module.__class__.__name__:
                suffixes[name.split(".")[-1]] += 1
        entry["linear_suffix_counts"] = dict(suffixes.most_common(40))
        entry["qwen_targets_present"] = {t: suffixes.get(t, 0) for t in QWEN_TARGETS}
        entry["non_qwen_linears"] = {k: v for k, v in suffixes.items()
                                     if k not in QWEN_TARGETS and v > 2}

    with probe.step("B4_json_inference") as entry:
        recs = load_bid_examples(3)
        gen_kwargs = probe.result.get("thinking_off_kwargs", {})
        outs = []
        text_tok.padding_side = "left"
        if text_tok.pad_token is None:
            text_tok.pad_token = text_tok.eos_token
        for rec in recs:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": rec["system"]},
                 {"role": "user", "content": rec["user"]}],
                tokenize=False, add_generation_prompt=True, **gen_kwargs)
            enc = text_tok(prompt, return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=300, do_sample=False,
                                     pad_token_id=text_tok.pad_token_id)
            latency = time.perf_counter() - t0
            text = text_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            has_json = bool(re.search(r'\{[^{}]*"decision"\s*:\s*"(BID|NO_BID)"', text, re.S))
            outs.append({"stage": rec["stage"], "latency_s": round(latency, 2),
                         "parseable_json": has_json, "new_tokens": int(out.shape[1] - enc["input_ids"].shape[1]),
                         "raw": text[:600]})
        entry["generations"] = outs
        entry["json_ok_count"] = sum(o["parseable_json"] for o in outs)
        with open(OUT_DIR / f"inference_{args.tag}.txt", "w", encoding="utf-8") as f:
            for o in outs:
                f.write(f"--- {o['stage']} ({o['latency_s']}s, json={o['parseable_json']}) ---\n{o['raw']}\n\n")

    ATTN_SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "out_proj", "in_proj"}
    MLP_SUFFIXES = {"gate_proj", "up_proj", "down_proj", "gate_up_proj", "fc1", "fc2",
                    "w1", "w2", "w3"}

    peft_model = None
    with probe.step("C1_lora_attach") as entry:
        from peft import LoraConfig, get_peft_model
        inv = probe.result["steps"]["B3_module_inventory"]
        strict = [t for t, c in inv["qwen_targets_present"].items() if c > 0]
        entry["strict_qwen_targets_present"] = strict
        entry["strict_qwen_targets_missing"] = [t for t in QWEN_TARGETS if t not in strict]

        # functional-equivalent set: every attention/mlp linear regardless of
        # fusion naming, classified by parent path so mamba/expert projections
        # are visible instead of silently swept in
        categories = {"attention": set(), "mlp": set(), "mamba_or_ssm": set(),
                      "expert": set(), "other_matched": set()}
        functional = set()
        for name, module in model.named_modules():
            if "Linear" not in module.__class__.__name__:
                continue
            suffix = name.split(".")[-1]
            if suffix not in ATTN_SUFFIXES | MLP_SUFFIXES:
                continue
            functional.add(suffix)
            path = name.lower()
            if "expert" in path:
                categories["expert"].add(suffix)
            elif "mamba" in path or "mixer" in path or "ssm" in path:
                categories["mamba_or_ssm"].add(suffix)
            elif suffix in ATTN_SUFFIXES:
                categories["attention"].add(suffix)
            else:
                categories["mlp"].add(suffix)
        entry["functional_targets"] = sorted(functional)
        entry["functional_categories"] = {k: sorted(v) for k, v in categories.items() if v}

        # strict-set trainable count, measured then detached, for the
        # technical-consistency question
        if strict:
            strict_model = get_peft_model(model, LoraConfig(
                r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                target_modules=strict, task_type="CAUSAL_LM"))
            entry["strict_trainable_params"] = sum(
                p.numel() for p in strict_model.parameters() if p.requires_grad)
            strict_model.unload()

        try:
            cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                             target_modules=sorted(functional), task_type="CAUSAL_LM")
            peft_model = get_peft_model(model, cfg)
        except ValueError as e:
            if "not supported" not in str(e):
                raise
            # wrapper classes (e.g. gemma4's clippable linear) hide the real
            # nn.Linear one level down, target exact names whose class is a
            # plain Linear, language-model paths only (text-only adaptation)
            import torch.nn as nn
            exact = []
            for name, module in model.named_modules():
                if type(module) is not nn.Linear:
                    continue
                parts = name.split(".")
                suffix = parts[-1]
                parent = parts[-2] if len(parts) > 1 else ""
                if suffix not in functional and not (suffix == "linear" and parent in functional):
                    continue
                lowered = name.lower()
                if any(t in lowered for t in ("vision", "audio", "image", "tower")):
                    continue
                exact.append(name)
            entry["wrapper_workaround_exact_names"] = len(exact)
            cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                             target_modules=exact, task_type="CAUSAL_LM")
            peft_model = get_peft_model(model, cfg)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        entry["trainable_params"] = trainable
        entry["trainable_pct_of_total"] = round(trainable / total * 100, 3)

    if not args.skip_train and peft_model is not None:
        with probe.step("C2_gradient_smoke") as entry:
            from datasets import Dataset
            from trl import SFTConfig, SFTTrainer
            gen_kwargs = probe.result.get("thinking_off_kwargs", {})
            recs = load_bid_examples(SMOKE_EXAMPLES)
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": r["system"]},
                 {"role": "user", "content": r["user"]},
                 {"role": "assistant", "content": r["completion"]}],
                tokenize=False, add_generation_prompt=False, **gen_kwargs) for r in recs]
            ds = Dataset.from_list([{"text": t} for t in texts])

            def run(batch_size):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cfg = SFTConfig(
                    output_dir=str(OUT_DIR / f"smoke_{args.tag}"),
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=GRAD_ACCUM,
                    max_steps=SMOKE_STEPS, learning_rate=2e-4, warmup_steps=2,
                    logging_steps=1, optim="adamw_8bit", lr_scheduler_type="linear",
                    seed=42, max_length=SEQ_LEN, dataset_text_field="text",
                    report_to="none", bf16=True, gradient_checkpointing=True,
                    save_strategy="no")
                trainer = SFTTrainer(model=peft_model, train_dataset=ds, args=cfg,
                                     processing_class=text_tok)
                times = []
                last = [None]

                from transformers import TrainerCallback

                class T(TrainerCallback):
                    def on_step_end(self, a, s, c, **k):
                        now = time.perf_counter()
                        if last[0] is not None:
                            times.append(now - last[0])
                        last[0] = now

                trainer.add_callback(T())
                out = trainer.train()
                losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
                return {"losses": losses, "median_step_s": round(statistics.median(times), 2) if times else None,
                        "peak_train_vram_gib": gib(torch.cuda.max_memory_reserved()),
                        "loss_changed": len(set(losses)) > 1,
                        "runtime_s": round(out.metrics["train_runtime"], 1)}

            try:
                entry.update(run(BATCH))
                entry["batch_size_used"] = BATCH
                entry["exact_qwen_config"] = True
            except torch.cuda.OutOfMemoryError:
                entry["exact_qwen_config"] = False
                for fallback in (4, 2, 1):
                    try:
                        entry.update(run(fallback))
                        entry["batch_size_used"] = fallback
                        entry["oom_note"] = f"batch {BATCH} OOM, max stable found: {fallback} (NOT equivalent to qwen config)"
                        break
                    except torch.cuda.OutOfMemoryError:
                        continue

        with probe.step("C3_adapter_save_reload_merge") as entry:
            adapter_dir = OUT_DIR / f"adapter_{args.tag}"
            peft_model.save_pretrained(str(adapter_dir))
            entry["adapter_dir"] = str(adapter_dir)
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                args.model, revision=args.revision, dtype=torch.bfloat16,
                trust_remote_code=probe.result["steps"]["B2_model_load"].get("trust_remote_code", False))
            reloaded = PeftModel.from_pretrained(base, str(adapter_dir))
            merged = reloaded.merge_and_unload()
            entry["merge_ok"] = True
            if args.merge_dir:
                merged.save_pretrained(args.merge_dir)
                text_tok.save_pretrained(args.merge_dir)
                entry["merged_saved_to"] = args.merge_dir

    probe.save()


if __name__ == "__main__":
    main()
