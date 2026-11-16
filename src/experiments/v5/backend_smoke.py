"""a backend-neutral training smoke run, ready for LUMI (AMD MI250X, ROCm) or this machine (NVIDIA, CUDA)

  python -m src.experiments.v5.backend_smoke --probe-only
  python -m src.experiments.v5.backend_smoke --path unsloth --mode sealed --steps 10
  python -m src.experiments.v5.backend_smoke --path unsloth --mode sealed --steps 10 --optim adamw_torch
  python -m src.experiments.v5.backend_smoke --path plain --mode sealed --steps 10

nothing here assumes a backend. torch exposes ROCm through the same torch.cuda namespace, so the
probe reads torch.version.hip to tell the two apart, and reports the GPU architecture string
(gcnArchName, gfx90a on MI250X) where the build provides it. every step records what it saw instead
of stopping at the first failure, so one run on LUMI answers every question below at once

  probe    torch, cuda or hip version, device and architecture, bf16 support; import status and
           version of unsloth, unsloth_zoo, bitsandbytes, xformers, flash_attn, triton,
           cut_cross_entropy, peft, trl, transformers
  kernels  tiny real operations on the device: a bf16 matrix product, scaled dot-product attention,
           a triton kernel, an xformers attention call, and one bitsandbytes 8-bit AdamW step, the
           optimizer every completed run used
  unsloth  src/experiments/finetune_lora.py unmodified, a handful of steps on real v5 train rows,
           --optim adamw_8bit by default or adamw_torch to take bitsandbytes out of the picture
  plain    the same rows with transformers and peft only, no unsloth and no bitsandbytes, torch
           AdamW and gradient checkpointing, so an unsloth-specific failure can be told apart from
           a basic ROCm training failure. text-only models only: gemma4 cannot be wrapped by
           plain peft at all (reports/feasibility/probe_gemma4-e2b.json), and the multimodal qwen
           checkpoint needs unsloth's language-tower targeting

writes reports/v5/lumi/backend_smoke_<backend>_<stamp>.json. this machine has no ROCm hardware;
a run here proves the script, not ROCm
"""

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO / "reports" / "v5" / "lumi"
SMOKE_DIR = REPO / "data" / "finetune" / ".v5_smoke"
WINDOWS = {"sealed": 1024, "reverse_auction": 1280, "supervisor": 2048}
UNSLOTH_MODEL = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
PLAIN_MODEL = ("ibm-granite/granite-4.1-3b", "c0650403e44e78ec0262dab1c90914c65b196c4e")
PACKAGES = ("torch", "unsloth", "unsloth_zoo", "bitsandbytes", "xformers", "flash_attn", "triton",
            "cut_cross_entropy", "peft", "trl", "transformers", "accelerate", "datasets")


def attempt(fn):
    started = time.perf_counter()
    try:
        return {"ok": True, "result": fn(), "seconds": round(time.perf_counter() - started, 2)}
    except BaseException as exc:  # a smoke run records failures, it does not stop at them
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:600],
                "trace_tail": traceback.format_exc().splitlines()[-6:]}


def probe():
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "torch_hip": getattr(torch.version, "hip", None),
            "backend": "rocm" if getattr(torch.version, "hip", None) else ("cuda" if torch.version.cuda else "cpu"),
            "accelerator_available": torch.cuda.is_available(),
            "environment": {k: os.environ.get(k) for k in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES",
                                                           "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                                           "PYTORCH_ROCM_ARCH", "PYTORCH_CUDA_ALLOC_CONF")}}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({"device_count": torch.cuda.device_count(), "device_name": props.name,
                     "architecture": getattr(props, "gcnArchName", None),
                     "capability": list(torch.cuda.get_device_capability(0)),
                     "memory_gib": round(props.total_memory / 1024 ** 3, 2),
                     "bf16_supported": torch.cuda.is_bf16_supported()})
    packages = {}
    for name in PACKAGES:
        outcome = attempt(lambda n=name: getattr(importlib.import_module(n), "__version__", "imported, no version"))
        packages[name] = outcome["result"] if outcome["ok"] else f"import failed: {outcome['error']}"
    info["packages"] = packages
    return info


def kernels():
    import torch
    device = "cuda"
    out = {}
    out["bf16_matmul"] = attempt(lambda: float((torch.randn(256, 256, device=device, dtype=torch.bfloat16)
                                                @ torch.randn(256, 256, device=device, dtype=torch.bfloat16)).float().abs().mean()))
    out["sdpa_attention"] = attempt(lambda: list(torch.nn.functional.scaled_dot_product_attention(
        *(torch.randn(1, 8, 128, 64, device=device, dtype=torch.bfloat16) for _ in range(3)), is_causal=True).shape))

    def triton_kernel():
        import triton
        import triton.language as tl

        # triton.jit compiles `add` by re-parsing its source and resolving free names against the
        # function's __globals__, not this enclosing scope, so a tl bound only as a local here is
        # invisible to the compiler and every tl.* reference fails with NameError. publish it to
        # module scope, where the jit will look. the imports stay lazy so this module still
        # imports on a machine with no triton
        globals().setdefault("tl", tl)

        @triton.jit
        def add(x_ptr, y_ptr, out_ptr, n, block: tl.constexpr):
            offsets = tl.arange(0, block)
            mask = offsets < n
            tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask), mask=mask)

        x, y = torch.ones(1000, device=device), torch.ones(1000, device=device)
        z = torch.empty_like(x)
        add[(1,)](x, y, z, 1000, block=1024)
        return float(z.sum())

    out["triton_kernel"] = attempt(triton_kernel)

    def xformers_attention():
        from xformers.ops import memory_efficient_attention
        q = torch.randn(1, 128, 8, 64, device=device, dtype=torch.bfloat16)
        return list(memory_efficient_attention(q, q, q).shape)

    out["xformers_attention"] = attempt(xformers_attention)

    def bnb_adamw8bit_step():
        import bitsandbytes as bnb
        param = torch.nn.Parameter(torch.randn(4096, 64, device=device))
        optimizer = bnb.optim.AdamW8bit([param], lr=1e-3)
        param.grad = torch.randn_like(param)
        optimizer.step()
        return float(param.detach().abs().mean())

    out["bitsandbytes_adamw8bit_step"] = attempt(bnb_adamw8bit_step)
    return out


def sample_file(mode):
    path = SMOKE_DIR / f"{mode}_smoke.jsonl"
    if not path.exists():
        subprocess.run([sys.executable, "-m", "src.experiments.v5.smoke_sample"], cwd=REPO, check=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    return path


def unsloth_run(args):
    model, revision = args.base_model or UNSLOTH_MODEL[0], args.revision or (None if args.base_model else UNSLOTH_MODEL[1])
    command = [sys.executable, "src/experiments/finetune_lora.py", "--train-jsonl", str(sample_file(args.mode)),
               "--base-model", model, "--max-steps", str(args.steps), "--batch-size", str(args.batch_size),
               "--grad-accum", str(args.grad_accum), "--max-seq-length", str(WINDOWS[args.mode]), "--optim", args.optim,
               "--output-dir", str(SMOKE_DIR / f"out_backend_smoke_{args.mode}")]
    if revision:
        command += ["--revision", revision]
    started = time.perf_counter()
    # unsloth prints emoji, so the streams are decoded as utf-8 with replacement: a locale code page
    # would fail to decode them and leave a stream empty
    run = subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"},
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = ((run.stdout or "") + (run.stderr or "")).splitlines()
    keep = [line for line in lines if any(k in line for k in ("per-step wall time", "peak vram", "train_runtime",
                                                              "Error", "error", "Traceback", "Unsloth:"))]
    return {"command": " ".join(command), "exit": run.returncode, "seconds": round(time.perf_counter() - started, 1),
            "key_lines": keep[-40:], "tail": lines[-15:]}


def plain_run(args):
    import torch
    from peft import LoraConfig, get_peft_model
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name, revision = args.base_model or PLAIN_MODEL[0], args.revision or (None if args.base_model else PLAIN_MODEL[1])
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    # transformers renamed torch_dtype to dtype in v5. both names reach **kwargs on either
    # version, but 4.x only consumes torch_dtype and passes an unrecognised dtype straight on to
    # the model constructor, which then dies with an unexpected-keyword TypeError. pick the name
    # the installed version actually consumes rather than assuming one of them
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                 **{dtype_kw: torch.bfloat16}).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    rows = [json.loads(line) for _, line in zip(range(args.steps * args.batch_size), open(sample_file(args.mode), encoding="utf-8"))]
    torch.cuda.reset_peak_memory_stats()
    step_seconds = []
    model.train()
    for step in range(args.steps):
        batch = rows[step * args.batch_size:(step + 1) * args.batch_size]
        texts = [tokenizer.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                                                {"role": "assistant", "content": r["completion"]}], tokenize=False) for r in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=WINDOWS[args.mode]).to("cuda")
        labels = encoded["input_ids"].masked_fill(encoded["attention_mask"] == 0, -100)
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = model(**encoded, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_seconds.append(round(time.perf_counter() - started, 3))
    return {"model": model_name, "revision": revision, "steps": args.steps, "batch_size": args.batch_size,
            "final_loss": round(float(loss), 4), "step_seconds": step_seconds,
            "median_step_seconds_excluding_first": sorted(step_seconds[1:])[len(step_seconds[1:]) // 2] if len(step_seconds) > 1 else None,
            "peak_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--path", choices=("unsloth", "plain"), default="unsloth")
    parser.add_argument("--mode", choices=tuple(WINDOWS), default="sealed")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--revision", default="")
    args = parser.parse_args()

    record = {"started_at": datetime.now().isoformat(timespec="seconds"), "arguments": vars(args)}
    record["probe"] = attempt(probe)
    backend = (record["probe"].get("result") or {}).get("backend", "unknown")
    if not args.probe_only:
        record["kernels"] = attempt(kernels)
        record[args.path] = attempt(lambda: unsloth_run(args)) if args.path == "unsloth" else attempt(lambda: plain_run(args))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"backend_smoke_{backend}_{args.path if not args.probe_only else 'probe'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, indent=2, default=str)[:6000])
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
