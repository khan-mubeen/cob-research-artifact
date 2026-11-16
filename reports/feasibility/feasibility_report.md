# Cross-model fine-tuning feasibility matrix — five candidates vs Qwen3.5-4B

2026-07-19 · RTX 5090 laptop (24 GiB, Windows) · isolated env `.venv-feas` (torch 2.10.0+cu130,
transformers 5.5.0, unsloth 2026.7.2, peft 0.19.1, trl 0.24.0; full freeze:
`env_feas_freeze.txt`) · conversion: llama.cpp `571d0d54` + b10068 quantize · project runtime:
llama-cpp-python **0.3.34** cu132 · machine-readable: `feasibility_matrix.json` / `.csv`.

Every claim below is labeled **[executed]** (directly run here, evidence path given) or
**[docs]** (official documentation, not independently executed).

`BIDDING_WINDOW_SECONDS` verified from the repo: **10** (supervisor.py:19 — note it was 60
briefly on 2026-07-16 and has been reverted; the headroom comment above it is now stale).

## Verdict table

| model | revision | status | evidence highlights |
|---|---|---|---|
| Qwen/Qwen3.5-4B (reference) | 851bf6e8 | **executed baseline** | trained 3 rounds, 6.6 s/step, 10.3 GiB; **Stage E: convert pass but 0.3.34 runtime load FAIL** (converter/runtime skew, see §E) |
| ibm-granite/granite-4.1-3b | c0650403 | **ready** | all stages pass; strict Qwen target names match exactly; E2E: Q8_0 loads in 0.3.34, 6/6 JSON, 0.84 s steady |
| microsoft/Phi-4-mini-instruct | cfbefacb | **ready with controlled adaptation** | fused qkv/gate_up needs functional target mapping; conversion needs source tokenizer files copied; then 6/6 JSON, 0.82 s |
| google/gemma-4-E2B-it | 179516f0 | **ready with controlled adaptation** | plain PEFT 0.19.1 can't wrap `Gemma4ClippableLinear`; **unsloth path (same stack as Qwen) works**: loss 1.43→0.81, 4.52 s/step, E2E 6/6 JSON at 0.67 s — fastest overall |
| LiquidAI/LFM2.5-8B-A1B | 5673e0de | **unsuitable for the frozen comparison** | always-on reasoning, no official off-switch [docs+executed]; 0/3 parseable JSON (CoT ate the 300-token budget); LoRA reaches only 5.7 M params (0.067%) — MoE experts aren't nn.Linear |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | dfaf35de | **blocked pending tooling** | model load requires `mamba-ssm` (no Windows wheels, source build needs nvcc toolchain); chat template independently crashes (`KeyError: '-'`); not disproven on Linux/WSL2 |

## Stage A — identity, access, licensing [docs unless noted]

All six repos exist at the pinned revisions above, all **ungated**; access verified by actual
download **[executed]**: phi 7.17 / granite 6.35 / nemotron 7.42 / gemma 9.57 / lfm 15.79 GiB.
No HF token is configured on this machine (works because nothing is gated; set one for rate
limits if desired).

- **Parameters**: qwen 4.56 B stored (multimodal); phi 3.836 B dense; granite 3.4 B dense;
  nemotron 3.97 B hybrid mamba2+attention; gemma **5.12 B stored / 2.3 B effective** (per-layer
  embeddings — report both, never just one); lfm **8.47 B total / 1.5 B active** (32 experts, 4 active).
- **trust_remote_code**: phi carries a custom_code tag but loads natively with
  `trust_remote_code=False` **[executed]** — the starting observation was wrong in our favor.
  granite/gemma/lfm load natively **[executed]**. nemotron requires it and *still* fails on
  mamba-ssm **[executed]**.
- **Licenses** (from actual texts): phi MIT; granite + gemma Apache-2.0 (Gemma 4 dropped the
  Gemma-Terms gate — verified by ungated download); nemotron NVIDIA Open Model License
  (commercial + derivatives OK with notice, Trustworthy-AI/guardrail clause, **repo LICENSE file
  is 0 bytes** — terms only on nvidia.com); lfm LFM Open License v1.0 (**$10 M annual revenue
  threshold** on commercial use; research permitted; derivatives redistributable with notices).
  All four viable candidates permit redistribution of fine-tuned weights and conversions with
  attribution; publishing **adapters only** is safest and permitted everywhere. No geography
  restrictions beyond export law; no publication restrictions found for any candidate.

## Stage B — templates and JSON inference [executed]

Rendered templates saved per model in `templates_<tag>.txt`; generation-vs-training assistant
boundary verified consistent for every model that loads. Thinking controls detected by
render-comparison (a kwarg that doesn't change the render is reported as ignored, not as a
control): qwen `enable_thinking=False` (verified July); phi/granite have no reasoning modes
(all kwargs ignored — they're plain instruct models); gemma controlled by `<|think|>` token
presence in system prompt [docs — our prompts don't include it, so thinking stays off];
lfm: **no control exists** — `<think>` always emitted [executed sample in `inference_lfm25-8b-a1b.txt`].
JSON on unmodified bidding prompts: phi 3/3, granite 3/3, gemma 3/3, lfm **0/3**, nemotron n/a.
No task-prompt changes were needed for any passing model — template-level only.

## Stage C — LoRA and gradient smoke [executed]

Strict Qwen target-name intersection vs functionally-equivalent projection set (r=16, α=16):

| model | strict-name trainable | functional trainable | note |
|---|---|---|---|
| qwen (reference) | 21.2 M (0.47%) | same | executed July |
| granite | 31.1 M | 31.1 M (0.85%) | naming identical to Qwen |
| phi | 8.9 M | 23.1 M (0.60%) | fused qkv_proj/gate_up_proj — strict naming silently drops attention+MLP |
| gemma (unsloth) | — | 25.3 M | plain PEFT unsupported; unsloth dispatch handles the wrapper |
| lfm | 0.88 M | 5.7 M (0.067%) | experts not nn.Linear → un-targetable; attention+shared-path only |

Gradient smokes (8 steps, seq 1024, batch 8, accum 2, bf16, checkpointing, adamw_8bit, seed 42):
loss decreased and grads flowed for granite (11.4 s/step), phi (16.8 s/step), lfm (12.0 s/step),
gemma via unsloth (**4.52 s/step**). **Measurement caveat**: probe "peak VRAM" is
`max_memory_reserved` and values of 37.9 (granite), 53.6 (phi), 56.9 (lfm) GiB exceed physical
24 GiB — Windows CUDA sysmem fallback spilled to shared memory. The exact Qwen config *runs*
on all three but only gemma (13.9 GiB) and Qwen itself (10.3 GiB) **fit natively**; candidate
step times include spill penalty and are not clean per-device comparisons. A no-fallback
max-batch search was not performed (driver-level toggle, not scriptable) — flagged as unverified.

## Stage E — merge → GGUF → quantize → project runtime [executed]

| model | convert | Q8_0 size | 0.3.34 load | JSON | steady latency |
|---|---|---|---|---|---|
| granite | pass | 3.37 GiB (sha `7f770a7b…`) | pass | 6/6 | **0.84 s** |
| phi | pass (after copying source BPE tokenizer files into merged dir — converter demands them) | 3.80 GiB | pass | 6/6 | **0.82 s** |
| gemma | pass | 4.63 GiB | pass | 6/6 | **0.67 s** |
| **qwen r1 (reference)** | pass | 4.17 GiB | **FAIL: `missing tensor 'blk.32.attn_norm.weight'`** | — | — |
| lfm / nemotron | not run — gated out earlier (B4 fail / B2 fail); official GGUFs exist [docs] but that does not prove our fine-tuned path | | | | |

The Qwen failure is **converter↔runtime version skew** on qwen35's MTP block (master converter
vs the older llama.cpp bundled in 0.3.34 — the base Q8_0 from July loads fine). Consequence for
the *existing* Qwen plan: re-quantization of round one must use a converter matched to the
runtime version, or the runtime wheel must be upgraded first.

> **Correction (2026-07-19, gguf-fix investigation — see `reports/gguf_fix/`):** the version-skew
> explanation above was incomplete. The decisive factor is the converter's `--no-mtp` flag, not
> the converter version: the merged checkpoint (saved through the causal-LM view) drops the MTP
> tensors but keeps `mtp_num_hidden_layers: 1` in config.json, so a default conversion writes
> `block_count = 33` with no `blk.32.*` tensors — an internally inconsistent file that fails in
> any runtime. Converting with `--no-mtp` yields byte-identical working Q8_0 artifacts from BOTH
> the vendored-commit converter (e3546c79) and the feasibility-era master (571d0d54), sha
> `4461e48a…`, loading 6/6 in the unmodified 0.3.34 runtime. A second packaging requirement was
> also found: the GGUF must embed the unsloth-style chat template (non-thinking default), because
> llama-cpp-python never passes `enable_thinking` and the original Qwen template defaults to
> thinking mode (0/6 JSON otherwise). Adapter behavior demonstrably
survives merge+conversion for the three candidates (fine-tuned-style JSON out of the quantized
artifacts). Logs: `convert_<tag>.log/.json`.

## Inference performance and bidding-window fit [executed]

Steady single-stream Q8_0 latencies on real bidding prompts (6 prompts each, first-call
excluded): gemma 0.67 s · phi 0.82 s · granite 0.84 s · qwen base ~1.05–1.5 s (July measurement).
Cold loads 2.2–6.4 s; one-time CUDA warmup applies per process (warmup hook already in the agents).
**Window fit at the current 10 s window**: each stage process serializes ~12 agent decisions on
one model lock → qwen ≈ 14 s **does not fit**; granite/phi ≈ 10 s borderline; gemma ≈ 8 s fits.
At the briefly-used 60 s window all fit comfortably. This is a system decision to make before
any live SLM run, independent of model choice.

## Cross-model fairness — proposed frozen protocol

**Technical consistency** (can the exact same config run?): yes for granite/phi/gemma with two
controlled adaptations that do not touch task content: (1) functionally-equivalent LoRA targets
instead of string-identical names (else phi trains 2.6× fewer params for no scientific reason);
(2) the unsloth stack for all models (Qwen already used it; gemma requires it). No for lfm
(reasoning) and nemotron (won't load) on this machine.

**Proposed pre-registered protocol** (recommendation, not implemented):
1. Data: identical 15,000 stratified examples + the frozen edge policy, seed 42, 1 epoch.
2. LoRA r=16 α=16, functional attention+MLP target set per model, trainable counts published.
3. batch 8 × accum 2, seq 1024, bf16, greedy decoding, unchanged prompts; per-device VRAM and
   sysmem-spill status reported per model rather than hidden.
4. Non-thinking mode for all primary arms; a model that cannot disable reasoning is excluded
   from the primary comparison, optionally reported as a clearly-labeled exploratory arm —
   never silently mixed in.
5. Evaluation: the frozen Qwen eval sets and four criteria + boundary holdouts; Stage E with a
   version-matched converter; latency in the 0.3.34 runtime (or a deliberately upgraded one,
   chosen once for all models).
6. Equal examples ⇒ equal optimizer steps at fixed effective batch 16; no per-model tuning
   allowance in round one (a pre-registered small LR sweep may be *discussed* for round two).

## Recommendations

- **Safest first**: granite-4.1-3b (nothing needed adaptation; even strict target names match).
- **Riskiest attempted**: nemotron (blocked outright on Windows); riskiest *trainable*: gemma
  (needed the most adaptation, though it ended fastest end-to-end).
- **Execution order**: granite → phi → gemma → then decide lfm (exploratory arm or drop) and
  nemotron (WSL2/Linux attempt or drop).
- **Open decisions before GPU training**: (1) bidding window 10 s vs 60 s; (2) lfm/nemotron
  disposition; (3) converter/runtime version pinning strategy — also unblocks the pending Qwen
  r1 re-quantization; (4) publication mode (adapters-only recommended); (5) protocol §above
  sign-off.

> **De-scoping addendum (2026-07-21):** Nemotron-3-Nano-4B and LFM2.5-8B-A1B are removed from
> the six-model slate and replaced by Ministral 3 3B Instruct and SmolLM3-3B (pending their own
> feasibility gates — see `reports/cross_model/model_slate.json`). Final Nemotron state: the
> chat-template crash reported above NO LONGER reproduces under transformers 5.5.0
> (renders cleanly, reasoning-off via `enable_thinking=False` → `<think></think>`), but model
> load remains blocked by the mamba-ssm dependency on native Windows
> (`nemotron_load_infer_result.json`, fails at import in 0.8 s, no VRAM touched). Final LFM
> state: loads and runs one controlled inference cleanly (`lfm_load_infer_result.json`, peak
> 17.6 GB VRAM, no crash), but reasoning remains always-on (leaked `<think>` on a trivial
> JSON instruction) and 91.5% of parameters sit in fused 3D expert tensors
> (`Lfm2MoeExperts.gate_up_proj/down_proj`, 32×3584×2048 / 32×2048×1792) outside standard
> LoRA reach. All Nemotron/LFM evidence in this directory is retained as de-scoped record;
> the superseded probe script `lfm_deep_probe.py` (training-smoke variant, never completed)
> was removed after a verified zero-reference check.

## Artifact inventory

Scripts (new, under `src/experiments/feasibility/`): `probe_model.py`, `merge_adapter.py`,
`convert_gguf_test.py`, `gemma_unsloth_smoke.py`, `build_feasibility_matrix.py`. Evidence
(under `reports/feasibility/`): probe/convert JSONs + logs, template dumps, inference samples,
env freeze, matrix JSON/CSV. Model artifacts: HF cache +46.3 GiB (5 checkpoints);
`models/feasibility/` 76.9 GiB (merged + GGUF, gitignored); `.venv-feas` 3.7 GiB;
`tools` 0.2 GiB. Disk free after: 1.51 TB. Environments: `.venv-feas` created (freeze
recorded); the Qwen `.venv` was **not modified**. No training runs started, no commits, no
pushes, orders.csv/metadata untouched, Qwen adapters and reports untouched.
