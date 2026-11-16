# Qwen r1 GGUF converter/runtime investigation — resolved

2026-07-19 · all evidence in `reports/gguf_fix/` · scripts in `src/experiments/gguf_fix/` ·
production runtime **llama-cpp-python 0.3.34 was not modified**.

## Reproduced failure

`repro_load_failure.log`: loading the feasibility artifact
`models/feasibility/qwen35-r1-Q8_0.gguf` in 0.3.34 fails with
`llama_model_load: error loading model: missing tensor 'blk.32.attn_norm.weight'`
(CPU load, so it is a file-parse failure, not a GPU issue).

## Root cause (corrects the feasibility report's "version skew" hypothesis)

`metadata_diff.json`: the failing file and the WORKING unsloth base GGUF contain the **same
426 tensors (blocks 0–31)**; they differ in one metadata field — `qwen35.block_count` is 33
in the failing file vs 32 in the working one.

Why: Qwen3.5 carries an MTP block (`mtp_num_hidden_layers: 1`). Merging the r1 adapter through
the causal-LM view drops the MTP tensors but keeps the config key. The converter's
`_Qwen35MtpMixin` then extends block_count to 33 while finding no `mtp.*` tensors to remap —
producing an internally inconsistent file that no runtime version loads. **Both converter
versions behave identically**; the missing piece was the `--no-mtp` flag.

Proof by cross-check: converting the same merged checkpoint with `--no-mtp` using
(a) llama.cpp `e3546c79` — the exact commit vendored by 0.3.34 — and
(b) the feasibility-era master `571d0d54`
produces **byte-identical Q8_0 files** (sha `4461e48a…`), both loading cleanly in 0.3.34.

## Second packaging requirement found: the chat template default

First `--no-mtp` artifact loaded but produced 0/6 JSON (chain-of-thought prose,
`convert_qwen35-r1-nomtp-matched_attempt1_thinking_template.*`). Cause: the merged dir carried
the original Qwen template, whose **default is thinking ON**; the working unsloth base GGUF
ships the same template with the default branch inverted (empty `<think>` block when
`enable_thinking` is undefined — `template_base_gguf.txt` vs `template_r1_gguf.txt`).
llama-cpp-python never passes `enable_thinking`, so only the non-thinking-default packaging
works. Training/eval renders are unaffected: they always pass `enable_thinking=False`, under
which both templates render identically. Fix: embed the unsloth-style template. Result: 6/6
JSON, 0.95 s median steady.

## Verified chain (every step recorded)

1. **Merge** (`merge_verify.py` → `merge_verify.json`): r1 adapter into pinned base
   `851bf6e8…`, with a hard weight-delta gate — probed q/o/down projections all changed
   (max |Δ| 7.3e-4 / 1.7e-4 / 2.4e-4), i.e. no silent no-op merge. Output sha `2da19075…`.
2. **Convert** `--no-mtp`, converter `e3546c79`: block_count 32, 426 tensors, no blk.32.
3. **Quantize** b10068 → Q8_0 4.17 GiB, sha `4461e48a…` (quantize verified metadata-neutral).
4. **Load + smoke** in unmodified 0.3.34: 6/6 JSON, load 3.0 s, 0.95 s steady.
5. **Full frozen eval** (`eval_gguf.py`, 1,590 generations, single-stream greedy):
   median latency **0.939 s**, 89.3 tokens/s overall.

## Native vs merged vs quantized (`native_merged_quantized_comparison.json`)

| set | native r1 (HF) | merged (HF) | quantized (0.3.34) |
|---|---|---|---|
| baseline six | 4/6 | 4/6 | 4/6 (same two DEADLINE false BIDs) |
| main holdout decisions | 99.67% | 99.67% | 99.67% |
| main holdout JSON | 100% | 100% | 100% |
| main holdout offer dev | 0.653% | 0.660% | 0.663% |
| edge holdout (12) | 91.67% | 83.33% | 83.33% |
| boundary stress (36, context) | 50.0% | 52.78% | 44.44% |
| final boundary (36) | 58.33% | 47.22% | 52.78% |

Reading: on everything the model is actually stable at (pricing, mainline decisions, JSON),
merge and quantization are transparent — offer deviation moves +0.01 pp end-to-end and not a
single mainline decision flips. The flips concentrate exclusively in the near-1.00 boundary
cases (row-level flip lists in the comparison JSON), where r1 was already unstable natively;
bf16 fusion rounding is enough to flip those near-ties in both directions. **No new failure
mode is introduced by deployment**; the boundary instability is a property of the model, and
r1 already failed the pre-registered boundary bar natively.

## Does the production runtime need upgrading?

No. The fix works entirely inside 0.3.34. Also verified: **0.3.34 is the newest prebuilt
cu132 wheel on the abetlen index** — no prebuilt Windows upgrade path currently exists, so the
matched-converter route is not just cleaner but the only practical one.

## Frozen deployment recipe for qwen (adopted into protocol_v1)

merge (bf16, causal view, delta-verified) → convert with llama.cpp `e3546c79` **`--no-mtp`**
+ non-thinking-default chat template in the merged dir → `llama-quantize` b10068 Q8_0 →
verify `block_count == 32` before shipping. Old broken artifact
`models/feasibility/qwen35-r1-Q8_0.gguf` left untouched as evidence.
