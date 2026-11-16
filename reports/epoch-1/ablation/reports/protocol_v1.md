# Frozen cross-model fine-tuning protocol v1

Frozen 2026-07-19, **before any candidate training run**. Machine-readable twin with all
hashes: `protocol_v1.json`. Reference: the official Qwen r1 run.

## Why this protocol is methodologically defensible

1. **Identical data, bit-for-bit.** Every model trains on the same 15,000-example stratified
   subsample (seed 42, scenario-share quotas from orders.csv) plus the same 40 round-one edge
   examples × 5 — the exact recipe of the official Qwen r1 checkpoint. All input files are
   SHA-256 pinned. The round-one edge files were preserved in `data/finetune/v4/round1_edge/`,
   so the comparison is official-vs-official, not official-vs-variant.
2. **Identical optimization.** Same LoRA rank/alpha (16/16), same functionally-equivalent
   target set (attention q/k/v/o + MLP gate/up/down), same optimizer, LR, schedule, seed,
   epoch count, sequence length, effective batch 16, same unsloth stack and versions.
   Trainable-parameter counts are published per model rather than pretending they can be equal
   across architectures.
3. **Batch policy: native fit, never deliberate spill.** Per-device batch stays at Qwen's 8
   when the model fits natively under the actual training stack (verified by a pre-launch smoke
   reading `max_memory_reserved`); otherwise the largest native-fit per-device batch is used
   with gradient accumulation keeping effective batch 16, and the deviation is disclosed.
   Windows sysmem spill silently changes step timing and memory behavior and is not accepted
   as a training configuration. (Granite turned out to fit natively at batch 8 — 16.9 GiB
   reserved — so no deviation was needed; the feasibility probe's 37.9 GiB reading came from
   the plain-PEFT probe stack, not the unsloth stack used for real training.)
4. **Template alignment is a launch gate, not an assumption.** Before training, each model
   must pass `template_check.py`: generation render is an exact prefix of the training render,
   the loss-mask markers occur exactly once, and the GGUF-embedded template renders
   identically to the HF template. (Granite: PASS, all seven checks.)
5. **Pre-registered acceptance criteria** (frozen here, before results):
   1. baseline six: 6/6 NO_BID;
   2. main-holdout decision agreement per stage ≥ the same model's own base checkpoint;
   3. per-stage mean offer deviation strictly improved vs the same model's base;
   4. 100% JSON validity on every set;
   5. final boundary holdout ≥ 9/12 **per class**, never aggregated.
6. **Known-defect disclosure instead of silent metric reuse.** 2 of the 12 current edge-holdout
   prompts are verbatim in the round-one edge train set (found 2026-07-19; ids
   SYNEDGE-T-INF-T2/T2 and SYNEDGE-T-INF-T11/T11). This taints the edge_holdout metric equally
   for Qwen r1 and any model trained on the r1 recipe, so edge_holdout is reported both as
   all-12 and clean-10 for every model. The decisive sets (final boundary holdout, boundary
   stress, main holdouts) have zero overlap with the round-one edge train set.
7. **Non-thinking arms only.** A model that cannot disable reasoning is excluded from the
   primary comparison (LFM), reported at most as a labeled exploratory arm.
8. **Deployment path pinned.** Conversion uses the llama.cpp commit vendored by the project
   runtime (e3546c79) + llama-quantize b10068 + Q8_0; Qwen additionally requires `--no-mtp`
   and the non-thinking-default chat template (see `reports/gguf_fix/`).
