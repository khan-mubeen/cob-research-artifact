# Protocol v2 — design document and critical assessment

2026-07-19 · frozen BEFORE any v2 training · data report with hashes:
`protocol_v2_data_report.json` · manifest: `protocol_v2.json` · generator:
`src/experiments/cross_model/generate_edge_v2.py`.

Labels used below: **[executed]** = run here with evidence path, **[docs]** = from
repository code/reports, **[interpretation]** = judgment, **[unresolved]** = needs user.

## 1. Is replacing Protocol v1 justified? Yes — with one reframing

**[executed]** The r1 recipe's weakness is confirmed from the preserved files, and is worse
than the motivation stated: the tailoring edge set has zero infeasible coverage in
(1.00, 1.315) **and the feasible side is a monoculture — all ten feasible examples sit at
exactly 0.97** (`round1_edge/edge_tailoring_train.jsonl`, realized ratios recomputed from
prompts). r2's set concentrates 22 of 32 infeasible rows at 1.03–1.15 against a 0.90–1.00
feasible side, ×5 repetition.

**[interpretation]** The user's hypothesis holds up with one precision: granite v1 does not
*reproduce* qwen r1's behavior so much as show the same *directional* failure with different
magnitude — qwen r1 moved above-boundary from 0/12 to 8/12 while granite v1 stayed at its
base's 1/12. What the controlled pair demonstrates is that the v1 recipe is *insufficient*
to teach the boundary in a second architecture; it does not demonstrate equal learning
dynamics. Either way, continuing v1 for Phi/Gemma would spend GPU hours confirming a known
data defect — v2 is justified. The claim "this is not an architecture-wide arithmetic
limitation" remains a **hypothesis** that v2 will test, not a settled fact.

**[interpretation]** The fairness correction is accepted and adopted: Qwen has had 3
iterations, Granite 1, Phi/Gemma 0. The primary comparison must be fresh same-budget v2 runs
for every model; qwen r1–r3 and granite v1 become secondary ablation/development results.

## 2. What v2 fixes, concretely

| defect | r1 | r2 | v2 |
|---|---|---|---|
| infeasible coverage in (1.00, 1.315) | none | 1.03–1.15 only | 1.01, 1.03, 1.07, 1.12, 1.20 |
| far infeasible | 1.32–1.84 random | same | 1.35, 1.60, 1.85 deterministic |
| feasible-side spread | single point (0.97!) | 0.90–1.00 | 0.70–1.00 at 7 distances |
| unique tailoring examples | 20 | 52 | **180** (every mfr × every ratio) |
| repetition | ×5 | ×5 | **×1** (breadth replaces repetition) |
| label balance (tailoring rows) | 50/50 of 100 | 62/38 of 260 | 84 BID / 96 NO_BID unique |
| deadlines | 3–6 d | 3–6 d | 7–10 d (novel, collision-free by construction) |

Not overfit to known eval tuples: no T9/T12 special-casing, no baseline-case parameters, all
overlap asserted at tuple level and rechecked at exact-prompt level (0 hits) **[executed]**.
Logistics kept light and separate (23 examples, same category logic as r2) because both
architectures already score 100% on logistics decisions **[docs: granite/qwen eval jsons]**.

## 3. Evaluation hierarchy (independence protection)

1. **Tier 1 — main in-distribution holdout** (1,500 unseen order/agent pairs): primary
   real-data quality. Gated.
2. **Tier 2 — historical regression sets** (baseline six, edge holdout 12 — 2 rows
   contaminated for v1-recipe models, reported all-12 and clean-10 —, dev stress 36, final
   boundary holdout 36): these influenced previous design decisions (r2/r3 and the v1
   criteria), so they are REPORTED as regression context, **not gates** for v2.
3. **Tier 3 — NEW locked robustness holdout** (36; 12 per class; band-drawn ratios;
   days 11–12; construction logic disjoint from the training ladder): the v2 robustness
   gate. Frozen and hashed (`cc65b091…`) before any v2 training; will not be revised after
   results, per directive.

## 4. Frozen equal budget (all v2 runs, all models)

Identical: real 15,000 stratified subsample (same files/hashes/seed as v1), identical v2
edge records (hashes above), edge-repeat 1 → 15,203 rows → 951 optimizer steps at effective
batch 16; seed 42; 1 epoch; LoRA r16/α16 on functionally-equivalent q/k/v/o/gate/up/down
targets with published trainable counts; seq 1024; unsloth response-only masking with
model-specific markers (template gate required before launch); adamw_8bit LR 2e-4 warmup 5
linear; bf16; greedy decoding, 200 max new tokens; identical eval sets. Per-device batch
follows the v1 native-fit rule (8 unless spill, then largest native fit at effective 16,
disclosed). No per-model tuning in the primary comparison.

**Seed plan [interpretation]:** one common seed (42) for the primary four-model comparison —
at ~1.5–1.75 h/run, multi-seed for all models (~12+ extra GPU-hours for 2 extra seeds) is not
worth it before the field is narrowed. Seeds 43 and 44 are PRE-REGISTERED as the confirmation
budget for the top one or two models only, to be run before any final architecture claim.

## 5. Pre-registered v2 acceptance criteria (frozen now)

1. JSON validity 100% on every set.
2. Main-holdout decision agreement per stage ≥ the same model's own base.
3. Per-stage mean offer deviation strictly improved vs own base.
4. **Tier-3 robustness holdout ≥ 9/12 in EACH class** (below_feasible, exact_feasible,
   above_infeasible) — reported per class, never aggregated.

Also reported (official metrics, not gates unless listed): false BID / false NO_BID counts
per set, completion-days agreement, latency/throughput in both runtimes. Tier-2 sets
reported for regression context. Criteria will not be softened after results.

## 6. Execution plan

Qwen v2 first (most-studied reference validates the v2 pipeline end to end), then Granite v2,
sequentially — no simultaneous GPU jobs. Phi and Gemma are NOT trained in this task: the
qwen-v2 vs granite-v2 pair is needed first to validate that the frozen v2 pipeline and
criteria behave sanely across two architectures before spending the remaining GPU budget
**[interpretation]**. Both v2 runs start from the original base checkpoints (qwen `851bf6e8…`,
granite `c0650403…`), never from prior adapters. Expected cost: qwen ≈ 105 min (6.6 s/step ×
951), granite ≈ 89 min (5.62 s/step × 951), plus ~40 min full eval each.

## 7. Protocol v1 preservation

Nothing is overwritten: qwen r1/r2/r3 and granite v1 adapters, all eval reports, protocol v1
manifests, and GGUF artifacts stay in place; v2 outputs use new `-v2` paths. Granite v1 is
re-labeled a **Protocol v1 ablation** (controlled second-architecture exposure to the r1
coverage gap), not a failed primary result.
