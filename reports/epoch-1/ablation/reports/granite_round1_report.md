# Granite-4.1-3B round one under the frozen cross-model protocol

2026-07-19 · protocol: `protocol_v1.md` / `.json` (frozen BEFORE training) · launch gates:
`granite_launch_gates.json` (4/4 PASS) · training console: `granite_train_console.log` ·
machine-readable results: `granite_round1_summary.json`.

## Headline

**Granite-FT round one: FAIL against the pre-registered criteria (2 of 5 passed).**
The pricing/decision transformation fully replicates the qwen result — offer deviation drops
from 46–80% to 1.0–1.6%, mainline decisions reach 99.2–100% — but granite learned the dynamic
NO_BID boundary even less than qwen r1 did on identical data, and one malformed JSON row
breaks the 100%-validity bar. This is a real cross-model negative finding under a clean
protocol, not a pipeline artifact.

## Training run (exactly as gated, no deviations)

- base `ibm-granite/granite-4.1-3b` @ `c0650403…` (original checkpoint, NOT the feasibility
  smoke adapter), output `models/finetune/granite-4.1-3b-bidding-lora/final`
- official r1 recipe: 15,000 stratified (seed 42) + 40 round-one edge examples × 5 = 15,200
  rows; LoRA r16/α16 on q/k/v/o/gate/up/down (31,129,600 trainable, 0.91%, equals the
  validated set); batch 8 × accum 2, seq 1024, bf16, adamw_8bit, LR 2e-4, 1 epoch, seed 42
- 950 optimizer steps in 88.5 min (median 5.62 s/step); peak 9.3 GiB allocated / **17.8 GiB
  reserved — native fit for the whole run, no sysmem spill** (the memory-spill concern from
  the feasibility probe was an artifact of the plain-PEFT probe stack; under the unsloth
  training stack granite fits at the exact qwen batch config, so the protocol's identical
  per-device batch applied with no deviation)
- final train loss 0.133

## Pre-registered criteria (frozen before results; verdicts stated plainly)

| # | criterion | granite-ft | verdict |
|---|---|---|---|
| 1 | baseline six 6/6 NO_BID | **2/6** (both logistics correct; all four tailoring deadline/workload cases BID) | **FAIL** |
| 2 | main-holdout decisions ≥ own base per stage | 99.2/99.8/100.0 vs base 88.8/95.4/54.8 | **PASS** |
| 3 | offer deviation improved vs own base per stage | 1.59/1.60/1.05% vs base 60.3/79.8/46.3% | **PASS** |
| 4 | JSON validity 100% on every set | 1 malformed row of 1,590 (ORD-18870/T2: `"estimated_completion_days": 1. sixth`) | **FAIL** |
| 5 | final boundary ≥9/12 per class | above_infeasible **1/12**, below_feasible 8/12, exact 10/12 | **FAIL** |

## Granite vs base granite vs qwen (same frozen sets, same decoding)

| metric | granite base | granite FT | qwen base | qwen r1 (official) |
|---|---|---|---|---|
| baseline six | 1/6 | 2/6 | 0/6 | 4/6 |
| main holdout decisions (mat/tail/log) | 88.8/95.4/54.8 | 99.2/99.8/100.0 | 92.2*/100*/100* | 99.0/100/100 |
| mean offer dev (mat/tail/log) | 60.3/79.8/46.3% | 1.59/1.60/1.05% | 36–54% | 0.55/0.65/0.77% |
| edge holdout all-12 / clean-10 | 50.0 / — | 75.0 / 70.0 | — | 91.7 / 90.0 |
| final boundary above/below/exact | 1/12 · 11/12 · 6/12 | 1/12 · 8/12 · 10/12 | 0/12 · 12/12 · 12/12 | 8/12 · 5/12 · 8/12 |
| deployment (Q8_0 in 0.3.34) | — | **loads, 6/6 JSON, 0.87 s steady** | — | loads after gguf-fix, 0.94 s |

\* qwen-base holdout numbers from the saved base-audit artifact
(`reports/reference/base-evaluations/raw/eval_base-audit_20260718_235632.json`).
**Correction 2026-07-25:** the material figure was published here as 94.9% — a transcription
error, not a different run. The cited artifact stores 92.2%, and recomputing from its 500 saved
main-holdout material rows gives 461/500 = 92.20%. The independent earlier base run
(`eval_base_20260718_140749.json`) gives the same 461/500. No evaluation anywhere in the project
produces 94.9% on this metric. Corrected to 92.2%; nothing else in this table changes.

Contamination disclosure (also in protocol §6): 2 of 12 edge-holdout prompts are verbatim in
the round-one edge train set (SYNEDGE-T-INF-T2/T2, SYNEDGE-T-INF-T11/T11 — both trained-on by
qwen r1 AND granite-ft equally); edge-holdout is therefore reported all-12 and clean-10.
Final boundary / stress / main holdouts have zero overlap.

## Reading the boundary result honestly

- Granite-FT is **permissive** at the boundary: above_infeasible 1/12 — unchanged from its
  base (1/12). On identical data, qwen r1 moved 0/12 → 8/12 on that class (while over-refusing
  feasible cases). Granite's edge-holdout errors are 3 false NO_BIDs (overcautious there) yet
  it stays permissive at the exact ratio boundary — the same ±3% instability seen across all
  three qwen rounds, now reproduced in a second architecture under a frozen protocol. This
  strengthens the "observed limitation under the current setup" conclusion, and remains an
  observed limitation — not proof of a fundamental limit.
- Granite's strengths are real: cleanest deployment chain (no MTP/no template surgery), the
  best exact-boundary pricing class (10/12), 100% logistics decisions, native-fit training,
  and the fastest FT inference measured so far (0.87 s steady in the production runtime).

## Status per required scale

- Granite training + evaluation pipeline: **PASS** (all gates, full run, all sets evaluated)
- Granite round-one model quality vs pre-registered bars: **FAIL** (criteria 1, 4, 5)
- Scientific comparability to qwen r1: **PASS** — identical data (bit-for-bit, hash-pinned),
  hyperparameters, stack, seed, eval sets, decoding; per-device batch also identical because
  granite fits natively (verified, not assumed).
