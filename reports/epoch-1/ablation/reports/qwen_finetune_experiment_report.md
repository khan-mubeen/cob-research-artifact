# Qwen3.5-4B bidding-agent fine-tuning experiment — consolidated report

Data version v4 · concluded 2026-07-19 · all numbers extracted from saved evaluation
artifacts in this directory (sources cited per table; no inference was rerun to produce
this report). Machine-readable versions: `qwen_comparison_summary.json` / `.csv`.

## 1. Selected pure fine-tuned checkpoint: round one

`models/finetune/qwen3.5-4b-bidding-lora/final`

Verified from artifacts (main holdout, 500 examples/stage, identical sets and decoding):

| checkpoint | decisions mat/tlr/log | mean offer dev mat/tlr/log | median dev mat/tlr/log | baseline six |
|---|---|---|---|---|
| **round 1** | **99.0 / 100 / 100 %** | **0.55 / 0.65** / 0.76 % | 0.36 / 0.49 / 0.56 % | 4/6 |
| round 2 | 98.4 / 100 / 100 % | 0.75 / 0.76 / **0.73** % | 0.61 / 0.61 / 0.57 % | 6/6 |
| round 3 | 98.2 / 100 / 100 % | 0.79 / 0.76 / 0.77 % | 0.62 / 0.61 / 0.59 % | 6/6 |

Rationale: best main-holdout decision agreement, lowest material and tailoring offer
deviation, and no disadvantage on the real v4 distribution, whose 246,025 eligible
tailoring pairs all have completion/deadline ratios ≤ 0.607828 (exhaustive scan,
`audit_guards.py --scan-band`) — the synthetic boundary band where round one loses is
unreachable in this data.

**Evidence against the selection, reported not hidden**: (a) round one scores 4/6 on the
six fixed baseline cases (misses the two ratio-1.06/1.12 deadline cases) where rounds
two/three score 6/6 — those cases sit in the zero-exposure band; (b) round two has
marginally lower logistics deviation (0.73 vs 0.76 %). Neither changes the selection
under the stated rationale, but both are real.

Rounds two and three are retained as boundary-focused experimental variants and negative
findings (§4), adapters preserved at `...-lora-r2/final` and `...-lora-r3/final`.

## 2. Official three-arm comparison (pure arms, no guards)

Sets: baseline six (6), main holdout (1,500 = 500/stage), edge holdout (12), final
boundary holdout (36). Sources: `eval_base-audit_20260718_235632.json`,
`eval_finetuned-r1-audit_20260719_001029.json`. The rule-based arm is the reference:
every label on every set *is* the rule-based output (asserted at generation), so its
agreement is 100 % and deviation 0 by construction.

**Main holdout** (false BID = rule says NO_BID, model bids; false NO_BID = the reverse):

| metric | rule-based | base Qwen | fine-tuned r1 |
|---|---|---|---|
| decision agreement mat/tlr/log | 100 % (ref) | 92.2 / 100 / 100 % | 99.0 / 100 / 100 % |
| false BIDs (of 1,500) | 0 | 39 | 4 |
| false NO_BIDs | 0 | 0 | 1 |
| JSON validity | — | 100 % | 100 % |
| mean abs offer dev mat/tlr/log | 0 (ref) | 53.9 / 36.3 / 24.6 % | 0.55 / 0.65 / 0.76 % |
| median abs offer dev | 0 (ref) | 41.5 / 19.3 / 21.4 % | 0.36 / 0.49 / 0.56 % |
| completion-days exact mat/tlr/log | 100 % (ref) | 97.2 / 0.2 / 99.8 % | 99.6 / 85.8 / 100 % |

**Boundary sets** (correct decisions):

| set | rule | base | fine-tuned r1 |
|---|---|---|---|
| baseline six (all true NO_BID) | 6/6 | 0/6 | 4/6 |
| edge holdout (12) | 12/12 | 6/12 (6 false BIDs) | 11/12 (1 false BID) |
| final boundary: below 0.97 / exact 1.00 / above 1.03 | 12/12/12 | 12 / 12 / 0 | 5 / 8 / 8 |

**Latency / throughput** (documented measurements, not re-benchmarked): rule-based
sub-millisecond per decision (pure formulas). Base GGUF Q8_0 on the RTX 5090, full
offload: 1.05–1.5 s per warm decision, 85.2 generated tok/s over 829 calls
(slm_check, 2026-07-15). HF bf16 evaluation runtime: ~1.9–2.0 examples/s at generation
batch 16 (audit evals). Fine-tuned adapter shares the architecture — equivalent
per-call cost; GGUF latency for r1 pends re-quantization (not yet performed).

Reproducibility note: the base arm's offer deviations differ ≤0.7 pp from its original
2026-07-18 eval because adding eval sets changed generation batch shapes (batched greedy
decoding is not bit-stable across batch composition). Both raw artifacts are preserved;
this table uses the audit rerun so all arms share one run configuration.

## 3. Hybrid engineering arm (separate — never a substitute for §2)

**`fine-tuned r1 + deterministic dynamic-feasibility guard (Guard B)`** — when the model
answers BID, the dynamic constraint is recomputed with the production formulas and the
bid is converted to NO_BID if infeasible. Offline audit only (`guard_audit.json`); the
guard is **not implemented in the live agent path**.

| set | r1 pure | r1 + Guard B |
|---|---|---|
| baseline six | 4/6 | 6/6 |
| main holdout decisions mat/tlr/log | 99.0 / 100 / 100 % | 99.8 / 100 / 100 % (false BIDs 4→0) |
| edge holdout | 11/12 | 12/12 |
| final boundary below/exact/above | 5 / 8 / 8 | 5 / 8 / **12** |

Stated plainly:
- The hybrid **does not pass the pre-registered synthetic boundary bar** (≥9/12 per
  class): below 5/12, exact 8/12. It is not claimed to.
- Its production recommendation is a **distribution-specific waiver**: all 246,025 real
  eligible manufacturer-order pairs have ratios ≤ 0.607828, so the failing band has zero
  exposure in v4.
- **False NO_BIDs remain unrecoverable** — the guard only removes invalid BIDs; there is
  no offer to restore on a wrongly refused order.
- Supervisor final-chain validation (budget + remaining-time check in
  `supervisor.py`) remains a separate, later safeguard and is unchanged.
- Context: round three + Guard B scores 7/12 · 11/12 · 12/12 on the final boundary
  holdout — closer to the bar (still failing `below`). **Round one is nonetheless
  selected for the hybrid** because pricing quality dominates once the boundary band has
  zero real exposure: r1's offer deviations (0.55/0.65/0.76 %) beat r3's
  (0.79/0.76/0.77 %), its main-holdout decisions are best, and under the guard both
  models' false-BID direction is fully closed anyway.

## 4. Boundary investigation — consolidated ablation

| round | change | outcome |
|---|---|---|
| 1 | no near-boundary NO_BID supervision (infeasible ratios ≥1.32 only) | permissive boundary (~1.15); missed 1.06/1.12 baseline cases; best pricing |
| 2 | deterministic boundary coverage 1.03–1.15 + at-boundary BIDs | 6/6 baseline but overly conservative: refuses feasible 0.97–1.00 (stress 8 % correct) |
| 3 | weighting only (boundary-infeasible ×4 vs feasible ×5) | falsified row imbalance as the main explanation; direction flipped between probe sets |

Consistent findings: fine-tuning improved pricing by ~50× (54 %→0.6 % deviation) and
made completion reporting honest (base misreports toward eligibility — 65 Guard-A/B
disagreements; r1 has 13, r3 has 4). Single-pass BID/NO_BID decisions remained unstable
on synthetic cases close to ratio 1.00 across all three rounds. This is an **observed
limitation under the current prompts, decoding, model size, and training setup** — not a
proven fundamental model limit. The current v4 distribution has zero exposure near that
boundary (max real ratio 0.6078).

## 5. Reproduction

Seeds: 42 everywhere (order sampling, splits, shuffles, LoRA init, subsampling).
Environment: Windows, RTX 5090 laptop, `.venv` Python 3.11, torch 2.10.0+cu130,
unsloth 2026.7.2, triton-windows 3.7.1.post27, transformers 5.5.0, trl 0.24.0.
Base checkpoint: `Qwen/Qwen3.5-4B` (safetensors, HF cache). All commands run from the
repo root with `PYTHONPATH=.` and `.venv\Scripts\python.exe`.

Datasets (`data/finetune/v4/`): train 44,342 (material 11,396 / tailoring 13,182 /
logistics 19,764), holdout 4,916, from 3,000 orders sampled of 50,000, split disjoint by
order id (2,700/300). Edge set v1 (48; round one; preserved in `round1_edge/`), edge set
v2 (84; rounds two/three; 72 train / 12 holdout). Boundary stress 36 (dev),
final boundary holdout 36 (untouched, pre-registered bar in its generator docstring).

```
# data
python src/experiments/generate_finetune_data.py
python src/experiments/generate_edge_examples.py           # v2 edge set (r2/r3)
python src/experiments/generate_boundary_stress.py
python src/experiments/generate_final_boundary_holdout.py

# training (r1 used the v1 edge set: restore data/finetune/v4/round1_edge/* first)
python src/experiments/finetune_lora.py --include-edge --edge-repeat 5 --subsample 15000 --batch-size 8 --grad-accum 2                                              # round 1
python src/experiments/finetune_lora.py --include-edge --edge-repeat 5 --subsample 15000 --batch-size 8 --grad-accum 2 --output-dir models/finetune/qwen3.5-4b-bidding-lora-r2   # round 2
python src/experiments/finetune_lora.py --include-edge --edge-repeat 5 --edge-weights "tailoring:boundary_infeasible=4" --subsample 15000 --batch-size 8 --grad-accum 2 --output-dir models/finetune/qwen3.5-4b-bidding-lora-r3  # round 3

# evaluation (writes eval_<tag>_<ts>.json here)
python src/experiments/eval_finetune.py --tag base-audit
python src/experiments/eval_finetune.py --tag finetuned-r1-audit --adapter models/finetune/qwen3.5-4b-bidding-lora/final

# guards + consolidation (offline, saved artifacts only)
python src/experiments/audit_guards.py --scan-band
python src/experiments/audit_guards.py --guards
python src/experiments/consolidate_qwen_report.py
```

## 6. Conclusions, limitations, next steps

**Conclusions.** Fine-tuning turns Qwen3.5-4B from an unusable pricer (36–54 % offer
error, bids on everything) into a near-rule-faithful bidder (≤0.8 % error, 99–100 %
decision agreement) at ~1 s/decision. Decision behavior within ±3 % of the
completion/deadline boundary is unstable in every configuration tested; a deterministic
post-decision guard closes the false-BID direction completely and cheaply.

**Limitations.** Boundary instability under the current setup (§4 phrasing); false
NO_BIDs are unguardable; hybrid results are engineering, not pure-SLM, evidence; GGUF
re-quantization of r1 not yet done (HF-runtime results only); single 24 GB GPU, one seed
per round.

**Recommendations.**
- *Research result*: **round one** represents fine-tuned Qwen in the official
  rule-based vs base SLM vs fine-tuned SLM comparison; rounds two/three reported as
  boundary ablations.
- *Engineering configuration*: **round one + Guard B** for the v4 distribution
  (waiver rationale in §3); integration point when approved:
  `dynamic_feasibility_reason()` in `bidding_formulas.py`, called from the agent
  callbacks after `slm_decide_*`, mirroring the existing static pre-filter — not
  `validate_bid_values()`, which stays structural.
- *Next experiment*: repeat this pipeline (data generation → LoRA → four-criteria +
  boundary evaluation) for the remaining five shortlisted models per Huong's
  comparison requirement, reusing the frozen eval sets and this report's format.
  Not started.
