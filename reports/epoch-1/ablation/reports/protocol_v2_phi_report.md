# Protocol v2 — Phi-4-mini added; three-architecture cross-model results + Guard B

2026-07-20. Scope of this run: sections 1–7 of the directive (state, lineage, diagnostics,
JSON audit, Guard B, framing, **Phi through full evaluation**). Gemma (section 8) is deliberately
NOT started — the directive says stop after Phi and report before Gemma.

Evidence labels: **[executed]** run here with path · **[docs]** repo artifact · **[interp]**
judgment · **[unresolved]** needs user.

Machine-readable: `final_comparison.json`, `diagnostics_results.json`, `decision_census_*.json`,
`baseline_six_lineage.json`, `protocol_v2_data_report.json`, `scale_diagnostic_data_report.json`,
`json_format_stress_data_report.json`.

## 1. Current-state verification [executed]

HEAD `d85c9bb`, working tree carries only the (previously untracked) experiment tree plus the
guard edits (§5). All v2 report numbers reproduced from saved JSON (qwen-v2 robustness 12/5/0,
granite-v2 15 malformed, per-adapter hashes recorded). Nothing overwritten; orders files
untouched.

## 2. Baseline-six lineage — the 6/6 belongs to r2/r3, NOT r1 [executed]

`baseline_six_lineage.json`. Round one is **4/6 in all four of its saved evals**
(finetuned, r1-stress, r1-audit, r1-merged-hf); **r2 is 6/6 (two evals), r3 is 6/6 (one eval)**.
Every raw artifact is internally self-consistent. **Verdict: documentation/memory-attribution
issue, not an artifact mismatch.** r1 was selected as the official pure-FT arm on its full metric
balance (r2/r3 hit 6/6 on baseline six but were over-conservative/erratic elsewhere), so the
selection stands; only summaries that pinned 6/6 to r1 were wrong.

## 3. Scale-vs-magnitude diagnostic — the hypothesis is architecture-specific [executed]

432 rows, 4 matched families, frozen sha `4a6bee8b…`. The clean test is **F2** (completion time
held constant, only deadline moves): a rule-follower's BID count must swing 0→12 across
deadlines 2..14 (spread 12); a magnitude-matcher that ignores the deadline gives spread ~0.

| checkpoint | F2 deadline-spread (0–12) | reading |
|---|---|---|
| qwen base | 0 | insensitive (bids uniformly) |
| qwen r1 | **12** | tracks the deadline (rule-like) |
| qwen v2 | **11** | tracks the deadline |
| granite base | 0 | insensitive |
| granite v1 | 6 | partial |
| granite v2 | **0** | **deadline-blind** (keys off fixed quantity) |
| phi base | 3 | mostly deadline-blind |
| phi v2 | 3 | mostly deadline-blind |

**The v2-report "models pattern-match magnitude" reading is refuted for Qwen and supported for
Granite/Phi.** Qwen's fine-tunes genuinely track completion-vs-deadline; their boundary errors
concentrate in the confounded families (F1/F4 ≈50%) — i.e. fine ratio resolution near 1.00, not
deadline blindness. Granite v2 and Phi ignore the deadline dimension. **I do not call the single
magnitude hypothesis proven** — the matched diagnostics show it is architecture-dependent.
(F3, profile-sensitivity with RFQ numbers pinned, agrees: qwen r1/v2 spread 11/7, granite/phi 1–3.)

## 4. Numeric JSON-format audit — non-Qwen fine-tune phenomenon, magnitude-triggered [executed]

144-row stress set, offers 440–262,841 EUR, sha `b71ffac7…`. Strict-parse validity by band:

| checkpoint | <1k | 1k–10k | 10k–100k | >100k |
|---|---|---|---|---|
| every Qwen (base/r1/v2) | 100 | 100 | 100 | 100 |
| granite base | 100 | 100 | 100 | 100 |
| granite v2 | 100 | 100 | 90 | 92 |
| phi base | 100 | 100 | 100 | 100 |
| phi v2 | 100 | 100 | 96 | **38** |

Cause: thousands-separator commas in `offer_amount` (e.g. `11,500.02`). **Not general** — absent
in every base model and every Qwen checkpoint. It is a **fine-tuned-non-Qwen** fault, appearing
only after FT in Granite and Phi and only at high magnitudes (worse the larger the offer). Per the
directive, it is **not silently repaired** in the pure arm and the frozen v2 prompt was **not
changed** before Phi; any normalization is deferred to a separate controlled change or the guard.

## 5. Guard B — implemented, disabled by default, proven [executed]

Shared post-eligibility feasibility helpers in `bidding_formulas.py`
(`material_/manufacturer_/logistics_dynamic_check`, `dynamic_feasibility_reason`,
`apply_dynamic_guard`, `dynamic_guard_enabled`), reused by both `decide_*` and the guard. Wired
into all three agent callbacks **after `slm_decide_*`, before publish, gated on
`SLM_DYNAMIC_GUARD` (default off)**; not placed in `validate_bid_values`.

- **Regression**: `decision_census.py` — 329,313 real+synthetic decisions **byte-identical**
  pre/post refactor (sha `9560c94a…`). Rule behavior unchanged.
- **Unit tests**: 15/15 pass — every dynamic reason (completion_exceeds_deadline,
  insufficient_stock_and_restock, no_daily_capacity, workload) + the contract (suppress
  infeasible BID; feasible BID byte-identical; NO_BID never recovered; off by default).
- Guard ≡ rule ≡ label equivalence documented (`experiment_framing.md`), used to score the
  guarded arm exactly.

## 6. Three-architecture comparison (pure Protocol-v2 FT), base → v2 [executed]

| metric | qwen base→v2 | granite base→v2 | phi base→v2 |
|---|---|---|---|
| trainable params | 21.2M (0.47%) | 31.1M (0.91%) | 23.1M (0.598%) |
| train loss / runtime | 0.104 / 104 min | 0.133 / 89 min | 0.147 / 82 min |
| peak reserved VRAM | 18.0 GiB (fit) | 24.6 GiB (marginal) | 26.3 GiB (marginal spill) |
| gen throughput (HF, ex/s) | ~2.5 | ~2.6 | ~2.6 |
| baseline six | 0→**4/6** | 1→**0/6** | 4→**1/6** |
| main decisions mat/tail/log | →98/100/100 | →98/99/100 | →98/100/100 |
| main offer dev % | →0.86/0.71/0.71 | →1.61/1.64/1.16 | →1.09/1.10/1.47 |
| main JSON | 100 | 99.0 | 100 |
| malformed (official sets) | **0** | **15** | 0 |
| robustness above/below/exact | 12/5/0 | 2/3/4 | 11/2/1 |
| final-boundary above/below/exact | 2/12·12/12·10/12 | 1/12·8/12·10/12 | 1/12·12/12·10/12 |

**Pre-registered criteria (pure arm):**
- **qwen v2: FAIL** — criterion 4 (robustness) only. json/regression/offer all PASS.
- **granite v2: FAIL** — criteria 1 (JSON, 15 malformed) and 4.
- **phi v2: FAIL** — criterion 4; criterion 3 **indeterminate** (phi-base produced *no* scorable
  tailoring offers → `None`, so "strictly improved on every stage" is undefined; substantively
  phi-v2 pricing is 1.1% vs base 29–52% where measurable — a ~30× improvement, not a regression).

All three transfer the **in-distribution** result excellently (98–100% decisions, ~1% pricing,
base pathologies removed) and all three **fail the synthetic near-1.00 robustness bar** — in three
different ways (qwen over-strict-at-scale, granite erratic + malformed, phi over-strict).

## 7. Guard B impact — perfect on false BIDs, powerless on the robustness failures [executed]

| set (error type) | qwen v2 | granite v2 | phi v2 |
|---|---|---|---|
| baseline six → guarded | 4→**6/6** | 0→**6/6** | 1→**6/6** |
| final-boundary above_infeasible false-BIDs removed | 10 | 10 | 11 |
| robustness false NO_BIDs **remaining** (unrecoverable) | 19 | 17 | 21 |
| robustness per class **after** guard | 12/5/0 | 11/3/4 | 12/2/1 |

**The guard makes every model 6/6 on baseline six and clears the false BIDs on the above-boundary
class, adding zero new errors.** But the robustness failures are **false NO_BIDs** (feasible cases
the model over-refused), which Guard B cannot recover — so the guarded robustness per class is
essentially unchanged and **the hybrid does NOT pass criterion 4.** The guard is a real
engineering safeguard, not a fix for the research limitation. Reported separately, never
substituted for the pure arm.

## Conclusions (§10)

1. **Does Protocol v2 transfer across architectures?** [interp] The *pricing/decision* result
   transfers cleanly to all three (and by earlier evidence, four) architectures; the *near-1.00
   boundary* result fails in every one, with architecture-specific failure modes. So "consistent
   transfer" is **partial** — robust where it matters for real v4 traffic, unreliable in the
   synthetic band with zero real support.
2. **Best explanation of boundary behavior?** [interp, diagnostics-supported] Not one factor.
   **Qwen** tracks completion-vs-deadline but under-resolves ratios within ±3% of 1.00.
   **Granite and Phi** are largely deadline-blind and key off quantity magnitude. Do not treat the
   single magnitude hypothesis as proven.
3. **Malformed numeric JSON?** [executed] Model-specific: fine-tuned Granite and Phi only,
   magnitude-triggered; every Qwen checkpoint and every base is immune.
4. **Is Guard B justified as production config?** [interp] For the current v4 distribution, yes as
   a labelled engineering configuration — the exhaustive census shows no eligible near-boundary
   pairs, the guard equals the rule (329k-pair proof), it makes the existing NO_BID cases perfect
   and adds zero errors. It does **not** repair the robustness (false-NO_BID) failures, so it is a
   safeguard, not a scientific fix.
5. **Best pure research model?** [interp] **Qwen v2** — the only architecture that tracks the rule
   dimensionally (F2 spread 11), zero malformed JSON, best pricing.
6. **Strongest guarded config for v4 deployment?** [interp] **Qwen v2 + Guard B** — best pricing,
   deadline-aware, guard makes baseline/above-boundary perfect, zero malformed output.
7. **Do Phi results change the stop-v3 decision?** [interp] They reinforce stopping v3-for-boundary:
   a third architecture fails the bar, and magnitude-blindness now appears in 2 of 3. One more data
   recipe is unlikely to fix an architecture-level representational limit for granite/phi. Gemma
   (section 8) is the remaining data point before a final call.
8. **Huong's six-model requirement** [interp/docs] Under frozen Protocol v2: Qwen, Granite, Phi
   complete (3/6). **Gemma remains** (next task). LFM unsuitable (always-on reasoning) and
   Nemotron blocked on Windows (mamba-ssm) per the feasibility report — realistically the
   comparison is 4 trainable architectures + 2 documented-infeasible.

## Statuses

- Guard B implementation + regression + tests: **PASS**
- Phi Protocol v2 pipeline (fused targets, template, native-fit smoke): **PASS WITH CONTROLLED
  ADAPTATION** (fused LoRA names 23.1M; pad token `<|endofprompt|>` since Phi ships pad==eos)
- Phi v2 model quality vs pre-registered bars: **FAIL** (criterion 4; 3 indeterminate)
- Cross-model comparability: **PASS** (identical data/seed/budget/eval; per-device batch identical;
  adaptations disclosed)
- Gemma: **not started** (out of this run's scope by directive)
