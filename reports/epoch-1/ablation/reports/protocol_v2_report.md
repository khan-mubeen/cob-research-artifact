# Protocol v2 — results: the coverage fix did not transfer; boundary behavior is deadline-scale-dependent

2026-07-20 · protocol frozen before training (`protocol_v2.json`, design `protocol_v2_design.md`,
data report + hashes `protocol_v2_data_report.json`) · machine-readable results
`protocol_v2_results.json` · training consoles + crash logs in `reports/cross_model/`.

Labels: **[executed]** evidence from runs here · **[docs]** repository artifacts ·
**[interpretation]** judgment · **[unresolved]** needs user decision.

## Runs [executed]

Both v2 runs from original base checkpoints, identical frozen recipe (15,000 stratified +
203 unique v2 edge rows ×1, 951 steps, seed 42, batch 8×2, LoRA r16/α16, bf16):

| run | wall time | train loss | peak reserved | note |
|---|---|---|---|---|
| qwen v2 | 104.1 min (6.58 s/step) | 0.104 | 18.0 GiB (native fit) | restarted once — OS crash at step 491 (no checkpoint reached); seed-42 rerun is the protocol run |
| granite v2 | 89.1 min (5.62 s/step) | 0.133 | **24.6 GiB (marginally over 23.9 physical — brief spill, disclosed; step time unaffected vs v1's 5.62)** | restarted once — OS crash at step 195 |

Two OS-level crashes during sustained GPU load in one night is a machine-stability risk,
not a pipeline defect (both reruns completed identically).

## Pre-registered criteria verdicts (frozen before results; stated plainly)

| criterion | qwen v2 | granite v2 |
|---|---|---|
| 1 JSON 100% every set | **PASS** (0 malformed / 1,626) | **FAIL** — 15 malformed: thousands-separator commas in offer_amount (e.g. `11,500.02`) on bids ≳10k EUR |
| 2 main-holdout decisions ≥ own base | **PASS** (98.0/100/100 vs 92.2/100/100 †) | **PASS** (98.0/99.0/100 vs 88.8/95.4/54.8) |
| 3 offer dev improved vs own base | **PASS** (0.86/0.71/0.71%) | **PASS** (1.61/1.64/1.16%) |
| 4 tier-3 robustness ≥9/12 each class | **FAIL** — 12/12 above, 5/12 below, **0/12 exact** | **FAIL** — 2/12 above, 3/12 below, 4/12 exact |
| **overall** | **FAIL (criterion 4)** | **FAIL (criteria 1, 4)** |

† **Correction 2026-07-25:** the qwen-base material figure was published as 94.9%, a
transcription error. The authoritative raw artifact
(`reports/reference/base-evaluations/raw/eval_base-audit_20260718_235632.json`) stores 92.2%,
and recomputation from its 500 saved rows gives 461/500 = 92.20%. The verdict is unaffected:
98.0 ≥ 92.2, so criterion 2 remains **PASS**.

## The central finding: deadline-scale dependence [executed, all six checkpoints]

Correct decisions /12 per class, on the two boundary sets that differ ONLY in deadline range:

| model | final boundary (days 3–6): above / below / exact | robustness v2 (days 11–12): above / below / exact |
|---|---|---|
| qwen base | 0 / 12 / 12 | 0 / 12 / 12 |
| qwen r1 (v1) | 8 / 5 / 8 | 11 / 3 / 5 |
| **qwen v2** | **2 / 12 / 10** | **12 / 5 / 0** |
| granite base | 1 / 11 / 6 | 4 / 10 / 10 |
| granite v1 | 1 / 8 / 10 | 9 / 4 / 1 |
| **granite v2** | **1 / 8 / 10** | **2 / 3 / 4** |

Every fine-tuned checkpoint (both architectures, all recipes) is more willing to refuse at
long deadlines than short ones, holding the ratio fixed. Near-boundary NO_BID supervision
generalizes **up** the deadline scale but not **down**: qwen v2 trained on days 7–10 becomes
perfectly strict at 11–12 d (12/12 above — at the cost of refusing everything at the exact
boundary) while staying permissive at 3–6 d (2/12). The models are not learning the ratio
rule `completion/deadline > 1`; they are pattern-matching magnitude cues that correlate with
it inside the trained range **[interpretation]**. Granite v2 additionally destabilized in
both directions plus JSON formatting — the identical data produced qualitatively different
failure modes in the two architectures.

## Did Protocol v2 correct the coverage weakness consistently? **No.**

The v1 gap (no infeasible coverage in (1.00, 1.315), feasible monoculture at 0.97) was real
and is fixed in the data [executed: `protocol_v2_data_report.json`]. But the corrected data
did not produce corrected behavior: qwen flipped from permissive to over-strict at the
robustness scale (r2's failure mode, now reproduced under a symmetric, repetition-free,
broad-coverage set — ruling out the "imbalance/repetition" explanations), and granite
regressed. Five boundary-training attempts across two architectures (qwen r1/r2/r3+v2,
granite v1/v2) have now failed the per-class bar in five different ways. This is an
**observed limitation under the current setup** (4B-class models, LoRA r16, 1 epoch, prompt
format as-is) — still not proof of a fundamental limit, but the marginal-return curve of
data-recipe iteration alone is now clearly flat **[interpretation]**.

## What stands regardless

In-distribution results are uniformly strong and deployment-ready for BOTH v2 models:
98–100% decisions, ≤1.6% offer deviation, base-level pathologies (46–80% pricing error,
0-refusal) fully corrected. The failures are confined to the synthetic boundary band that
has **zero support in real v4 data** (max eligible ratio 0.608 [docs]). Guard B
(authoritative recompute) remains proven to eliminate 100% of false BIDs with zero added
errors on saved artifacts [docs: guard audit].

## Recommendation for Phi and Gemma [interpretation; unresolved — user decision]

**Hold Phi/Gemma full training.** Running them under v2 as-is would likely document two more
architecture-specific boundary failures at ~2 h GPU each without advancing the main question.
Options to discuss, in my recommended order:
1. **Accept the architectural split**: SLMs own pricing + mainline decisions (excellent in
   every run); the boundary rule is enforced by Guard B at the supervisor validation point.
   Then Phi/Gemma CAN run under frozen v2 cheaply, scored primarily on tiers 1–2, with tier 3
   reported as context — a protocol-consistent, honest comparison of what SLMs are actually
   good at here.
2. Protocol v3 with deadline-scale-invariant supervision (boundary ladder crossed with the
   FULL deadline range 2–14 d in both training and a new gate set) — one more data iteration,
   justified only because scale-dependence is a *new, specific, testable* hypothesis, unlike
   the exhausted balance/coverage hypotheses.
3. Both: v3 data + Guard B as belt-and-braces.

## Preservation [executed]

Untouched: qwen r1/r2/r3 + granite v1 adapters, all prior reports and manifests, protocol v1
files, all GGUF artifacts, orders files. New artifacts under `-v2` paths only. Granite v1 is
re-labeled a Protocol v1 ablation per the directive.
