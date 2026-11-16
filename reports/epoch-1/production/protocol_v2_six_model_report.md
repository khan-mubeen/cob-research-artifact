# Protocol v2 — Six-Model Cross-Architecture Comparison

**Supervisor report · 2026-07-22 · chain-of-bid**
All figures below were read from saved machine-readable evaluation artifacts. No training or
evaluation was rerun to produce this document.

---

## 1. Executive summary

All six base checkpoints and their six Protocol v2 fine-tuned adapters were evaluated on the
identical frozen evaluation suite.

- **Fine-tuning worked, and worked consistently.** Every one of the six models improved sharply
  on normal (in-distribution) decision agreement and pricing. Mean offer deviation fell from a
  24–91% band at base to **0.7–2.6%** after fine-tuning; main-holdout decision agreement reached
  **97.4–100%** on every stage for every model.
- **Every model failed the pre-registered synthetic near-boundary robustness requirement**
  (≥9/12 correct in each of the three classes on the locked Protocol v2 robustness holdout).
  Six out of six. This is a real, pre-registered result and was not softened after the fact.
- **Guard B corrected false BIDs but could not recover false NO_BIDs.** It lifted every model to
  6/6 on the baseline-six set with **zero new errors introduced**, but it left the robustness
  failures essentially untouched, because those failures are over-refusals — a class of error the
  guard is structurally incapable of repairing.
- **Qwen-v2 is the strongest pure research model** (best pricing, cleanest JSON, the only clear
  rule-following behaviour in the matched diagnostics).
- **Qwen-v2 + Guard B is the recommended engineering configuration for the current v4
  distribution.**

> The guarded configuration **does not** pass the pure-SLM robustness bar. It is reported as a
> separate engineering arm and must never be presented as the pure fine-tuned result.

---

## 2. Models and experimental setup

**Final six-model slate**

| # | Model | Revision |
|---|---|---|
| 1 | Qwen/Qwen3.5-4B | `851bf6e806ef` |
| 2 | ibm-granite/granite-4.1-3b | `c0650403e44e` |
| 3 | microsoft/Phi-4-mini-instruct | `cfbefacb9925` |
| 4 | google/gemma-4-E2B-it | `179516f0c449` |
| 5 | mistralai/Ministral-3-3B-Instruct-2512-BF16 | `b6d637bef239` |
| 6 | HuggingFaceTB/SmolLM3-3B | `a07cc9a04f16` |

**Frozen Protocol v2 (identical for all six).** 15,000 stratified real training examples plus 203
unique synthetic edge examples (repeat ×1) = **15,203 rows**; **seed 42**; **1 epoch**;
**951 optimizer steps**; **effective batch 16** (per-device 8 × grad-accum 2); **LoRA rank 16,
alpha 16**; sequence length 1024; bf16; adamw_8bit at LR 2e-4. All dataset files are SHA-256
pinned in `protocol_v2.json`.

**Functionally equivalent, model-specific LoRA targets.** Architectures differ, so target *names*
differ while the adapted *function* (attention + MLP projections) is held constant; exact
trainable counts are published per model in Table A. Phi required fused `qkv_proj`/`gate_up_proj`
names; Ministral required language-tower-only targeting to exclude its vision tower.

**Non-thinking decoding for all primary arms**, greedy, 200 max new tokens. Where a model has a
reasoning mode it was disabled through the documented mechanism and *verified by generation*, not
assumed from the flag.

**Identical evaluation sets** for all twelve checkpoints: baseline six, main holdout (1,500),
edge holdout (12), developer boundary stress (36), locked Protocol v2 robustness holdout
(36, SHA `cc65b091cc10887f`, frozen before any v2 training), and the high-value numeric JSON
stress set by offer-value range.

**Guard B is disabled by default** (`SLM_DYNAMIC_GUARD` unset). All primary results are pure-SLM.

### Slate changes and context

**Nemotron-3-Nano-4B and LFM2.5-8B-A1B were de-scoped** on schedule, tooling, architecture, and
environment risk. Nemotron requires a `mamba-ssm` build unavailable on native Windows (its earlier
chat-template blocker did resolve, and it fails cleanly at import in 0.8 s without touching the
GPU). LFM2.5 loads and runs, but its reasoning cannot be disabled and 91.5% of its parameters sit
in fused 3-D expert tensors outside standard LoRA reach. **Neither was proven impossible**, and
**neither was proven responsible for any hardware crash.** Their evidence is retained, labelled
de-scoped, under `reports/feasibility/`.

**Ministral 3 3B Instruct and SmolLM3-3B replaced them. Huong has confirmed approval of this
substitution; it is settled.** (This confirmation is organisational and is not recorded in any
repository artifact.)

**Infrastructure risk, recorded separately.** Seven crash/interruption incidents occurred across
the project, with six different or absent bugcheck codes, **three of them while no model workload
was running at all**, against a continuous background of WHEA corrected PCI Express Root Port
errors (8,015 on 19 July; ~48 in ten idle minutes on 21 July) while the GPU driver itself showed
zero XID faults. These affected **execution continuity** — one Ministral run was interrupted at
step 834/951 and resumed from a recovery checkpoint. **No claim is made that any model caused
them.** Full record: `reports/feasibility/hardware_incident_log.json`.

---

## 3. Main six-model results

### Table A — training and resources

| Model | Revision | Trainable params | % | Train loss | Runtime | Peak VRAM (alloc / reserved) | Controlled adaptation |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | `851bf6e8` | 21,233,664 | 0.47% | 0.104 | 104 min | 10.8 / 18.0 GiB | multimodal; language tower only |
| Granite-4.1-3B | `c0650403` | 31,129,600 | 0.91% | 0.133 | 89.1 min | 9.3 / 24.6 GiB | none (names matched directly) |
| Phi-4-mini | `cfbefacb` | 23,068,672 | 0.60% | 0.147 | 82 min | 11.2 / 26.3 GiB | fused `qkv_proj`/`gate_up_proj`; pad token ≠ eos |
| Gemma-4-E2B | `179516f0` | 25,337,856 | 0.49% | 0.059 | 72 min | 12.5 / 22.2 GiB | unsloth-only dispatch; `<\|turn\|>` markers |
| Ministral-3-3B | `b6d637be` | 24,707,072 | 0.64% | — ¹ | 75 min + 14 min resume | 10.7 / 14.4 GiB | BF16 repo (not FP8 default); multimodal loader; **vision tower frozen** |
| SmolLM3-3B | `a07cc9a0` | 30,228,480 | 0.97% | 0.136 | 72 min | 9.0 / 23.9 GiB | pad token ≠ eos; non-thinking render at train time |

¹ Ministral's run was interrupted by a host crash at step 834/951 and resumed once from
checkpoint-800; the reported tail-segment loss (0.017) covers only the final 151 steps and is not
comparable to the other models' full-run losses. All other protocol variables were unchanged and
optimizer/RNG/scheduler state were restored exactly.

Phi (26.3 GiB), Granite (24.6 GiB) and SmolLM3 (23.9 GiB) touched or briefly exceeded the 23.9 GiB
physical ceiling at peak reserved; step times were unaffected, indicating transient allocator
caching rather than sustained system-memory spill. For Granite specifically the v2 run reserved
24.6 GiB against v1's 17.8 GiB on the same configuration, yet held the same 5.62 s/step — the
brief spill cost no throughput. Every figure in this table is the value printed at the end of that
model's own training log under `reports/epoch-1/production/training-logs/`.

### Table B — evaluation results (fine-tuned arms)

Reference points: the **rule-based system** is the label oracle (100% by construction).
**Base models** scored 0–4/6 on baseline six, 24.0–100% main-holdout decision agreement, and
**24–91% mean offer deviation** — the pathology fine-tuning removes.

| Fine-tuned model | Main decisions mat/tail/log | Mean offer dev mat/tail/log | JSON validity | Baseline six (pure → +Guard B) | Robustness v2 above / below / exact |
|---|---|---|---|---|---|
| **Qwen-v2** | 98.0 / 100 / 100 | 0.86 / 0.71 / 0.71% | **100%** ³ | 4/6 → **6/6** | 12/12 · 5/12 · 0/12 |
| Granite-v2 | 98.0 / 99.0 / 100 | 1.61 / 1.64 / 1.16% | **99.08%** ³ (15 malformed of 1,626); 99.67% on the main holdout alone (5 of 1,500) | 0/6 → **6/6** | 2/12 · 3/12 · 4/12 |
| Phi-v2 | 98.0 / 100 / 100 | 1.09 / 1.10 / 1.47% | **100%** ³ (but 34 malformed of 576 diagnostic rows — see ³) | 1/6 → **6/6** | 11/12 · 2/12 · 1/12 |
| Gemma-v2 | 97.4 / 100 / 100 | 2.41 / 2.62 / 1.86% | **99.14%** ³ (14 malformed of 1,626); 100% on the main holdout alone | 3/6 → **6/6** | 8/12 · 7/12 · 6/12 |
| Ministral-v2 | 98.0 / 100 / 100 | 1.23 / 1.14 / 0.82% | **100%** ³ | 2/6 → **6/6** | 12/12 · 0/12 · 0/12 |
| SmolLM3-v2 | 98.4 / 100 / 100 | 1.17 / 1.10 / 1.44% | **100%** ³ | 1/6 → **6/6** | 12/12 · 0/12 · 0/12 |

³ **JSON-validity denominator.** All percentages in this column are over the **six sets the
pre-registered criteria are scored on** — baseline six, edge holdout, main holdout, boundary
stress, final boundary holdout, robustness holdout v2 = **1,626 rows per model**, identical for
all six architectures (`ALL_SETS` in `consolidate_final.py`). Percentages over the main holdout
alone are higher and are *not* the criterion; both are given above where they differ. The
`scale_diagnostic`, `json_format_stress` and `magnitude_diagnostic` sets are section-3/4
explanatory evidence and are outside this denominator — which is why Phi is 100% here while
section 4 correctly reports it emitting thousands-separator numbers: all 34 of Phi's malformed
rows fall in those diagnostic sets, none in the six criteria sets.

**Pre-registered criteria (pure arm), by name**

| Model | JSON validity 100% | No decision regression vs own base | Offer-deviation improvement vs own base | Near-boundary robustness ≥9/12 per class | Overall |
|---|---|---|---|---|---|
| Qwen-v2 | PASS | PASS | PASS | **FAIL** | **FAIL** |
| Granite-v2 | **FAIL** (15) | PASS | PASS | **FAIL** | **FAIL** |
| Phi-v2 | PASS | PASS | indeterminate ² | **FAIL** | **FAIL** |
| Gemma-v2 | **FAIL** (14) | PASS | PASS | **FAIL** | **FAIL** |
| Ministral-v2 | PASS | PASS | PASS | **FAIL** | **FAIL** |
| SmolLM3-v2 | PASS | PASS | PASS | **FAIL** | **FAIL** |

² Phi-base produced no scorable tailoring offers, so "strictly improved on every stage" is
undefined there. Substantively Phi-v2 improved pricing by roughly 30× on the measurable stages.

---

## 4. Key cross-model findings

**No single mechanism explains all six boundary failures.** Matched counterfactual diagnostics
(holding ratio, deadline, quantity, workload or capacity fixed in turn) give a different answer
per architecture:

- **Qwen-v2 is the clearest deadline/completion rule follower.** It tracks the deadline and the
  agent profile, and is invariant to both absolute magnitude scaling and quantity/workload
  re-splitting — the behaviour expected of a model applying the rule.
- **Granite-v2 is deadline- and capacity-insensitive near the boundary**, but direct tests show it
  is invariant to absolute magnitude and to the quantity/workload split. **It is not proven to be
  a magnitude matcher**; the accurate statement is that it fails to use the deadline dimension.
- **Phi-v2 remains sensitive to the quantity/workload decomposition** — its decisions shift when
  the same total is re-split between backlog and order quantity, which the rule says is irrelevant.
- **Gemma-v2 shows partial rule tracking** but is undermined by reasoning-leakage failures on
  difficult cases.
- **SmolLM3-v2 and Ministral-v2 became highly conservative on the Protocol v2 robustness set** —
  both scored a perfect 12/12 above the boundary while refusing every feasible case below and at
  it (0/12, 0/12). **This is the same phenomenon as the directional instability seen elsewhere,
  viewed from one side:** the very same checkpoints behave *permissively* on the older final
  boundary holdout (each scoring 12/12 below-feasible and 11–12/12 exact-feasible there) while
  being *maximally strict* on the robustness holdout. A single checkpoint therefore swings between
  over-permissive and over-strict depending on which near-boundary set it sees.

  This demonstrates **unstable single-pass behaviour near ratio 1.00** under the tested prompts,
  decoding, model sizes, and training protocol. **It is not claimed to be a proven fundamental
  arithmetic-resolution limit** of these architectures.

**JSON output quality after fine-tuning splits three ways:**

- **Clean:** Qwen, SmolLM3, and Ministral — 100% validity on every set and every offer-value band.
- **Magnitude-triggered comma formatting:** Granite and Phi emitted thousands-separator numbers
  (e.g. `25,800.6`) that break strict parsing, only at offers ≳ €10,000. Both bases were clean, so
  fine-tuning introduced the habit.
- **Reasoning leakage and numeric runaway:** Gemma, on difficult near-boundary cases — a
  difficulty-triggered failure rather than a magnitude-triggered one, and distinct from the comma
  fault.

---

## 5. Guard B and result families

The two families are reported separately and must stay separate.

**Primary research comparison** — rule-based system · base SLM · **pure Protocol v2 fine-tuned
SLM** (no guard, no post-processing). The architecture ranking is decided here.

**Additional engineering configuration** — **fine-tuned SLM + Guard B.**

How Guard B behaves:

- It recomputes authoritative dynamic feasibility from the agent profile and RFQ, **only when the
  SLM has produced a BID**.
- It **suppresses** an infeasible BID so nothing is published.
- It **does not replace model pricing** — an accepted BID passes through byte-for-byte.
- It **cannot recover a false NO_BID**; a model refusal stays a refusal.
- It **introduced zero new errors** across all verified tests on all six models.
- **Supervisor final-chain validation remains a separate, later safeguard.**

Verified effect: baseline six rose to **6/6 for all six models** (2–6 false BIDs corrected each,
0 remaining false NO_BIDs, 0 new errors). On the robustness holdout the guard changed little —
for Qwen-v2, SmolLM3-v2 and Ministral-v2 it removed **zero** false BIDs because those models were
already perfect above the boundary; their 24 remaining errors are all false NO_BIDs, which the
guard cannot repair. **No model passes the robustness criterion even with the guard enabled.**

---

## 6. Conclusions, recommendation, and limitations

- **All six models transferred normal bidding and pricing behaviour successfully.** This is the
  substantive positive result: a ~3–4 B parameter local model, LoRA-tuned for one epoch, reaches
  97–100% decision agreement and ~1% pricing error on real v4 traffic.
- **No model passed the synthetic near-boundary robustness bar** — 0 of 6.
- **No further boundary-focused fine-tuning is recommended during this internship phase.** Six
  architectures and five data-recipe iterations have now failed this bar in five different ways;
  the marginal return on another data iteration is low.
- **Qwen-v2 best represents pure fine-tuned performance.**
- **Qwen-v2 + Guard B is recommended for the current v4 engineering configuration.** This is
  defensible specifically because the exhaustive real-data census contains **no eligible
  manufacturer pair near the failing boundary band** (max realised ratio 0.608), so the guard's
  practical job is to catch off-distribution cases, which it does perfectly and without side
  effects.
- **Guarded and pure results must remain separately labelled** in every downstream report.

**Limitations**

- One primary training seed (42); seeds 43/44 are pre-registered but unused.
- Small synthetic robustness sets (36 cases, 12 per class).
- No real v4 manufacturer/order pair lies in the failing boundary band, so the failure has no
  demonstrated effect on current production traffic.
- Architecture-specific adaptations were required (fused target names, pad-token fixes,
  multimodal loaders, unsloth-only dispatch, non-thinking render alignment) — all disclosed.
- Two model replacements were made under schedule and infrastructure constraints, not because the
  originals were proven infeasible.
- Hardware instability interrupted execution; one run required a checkpoint resume.

**Key artifacts for reproducibility**

```
reports/cross_model/protocol_v2.json               frozen protocol + dataset SHA-256 hashes
reports/cross_model/protocol_v2_design.md          data design and rationale
reports/cross_model/final_comparison.json          machine-readable four-model comparison
reports/cross_model/diagnostics_results.json       scale / magnitude diagnostic results
reports/cross_model/*_launch_gates.json            per-model pre-launch gate evidence
reports/cross_model/*_template_alignment.json      train vs GGUF render alignment checks
reports/slm_comparison/v4/eval_*.json              raw per-row evaluation records (all 12 arms)
reports/feasibility/feasibility_report.md          six-candidate feasibility matrix + de-scoping
reports/feasibility/hardware_incident_log.json     consolidated hardware incident record
data/finetune/v4/protocol_v2/                      frozen training + locked robustness holdout
models/finetune/*-bidding-lora-v2/final/           the six Protocol v2 adapters
```
