# Six models, two bidding modes, 150 orders, dynamic resources

**Written:** 28 August 2026, 07:30
**Branch:** `main`, nothing committed or staged
**Scope:** twelve SLM batches plus two derived rule-based comparison rows, all on
one 150-order subset with dynamic agent state active.

All twelve runs completed. All twelve pass every invariant check. No model was
dropped and nothing was compromised to fit the night.

---

## 1. What was run

Every run used the same 150 orders, in the same `order_created` sequence, against
the `orders.csv` on disk. The subset is the first 150 of the previously verified
300-order SLM holdout, ordered by `order_created` then `order_id`, running
`ORD-15478` (2026-08-20 11:00) to `ORD-25134` (2026-11-13 12:00).

```
subset sha256  d566da1fcf6276ebed8a874ca14cf76c2a2f0e5bb626c883c980cec97ba095c4
parent  sha256 0e64da1c70239fa7856130d943c703e70cedf759a54af90c83ae214bb0491983
```

Both recorded with the ids in `reports/bidding/holdout_150_subset.json`, and the
parent hash re-verified against the holdout files at the start of the night.

Every batch ran through the real NATS pipeline with freshly started agent and
supervisor processes. Dynamic state lives only in process memory, so a reused
process would carry the previous batch's stock and clock into the next.

The rule-based rows are **derived, not rerun**: the canonical 300-order runs from
25 August were filtered to the same 150 ids. Those originals are untouched.

---

## 2. Sealed bidding

| model | done | infeas | cost mean | cost med | days | trig | cmt | thr | landed | blocked | rej | rel | pairs | min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen | **132** | 18 | 9220.65 | 5290.27 | 6.28 | 50 | 28 | 22 | 33 | 6 | 47 | 45 | 16 | 77 |
| ministral | **132** | 18 | 9376.01 | 5307.93 | **4.71** | 44 | 23 | 21 | 34 | 7 | 44 | 35 | 16 | 76 |
| rule baseline | 126 | 24 | 8489.83 | 5075.75 | 7.58 | 35 | 19 | 16 | 31 | 22 | 0 | 0 | 16 | 15 |
| phi | 126 | 24 | 9446.98 | 5276.35 | 5.07 | 61 | 36 | 25 | 32 | 4 | 114 | 102 | 15 | 76 |
| smollm3 | 126 | 24 | 9308.64 | 5238.50 | 6.21 | 80 | 57 | 23 | 33 | 8 | 159 | 147 | 17 | 77 |
| gemma | 124 | 26 | 8520.83 | 5126.78 | 6.40 | 64 | 45 | 19 | 29 | 18 | 100 | 91 | 18 | 76 |
| granite | 120 | 30 | 8115.35 | 5002.32 | 6.00 | 64 | 51 | 13 | 24 | 11 | 131 | 121 | 16 | 75 |

## 3. Reverse auction

| model | done | infeas | cost mean | cost med | days | trig | cmt | thr | landed | blocked | rej | rel | pairs | min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen | **135** | 15 | 8838.93 | 5259.33 | 4.99 | 38 | 19 | 19 | 34 | 7 | 19 | 12 | 19 | 211 |
| ministral | 134 | 16 | 9188.09 | 5181.79 | 4.19 | 45 | 19 | 26 | 35 | 10 | 30 | 23 | 14 | 209 |
| granite | 132 | 18 | 8869.81 | 5213.31 | 5.16 | 45 | 26 | 19 | 32 | 6 | 57 | 46 | 18 | 210 |
| rule baseline | 130 | 20 | 8951.32 | 5153.74 | 7.04 | 36 | 25 | 11 | 34 | 16 | 0 | 0 | 20 | 211 |
| smollm3 | 129 | 21 | 8948.35 | 5158.63 | 5.17 | 59 | 34 | 25 | 31 | 8 | 93 | 86 | 16 | 210 |
| gemma | 126 | 24 | 9099.36 | 5175.38 | 5.19 | 66 | 38 | 28 | 32 | 9 | 84 | 81 | 18 | 212 |
| phi | 123 | 27 | 8944.62 | 4973.78 | **4.04** | 65 | 24 | 41 | 31 | 8 | 110 | 107 | 16 | 213 |

---

## 4. What the numbers say

**Every SLM finishes chains faster than the rule baseline, in both modes.** Sealed
4.71 to 6.40 days against 7.58; auction 4.04 to 5.19 against 7.04. Mean chain cost
stays within about 11% of baseline everywhere, so this is not speed bought by
overspending.

**Qwen and ministral lead in both modes.** Qwen takes sealed and auction on
completions, 132 and 135. Ministral matches it sealed at 132 and is one behind in
the auction at 134, while posting the fastest chains of any run in the set, 4.19
days.

**The reservation-ack path only does real work under a model.** Rule-based agents
record zero rejections in either mode: their promises are computed from the same
state the reservation re-checks, so they cannot fail. Every SLM records rejections,
19 to 159, and in every case the overwhelming majority are
`promised_completion_no_longer_met` rather than stock shortfalls. Models promise
delivery dates the live state cannot hold; the supervisor catches it at the ack,
rolls the hold back, and tries the next chain.

**Rejection count does not predict completions, and that is the more useful
finding.** The relationship holds only at the low end:

| model | sealed rejections | sealed completed |
|---|---:|---:|
| ministral | 44 | 132 |
| qwen | 47 | 132 |
| gemma | 100 | 124 |
| phi | 114 | 126 |
| granite | 131 | 120 |
| smollm3 | 159 | 126 |

The two models that rarely over-promise land on the same completion rate. Above
roughly 100 rejections the count orders nothing: smollm3 draws the most of any
model and still finishes six ahead of granite, and phi at 114 matches the rule
baseline outright while running one retry chain eight deep. How often a model
over-promises is a property of the model. Whether the order still completes is a
property of the supervisor's fallback. Only the low end couples them.

An earlier reading of three models suggested a clean monotone relationship, and a
later one suggested the auction absorbs over-promising because granite recovered
from 120 sealed to 132. Both were contradicted by the next model in and should not
be carried forward. Gemma went 124 to 126 between modes, and phi went 126 down to
123.

**Cross-round state consistency holds for all six.** State does not move between
rounds of one order: rounds share a single `sim_now`, and reservation happens once
after the auction closes.

| model | multi-round auctions | offer moves | completion drift | tailoring | material | orders spanning >1 sim_now |
|---|---:|---:|---:|---:|---:|---:|
| rule | 432 | 1564 | 0 | 0 | 0 | **0** |
| qwen | 426 | 1545 | 34 | 30 | 4 | **0** |
| ministral | 423 | 1526 | 51 | 51 | 0 | **0** |
| granite | 426 | 1545 | 66 | 60 | 6 | **0** |
| smollm3 | 426 | 1549 | 89 | 85 | 4 | **0** |
| phi | 432 | 1561 | 113 | 112 | 1 | **0** |
| gemma | 429 | 1565 | 245 | 238 | 7 | **0** |

The offer-move column matters: it shows rounds genuinely competed rather than the
check passing vacuously. Drift is the model regenerating a different completion
time between rounds, not state changing, and the rule row confirms that by
recording exactly zero while running the same number of rounds. It concentrates in
tailoring for every model, and gemma is an outlier at roughly four times the next
worst.

---

## 5. Invariants

Every one of the fourteen run files passes, including the two 300-order originals:

- a hold is never taken twice without a release in between
- a restock never lands before the moment it was triggered, and never lands twice
- never two restocks in flight for one agent and material
- `sim_now` never moves backwards across the batch

The first two checks were corrected mid-project. The original versions counted
events, which is fine under rule-based agents where no reservation is ever refused,
but reported 32 false repeated-reservation failures and several false
two-in-flight failures on the first SLM run. Every one had an intervening
`reservation_released`, or a release carrying `restock_cancelled`. The checks now
track held/released state. Genuine violations across all fourteen files: zero.

---

## 6. Build notes, and why the documented recipe needed generalising

The recipe that worked for Qwen turned out to be largely Qwen-specific. Five
assumptions had to be made architecture-aware rather than applied blindly.

**`--no-mtp` is Qwen-only.** The converter rejects it outright for anything else:
"--mtp / --no-mtp are only supported for Qwen3.5/3.6 and Step3.5 text variants
today". Now applied by architecture, matched on prefix, because Qwen reports
`qwen3_5_text` and an exact match would have silently dropped the flag from Qwen
builds and reintroduced the broken block shape the flag exists to prevent.

**The chat-template swap is Qwen-only.** Granite, gemma, smollm3, phi and ministral
templates never mention thinking. Copying Qwen's in would have corrupted them. The
swap now runs only where the merged template actually branches on `enable_thinking`
and defaults to thinking.

**The metadata gate was Qwen-only.** It read `qwen35.block_count` by name and
asserted 32. It now reads `general.architecture` from the file and checks against
each model's own config: qwen 32, granite 40, gemma 35, smollm3 36, phi 32,
ministral 26. Gemma nests its layer count under `text_config`, which the lookup
handles.

**The merge probe was Qwen-only.** It asserted on `q_proj`, `o_proj`, `down_proj`.
Phi fuses to `qkv_proj` and `gate_up_proj`, so the probe found nothing and aborted
before doing any work. It now probes whichever of seven candidate projections exist
and requires at least two. It also prefers text-tower parameters, because on a
vision-language base it had selected a vision weight the adapter never targets,
which would have read zero delta and **false-failed a correct merge**.

**Three separate tokenizer fields are mangled by transformers 5.5
`save_pretrained`.** This is one root cause with three faces:

- `chat_template.jinja` rewritten so thinking defaults on, the known 0/6 JSON failure
- `tokenizer_class` rewritten `GPT2Tokenizer` to `TokenizersBackend`, which sends
  phi's converter down a sentencepiece path demanding a `tokenizer.model` this
  checkpoint has never had
- `added_tokens_decoder` written as `null` where ministral's base has 1000 entries,
  so `LlamaTokenizer.__init__` calls `add_tokens(None)` and raises
  `TypeError: Input must be a List[Union[str, AddedToken]]`

The general lesson is to take tokenizer artifacts from the base rather than from
the merged copy. A LoRA merge never touches the tokenizer, so the round trip is
pure risk.

### Ministral needed more than a metadata fix

Its base is `Mistral3ForConditionalGeneration`, a vision-language wrapper around a
26-layer `ministral3` text tower, and it failed three ways in sequence. Loading
(`AutoModelForCausalLM` has no mapping for `Mistral3Config`), then tensor layout
(`Can not map tensor 'model.model.embed_tokens.weight'`, because the base is in
Mistral's native format and the round trip produced triple-nested keys plus 218
vision tensors), then the tokenizer field above.

`src/experiments/dynamic_resources/flatten_text_tower.py` resolves the second:
it keeps the language tower, strips the wrapper level, drops vision and projector
tensors, and rebuilds the config from `text_config` with
`architectures: ["Ministral3ForCausalLM"]`, which is what `conversion/mistral3.py`
registers. 236 text tensors kept, 222 dropped, rope preserved as yarn with the
`mscale_all_dim` and `llama_4_scaling_beta` the converter asserts on.

**This is the least conventional artefact in the set and the write-up should say
so.** There is no native conversion of this checkpoint to compare against, so what
the results demonstrate is that the flattened model behaves sensibly and
competitively, not that it is identical to a hypothetical native build. It placed
first or second on speed in both modes and second overall on completions.

---

## 7. Verification gates

No model was used for a batch until its GGUF passed all four:

- `block_count` matches the model's own config
- no orphan block one past the end
- JSON 6/6 from real generations
- pricing spot-check against the rule figure on real holdout orders

The last gate is new and exists because of a specific failure: an earlier 300-order
run shipped a model priced two orders of magnitude low, and it only surfaced after
a 2.5 hour batch. `spot_check_prices.py` compares a handful of orders before any
batch starts and fails on anything more than 10x off. Measured medians this time:
qwen 0.999, granite 0.996, gemma 0.980, smollm3 1.000, ministral 1.007, phi 0.997.

Steady-state inference, full GPU offload: 0.60 to 0.83 s per decision across the
six.

---

## 8. Scheduling and hardware

Sealed batches ran 75 to 77 minutes, auction batches 209 to 213. The auction costs
roughly 2.8x sealed because it serves nine round windows per order instead of
three, and the window is a fixed sleep rather than a deadline agents race.

**Concurrent CPU builds alongside a GPU batch were measured, not assumed.**
Baseline with a batch alone: 21.1 GB VRAM, 93% GPU utilisation, CPU 9-12%. With a
full model merge running beside it: CPU 86-90%, GPU utilisation momentarily reading
0%. Those readings look alarming, but orders during the merge took 30.9, 31.7 and
31.5 s against a 30.5 s baseline. A batch is window-bound, inference needs only
4-9 s of each 10 s window, so contention eats slack rather than wall time. Ten of
the twelve GGUFs were built this way at no cost.

The exception is the verification generation at the end of a convert. Only about
3.4 GB of VRAM is free during a batch and the load test needs roughly 5 GB, so those
are deferred to gaps between batches.

Disk: intermediate merged directories are deleted once a GGUF verifies. 84.9 GB
freed during the night, 1231 GB free at the end.

---

## 9. Files

Twelve SLM runs plus summaries:

```
reports/bidding/sealed/dynamic_resources/slm_{qwen,granite,gemma,smollm3,ministral,phi}_run_*.jsonl
reports/bidding/reverse_auction/dynamic_resources/slm_{...}_run_*.jsonl
```

Derived rule-based comparison rows, and the untouched 300-order originals they came
from:

```
reports/bidding/sealed/dynamic_resources/rule_run_20260825_200931_first150.jsonl
reports/bidding/reverse_auction/dynamic_resources/rule_run_20260825_204133_first150.jsonl
```

Naming is fixed in `reports/bidding/NAMING.md`; the milestone log written as each
run landed is `reports/bidding/dynamic_resources_run_status.md`.

Reusable tooling, moved out of session scratch into the tree:

```
src/experiments/dynamic_resources/run_case.py            one case, fresh stack, verify, status note
src/experiments/dynamic_resources/check_run.py           invariant checks
src/experiments/dynamic_resources/spot_check_prices.py   pricing gate before a batch
src/experiments/dynamic_resources/build_ggufs.py         merge and convert queue, cpu only
src/experiments/dynamic_resources/verify_ggufs.py        deferred gpu verification
src/experiments/dynamic_resources/flatten_text_tower.py  vision-language wrapper to text tower
src/experiments/dynamic_resources/metrics.py             the tables in this report
```

`run_case.py` sets `SLM_ORDER_SAMPLE_SIZE` to the batch size on every SLM run. Its
default of 200 silently capped an earlier 300-order run at order 201 and timed out
the remaining 99.

---

## 10. Open items

**Nothing blocks the frontend work.** All twelve runs are complete and verified.

Two things worth a decision before the paper:

1. Ministral's flattened-text-tower provenance, section 6, needs an explicit
   sentence in the methods rather than a footnote.
2. No six-model comparison chart has been built. The two existing charts still draw
   from the 300-order rule-based sealed run, which remains correct for what they
   show. Whether the paper wants one figure carrying all twelve runs is a design
   question best settled once the narrative is fixed, and section 2 and 3 read
   perfectly well as tables in the meantime.

Nothing is committed or staged. `data/synthetic/v4/orders.csv` and
`orders_metadata.json` show as modified in `git status`; they were already modified
before this work and were not touched.
