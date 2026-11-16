# Naming for the six-model, 150-order dynamic-resources runs

Model slugs, used everywhere: `qwen`, `granite`, `phi`, `ministral`, `gemma`, `smollm3`.

## Run files, twelve of them

```
reports/bidding/sealed/dynamic_resources/slm_<slug>_run_<YYYYMMDD_HHMMSS>.jsonl
reports/bidding/sealed/dynamic_resources/slm_<slug>_run_<YYYYMMDD_HHMMSS>_summary.json
reports/bidding/reverse_auction/dynamic_resources/slm_<slug>_run_<YYYYMMDD_HHMMSS>.jsonl
reports/bidding/reverse_auction/dynamic_resources/slm_<slug>_run_<YYYYMMDD_HHMMSS>_summary.json
```

The mode is the directory, as before. `slm_` keeps the existing agent-kind prefix
and the slug sits between it and `run_`, so a plain sort groups by agent kind then
model. Example: `slm_qwen_run_20260827_023000.jsonl`.

## Rule-based comparison rows, derived not rerun

Cases 1 and 2 stay canonical at 300 orders and are not regenerated. The 150-order
comparison rows are filtered copies, named after the run they derive from:

```
reports/bidding/sealed/dynamic_resources/rule_run_20260825_200931_first150.jsonl
reports/bidding/reverse_auction/dynamic_resources/rule_run_20260825_204133_first150.jsonl
```

with matching `_summary.json`. The `_first150` suffix marks them derived; the
timestamp preserves provenance.

## GGUFs and build reports, per model per mode

```
models/epoch-2/deployment/<slug>-sealed/<slug>-sealed-nomtp-matched-Q8_0.gguf
reports/epoch-2/deployment/<slug>-sealed/convert_<slug>-sealed-nomtp-matched.json

models/reverse_auction/epoch-1/v4/deployment/<slug>-auction/<slug>-auction-nomtp-matched-Q8_0.gguf
reports/reverse_auction/v4_deployment/<slug>-auction/convert_<slug>-auction-nomtp-matched.json
```

Sealed GGUFs come from the `*-bidding-lora-v2` adapters under
`models/epoch-2/production`, auction GGUFs from the `*-reverse-auction-lora`
adapters under `models/reverse_auction/epoch-1/v4`. Each build directory also holds
`merge_verify.json` and the convert log.

Qwen's auction GGUF was built before this convention and sat directly in
`models/reverse_auction/epoch-1/v4/deployment/`. It has been moved into
`qwen-auction/` so all twelve match. The merged intermediate directory for each
model is deleted once its GGUF verifies, keeping only the GGUF and the reports.

## The subset

`reports/bidding/holdout_150_subset.json` records the 150 ids, their
`sha256 d566da1fcf6276ebed8a874ca14cf76c2a2f0e5bb626c883c980cec97ba095c4`, and the
parent 300-set hash they were taken from. The subset is the first 150 of the
verified holdout ordered by `order_created` then `order_id`, running
`ORD-15478` (2026-08-20 11:00) to `ORD-25134` (2026-11-13 12:00).
