# Latency records

The per-request records behind the latency charts, the radar latency axis and `tables/latency_breakdown_*`. One
gzip CSV per arm, `<mode>_<model>_<arm>.csv.gz`: `arm` is `base` (the base model) or `selected` (the fine-tuned
model at the epoch selected on validation, LoRA adapter attached and not merged). Counts and timings only; no
generated text and no labels.

| column | meaning |
|---|---|
| `row_index` | position of the row in the test split; the same index as in `reports/v5/results/latency/` |
| `row_id` | the row's order id, as in `reports/v5/results/latency/` |
| `input_tokens` | rendered prompt tokens |
| `output_tokens` | generated tokens, counted from the generated ids, the stop token included |
| `total_ms` | latency of the decision in ms: the `generate` call between two device synchronisations |
| `ttft_ms` | time to first token in ms, from the start of `generate` to the first generated token |
| `tpot_ms` | time per output token in ms, (total - TTFT) / (output tokens - 1); empty when only one token was generated |
| `warmup` | 1 for a warm-up row, else 0 |

Rows per arm: sealed bid 67,597, reverse auction 75,397, supervisor 4,435 (the whole test split; 1,769,148
rows over the 36 files). Each request ran alone (batch size 1, greedy decoding, at most 200 new tokens) on one
GCD of an AMD MI250X, eight measuring processes per node (the last 680 rows of reverse-auction Granite base
shard 0 ran with one process on its node). The split was cut into the same shards as the
original profile; a warm-up row is one of the first three rows of a shard or the first row after a resume.
Latency statistics leave warm-up rows out; token statistics use every row.

The files are written by `src/experiments/v5/latency_measurement/summarize.py --export` from the measurement's
own records. To rebuild the main latency tables from the included files, run from the repository root:

    python -m src.experiments.v5.latency_measurement.summarize --publish reports/v5/evaluation/latency_records --skip-original-profile --out build/latency-tables

Without `--skip-original-profile`, the command also compares against the original profiling JSONL files,
whose sealed-bid and reverse-auction records are not distributed. The flag skips only that comparison.
For the complete figure and table rebuild, see [setup and reproducibility](../../../../docs/reproducibility.md).
