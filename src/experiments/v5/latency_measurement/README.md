# Latency measurement

The tooling behind `reports/v5/evaluation/latency_records/` and `reports/v5/evaluation/tables/latency_breakdown_*`.
It times every row of the test split for all 36 arms (base and the selected fine-tuned epoch of the 18 mode and model
cells), one request at a time, and splits each decision into time to first token (TTFT) and time per output token
(TPOT). Load path, generation settings (greedy, at most 200 new tokens), stop rule, shard plan and timed region are
those of `src/experiments/v5/inference_profile.py`; the only addition is a stopping-criterion hook that records the
first token and never stops generation. Records hold counts and timings only, never generated text.

| file | purpose |
|---|---|
| `measure.py` | splits the shards into parts (`--make-parts`), measures one shard per call (`--claim`), shows progress (`--status`), estimates a layout (`--list`) |
| `run.sbatch` | Slurm launcher: one task per node, eight measuring processes per node, one per GCD, each claiming shards under a lock |
| `submit_all.sh` | submits a self-checking pilot, then the parts, each part after the pilot succeeded |
| `summarize.py` | summary and checks against the original profiling records; `--export` writes the published records, `--publish` rebuilds the published tables |
| `adapter_merge_diagnostic/` | the adapter-merge timing control (below) |

Usage. Copy this folder to a working location: records, locks and logs are written next to the scripts. Set
`COB_REPO` (repository root), `LFT_DIR` (the working copy), `COB_PROFILE_ROOT` (model, cache and Python environment
root), `COB_EB_USER_PREFIX` (EasyBuild prefix of the container module), `COB_TORCH_MODULE` (the LUMI PyTorch 2.7.1,
ROCm 6.2.4, Python 3.12 singularity module) and `SBATCH_ACCOUNT`. Another cluster needs its own module lines.

    PYTHONDONTWRITEBYTECODE=1 COB_DATASET_VERSION=v5 python3 "$LFT_DIR/measure.py" --list --parts 8 --nodes 8
    bash "$LFT_DIR/submit_all.sh"
    watch -n 60 'PYTHONDONTWRITEBYTECODE=1 python3 "$LFT_DIR/measure.py" --status'
    PYTHONDONTWRITEBYTECODE=1 python3 "$LFT_DIR/summarize.py"
    PYTHONDONTWRITEBYTECODE=1 python3 "$LFT_DIR/summarize.py" --export reports/v5/evaluation/latency_records

To rebuild the four latency tables from the included CSV records, omitting only the original-profile comparison:

    python -m src.experiments.v5.latency_measurement.summarize --publish reports/v5/evaluation/latency_records --skip-original-profile --out build/latency-tables

Use a separate output directory. A folder containing an older original-profile comparison is refused in
this mode, so it cannot be mistaken for a newly generated comparison. The aggregation rules are unchanged.

The historical table-rebuild command below also compares the measurements with the original profiling records in
`reports/v5/results/latency/`. The original sealed-bid and reverse-auction `.jsonl` files are not included;
without them the command stops with a missing-file error. With those files restored, run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 python3 src/experiments/v5/latency_measurement/summarize.py \
        --publish reports/v5/evaluation/latency_records --out reports/v5/evaluation/tables

To check the published latency tables using only the included files and the Python standard library:

    python3 src/experiments/v5/check_latency_tables.py

## Adapter-merge diagnostic

A timing control only. For each of the 18 cells, at the epoch selected on validation, it times the base model, the
model with the LoRA adapter attached and a copy with the adapter merged into the weights in memory, on the same 300
validation rows per arm, interleaved in blocks. It shows whether the attached adapter, rather than fine-tuning
itself, causes the slower decoding. The merged model is not output-identical to the attached one, so the diagnostic
compares timing, not outputs, and it is separate from the whole-test-split tables. Its results are
`reports/v5/evaluation/tables/adapter_merge_diagnostic_paper_table.{md,tex}` and
`adapter_merge_diagnostic_summary.csv`. `run_diagnostic.py` keeps the generated validation text in its own records
to compare merged and attached outputs; those records are not published.

    PYTHONDONTWRITEBYTECODE=1 COB_DATASET_VERSION=v5 python3 adapter_merge_diagnostic/run_diagnostic.py --mode sealed --model gemma
    PYTHONDONTWRITEBYTECODE=1 python3 adapter_merge_diagnostic/summarize.py --dir runs
