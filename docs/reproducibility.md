# Setup and reproducibility

Run Python commands from the repository root. Python 3.11 is used for the audit's local checks; the recorded paper-rendering environment used Python 3.12. A GPU is needed for model training and inference, not for inspecting results or rebuilding reports.

## What can be reproduced

| Task | Included inputs | Additional requirements |
| --- | --- | --- |
| Check published latency tables | Tables and CSV summaries | Python standard library only |
| Rebuild main figures and tables | Saved v5 results, metadata, token summaries and latency CSVs | Rendering environment; use `--skip-original-profile` |
| Rebuild original-profile comparison | Summary tables are included | Original sealed-bid and reverse-auction profiling JSONL files, not distributed |
| Run focused rule/schema tests | Source, synthetic data and fixtures | Runtime environment |
| Run complete regression suite | Test source is included | Fine-tuning corpus and some local tooling/assets; `--fast` does not remove these requirements |
| Rerun training or inference | Code, model revisions, protocols and selection records | Corpus, model weights/adapters, suitable GPU environment and access to the original execution setup |

A report rebuild reads saved decisions and timings. It does not rerun inference, retrain a model or remeasure latency. Recreating those measurements is a separate experiment.

## Environments

### Runtime and local tests

```sh
python -m venv .venv
```

Activate the environment:

- Linux/macOS: `source .venv/bin/activate`
- PowerShell: `.venv\Scripts\Activate.ps1`

Then install:

```sh
python -m pip install -r requirements.txt
```

Activation is optional if you call the environment's interpreter directly. For example, PowerShell users whose execution policy blocks activation can use `& .venv\Scripts\python.exe -m pip install -r requirements.txt`.

The runtime file pins the package versions used for the audit's local checks. It is not a claim that these were the paper's training or rendering versions.

### Report rendering

Create and activate a separate environment named `.venv-report`, using the same steps above with that name:

```sh
python -m venv .venv-report
```

After activating it:

```sh
python -m pip install -r requirements-evaluation.txt
```

This file pins matplotlib and the recorded rendering dependencies. The added `six` pin fixes a transitive dependency for the rebuild environment. Use Python 3.12 when comparing figure pixels with the original rendering environment; matching one package version alone is not a guarantee of identical files across platforms.

### GGUF demo

In the runtime environment:

```sh
python -m pip install -r requirements-slm.txt
```

This adds `llama-cpp-python`; it is not a training requirements file. Its build and GPU support depend on the local compiler and accelerator setup. The full demo also needs the GGUF files listed in the [API README](../src/api/README.md). The rule-based demo does not need them.

### GPU training and evaluation

The recorded latency environment was PyTorch 2.7.1 with ROCm 6.2.4 and Python 3.12 on AMD MI250X hardware. Package records are retained in `requirements-evaluation.txt` comments, `src/experiments/v5/execution.py` (`LUMI_VENV_PINS`) and the measurement metadata.

Do not install those comments as a generic GPU environment. CUDA and ROCm use different platform builds; cluster-specific Torch and bitsandbytes builds were part of the original setup. The existing code accepts explicit execution profiles and roots through `COB_PROFILE_ROOT` / `--profile-root`. The corpus and models are not downloaded by the runtime setup.

`config.py` retains the historical v4 default. The v5 generators require an explicit v5 context, usually `COB_DATASET_VERSION=v5`. Do not change the global default to reproduce a single experiment, and do not rerun data generators over the saved inputs as a setup step.

## Verify the included records

This command reads published tables without modifying them:

```sh
python src/experiments/v5/check_latency_tables.py
```

It checks Markdown/LaTeX consistency, CSV rounding, token totals, latency decomposition and warm-up statistics. It does not verify model decisions or establish new experimental results.

## Rebuild into a separate directory

In the rendering environment:

```sh
python -m src.experiments.v5.render_report --skip-original-profile --out build/evaluation --csv-out build/evaluation/tables
python src/experiments/v5/check_latency_tables.py --tables build/evaluation/tables
```

Use a new directory for each run, or choose another `--out` and matching `--csv-out`. Existing CSVs are deliberately not overwritten. The new flag leaves all metric calculations, latency aggregation, epoch selection and chart logic unchanged. It omits only `latency_breakdown_vs_original_profile.{md,csv}`, whose inputs are not distributed.

To rebuild just the latency tables from the included per-request records:

```sh
python -m src.experiments.v5.latency_measurement.summarize --publish reports/v5/evaluation/latency_records --skip-original-profile --out build/latency-tables
```

The full report command also creates token summaries needed by the table checker; the latency-only command does not create those summaries.

The historical commands without `--skip-original-profile` still require original profiling JSONL records and preserve the previous behavior. Training-status provenance can differ when local checkpoints are absent; it is not inferred from the published metrics. See the [evaluation README](../reports/v5/evaluation/README.md#regenerate) for the full file index and disclosures.

The rebuild creates figures and tables, not a new copy of the archived captions or provenance sidecars. Those remain in the published evaluation directory.

## Tests

Focused tests that use included inputs:

```sh
python -m tests.test_auction_rounds
python -m tests.test_dataset_schemas
python -m unittest tests.test_report_rebuild
```

The new reporting tests check that skipping original-profile input produces the same metric tables as the comparison-enabled path, that the default still requires the original records, and that an old comparison cannot be mistaken for a newly generated one.

With the omitted fine-tuning data restored:

```sh
python -m tests.run_all --fast
```

Without `--fast`, the runner also performs the exhaustive v4 census. Some tests use site-specific tools or recorded corpus hashes; failures and unavailable checks from this audit are listed in the [audit report](anonymity-audit.md). They must not be treated as passing or fixed by changing the saved evidence.

## Demo and containers

The [API README](../src/api/README.md) lists demo commands. A rule-only session uses:

```sh
python -m src.api up --configs rule
```

This starts local brokers and the API; the broker tooling may download its pinned NATS binary. Stop the foreground session with Ctrl+C. Use a second terminal for status, submissions or reset. The demo is development tooling, without an authentication or production deployment claim.

Docker Compose is an optional earlier multi-process agent setup, not the v5 evaluation environment:

```sh
docker compose config --quiet
docker compose up --build
```

It requires a running Docker daemon. It reads the checked-in synthetic data and writes runtime state through its existing mounts. Review the configuration before a new experiment; do not use a container restart as a way to reproduce paper measurements.

## Review-copy provenance

Names, personal paths, hostnames, scheduler identifiers, workflow timestamps and original Git references are withheld. Some original byte-level provenance hashes are withheld because anonymization changes the files they described. Per-row values, model revisions, scientific source records, selection rules and measured timings are retained. The packaged SHA256SUMS.txt checks the distributed files, not the undistributed originals.

The DataCo raw customer CSV is omitted because it is not ingested by the synthesis pipeline and contains identifying fields. Its public source citation remains. Full historical corpus and site checks still need inputs not distributed here.

Historical freeze and hash-gate checks that depend on withheld provenance ledgers are not supported by the anonymous package. The reporting and focused test commands above remain the supported review checks.
