# Chain-of-Bid: anonymous review materials

Chain-of-Bid studies multi-stage bidding for garment order fulfilment. An order passes through material supply, manufacturing or tailoring, and logistics. Agents bid under cost, time, quality and reliability constraints; a supervisor checks whether their combined bids meet the order's budget and deadline.

The v5 evaluation compares Gemma, Granite, Ministral, Phi, Qwen and SmolLM3 before and after fine-tuning on three tasks: sealed bidding, reverse auctions and supervisor chain selection. Model revisions are recorded in the [v5 registry](src/experiments/v5/model_registry.py). This review copy retains research code, measurements and public source attribution. Contributor identities and private working records are omitted.

## Start here

- **Paper results:** [v5 evaluation](reports/v5/evaluation/README.md), including metric definitions, epoch selection, provenance and the Qwen stop-token correction.
- **Reproduction:** [setup and reproducibility guide](docs/reproducibility.md), with separate instructions for checking saved results, rebuilding reports and rerunning experiments.
- **Methods:** [documentation index](docs/README.md), including data synthesis and agent design.
- **Interactive demo:** [API backend](src/api/README.md) and [frontend](ui/frontend/README.md). This uses earlier GGUF fine-tunes and is separate from the v5 evaluation.

## What is included

Saved evaluation results, figures, tables, provenance records, synthetic datasets and the compressed per-request latency records are included. The fine-tuning corpus (`data/finetune/`), model weights and adapters (`models/`), and the large original sealed-bid and reverse-auction profiling JSONL files are not distributed here.

You can inspect the published results and rebuild the main figures and tables without a GPU or the omitted files. The original-profile comparison and a new training or inference run need additional inputs; see the [reproduction limits](docs/reproducibility.md#what-can-be-reproduced).

## Check the published tables

From the repository root, with Python installed:

```sh
python src/experiments/v5/check_latency_tables.py
```

This checks consistency between the published latency tables and their CSV summaries. It uses only the Python standard library and leaves the results unchanged.

## Python setup

Use Python 3.11 or newer; the recorded figure-rendering environment used Python 3.12. Create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in PowerShell, then install the runtime dependencies:

```sh
python -m pip install -r requirements.txt
```

These packages cover data tools, rule-based agents, the API and local tests. GGUF inference is optional and uses `requirements-slm.txt`. GPU training needs a separate, hardware-specific environment; it is not installed by either file.

For the paper's rendering versions, create a **separate environment** and install `requirements-evaluation.txt` instead. Do not combine the runtime and rendering dependency sets. See the [setup guide](docs/reproducibility.md#environments) for shell-specific instructions.

## Rebuild the main figures and tables

In the rendering environment, run from the repository root:

```sh
python -m src.experiments.v5.render_report --skip-original-profile --out build/evaluation --csv-out build/evaluation/tables
python src/experiments/v5/check_latency_tables.py --tables build/evaluation/tables
```

The rebuild writes to the ignored `build/` directory, leaving the published results intact. It uses the existing metric and latency calculations. The flag skips only the comparison against the omitted original profiling JSONL files; it does not skip the latency tables or charts.

Use a new output directory for each rebuild: `--csv-out` refuses to replace existing CSV files. Without the flag, the historical workflow still requires the original profiling files. The [evaluation README](reports/v5/evaluation/README.md#regenerate) lists the detailed inputs and outputs.

## Bidding tasks

- **Sealed bid:** each stage agent responds once with a bid or no bid, checking its own stage against the order deadline.
- **Reverse auction:** round 1 uses sealed-bid eligibility and opening prices. Later bids must undercut the best offer by the platform decrement. The auction ends after a round with no improvement or at the round cap (default 3). Each agent keeps its best standing offer; stage winners are ordered by price, then arrival, then agent ID.
- **Supervisor:** the learned task selects a chain from candidate stage bids. Its reference, `decide_chain_selection` in `src/common/bidding_formulas.py`, rejects chains over budget or deadline and ranks feasible chains by order priority. The [evaluation README](reports/v5/evaluation/README.md#epoch-selection-and-test-reads) explains the objective and tie-breaking rules. In the running agent system, stage winners remain provisional until the combined cost and time pass rule-based validation.

## Tests

With runtime dependencies installed, these focused checks use files included in Git:

```sh
python -m tests.test_auction_rounds
python -m tests.test_dataset_schemas
python -m unittest tests.test_report_rebuild
```

For a development checkout with the fine-tuning corpus restored:

```sh
python -m tests.run_all --fast
```

Omit `--fast` for the full v4 census. The flag only reduces that census; it does not remove corpus-dependent tests. Make users can also use `make test`, `make test_fast`, `make check_report` and `make test_report`. Known validation limits are recorded in the [repository audit](docs/anonymity-audit.md).

## Project layout

```text
chain-of-bid/
|-- data/
|   |-- raw/                 Source datasets and source metadata
|   |-- processed/           Derived inputs and calibration evidence
|   |-- synthetic/           Versioned orders and agent profiles (v1-v5)
|   `-- demo/                Example orders for the interactive demo
|-- src/
|   |-- agents/              Material, manufacturing and logistics agents
|   |-- supervisor/          Stage bidding, auctions and chain validation
|   |-- common/              Shared formulas, messaging and model helpers
|   |-- dataset/             Source acquisition and versioned data context
|   |-- data_synthesis/      Synthetic data generators
|   |-- experiments/         Training, evaluation and reporting; v5/ is the paper pipeline
|   |-- api/                 Interactive demo backend
|   `-- visualization/       Earlier diagnostic and plotting tools
|-- reports/                 Experimental results and scientific records
|   `-- v5/evaluation/       Paper figures, tables, timings and provenance
|-- docs/                    Methodology, reproduction guide and anonymity notes
|-- tests/                   Regression tests and v4 golden baselines
|-- slurm/                   Cluster launch scripts
|-- ui/frontend/             Results dashboard, replay and live demo client
|-- requirements*.txt        Separate runtime, rendering and GGUF dependency sets
|-- Makefile                 Convenience commands
`-- LICENSE                  MIT licence for project code
```

The omitted `data/finetune/` and `models/` directories may also exist in a local development checkout. Generated runtime logs, environments, model caches and build outputs are ignored. Private meeting notes and planning documents are omitted. Scientific records retain their relative paths; identifying provenance fields are withheld.

## Licence and source data

Project code is released under the [MIT licence](LICENSE). The bundled Source Sans 3 fonts have a separate [SIL Open Font License 1.1](src/experiments/v5/fonts/LICENSE.md). Models and third-party datasets retain their own terms.

The identifying raw DataCo CSV is omitted. It was a qualitative reference, not an input to the synthesis or report pipeline. Its public source citation remains in the methodology.

## Package for review

Run the following from this directory; a commit is not required:

    python package_review.py

This creates submission/chain-of-bid-review.zip, excluding Git history, private files, environments and generated outputs. It contains a SHA-256 manifest and fixed file timestamps.

Submit that ZIP rather than the working directory or a clone. The local Git directory retains identifying history and has not been rewritten. Git archive of HEAD writes a commit identifier into the ZIP comment; use the packaging command above instead. See [anonymity notes](docs/anonymity-audit.md).
