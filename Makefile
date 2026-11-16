PYTHON ?= python
PYTHONPATH := .
export PYTHONPATH

# ── data synthesis ────────────────────────────────────────────────────────────

data: product material manufacturers logistics orders
	@echo "Data synthesis completed"

product:
	$(PYTHON) src/data_synthesis/synthesize_product.py

material:
	$(PYTHON) src/data_synthesis/synthesize_material_providers.py

manufacturers:
	$(PYTHON) src/data_synthesis/synthesize_manufacturers.py

logistics:
	$(PYTHON) src/data_synthesis/synthesize_logistics.py

orders:
	$(PYTHON) src/data_synthesis/synthesize_orders.py

# ── docker ────────────────────────────────────────────────────────────────────

up:
	docker compose up --build

down:
	docker compose down

# ── live bidding ──────────────────────────────────────────────────────────────
# every process reads BIDDING_MODE, unset means sealed. run each in its own shell
# against a running broker, then drive orders with bidding_run
#   BIDDING_MODE=reverse_auction make supervisor

supervisor:
#   BIDDING_MODE=reverse_auction DYNAMIC_RESOURCES=1 make supervisor
	$(PYTHON) src/supervisor/supervisor.py

agent_material:
#   ACTIVE_AGENT_IDS=M01,M02 make agent_material
	$(PYTHON) src/agents/material/agent.py

agent_manufacturer:
	$(PYTHON) src/agents/manufacturer/agent.py

agent_logistics:
	$(PYTHON) src/agents/logistics/agent.py

bidding_run:
#   make bidding_run ARGS="--holdout --orders 150 --quiet"
	$(PYTHON) src/experiments/run_bidding_experiment.py $(ARGS)

# ── live demo ─────────────────────────────────────────────────────────────────
# five configurations at once, rule-based plus the base and fine-tuned gguf of
# each live architecture, over their own brokers. see src/api/README.md

demo:
#   make demo
#   make demo ARGS="--configs rule"          one configuration, no gpu
#   BIDDING_MODE=reverse_auction make demo   start straight into the auction
	$(PYTHON) -m src.api up $(ARGS)

demo_down:
#   make demo_down
	$(PYTHON) -m src.api down

demo_reset:
#   make demo_reset
	$(PYTHON) -m src.api reset

# switch bidding mode on a running session; model loading determines restart time.
# refused while an order is in flight
#   make demo_mode MODE=reverse_auction
#   make demo_mode MODE=sealed
demo_mode:
	$(PYTHON) -m src.api mode $(MODE)

# ── evaluation entry points ───────────────────────────────────────────

slm:
	@echo "For the GGUF demo, install requirements-slm.txt and see src/api/README.md."
	@echo "For v5 training and inference, see docs/reproducibility.md."

evaluate:
	@echo "No inference was started. Published results are in reports/v5/evaluation/."
	@echo "See docs/reproducibility.md before running new evaluations."

# rebuild into ignored build/, leaving the published scientific records untouched
report:
	$(PYTHON) -m src.experiments.v5.render_report --skip-original-profile --out build/evaluation --csv-out build/evaluation/tables

check_report:
	$(PYTHON) src/experiments/v5/check_latency_tables.py

test_report:
	$(PYTHON) -m unittest tests.test_report_rebuild

# ── visualization ─────────────────────────────────────────────────────────────

diagnose:
	$(PYTHON) src/visualization/data_diagnostics.py

bid_check:
	$(PYTHON) src/visualization/bid_competition_diagnostics.py

slm_check:
	$(PYTHON) src/visualization/slm_competition_diagnostics.py

figures:
	$(PYTHON) src/visualization/plot_generated_data.py

# ── ui ────────────────────────────────────────────────────────────────────────
# frontend in ui/frontend: saved-result dashboard, replay and optional live API client.
# the published dashboard data is already included; npm ci then npm run build inside ui/frontend

ui_data:
	$(PYTHON) src/experiments/export_ui_data.py

# ── tests ─────────────────────────────────────────────────────────────────────
# one command for the whole suite: the v4 golden regression harness plus the v5
# foundation layers. test_fast skips the exhaustive 50,000-order census

test:
	$(PYTHON) tests/run_all.py

test_fast:
	$(PYTHON) tests/run_all.py --fast

# rebuild data/raw/v5 from src/dataset/v5_sources.py. the only step that opens a
# network connection; re-running without --refresh changes nothing
v5_sources:
#   make v5_sources ARGS="--refresh"
	$(PYTHON) src/dataset/acquire_v5_sources.py $(ARGS)

# ── utility ───────────────────────────────────────────────────────────────────

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [p.unlink() for p in Path('.').rglob('*.pyc')]; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
	@echo "Python cache cleaned"

.PHONY: data product material manufacturers logistics orders up down slm evaluate diagnose bid_check slm_check figures clean \
	supervisor agent_material agent_manufacturer agent_logistics bidding_run ui_data \
	demo demo_down demo_reset demo_mode test test_fast v5_sources report check_report test_report
