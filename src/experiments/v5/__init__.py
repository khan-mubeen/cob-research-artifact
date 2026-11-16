"""v5 fine-tuning data generation, parallel to the v4 generators and never through them

the v4 generators in src/experiments/generate_*.py are read as the source of truth for the
sampling and labeling methodology and are not edited. every module here reuses their
price-agnostic pieces (completion_json, the auction replay, the chain shortlist) and differs
only where v5 has to: USD prices from the dispatched v5 formulas, rfqs carrying dated inputs,
the full 50,000-order pool, and one global stratified train/validation/test split by order id

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.split
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_finetune_data
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_reverse_auction_data
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_supervisor_data
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_edge_examples
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.generate_floor_boundary
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.manifest
"""
