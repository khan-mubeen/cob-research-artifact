#!/bin/bash
# submit every training job for this backend and profile
# the log paths are given here rather than in the scripts because slurm does not expand
# shell variables inside #SBATCH lines
set -euo pipefail
if [ -z "${COB_PROFILE_ROOT:-}" ]; then echo "COB_PROFILE_ROOT is not set" >&2; exit 2; fi
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# the freeze gate runs first and aborts the whole batch rather than one job at a time: if
# this checkout, the transferred corpus or the base checkpoints disagree with the frozen
# record, nothing should be queued at all
echo "freeze gate: checking this machine against reports/v5/freeze_check_local.json"
if ! python -m src.experiments.v5.freeze_check --remote; then
  echo "freeze gate FAILED, no jobs submitted" >&2
  exit 1
fi

mkdir -p "${COB_PROFILE_ROOT}/logs"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_gemma_sealed.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_gemma_reverse_auction.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_granite_sealed.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_granite_reverse_auction.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_ministral_sealed.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_ministral_reverse_auction.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_phi_sealed.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_phi_reverse_auction.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_qwen_sealed.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_qwen_reverse_auction.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_smollm3_sealed.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_smollm3_reverse_auction.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_gemma_supervisor.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_granite_supervisor.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_ministral_supervisor.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_phi_supervisor.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_qwen_supervisor.sh"
sbatch --output="${COB_PROFILE_ROOT}/logs/%x-%j.out" --error="${COB_PROFILE_ROOT}/logs/%x-%j.err" "$(dirname "$0")/train_smollm3_supervisor.sh"
