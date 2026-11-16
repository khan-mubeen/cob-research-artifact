#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --mem=0
#
# full test split latency with first-token time, in the pattern of the original profile's launcher:
# one srun task per node, eight measuring processes per node, one per gcd (ROCR_VISIBLE_DEVICES=slot), each
# with its own compile caches; environment copied from pack.sh. --mem=0 (the whole node) as the official
# campaign asked: dev-g otherwise gives 64 GiB, too little for eight model loads. every gcd slot runs
# `measure.py --claim` again and again: each call takes the next free shard of its part (longest first)
# under a lock in logs/locks/, measures it in that fresh process and exits, until nothing is left.
#
# set COB_REPO (repository root), LFT_DIR (the working copy of this folder), COB_PROFILE_ROOT (caches, models,
# python environment), COB_EB_USER_PREFIX (easybuild prefix) and SBATCH_ACCOUNT, then submit with submit_all.sh
# or by hand:
#   pilot     sbatch -p dev-g -N 1 --time=01:00:00 --job-name=lft-pilot \
#                 --output="$LFT_DIR/logs/%x-%j.out" "$LFT_DIR/run.sbatch" --pilot
#   one part  sbatch -p standard-g -N 8 --time=10:00:00 --job-name=lft-part-3-of-8 \
#                 --output="$LFT_DIR/logs/%x-%j.out" "$LFT_DIR/run.sbatch" --part 3/8
#
# the pilot job ends with the pilot checks and fails when they fail. a part rerun resumes: locks of jobs no
# longer in squeue are cleared first, finished shards are skipped, a started shard continues.
set -uo pipefail
REPO=${COB_REPO:?set COB_REPO to the repository root}
LFT=${LFT_DIR:?set LFT_DIR to the folder that holds measure.py}
export COB_PROFILE_ROOT=${COB_PROFILE_ROOT:?set COB_PROFILE_ROOT}

if ! type module >/dev/null 2>&1; then
  source "${LMOD_PKG:-/opt/cray/pe/lmod/lmod}/init/bash"
fi
module purge
export EBU_USER_PREFIX="${COB_EB_USER_PREFIX:?set COB_EB_USER_PREFIX}"
module load LUMI/25.03 partition/container
module use "${EBU_USER_PREFIX}/modules/container"
module load "${COB_TORCH_MODULE:?set COB_TORCH_MODULE to the pytorch container module}"
if [ -z "${SIF:-}" ]; then echo "SIF is empty" >&2; exit 2; fi
export GPU_ARCHS=gfx90a PYTORCH_ROCM_ARCH=gfx90a
export HF_HOME="${COB_PROFILE_ROOT}/hf-cache" HF_HUB_CACHE="${COB_PROFILE_ROOT}/hf-cache/hub"
export HF_XET_CACHE="${COB_PROFILE_ROOT}/hf-cache/xet" HF_HUB_OFFLINE=1
export MIOPEN_USER_DB_PATH="${COB_PROFILE_ROOT}/.miopen" MIOPEN_CUSTOM_CACHE_DIR="${COB_PROFILE_ROOT}/.miopen"
export COB_DATASET_VERSION=v5 PYTHONPATH="$REPO" COB_NO_GIT=1 PYTHONDONTWRITEBYTECODE=1
export LFT_DIR="$LFT" LFT_ARGS="$*"

if [ -z "${LFT_NODE_STEP:-}" ]; then
  echo "latency full test ($*): ${SLURM_NNODES} node(s): ${SLURM_JOB_NODELIST}"
  # stale locks: an empty answer from squeue clears nothing, and locks newer than this snapshot are kept
  export LFT_SNAPSHOT="$(date +%s)"
  export LFT_LIVE="$(squeue -h -u "$USER" -o %i 2>/dev/null | tr '\n' ' ')"
  singularity exec "$SIF" bash -c 'source "${COB_PROFILE_ROOT}/unsloth-env/bin/activate" && cd "$LFT_DIR" &&
    python measure.py --clear-stale --live "$LFT_LIVE" $LFT_ARGS' || { echo "refused, nothing started"; exit 2; }
  export LFT_NODE_STEP=1
  # --kill-on-bad-exit=0: one failed node must not end the others
  srun --kill-on-bad-exit=0 --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=56 \
       bash "$LFT/run.sbatch" "$@"
  rc=0
  if [ "${1:-}" = "--pilot" ]; then
    singularity exec "$SIF" bash -c 'source "${COB_PROFILE_ROOT}/unsloth-env/bin/activate" && cd "$LFT_DIR" &&
      python summarize.py --pilot --check'
    rc=$?
  fi
  echo "latency full test: all nodes finished $(date +%T), exit $rc"
  exit $rc
fi

# the repeat check keeps its slot logs under repeat_check/, away from the main run's logs
LOGROOT="$LFT/logs"; case " $* " in *" --check "*) LOGROOT="$LFT/repeat_check/logs" ;; esac
export LFT_LOGDIR="$LOGROOT/${SLURM_JOB_ID:-local}/node${SLURM_PROCID:-0}"
mkdir -p "$LFT_LOGDIR"
# node-local compile caches, removed on exit, as in pack.sh
export COB_JOB_CACHE="${TMPDIR:-/tmp}/lft-${SLURM_JOB_ID:-$$}-${SLURM_PROCID:-0}"
mkdir -p "${COB_JOB_CACHE}/bin"
echo "node ${SLURMD_NODENAME:-?}: TMPDIR ${TMPDIR:-/tmp} on $(df -h --output=fstype,size "${TMPDIR:-/tmp}" | tail -1)"
trap 'rm -rf "${COB_JOB_CACHE}"' EXIT
export COB_GPUS="${SLURM_GPUS_ON_NODE:-8}"

singularity exec "$SIF" bash -c '
  set -uo pipefail
  export HIP_CLANG_PATH="${ROCM_PATH:-/opt/rocm}/llvm/bin"
  for t in c++ g++; do ln -sf "${HIP_CLANG_PATH}/clang++" "${COB_JOB_CACHE}/bin/$t"; done
  for t in cc gcc; do ln -sf "${HIP_CLANG_PATH}/clang" "${COB_JOB_CACHE}/bin/$t"; done
  export PATH="${COB_JOB_CACHE}/bin:${PATH}"
  source "${COB_PROFILE_ROOT}/unsloth-env/bin/activate"
  cd "$LFT_DIR"
  G=${COB_GPUS%%(*}; G=${G##*:}; [ -z "$G" ] && G=8
  echo "node ${SLURMD_NODENAME:-?}: $G gcd slot(s) claiming with: $LFT_ARGS"
  worker() {  # one gcd slot: claim, measure and exit, again until nothing is left or three failures in a row
    local s=$1 n=0 fails=0 rc c="${COB_JOB_CACHE}/slot$1"
    mkdir -p "$c"/{unsloth,inductor,triton,aiter}
    export ROCR_VISIBLE_DEVICES=$s HIP_VISIBLE_DEVICES=0
    export UNSLOTH_COMPILE_LOCATION="$c/unsloth" TORCHINDUCTOR_CACHE_DIR="$c/inductor"
    export TRITON_CACHE_DIR="$c/triton" AITER_JIT_DIR="$c/aiter"
    while [ "$fails" -lt 3 ]; do
      n=$((n+1)); t0=$(date +%s)
      python measure.py --claim $LFT_ARGS > "${LFT_LOGDIR}/gcd${s}-${n}.log" 2>&1
      rc=$?
      echo "=== gcd $s call $n end $(date +%T) rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
      [ "$rc" -eq 3 ] && break
      if [ "$rc" -eq 0 ]; then fails=0; else fails=$((fails+1)); fi
    done
  }
  for ((s=0; s<G; s++)); do worker "$s" & done
  wait
  echo "node ${SLURMD_NODENAME:-?}: all slots done $(date +%T)"
'
