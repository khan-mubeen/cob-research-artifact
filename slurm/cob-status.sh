#!/bin/bash
# on-demand status sweep for the twelve v5 training runs. run it by hand every few hours;
# it is deliberately not a daemon and keeps no state of its own.
#
#   COB_PROFILE_ROOT=. ./slurm/cob-status.sh [sacct-starttime]
#
# VERDICT column:
#   DONE        TRAINING_DONE exists, nothing more to do
#   RUNNING     a job for this run is in the queue right now
#   CONTINUING  last job hit the wall clock and a successor is already queued (normal)
#   STALLED     last job ended and nothing is queued: needs a resubmit
#   ATTENTION   a real failure (FAILED / OOM / NODE_FAIL): needs a human, do NOT just resubmit
#   NOTSTARTED  no job has ever run for this run
set -uo pipefail
ROOT="${COB_PROFILE_ROOT:?set COB_PROFILE_ROOT first}"
SINCE="${1:-now-7days}"
TAG="v5-r01"

printf '%-10s %-16s %-11s %-10s %8s  %s\n' MODEL MODE LASTJOB STATE CKPT VERDICT
printf '%.0s-' {1..78}; echo

alert=0
# the twelve bidder runs plus the single qwen supervisor run
runs=""
for mode in sealed reverse_auction; do
  for model in gemma granite ministral phi qwen smollm3; do runs="$runs ${model}:${mode}"; done
done
for model in gemma granite ministral phi qwen smollm3; do runs="$runs ${model}:supervisor"; done

for entry in $runs; do
    model=${entry%%:*}; mode=${entry##*:}
    name="cob-${model}-${mode}"
    dir="${ROOT}/models/training/${mode}/${model}/${model}-${mode}-${TAG}"

    # most recent job of this name, and whether anything of that name is queued now
    line=$(sacct --name="$name" --starttime="$SINCE" -X -n -P -o JobID,State 2>/dev/null | tail -1)
    jid=${line%%|*}; state=${line##*|}; state=${state%% *}
    queued=$(squeue -h -n "$name" -o %i 2>/dev/null | wc -l)

    ck=$(ls -d "${dir}/checkpoints/checkpoint-"* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    [ -z "$ck" ] && ck="-"

    if   [ -f "${dir}/TRAINING_DONE" ];                 then verdict=DONE
    elif [ -z "$line" ];                                then verdict=NOTSTARTED; jid="-"; state="-"
    elif [ "$queued" -gt 0 ];                           then verdict=RUNNING
    else case "$state" in
           TIMEOUT|PREEMPTED)   verdict=STALLED ;;   # queued==0 already checked
           FAILED|OUT_OF_ME*|NODE_FAIL|BOOT_FAIL) verdict=ATTENTION ;;
           CANCELLED*)          verdict=STALLED ;;
           COMPLETED)           verdict=STALLED ;;   # exited 0 but no DONE marker
           *)                   verdict=STALLED ;;
         esac
    fi
    [ "$verdict" = ATTENTION ] && alert=$((alert+1))
    printf '%-10s %-16s %-11s %-10s %8s  %s\n' "$model" "$mode" "$jid" "$state" "$ck" "$verdict"
done

echo
if [ "$alert" -gt 0 ]; then
  echo "$alert run(s) marked ATTENTION: a genuine failure, not a wall-time continuation."
  echo "Investigate the job log before resubmitting; do not loop on it."
fi
echo "STALLED runs need one manual: sbatch slurm/rocm-lumi/train_<model>_<mode>.sh"
