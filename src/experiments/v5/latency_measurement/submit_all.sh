#!/bin/bash
# submits the pilot (dev-g, one node), then the full run as K parts (standard-g, N exclusive nodes each); every
# part waits for the pilot with --dependency=afterok, so it starts only when the pilot checks pass.
#
#   COB_REPO=... LFT_DIR=... COB_PROFILE_ROOT=... COB_EB_USER_PREFIX=... SBATCH_ACCOUNT=... bash "$LFT_DIR/submit_all.sh"
#
# defaults: LFT_PARTS=8 LFT_NODES=8 LFT_TIME=11:00:00 (about twice the estimated 5 h per part). job ids go to
# logs/job_ids.txt; the script refuses to run while that file exists, so a second copy cannot start.
set -uo pipefail
REPO=${COB_REPO:?set COB_REPO to the repository root}
LFT=${LFT_DIR:?set LFT_DIR to the folder that holds measure.py}
K=${LFT_PARTS:-8} N=${LFT_NODES:-8} T=${LFT_TIME:-11:00:00}
IDS="$LFT/logs/job_ids.txt"
CANCEL="scancel \$(awk '{print \$NF}' $IDS)"

# created exclusively before anything is submitted
if ! ( set -o noclobber; : > "$IDS" ) 2>/dev/null; then
  echo "refusing: $IDS exists, so this run was already submitted."
  echo "check with: squeue -u \$USER; cancel with: $CANCEL; then remove $IDS before submitting again."
  exit 1
fi
cd "$REPO"
PYTHONDONTWRITEBYTECODE=1 COB_DATASET_VERSION=v5 python3.11 "$LFT/measure.py" --make-parts --parts "$K" \
  || { rm -f "$IDS"; echo "nothing submitted"; exit 1; }

sub() { sbatch --parsable --output="$LFT/logs/%x-%j.out" "$@" | cut -d';' -f1; }
pilot=$(sub -p dev-g -N 1 --time=01:00:00 --job-name=lft-pilot "$LFT/run.sbatch" --pilot)
if [ -z "$pilot" ]; then rm -f "$IDS"; echo "pilot submission failed, nothing submitted"; exit 1; fi
echo "pilot $pilot" >> "$IDS"
for ((i=0; i<K; i++)); do
  id=$(sub -p standard-g -N "$N" --time="$T" --job-name="lft-part-$i-of-$K" --dependency=afterok:"$pilot" \
       "$LFT/run.sbatch" --part "$i/$K")
  if [ -z "$id" ]; then echo "submission of part $i/$K failed; the parts listed below were submitted"; break; fi
  echo "part $i/$K $id" >> "$IDS"
done

cat "$IDS"
cat <<EOF
job ids written to $IDS
status:   watch -n 60 'PYTHONDONTWRITEBYTECODE=1 python3.11 $LFT/measure.py --status'
if the pilot fails, the parts stay pending on an unsatisfied dependency (DependencyNeverSatisfied).
cancel everything:   $CANCEL
then remove $IDS before submitting again.
EOF
