"""write the scheduler job scripts for one backend and one execution profile

  PYTHONPATH=. python -m src.experiments.v5.slurm --backend rocm --profile lumi --account your_account
  PYTHONPATH=. python -m src.experiments.v5.slurm --backend cuda --profile roihu --account xxx --partition gpu

this generates, it does not launch, and nothing here runs a model. one script per model and mode,
six models x sealed and reverse auction, each a single continuous three-epoch run that saves a
loadable adapter at every epoch boundary (finetune_lora.py --save-epoch-adapters), which is what the
v5 validation-based epoch selection reads.

surviving the wall time is the point of the resume block. LUMI's queues cap a job well below a full
run (dev-g 3 h, small-g 3 days, standard-g 2 days), so every script:
  saves a recovery checkpoint every 250 optimizer steps, the protocol's interval
  starts with --resume, which picks the newest checkpoint that has a complete resume state
  writes a DONE marker only when training actually finished
  resubmits itself with sbatch --dependency=afterany when the marker is absent, up to a bounded
  number of continuations, so a run crosses as many wall times as it needs without a human in the
  loop and without ever restarting from zero

backend and profile are required and never detected. a training launch that guesses its hardware is
exactly what this project's earlier reports warn against; only backend_smoke.py --probe-only detects
anything, and it is a diagnostic

the roihu profile is a stub and is exercised here on purpose: generating a cuda + roihu script that
calls the identical finetune_lora.py command proves the experiment logic does not move when the site
does. its partitions and module lines are placeholders, marked as such in the generated file
"""

import argparse
import stat
import sys
from pathlib import Path

from config import PROJECT_ROOT
from src.experiments.v5 import execution, model_registry

SCRIPT_ROOT = PROJECT_ROOT / "slurm"
TRAIN_MODES = ("sealed", "reverse_auction")

# the supervisor arm now covers every architecture. the directory schema always accommodated it
# (models/training/supervisor/<model>/<run>); qwen simply ran first
SUPERVISOR_MODE = "supervisor"
DEFAULT_EPOCHS = 3
MAX_CONTINUATIONS = 12

# 0 means train on the whole train split: 541,545 sealed / 601,161 reverse-auction rows. the
# 40,000-row subsample runs are complete and archived separately; full corpus is the production
# configuration now. this lives here, beside DEFAULT_EPOCHS, because the freeze gate reads it as
# the expected value the same way, so changing it keeps the gate and the generator in step
DEFAULT_SUBSAMPLE = 0

# finetune_lora.py is a single process with no distributed logic of any kind, so a job uses one
# GCD however many are allocated. job 22194724 requested 8 and the trainer reported
# "Data Parallel GPUs = 1", leaving seven idle for three hours. the profile's gpus_per_node still
# describes the hardware; this is what a job asks for
DEFAULT_GPUS = 1

# how long before the wall clock Slurm sends the warning signal. it only has to cover one sbatch,
# so it is generous rather than tuned
SIGNAL_LEAD_SECONDS = 120

# ── where one training attempt writes ───────────────────────────────────────
#
#   run      models/training/<mode>/<model>/<run>/ with the trainer's resume state in
#            checkpoints/ and the evaluable epoch-1..3/ and final/ adapters beside it. this is
#            what v5 generates
#   legacy   models/v5/<mode>/<model>-3epoch/ with checkpoint-<step> loose in the run root, which
#            is what the first generated set used. kept so that set can be re-rendered byte for
#            byte rather than only described
#
# the run segment is a real directory, and the same string goes on all three of an attempt's
# validation evaluations: epoch_selection.select() groups results by an exact match on the run
# field and normalizes nothing, so two spellings of one run are two runs to it

LAYOUT_CHOICES = ("run", "legacy")
DEFAULT_LAYOUT = "run"
DEFAULT_RUN_TAG = "r01"
CHECKPOINT_DIRNAME = "checkpoints"


def run_name(model, mode, tag=DEFAULT_RUN_TAG):
    """the globally unique label for one training attempt, e.g. qwen-sealed-v5-r01

    model and mode are both in it because selection compares run strings and does not compare
    model identity: two architectures sharing a label would silently mix results. the tag is what
    a second attempt at the same cell changes"""
    return f"{model}-{mode}-v5-{tag}"


def run_paths(root_expr, model, mode, layout=DEFAULT_LAYOUT, tag=DEFAULT_RUN_TAG):
    """(run label, run root, trainer checkpoint dir) for one attempt under one layout

    the legacy layout has no separate checkpoint directory, and returns None for it so the caller
    emits no --checkpoint-dir at all and the old command line is reproduced exactly"""
    if layout not in LAYOUT_CHOICES:
        raise ValueError(f"unknown layout {layout!r}, expected one of {', '.join(LAYOUT_CHOICES)}")
    if layout == "legacy":
        return None, f"{root_expr}/models/v5/{mode}/{model}-3epoch", None
    run = run_name(model, mode, tag)
    output = f"{root_expr}/models/training/{mode}/{model}/{run}"
    return run, output, f"{output}/{CHECKPOINT_DIRNAME}"


def default_partition(profile):
    """the partition with the most wall time, not the first one listed

    a three-epoch run does not fit LUMI's 3-hour dev-g queue, so defaulting to it would send every
    real job straight into a continuation loop"""
    if not profile.partitions:
        return ""
    return max(profile.partitions.items(), key=lambda kv: kv[1].get("max_hours", 0))[0]


def sbatch_header(args, context, job_name):
    partition = args.partition or default_partition(context.profile)
    hours = args.hours or (context.profile.partitions.get(partition, {}).get("max_hours") or 3)
    account = args.account or f"${{{context.profile.account_env}}}"
    return [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={hours:02d}:00:00",
        "#SBATCH --nodes=1",
        f"#SBATCH --gpus-per-node={args.gpus}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        "#SBATCH --ntasks-per-node=1",
        # B: sends the signal to the batch shell itself rather than the job steps, which is what
        # lets the trap below run. without it Slurm SIGTERMs the shell at the wall clock and the
        # resubmission never happens, which is exactly how job 22194724 was lost
        f"#SBATCH --signal=B:USR1@{SIGNAL_LEAD_SECONDS}",
        # slurm does not expand shell variables in #SBATCH lines, so these stay relative to the
        # submission directory; submit_all.sh overrides them with the profile's log directory
        "#SBATCH --output=%x-%j.out",
        "#SBATCH --error=%x-%j.err",
    ]


def training_command(model, mode, context, output_dir, epochs, root_expr,
                     checkpoint_dir=None, base_layout=model_registry.BASE_LAYOUT_DEFAULT):
    flags = model_registry.train_flags(model, mode, root_expr=root_expr, base_layout=base_layout)
    command = ["python", "src/experiments/finetune_lora.py", *flags,
               "--num-train-epochs", str(epochs),
               "--save-epoch-adapters",
               "--resume",
               "--optim", context.backend.default_optim,
               "--output-dir", str(output_dir)]
    if checkpoint_dir:
        command += ["--checkpoint-dir", str(checkpoint_dir)]
    return command


def container_block(profile, command_line):
    """the training command, run inside the profile's container when it has one

    a heredoc rather than `bash -c '...'` on purpose: the command already carries single-quoted
    arguments (smollm3's --template-kwargs) and ANSI-C $'...' markers, which a quoted -c string
    would have to escape. inside a quoted heredoc every line is literal, and the shell that reads
    it expands ${COB_PROFILE_ROOT} and friends from the environment singularity passed through.

    the block is written to a file rather than piped straight into the shell so the training run
    can be backgrounded and waited on: a synchronous heredoc blocks this shell, and a blocked
    shell cannot service the wall-time signal. the file lives in the job's own scratch and goes
    away with it"""
    marker = "COB_CONTAINER"
    return ["# the training command, staged to a file so it can run in the background",
            f'cat > "$COB_INNER" <<\'{marker}\'',
            *profile.container_prologue,
            command_line,
            marker]


def runner(profile):
    """how the staged inner script is executed: inside the container, or plainly"""
    return f'{profile.container_exec or "bash -s"} < "$COB_INNER"' 


def quote(part):
    """one shell-safe argument

    three cases, and the first is not hypothetical: qwen's and smollm3's loss-masking markers end in
    a real newline ("<|im_start|>user\\n"), which a plain quoted string would break the script line
    on, so those become ANSI-C $'...' strings. arguments carrying ${VAR} must stay double-quoted so
    the shell still expands them, and everything else is single-quoted, which keeps | < > from being
    read as shell syntax"""
    if "\n" in part or "\t" in part:
        escaped = (part.replace("\\", "\\\\").replace("'", "\\'")
                   .replace("\n", "\\n").replace("\t", "\\t"))
        return f"$'{escaped}'"
    if "${" in part:
        return f'"{part}"'
    if not part or any(c in part for c in ' <>|{}"$&;()*?[]#~`\\\''):
        return "'" + part.replace("'", "'\\''") + "'"
    return part


def script_text(args, context, model, mode):
    job_name = f"cob-{model}-{mode}"
    root = "${COB_PROFILE_ROOT}"
    log_dir = f"{root}/logs"
    layout = getattr(args, "layout", DEFAULT_LAYOUT)
    run, output_dir, checkpoint_dir = run_paths(root, model, mode, layout,
                                                getattr(args, "run_tag", DEFAULT_RUN_TAG))
    done_marker = f"{output_dir}/TRAINING_DONE"
    # a site that has materialized its base tree pins the layout on its profile, so neither the
    # operator nor the freeze gate has to remember a flag for it
    base_layout = (getattr(args, "base_layout", None) or context.profile.base_layout
                   or model_registry.BASE_LAYOUT_DEFAULT)
    command = training_command(model, mode, context, output_dir, args.epochs, root,
                               checkpoint_dir=checkpoint_dir, base_layout=base_layout)
    # the supervisor training set is 35,552 rows, already smaller than the bidder subsample, so it
    # trains on all of it. --subsample applies to the two bidder modes only
    if args.subsample and mode in TRAIN_MODES:
        command += ["--subsample", str(args.subsample)]
    made = [log_dir, output_dir] + ([checkpoint_dir] if checkpoint_dir else [])

    environment = context.job_environment(root)
    lines = sbatch_header(args, context, job_name)
    lines += [
        "",
        f"# generated by src/experiments/v5/slurm.py for backend {context.backend.name}, "
        f"profile {context.profile.name}",
        f"# {context.profile.notes}",
        f"# backend: {context.backend.notes}",
    ]
    # the legacy layout renders exactly the lines it always rendered, so a re-render of the first
    # generated set is a byte comparison rather than a description of one
    if run:
        lines += [
            f"# layout {layout}, run {run}",
            f"# all three validation evaluations of this attempt must pass --run {run}:",
            "# selection groups results by an exact match on that string and normalizes nothing",
        ]
    lines += [
        "",
        "set -euo pipefail",
        "",
        f'if [ -z "${{COB_PROFILE_ROOT:-}}" ]; then',
        f'  echo "COB_PROFILE_ROOT is not set: it must point at this profile\'s root '
        f'(caches, checkpoints, logs)" >&2',
        "  exit 2",
        "fi",
        "mkdir -p " + " ".join(f'"{path}"' for path in made),
        "",
        "# site setup",
    ]
    lines += [f"{line}" for line in context.profile.setup]
    if context.profile.runtime_setup:
        lines += ["", "# settings only the compute node can resolve, evaluated here and not at "
                      "generation time"]
        lines += [f"{line}" for line in context.profile.runtime_setup]
    lines += [
        "",
        "# every path comes from the execution profile, none is hardcoded in the python",
    ]
    for key, value in sorted(environment.items()):
        lines.append(f'export {key}="{value}"')
    lines += [
        'export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"',
        'export COB_DATASET_VERSION=v5',
        "",
        f'if [ -f "{done_marker}" ]; then',
        '  echo "training already finished, nothing to do"',
        "  exit 0",
        "fi",
        "",
        "# the run itself: one continuous 3-epoch run, an adapter saved at every epoch boundary,",
        "# resuming from the newest complete checkpoint if this is a continuation.",
        "#",
        "# recorded caveat, carried here and on every result record (results_store.py",
        "# EPOCH_COMPARABILITY_NOTE): the linear LR schedule decays across the whole 3-epoch",
        "# horizon, so this run's epoch-1 and epoch-2 adapters are NOT directly comparable to v4's",
        "# separately trained epoch-1 and epoch-2 families. compare v5 epochs with each other, and",
        "# v4 against v5 only at the final selected checkpoint.",
        "# errexit is lifted around it on purpose: a non-zero exit is the wall-time case this",
        "# script exists to handle, and under set -e the shell would die before reading it",
        "# one resubmission per job, whoever asks for it. the wall-time trap and the non-zero",
        "# exit path below can both reach this; the guard is what stops them queueing two",
        "# successors for the same job",
        "COB_RESUBMITTED=0",
        "cob_resubmit() {",
        '  if [ "$COB_RESUBMITTED" -ne 0 ]; then return 0; fi',
        "  COB_RESUBMITTED=1",
        '  attempt="${COB_CONTINUATION:-0}"',
        f'  if [ "$attempt" -ge "{MAX_CONTINUATIONS}" ]; then',
        f'    echo "reached {MAX_CONTINUATIONS} continuations without finishing, stopping" >&2',
        "    return 1",
        "  fi",
        "  export COB_CONTINUATION=$((attempt + 1))",
        '  echo "resubmitting continuation $COB_CONTINUATION"',
        '  sbatch --dependency=afterany:${SLURM_JOB_ID} --export=ALL "$0"',
        "}",
        "",
        "# Slurm signals the batch shell shortly before the wall clock. resubmit from here, then",
        "# stop the trainer and leave cleanly: the successor picks up from the newest checkpoint",
        "cob_on_walltime() {",
        '  echo "wall time approaching, resubmitting before the kill"',
        "  cob_resubmit || true",
        '  kill "${train_pid:-0}" 2>/dev/null || true',
        "  exit 0",
        "}",
        "trap cob_on_walltime USR1",
        "",
        'COB_INNER="${COB_JOB_CACHE:-${TMPDIR:-/tmp}}/cob-inner-${SLURM_JOB_ID:-$$}.sh"',
    ]
    lines += container_block(context.profile, " ".join(quote(part) for part in command))
    lines += [
        "",
        "# backgrounded on purpose: `wait` is interruptible, a synchronous call is not, and this",
        "# shell has to stay free to run the trap above while training is still going",
        "set +e",
        f"{runner(context.profile)} &",
        "train_pid=$!",
        'wait "$train_pid"; status=$?',
        "# wait returns >128 when a handled signal interrupts it, so go back to waiting",
        'while [ "$status" -gt 128 ] && kill -0 "$train_pid" 2>/dev/null; do',
        '  wait "$train_pid"; status=$?',
        "done",
        "set -e",
        "",
        "if [ $status -eq 0 ]; then",
        f'  touch "{done_marker}"',
        '  echo "training finished"',
        "  exit 0",
        "fi",
        "",
        "# died for some other reason: continue from the last checkpoint",
        "cob_resubmit || exit 1",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_scripts(args, context):
    out_dir = Path(args.out_dir or (SCRIPT_ROOT / f"{context.backend.name}-{context.profile.name}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    pairs = [(model, mode) for model in sorted(model_registry.MODELS) for mode in TRAIN_MODES]
    pairs += [(model, SUPERVISOR_MODE) for model in sorted(model_registry.MODELS)]
    for model, mode in pairs:
        path = out_dir / f"train_{model}_{mode}.sh"
        path.write_text(script_text(args, context, model, mode), encoding="utf-8", newline="\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        written.append(path)
    submit = out_dir / "submit_all.sh"
    submit.write_text(
        "#!/bin/bash\n"
        "# submit every training job for this backend and profile\n"
        "# the log paths are given here rather than in the scripts because slurm does not expand\n"
        "# shell variables inside #SBATCH lines\n"
        "set -euo pipefail\n"
        'if [ -z "${COB_PROFILE_ROOT:-}" ]; then echo "COB_PROFILE_ROOT is not set" >&2; exit 2; fi\n'
        'export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"\n'
        "\n"
        "# the freeze gate runs first and aborts the whole batch rather than one job at a time: if\n"
        "# this checkout, the transferred corpus or the base checkpoints disagree with the frozen\n"
        "# record, nothing should be queued at all\n"
        'echo "freeze gate: checking this machine against reports/v5/freeze_check_local.json"\n'
        "if ! python -m src.experiments.v5.freeze_check --remote; then\n"
        '  echo "freeze gate FAILED, no jobs submitted" >&2\n'
        "  exit 1\n"
        "fi\n"
        "\n"
        'mkdir -p "${COB_PROFILE_ROOT}/logs"\n'
        + "".join(f'sbatch --output="${{COB_PROFILE_ROOT}}/logs/%x-%j.out" '
                  f'--error="${{COB_PROFILE_ROOT}}/logs/%x-%j.err" "$(dirname "$0")/{p.name}"\n'
                  for p in written if p.name.startswith("train_")),
        encoding="utf-8", newline="\n")
    submit.chmod(submit.stat().st_mode | stat.S_IEXEC)
    written.append(submit)
    return out_dir, written


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    execution.add_arguments(ap)
    ap.add_argument("--account", default="", help="slurm account; defaults to the profile's account variable")
    ap.add_argument("--partition", default="", help="slurm partition; defaults to the profile's first")
    ap.add_argument("--hours", type=int, default=0, help="wall time; defaults to the partition's maximum")
    ap.add_argument("--gpus", type=int, default=0,
                    help=f"gpus per node, default {DEFAULT_GPUS}: the training process is single "
                         f"device, so asking for the whole node just idles the rest")
    ap.add_argument("--cpus", type=int, default=7, help="cpus per task")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE,
                    help=f"stratified subsample of the train split for the bidder modes, default "
                         f"{DEFAULT_SUBSAMPLE}; 0 trains on the whole split. the supervisor job "
                         f"ignores this and uses its full set")
    ap.add_argument("--out-dir", default="", help="where the scripts land; defaults to slurm/<backend>-<profile>/")
    ap.add_argument("--layout", choices=LAYOUT_CHOICES, default=DEFAULT_LAYOUT,
                    help=f"training output hierarchy, default {DEFAULT_LAYOUT}: models/training/"
                         f"<mode>/<model>/<run>/ with trainer state in {CHECKPOINT_DIRNAME}/. "
                         f"legacy re-renders the first generated set, models/v5/<mode>/<model>-3epoch/")
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG,
                    help=f"the attempt segment of the run label, default {DEFAULT_RUN_TAG}; a second "
                         f"attempt at the same model and mode uses a new one so the two never share "
                         f"a run directory or a selection record")
    ap.add_argument("--base-layout", choices=model_registry.BASE_LAYOUT_CHOICES, default=None,
                    help="where --base-model points; defaults to the profile's own base_layout, "
                         "else registry. registry is the hub id or local copy pinned by --revision, "
                         "profile is <root>/models/base/<model>, a directory that carries no revision")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    context = execution.require(args.backend, args.profile, "slurm script generation",
                                root=args.profile_root or None)
    if not args.gpus:
        args.gpus = DEFAULT_GPUS
    if context.profile.scheduler != "slurm":
        raise execution.ExecutionError(
            f"profile {context.profile.name!r} has no scheduler: these scripts are for a slurm site "
            f"({', '.join(p for p, s in execution.PROFILES.items() if s.scheduler == 'slurm')})")
    out_dir, written = write_scripts(args, context)
    print(f"backend {context.backend.name}, profile {context.profile.name}, "
          f"optimizer {context.backend.default_optim}")
    print(f"layout {args.layout}, base layout "
          f"{args.base_layout or context.profile.base_layout or model_registry.BASE_LAYOUT_DEFAULT}"
          + (f", run tag {args.run_tag}" if args.layout == "run" else ""))
    print(f"{len(written)} script(s) written to {out_dir}")
    for path in written:
        print(f"  {path.name}")
    print("\nnothing was submitted; review the scripts, set COB_PROFILE_ROOT and the account, then "
          "run submit_all.sh on the cluster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
