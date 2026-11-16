"""phase 6: end-to-end training-time estimates for the six architectures and both modes

  PYTHONPATH=. python -m src.experiments.v5.training_estimate

the primary basis is tonight's measurement: qwen3.5-4b trained for a handful of real steps on
real v5 train rows of each mode (src/experiments/v5/smoke_sample.py), at the protocol's batch
8 x 2 and the sequence window the v5 token lengths require. its median seconds per step are read
from the smoke logs in reports/v5/phase6/

the other five models are scaled from qwen by their own measured v4 step times, mode by mode:
  s_per_step(model, mode) = qwen_v5_measured(mode) x v4_history(model, mode) / v4_history(qwen, mode)
the v4 history is read from disk, not typed in: the median interval between consecutive
checkpoints' trainer_state.json files of each model's committed run, divided by the steps between
them, which is the real wall time the run spent per step. sealed uses the epoch-2 production runs
(protocol v2, 8 x 2, window 1024) and reverse auction the epoch-1 v4 runs (the worded set that
data/finetune/v5/reverse_auction mirrors). qwen's v4 console log is quoted as a cross-check of the
checkpoint arithmetic

  steps per epoch = ceil(train rows / 16)
  hours           = steps x s_per_step / 3600, for one epoch and for the epoch-2 family's two

train rows come from data/finetune/v5/manifest.json. validation and test rows are never trained on

writes reports/v5/phase6_training_estimate.json
"""

import json
import math
import re
import statistics
import sys
from pathlib import Path

from config import PROJECT_ROOT

MANIFEST = PROJECT_ROOT / "data" / "finetune" / "v5" / "manifest.json"
SMOKE_DIR = PROJECT_ROOT / "reports" / "v5" / "phase6"
REPORT_PATH = PROJECT_ROOT / "reports" / "v5" / "phase6_training_estimate.json"
EFFECTIVE_BATCH = 16
MODELS = ("qwen", "granite", "phi", "ministral", "gemma", "smollm3")
V4_RUNS = {
    "sealed": {
        "qwen": "models/epoch-2/production/qwen3.5-4b-bidding-lora-v2",
        "granite": "models/epoch-2/production/granite-4.1-3b-bidding-lora-v2",
        "phi": "models/epoch-2/production/phi4mini-bidding-lora-v2",
        "ministral": "models/epoch-2/production/ministral3-bidding-lora-v2",
        "gemma": "models/epoch-2/production/gemma4-e2b-bidding-lora-v2",
        "smollm3": "models/epoch-2/production/smollm3-bidding-lora-v2",
    },
    "reverse_auction": {
        "qwen": "models/reverse_auction/epoch-1/v4/qwen3.5-4b-reverse-auction-lora",
        "granite": "models/reverse_auction/epoch-1/v4/granite-4.1-3b-reverse-auction-lora",
        "phi": "models/reverse_auction/epoch-1/v4/phi-reverse-auction-lora",
        "ministral": "models/reverse_auction/epoch-1/v4/ministral-reverse-auction-lora",
        "gemma": "models/reverse_auction/epoch-1/v4/gemma-reverse-auction-lora",
        "smollm3": "models/reverse_auction/epoch-1/v4/smollm3-reverse-auction-lora",
    },
}
QWEN_V4_CONSOLE_LOG = "reports/epoch-1/production/training-logs/qwen_v2_train_console.log"
STEP_LINE = re.compile(r"per-step wall time: median ([0-9.]+)s")


def checkpoint_seconds_per_step(run):
    points = []
    for state in (PROJECT_ROOT / run).glob("checkpoint-*/trainer_state.json"):
        points.append((json.loads(state.read_text(encoding="utf-8"))["global_step"], state.stat().st_mtime))
    points.sort()
    intervals = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(points, points[1:]) if b[0] > a[0] and b[1] > a[1]]
    return {"run": run, "checkpoints": len(points), "median_s_per_step": round(statistics.median(intervals), 3),
            "intervals": len(intervals)}


def smoke_seconds_per_step(mode):
    path = SMOKE_DIR / f"smoke_qwen_{mode}.log"
    match = STEP_LINE.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        raise SystemExit(f"{path} has no per-step wall time line; run the smoke run first")
    return float(match[-1]), str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = {"sealed": manifest["main_corpus_rows"]["sealed:train"],
            "reverse_auction": manifest["main_corpus_rows"]["reverse_auction:train"],
            "supervisor": manifest["main_corpus_rows"]["supervisor:train"]}
    history = {mode: {m: checkpoint_seconds_per_step(run) for m, run in runs.items()} for mode, runs in V4_RUNS.items()}
    qwen_log = STEP_LINE.findall((PROJECT_ROOT / QWEN_V4_CONSOLE_LOG).read_text(encoding="utf-8", errors="replace"))

    estimates, measured = {}, {}
    for mode in ("sealed", "reverse_auction"):
        qwen_v5, source = smoke_seconds_per_step(mode)
        measured[mode] = {"qwen_s_per_step": qwen_v5, "log": source}
        steps = math.ceil(rows[mode] / EFFECTIVE_BATCH)
        for model in MODELS:
            ratio = history[mode][model]["median_s_per_step"] / history[mode]["qwen"]["median_s_per_step"]
            seconds = qwen_v5 * ratio
            estimates[f"{model}:{mode}"] = {
                "train_rows": rows[mode], "steps_per_epoch": steps,
                "v4_ratio_to_qwen": round(ratio, 4), "s_per_step": round(seconds, 3),
                "hours_one_epoch": round(steps * seconds / 3600, 1), "hours_two_epochs": round(2 * steps * seconds / 3600, 1),
                "arithmetic": (f"ceil({rows[mode]} / {EFFECTIVE_BATCH}) = {steps} steps; {qwen_v5} s x "
                               f"{history[mode][model]['median_s_per_step']} / {history[mode]['qwen']['median_s_per_step']} = "
                               f"{seconds:.3f} s/step; {steps} x {seconds:.3f} / 3600 = {steps * seconds / 3600:.1f} h"),
            }
    supervisor = None
    if (SMOKE_DIR / "smoke_qwen_supervisor.log").exists():
        seconds, source = smoke_seconds_per_step("supervisor")
        steps = math.ceil(rows["supervisor"] / EFFECTIVE_BATCH)
        supervisor = {"model": "qwen", "train_rows": rows["supervisor"], "s_per_step": seconds, "log": source,
                      "steps_per_epoch": steps, "hours_one_epoch": round(steps * seconds / 3600, 1),
                      "hours_two_epochs": round(2 * steps * seconds / 3600, 1)}

    totals = {"one_epoch_hours_all_six_both_modes": round(sum(e["hours_one_epoch"] for e in estimates.values()), 1),
              "two_epoch_hours_all_six_both_modes": round(sum(e["hours_two_epochs"] for e in estimates.values()), 1)}
    result = {"basis": __doc__.strip(), "effective_batch": EFFECTIVE_BATCH, "train_rows": rows,
              "measured_tonight": measured, "v4_history_checkpoint_intervals": history,
              "qwen_v4_console_log_cross_check": {"log": QWEN_V4_CONSOLE_LOG, "median_s_per_step": float(qwen_log[-1]) if qwen_log else None},
              "estimates": estimates, "supervisor": supervisor, "totals": totals}
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    for key, e in estimates.items():
        print(f"{key:26s} {e['s_per_step']:7.3f} s/step  {e['steps_per_epoch']:7d} steps  "
              f"{e['hours_one_epoch']:7.1f} h/epoch  {e['hours_two_epochs']:7.1f} h/2 epochs")
    print(f"supervisor {supervisor}")
    print(f"totals {totals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
