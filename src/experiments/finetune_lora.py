"""
lora fine-tuning of Qwen3.5-4B on the rule-based bidding labels via unsloth.

trains one adapter on the combined material + tailoring + logistics training
sets of one named dataset, against the original safetensors checkpoint, not the
gguf. the holdout files are never read here.

--dataset picks which set: sealed is the round-1 data from
generate_finetune_data.py and is the default, reverse_auction is the round 2/3
data from generate_reverse_auction_data.py. the supervisor set goes through
--train-jsonl and is unaffected by either.

--smoke validates the local pipeline on a small sample, then prints per-step
timing and an estimated duration for the full training set.

usage:
    python src/experiments/finetune_lora.py --smoke   # pipeline check + estimate
    python src/experiments/finetune_lora.py           # full training; requires explicit approval
    python src/experiments/finetune_lora.py --dataset reverse_auction --subsample 600
    COB_DATASET_VERSION=v5 python src/experiments/finetune_lora.py --dataset v5_sealed --subsample 15000

the v5_* datasets are the train files of data/finetune/v5, handled by
src/experiments/v5/training_data.py: every record must come from the train split, the window must
fit the mode's measured length, and --subsample stratifies on the split's own scenario_type x
priority. the v4 datasets and their --subsample are unchanged
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from unsloth import FastLanguageModel  # must import before transformers/trl

from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from config import DATA_VERSION, PROJECT_ROOT, REVERSE_AUCTION_VERSION
from src.experiments.v5 import training_data as v5_training_data


BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_EPOCHS = 1
RANDOM_SEED = 42

SMOKE_EXAMPLES = 48
SMOKE_STEPS = 8

DATA_DIR = PROJECT_ROOT / "data" / "finetune" / DATA_VERSION
ORDERS_CSV = PROJECT_ROOT / "data" / "synthetic" / DATA_VERSION / "orders.csv"
DEFAULT_EPOCH = 1
DEFAULT_RESULT_CATEGORY = "production"
OUTPUT_DIR = (PROJECT_ROOT / "models" / f"epoch-{DEFAULT_EPOCH}" /
              DEFAULT_RESULT_CATEGORY / "qwen3.5-4b-bidding-lora-r1")

STAGES = ("material", "tailoring", "logistics")

# named training sets, each the per-stage jsonl files that make one set. sealed
# is the original round-1 layout and stays the default, so an existing command
# line reads exactly the same files in the same order as before. a new dataset
# is one entry here rather than another flag
TRAIN_SETS = {
    "sealed": lambda: [DATA_DIR / f"{stage}_train.jsonl" for stage in STAGES],
    "reverse_auction": lambda: [DATA_DIR / "reverse_auction" / REVERSE_AUCTION_VERSION
                                / stage / "train.jsonl" for stage in STAGES],
}


def train_set_paths(dataset):
    if v5_training_data.is_v5_dataset(dataset):
        return v5_training_data.train_set_paths(dataset)
    try:
        return TRAIN_SETS[dataset]()
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}, expected one of {sorted(TRAIN_SETS)}"
        ) from None


def load_train_records(dataset="sealed"):
    """combined per-stage records for one named training set

    the seeded shuffle runs after concatenation, so a dataset and a seed always
    produce the same order"""
    records = []
    for path in train_set_paths(dataset):
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    random.Random(RANDOM_SEED).shuffle(records)
    return records


def stratified_subsample(records, n):
    """subsample n examples matching the full orders.csv scenario mix

    a plain records[:n] slice overweights random_realistic at the example
    level because those orders have more eligible agents per order, so quotas
    here follow the population scenario shares instead. records arrive
    seeded-shuffled, quota filling preserves that order, any shortfall in a
    scenario is topped up from the remaining shuffle"""
    scenario_of = {}
    with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scenario_of[row["order_id"]] = row["scenario_type"]

    unknown = [r["order_id"] for r in records if r["order_id"] not in scenario_of]
    if unknown:
        raise ValueError(
            f"stratified_subsample got {len(unknown)} record(s) with order ids not in "
            f"{ORDERS_CSV.name}, e.g. {unknown[:3]}; synthetic or foreign records must "
            "be mixed in after subsampling, not before"
        )

    population = Counter(scenario_of.values())
    total = sum(population.values())
    quota = {s: round(c / total * n) for s, c in population.items()}
    quota[max(quota, key=quota.get)] += n - sum(quota.values())  # rounding drift

    taken, leftover = [], []
    for r in records:
        s = scenario_of[r["order_id"]]
        if quota[s] > 0:
            taken.append(r)
            quota[s] -= 1
        else:
            leftover.append(r)

    # a scenario can run out of examples before its quota fills (high_quality
    # breaks first, around n=28,600 with the current data), top up from the
    # rest of the shuffle but never silently
    shortfall = {s: q for s, q in quota.items() if q > 0}
    if shortfall:
        print("WARNING: scenario quotas not fillable from available records, "
              "topping up from other scenarios, shares below deviate from target:")
        for s, q in sorted(shortfall.items(), key=lambda kv: -kv[1]):
            print(f"  {s}: short by {q}")
    taken.extend(leftover[: n - len(taken)])
    random.Random(RANDOM_SEED).shuffle(taken)

    achieved = Counter(scenario_of[r["order_id"]] for r in taken)
    print("stratified subsample scenario shares:")
    for s, c in achieved.most_common():
        print(f"  {s:20s} {c:6d}  {c / len(taken) * 100:5.2f}%")
    return taken


def load_edge_train_records(edge_dir=None):
    """synthetic dynamic-NO_BID and near-boundary coverage from
    generate_edge_examples.py, the edge holdout files are never read here"""
    records = []
    for stage in ("tailoring", "logistics"):
        path = Path(edge_dir or DATA_DIR) / f"edge_{stage}_train.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f)
    return records


# a checkpoint written while the machine died mid-save keeps its adapter weights
# but loses the resume state, which is written last. get_last_checkpoint() returns
# it anyway and trainer.train() then dies on the missing trainer_state.json, so an
# unattended restart loop never makes progress
_RESUME_STATE_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")


def last_complete_checkpoint(output_dir):
    """newest checkpoint carrying a full resume state

    return (path_or_None, incomplete_paths_newest_first)"""
    if not os.path.isdir(output_dir):
        return None, []
    numbered = []
    for path in Path(output_dir).glob("checkpoint-*"):
        suffix = path.name.split("-", 1)[1]
        if path.is_dir() and suffix.isdigit():
            numbered.append((int(suffix), path))
    incomplete = []
    for _, path in sorted(numbered, reverse=True):
        if all((path / name).is_file() for name in _RESUME_STATE_FILES):
            return path, incomplete
        incomplete.append(path)
    return None, incomplete


def to_text(record, tokenizer, template_kwargs=None):
    messages = [
        {"role": "system", "content": record["system"]},
        {"role": "user", "content": record["user"]},
        {"role": "assistant", "content": record["completion"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         **(template_kwargs or {}))


class SaveIntervalEnforcer(TrainerCallback):
    """keep --save-steps authoritative across a resume.

    transformers restores the ENTIRE TrainerState from the checkpoint's
    trainer_state.json (trainer.py: `self.state = TrainerState.load_from_json(...)`),
    and DefaultFlowCallback decides when to checkpoint from `state.save_steps`,
    not `args.save_steps`. compare_trainer_and_checkpoint_args() only *warns*
    about the mismatch, it never reconciles it, so a tightened interval passed
    on the command line is silently discarded the moment you resume and the run
    reverts to whatever interval the checkpoint was written with.

    that matters here because crashes are an accepted fact of this host: an
    interval that only holds until the first crash is not a recovery guarantee.

    the restore happens AFTER on_train_begin fires, so re-asserting there does
    nothing. on_epoch_begin is the first hook that runs after it and before any
    on_step_end, which is where the save decision is made. re-asserting is
    idempotent, and because subsequent checkpoints then serialize the corrected
    value, later resumes inherit it without needing the fix again.
    """

    def __init__(self, save_steps):
        self.save_steps = save_steps
        self._reported = False

    def _enforce(self, args, state):
        if self.save_steps and state.save_steps != self.save_steps:
            print(f"SaveIntervalEnforcer: state.save_steps={state.save_steps} restored from "
                  f"checkpoint, overriding to the requested {self.save_steps}", flush=True)
            state.save_steps = self.save_steps
        elif not self._reported:
            print(f"SaveIntervalEnforcer: save interval {state.save_steps} steps (as requested)",
                  flush=True)
        self._reported = True

    def on_train_begin(self, args, state, control, **kwargs):
        self._enforce(args, state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._enforce(args, state)


class EpochAdapterSaver(TrainerCallback):
    """save one independently loadable adapter at the end of every epoch

    the trainer's own checkpoint-<step> directories carry optimizer and scheduler state for
    resuming, not a clean adapter to evaluate, and only the last one survives as final/. this writes
    <output-dir>/epoch-<n>/ after each epoch, which is exactly what the v5 validation-based epoch
    selection compares, so one run produces epoch 1, 2 and 3 instead of three cold starts.

    it changes no training math: saving is a side effect of on_epoch_end and the optimizer, the
    schedule and the data order continue untouched.

    what it does NOT give you is three runs' worth of independence. the linear LR schedule spans the
    WHOLE run, so epoch 1 of a 3-epoch run has been decaying toward a 3-epoch horizon and is not the
    same model as a standalone --num-train-epochs 1 run. that is a protocol choice, not a detail"""

    def __init__(self, model, tokenizer, output_dir):
        self.model, self.tokenizer, self.output_dir = model, tokenizer, output_dir
        self.saved = []

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        if epoch < 1 or any(entry["epoch"] == epoch for entry in self.saved):
            return
        path = Path(self.output_dir) / f"epoch-{epoch}"
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        self.saved.append({"epoch": epoch, "path": str(path), "global_step": state.global_step})
        print(f"EpochAdapterSaver: epoch {epoch} adapter saved to {path} "
              f"at global step {state.global_step}", flush=True)


class StepTimer(TrainerCallback):
    """wall time per optimizer step, so the estimate can use the median and
    ignore the slow compile/warmup steps at the start"""

    def __init__(self):
        self.step_seconds = []
        self._last = None

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self._last is not None:
            self.step_seconds.append(now - self._last)
        self._last = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"pipeline check: {SMOKE_EXAMPLES} examples, {SMOKE_STEPS} steps")
    parser.add_argument("--subsample", type=int, default=0,
                        help="train on the first N examples of the seeded shuffle instead of all")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
                        help="training sequence window. protocol v2 froze this at 1024 for the "
                             "stage agents; the supervisor needs 2048 because a 12-candidate "
                             "chain-selection prompt runs ~1600 tokens")
    parser.add_argument("--train-jsonl", default="",
                        help="train on this jsonl instead of the three stage files (used for the "
                             "supervisor set); --subsample/--include-edge do not apply to it")
    parser.add_argument("--dataset", default="sealed",
                        choices=sorted(TRAIN_SETS) + sorted(v5_training_data.V5_TRAIN_SETS),
                        help="named training set: sealed is the round-1 stage data, "
                             "reverse_auction is the round 2/3 data; ignored with --train-jsonl")
    parser.add_argument("--num-train-epochs", type=int, default=NUM_EPOCHS,
                        help="passes over the training set (protocol v2 froze this at 1; the "
                             "epoch-2 family raises it to 2). the linear LR schedule spans the "
                             "WHOLE run, so N epochs is not N-1 epochs plus one more -- the "
                             "optimization trajectory differs throughout")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stop after N steps for timing, extrapolate, save nothing")
    parser.add_argument("--include-edge", action="store_true",
                        help="mix the synthetic edge training subset into the set")
    parser.add_argument("--edge-repeat", type=int, default=1,
                        help="repeat the edge training subset this many times (with --include-edge)")
    parser.add_argument("--edge-weights", default="",
                        help="per-category repeat overrides, e.g. "
                             "tailoring:boundary_infeasible=4, others keep --edge-repeat")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="checkpoint and adapter output dir, use a fresh path per round")
    parser.add_argument("--checkpoint-dir", default="",
                        help="put the trainer's resume checkpoint-<step> directories here instead of "
                             "directly in --output-dir, so the evaluable epoch-<n>/ and final/ adapters "
                             "are not siblings of trainer state (the v5 layout uses "
                             "<output-dir>/checkpoints). empty = the old behavior, checkpoints in "
                             "--output-dir, which is what every completed run used")
    # cross-model protocol args, defaults reproduce the qwen runs exactly
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="hf checkpoint to fine-tune (frozen protocol: pin with --revision)")
    parser.add_argument("--revision", default=None,
                        help="exact hf revision of --base-model")
    parser.add_argument("--edge-dir", default="",
                        help="directory with edge_{stage}_train.jsonl, default data dir; "
                             "the round-one files live in data/finetune/v4/round1_edge")
    parser.add_argument("--instruction-part", default="<|im_start|>user\n",
                        help="template marker before user turns (model-specific)")
    parser.add_argument("--response-part", default="<|im_start|>assistant\n",
                        help="template marker before assistant turns, loss starts after it")
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="comma-separated LoRA target module names; use the functionally-"
                             "equivalent fused names for models like phi (qkv_proj,o_proj,gate_up_proj,down_proj)")
    parser.add_argument("--save-steps", type=int, default=0,
                        help="save a recovery checkpoint every N steps (0=off); does not alter "
                             "training math, only lets a crashed run resume with --resume")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the last checkpoint in --output-dir if one exists")
    parser.add_argument("--template-kwargs", default="",
                        help="JSON kwargs for apply_chat_template during training render, e.g. "
                             '\'{"enable_thinking": false}\' for models whose non-thinking mode '
                             "is template-controlled (smollm3); empty = prior behavior, "
                             "all completed model pipelines unchanged")
    parser.add_argument("--save-epoch-adapters", action="store_true",
                        help="also save a loadable adapter at the end of each epoch into "
                             "<output-dir>/epoch-<n>/, so one run yields the epoch 1/2/3 checkpoints "
                             "the v5 selection compares; off by default and it alters no training math. "
                             "note the LR schedule spans the whole run, so these are not equivalent to "
                             "separate 1/2/3-epoch runs")
    parser.add_argument("--optim", default="adamw_8bit",
                        help="trainer optimizer; adamw_8bit (bitsandbytes) is every completed run's setting, "
                             "adamw_torch is the fallback where bitsandbytes is unavailable")
    args = parser.parse_args()
    template_kwargs = json.loads(args.template_kwargs) if args.template_kwargs else None
    output_dir = args.output_dir
    # the trainer's resume state and the adapters a human evaluates are two different things that
    # happened to share a directory. they still do unless --checkpoint-dir says otherwise, so an
    # existing command line writes byte for byte where it always did
    checkpoint_dir = args.checkpoint_dir or output_dir

    if args.train_jsonl:
        with open(args.train_jsonl, encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
        total_available = len(records)
        print(f'loaded {total_available} records from {args.train_jsonl}')
    else:
        v5_dataset = v5_training_data.is_v5_dataset(args.dataset)
        if v5_dataset:
            from src.dataset import context as dataset_context
            if dataset_context.resolve().version != "v5":
                raise SystemExit(f"--dataset {args.dataset} needs {dataset_context.ENV_VAR}=v5")
            v5_training_data.check_window(args.dataset, args.max_seq_length)
        records = load_train_records(args.dataset)
        total_available = len(records)
        if v5_dataset:
            v5_training_data.assert_train_only(records, v5_training_data.read_split())
            print(f"dataset {args.dataset}: {total_available} records, all from the train split, from "
                  f"{', '.join(str(p.relative_to(PROJECT_ROOT)) for p in train_set_paths(args.dataset))}")
        else:
            # relative to the data dir, the per-stage files share a basename
            print(f"dataset {args.dataset}: {total_available} records from "
                  f"{', '.join(str(p.relative_to(DATA_DIR)) for p in train_set_paths(args.dataset))}")
    if args.subsample and not args.train_jsonl:
        if v5_training_data.is_v5_dataset(args.dataset):
            records, subsample_report = v5_training_data.stratified_subsample(records, args.subsample)
            print(f"v5 stratified subsample (scenario_type x priority, train split only): "
                  f"{subsample_report['rows']} rows from {subsample_report['orders']} orders")
            for stratum, s in subsample_report["strata"].items():
                print(f"  {stratum:28s} {s['achieved']:6d}  {s['achieved_share_pct']:6.2f}% "
                      f"(train population {s['population_share_pct']:.2f}%)")
            if subsample_report["shortfall_topped_up"]:
                print(f"WARNING: strata short of quota, topped up from others: {subsample_report['shortfall_topped_up']}")
        else:
            records = stratified_subsample(records, args.subsample)
    if args.include_edge and not args.train_jsonl and v5_training_data.is_v5_dataset(args.dataset):
        raise SystemExit("--include-edge does not apply to v5: the v5 edge rows are a robustness evaluation set, "
                         "never training data")
    if args.include_edge and not args.train_jsonl:
        if args.dataset != "sealed":
            raise SystemExit(
                f"--include-edge does not apply to --dataset {args.dataset}: the edge files are "
                "sealed-round prompts with no auction fields"
            )
        edge = load_edge_train_records(args.edge_dir or None)
        if args.edge_weights:
            overrides = {}
            for part in filter(None, args.edge_weights.split(",")):
                key, _, value = part.partition("=")
                stage, _, category = key.partition(":")
                overrides[(stage.strip(), category.strip())] = int(value)
            added = Counter()
            for r in edge:
                weight = overrides.get((r["stage"], r.get("category", "")), args.edge_repeat)
                records.extend([r] * weight)
                added[(r["stage"], r.get("category", ""))] += weight
            print(f"mixed in {len(edge)} synthetic edge examples with per-category "
                  f"weights = {sum(added.values())} rows:")
            for (stage, category), n in sorted(added.items()):
                print(f"  {stage}:{category} -> {n} rows")
        else:
            records.extend(edge * args.edge_repeat)
            print(f"mixed in {len(edge)} synthetic edge examples x {args.edge_repeat} "
                  f"= {len(edge) * args.edge_repeat} rows")
        random.Random(RANDOM_SEED + 1).shuffle(records)
    # the dataset a real run at these settings would train on
    planned_count = len(records)
    if args.smoke:
        records = records[:SMOKE_EXAMPLES]
    print(f"training examples this run: {len(records)} "
          f"(planned run: {planned_count}, available: {total_available})")

    load_started = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        revision=args.revision,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )
    print(f"base model {args.base_model} (rev {args.revision or 'default'}) "
          f"loaded in {time.perf_counter() - load_started:.1f}s")

    lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=lora_targets,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA targets {lora_targets}: {trainable:,} trainable "
          f"({trainable/total*100:.3f}% of {total:,})")

    dataset = Dataset.from_list([{"text": to_text(r, tokenizer, template_kwargs)} for r in records])

    timing_only = args.smoke or args.max_steps > 0
    max_steps = SMOKE_STEPS if args.smoke else (args.max_steps or -1)

    sft_config = SFTConfig(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        warmup_steps=5,
        logging_steps=1,
        optim=args.optim,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=RANDOM_SEED,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        **({"save_steps": args.save_steps, "save_strategy": "steps"}
           if args.save_steps > 0 and not timing_only else {}),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    # loss only on the assistant response, the prompt tokens are given
    trainer = train_on_responses_only(
        trainer,
        instruction_part=args.instruction_part,
        response_part=args.response_part,
    )

    timer = StepTimer()
    trainer.add_callback(timer)
    if args.save_steps > 0 and not timing_only:
        trainer.add_callback(SaveIntervalEnforcer(args.save_steps))
    epoch_saver = None
    if args.save_epoch_adapters and not timing_only:
        epoch_saver = EpochAdapterSaver(model, tokenizer, output_dir)
        trainer.add_callback(epoch_saver)

    if checkpoint_dir != output_dir:
        print(f"trainer checkpoints -> {checkpoint_dir}")
        print(f"epoch adapters and final -> {output_dir}")

    resume = False
    if args.resume:
        last, incomplete = last_complete_checkpoint(checkpoint_dir)
        for path in incomplete:
            print(f"skipping incomplete checkpoint {path.name}, "
                  f"no resume state (crash during save)")
        if last:
            # the exact path, not True: transformers would otherwise pick the
            # newest checkpoint again, including an incomplete one
            resume = str(last)
            print(f"resuming from checkpoint {last}")
        else:
            print("no complete checkpoint found, starting fresh")
    stats = trainer.train(resume_from_checkpoint=resume)

    import torch
    print(f"\ntrain_runtime: {stats.metrics['train_runtime']:.1f}s "
          f"for {stats.metrics.get('train_steps_per_second', 0) * stats.metrics['train_runtime']:.0f} steps")
    print(f"peak vram: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB allocated, "
          f"{torch.cuda.max_memory_reserved() / 1024**3:.1f} GiB reserved "
          f"(reserved > 24 GiB means windows sysmem spill, not a native fit)")

    if timer.step_seconds:
        median_step = statistics.median(timer.step_seconds)
        print(f"per-step wall time: median {median_step:.2f}s, "
              f"min {min(timer.step_seconds):.2f}s, max {max(timer.step_seconds):.2f}s "
              f"(over {len(timer.step_seconds)} measured steps, first step excluded)")

        if timing_only:
            effective_batch = args.batch_size * args.grad_accum
            full_steps = math.ceil(planned_count / effective_batch) * args.num_train_epochs
            est_seconds = full_steps * median_step
            print(f"\nextrapolation for a real run on {planned_count} examples "
                  f"({args.num_train_epochs} epoch(s), batch {args.batch_size} x accum {args.grad_accum} "
                  f"= effective {effective_batch}):")
            print(f"  steps needed:    {full_steps}")
            print(f"  estimated time:  {est_seconds / 3600:.1f} hours "
                  f"({est_seconds / 60:.0f} minutes) at {median_step:.2f}s/step")

    if timing_only:
        print("\ntiming run only, no adapter saved, full run needs an explicit go-ahead")
    else:
        final_dir = str(Path(output_dir) / "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"adapter saved to {final_dir}")
        if epoch_saver is not None:
            for entry in epoch_saver.saved:
                print(f"  epoch {entry['epoch']} adapter: {entry['path']} (step {entry['global_step']})")


if __name__ == "__main__":
    main()
