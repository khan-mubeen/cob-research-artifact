"""inference latency and token counts, one example at a time

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.inference_profile \
      --mode sealed --model qwen --epoch 1 --adapter models/v5/sealed/qwen-epoch1/final \
      --backend cuda --profile local --rows 30

this measures something the evaluators do not, and the two must never be reported as one number:

  training step time   seconds per optimizer step, batch 8 x grad accum 2, reported by
                       finetune_lora.py and src/experiments/v5/training_estimate.py
  eval throughput      examples per second at generation batch 16, printed by the eval scripts
  THIS                 wall time for ONE decision, batch size 1, nothing else in flight, which is
                       what a deployed agent actually waits for

per example it records the rendered input tokens, the output tokens really generated (counted from
the generated ids, not from the decoded string), the wall time, and tokens per second. the first
rows of any run are slower while kernels compile, so a warmup count is excluded from the medians
and reported separately rather than quietly dropped

the corpus-wide INPUT length distribution is already measured for every model and mode in
reports/v5/phase6_token_lengths.json and is not re-derived here; this run measures the input of the
rows it actually generates from, plus the output, which nothing measured before

every record carries the backend, execution profile and hardware identity from
src/experiments/v5/execution.py, so a LUMI latency and this machine's latency cannot end up in one
column unlabelled. results land through src/experiments/v5/results_store.py, appended one row at a
time, so a job killed by a wall-time limit resumes and finishes the rest

generation uses the v5 stop rule (stop_rule.py): the chat template's end-of-turn id is a stop id,
added only when the model's generation config does not already stop on it. output tokens are
counted from the real generated ids at batch size 1 (no padding exists), INCLUDING the stop token

two ways to run it:
  single run    the original --rows N measurement, written to results/_scratch/latency_profile/
  full split    shard / merge, the reported latency and token numbers:
    shard   --arm base|selected --split test --shard i/n --stage-dir <dir>
            times every row of a contiguous slice one at a time; per-row records are appended to
            <dir>/<mode>/<model>/<arm>-shard-<i>-of-<n>.jsonl with the host, slurm job and GCD, so
            a killed shard resumes. the first WARMUP rows of each shard are flagged warmup
    merge   --arm base|selected --split test --shards n --stage-dir <dir>
            checks every row of the split is timed exactly once, then writes
            results/latency/<mode>/<model>/<arm>.{jsonl,csv,meta.json}. latency statistics exclude
            the warmup rows of every shard (the count is recorded); token statistics use all rows
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from config import PROJECT_ROOT
from src.experiments.v5 import execution, model_registry, results_store, stop_rule
from src.experiments.v5.evaluate import MODE_FILES, load_rows

FAMILY = "latency"
DEFAULT_ROWS = 30
DEFAULT_WARMUP = 3
MAX_NEW_TOKENS = 200


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=tuple(MODE_FILES))
    ap.add_argument("--model", required=True, choices=sorted(model_registry.MODELS),
                    help="which architecture, for the record and for its base checkpoint")
    ap.add_argument("--epoch", type=int, default=0,
                    help="which epoch's adapter is being measured; 0 means the base model")
    ap.add_argument("--adapter", default="", help="adapter directory; empty measures the base checkpoint")
    ap.add_argument("--base-layout", choices=model_registry.BASE_LAYOUT_CHOICES,
                    default=model_registry.BASE_LAYOUT_DEFAULT,
                    help=f"where the base checkpoint is read from, default "
                         f"{model_registry.BASE_LAYOUT_DEFAULT}: the registry's pinned hub id or "
                         f"local copy. profile reads <root>/models/base/<model> under the resolved "
                         f"profile instead; it has no effect when --adapter is given")
    ap.add_argument("--split", default="validation", choices=("validation", "test"))
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="examples to time, one at a time")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help="leading examples excluded from the medians; still recorded")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--run", default="", help="result file stem; default <model>-<mode>-epoch<n>")
    ap.add_argument("--dry-run", action="store_true", help="resolve, load rows and stop, no model")
    ap.add_argument("--action", choices=("single", "shard", "merge"), default="single")
    ap.add_argument("--arm", choices=("base", "selected"), default="", help="shard/merge: which arm")
    ap.add_argument("--shard", default="", help="shard: i/n")
    ap.add_argument("--shards", type=int, default=0, help="merge: n")
    ap.add_argument("--stage-dir", default="", help="shard/merge: where per-shard records are kept")
    execution.add_arguments(ap)
    return ap.parse_args(argv)


def run_name(args):
    return args.run or f"{args.model}-{args.mode}-epoch{args.epoch}"


def select_rows(mode, split, count):
    """the first N rows of the mode's split, deterministic and shared by every model

    the same rows for every model and epoch is what makes the latencies comparable; a seeded sample
    would also work but would have to be pinned, and the file order already is"""
    rows = load_rows(mode, split)
    return rows[:count]


def time_one(model, tokenizer, text_tok, row, max_new_tokens, sync, stop_ids=None):
    """one unbatched decision: render, tokenise, generate, count what came back"""
    import torch
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = text_tok(prompt, return_tensors="pt").to(model.device)
    input_tokens = int(encoded["input_ids"].shape[1])
    sync()
    began = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=text_tok.pad_token_id,
                             **({"eos_token_id": list(stop_ids)} if stop_ids else {}))
    sync()
    seconds = time.perf_counter() - began
    generated = out[0][input_tokens:]
    # batch size 1 never pads, so every generated id is a token the model produced, the stop
    # token included
    ids = [int(t) for t in generated.tolist()]
    output_tokens = len(ids)
    stops = set(stop_ids or stop_rule.generation_stop_ids(model))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "seconds": round(seconds, 4),
        "ms_per_decision": round(seconds * 1000, 2),
        "output_tokens_per_second": round(output_tokens / seconds, 2) if seconds > 0 else None,
        "hit_token_cap": output_tokens >= max_new_tokens,
        "stopped_on_stop_id": bool(ids) and ids[-1] in stops,
        # collapsed to one line: the generated json carries newlines, and a raw copy would turn one
        # csv record into several rows. the text is a sample for eyeballing, not the measurement
        "text": " ".join(text_tok.decode(generated, skip_special_tokens=True).split())[:200],
    }


def summarise(records, warmup):
    """medians over the timed rows, warmup rows reported apart rather than dropped silently"""
    timed = [r for r in records if not r["warmup"]]
    warm = [r for r in records if r["warmup"]]

    def stats(rows, field):
        values = [r[field] for r in rows if r.get(field) is not None]
        if not values:
            return None
        return {"n": len(values), "median": round(statistics.median(values), 2),
                "mean": round(statistics.mean(values), 2), "min": round(min(values), 2),
                "max": round(max(values), 2),
                "p90": round(sorted(values)[int(len(values) * 0.9)], 2) if len(values) > 1 else round(values[0], 2)}

    return {
        "rows_timed": len(timed),
        "rows_warmup_excluded": len(warm),
        "warmup_median_ms": round(statistics.median([r["ms_per_decision"] for r in warm]), 2) if warm else None,
        "ms_per_decision": stats(timed, "ms_per_decision"),
        "input_tokens": stats(timed, "input_tokens"),
        "output_tokens": stats(timed, "output_tokens"),
        "total_tokens": stats(timed, "total_tokens"),
        "output_tokens_per_second": stats(timed, "output_tokens_per_second"),
        "rows_hitting_the_token_cap": sum(r["hit_token_cap"] for r in timed),
    }


def arm_checkpoint(args, context):
    """(checkpoint, epoch, is_directory) for --arm base or --arm selected"""
    from src.experiments.v5 import epoch_selection
    run = f"{args.model}-{args.mode}-v5-r01"
    if args.arm == "selected":
        selection = epoch_selection.run_dir(run) / epoch_selection.SELECTION_FILE
        if not selection.exists():
            raise SystemExit(f"{selection} does not exist; the selected arm needs a selection")
        chosen = json.loads(selection.read_text(encoding="utf-8"))["chosen"]
        return chosen["adapter"], chosen["epoch"], True, str(selection)
    base_layout = getattr(args, "base_layout", model_registry.BASE_LAYOUT_DEFAULT)
    spec = model_registry.model(args.model)
    if base_layout == "profile":
        return str(context.base_model_dir(args.model)), 0, True, None
    return spec["checkpoint"], 0, spec["local"], None


def bounds(n_rows, n_shards):
    per = -(-n_rows // n_shards)
    return [(i * per, min((i + 1) * per, n_rows)) for i in range(n_shards) if i * per < n_rows]


def shard_set(args, i, n):
    return results_store.ResultSet("shards", f"{args.arm}-shard-{i:03d}-of-{n:03d}", root=Path(args.stage_dir),
                                   subdir=f"{args.mode}/{args.model}")


def shard(args, context):
    import os
    import socket
    i, n = (int(x) for x in args.shard.split("/"))
    rows = load_rows(args.mode, args.split)
    b = bounds(len(rows), n)
    if i >= len(b):
        print(f"shard {i}/{n}: empty")
        return 0
    lo, hi = b[i]
    out = shard_set(args, i, n)
    done = out.done_keys()
    if len(done) >= hi - lo:
        print(f"{out.path}: complete")
        return 0
    checkpoint, epoch, is_dir, selection = arm_checkpoint(args, context)
    spec = model_registry.model(args.model)
    provenance = results_store.provenance(args.model, args.mode, epoch, checkpoint,
                                          backend=context.backend.name, profile=context.profile.name)
    from unsloth import FastLanguageModel
    window = model_registry.MODES[args.mode]["window"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint, revision=None if is_dir else spec["revision"],
        max_seq_length=window, load_in_4bit=False, dtype=None)
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token
    stops, stop_info = stop_rule.stop_ids(model, tokenizer, args.model, args.mode)
    import torch
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    node = {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gcd": os.environ.get("ROCR_VISIBLE_DEVICES")}
    print(f"{args.model} {args.mode} {args.arm} {args.split} shard {i}/{n}: rows [{lo}, {hi}), "
          f"{len(done)} done, node {node}", flush=True)
    session_first = True
    for index in range(lo, hi):
        key = results_store.record_key(args.model, args.mode, args.arm, args.split, index)
        if key in done:
            continue
        measured = time_one(model, tokenizer, text_tok, rows[index], args.max_new_tokens, sync, stops)
        # warmup: the first rows of the shard, and the first row after any resume (a fresh process
        # recompiles kernels), are flagged and kept out of the latency statistics
        warm = (index - lo) < args.warmup or session_first
        session_first = False
        out.append({"record_key": key, "row_index": index, "row_id": rows[index].get("order_id"),
                    "stage": rows[index].get("stage"), "arm": args.arm, "epoch": epoch, "split": args.split,
                    "shard": i, "shards": n, "warmup": warm, "node": node, "window": window,
                    "max_new_tokens": args.max_new_tokens, "generation_stop_rule": stop_info,
                    "selection": selection, **measured, **provenance})
        if (index - lo + 1) % 50 == 0:
            print(f"  {index - lo + 1}/{hi - lo} timed, last {measured['ms_per_decision']} ms", flush=True)
    print(f"shard {i}/{n} done -> {out.path}")
    return 0


def _stats(values):
    import math
    values = [v for v in values if v is not None]
    if not values:
        return None
    s = sorted(values)
    mean = statistics.mean(s)
    return {"n": len(s), "sum": round(sum(s), 4), "mean": round(mean, 4),
            "std": round(statistics.stdev(s), 4) if len(s) > 1 else 0.0,
            "median": round(statistics.median(s), 4), "p95": round(s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)], 4),
            "min": round(s[0], 4), "max": round(s[-1], 4)}


def merge(args, context):
    rows = load_rows(args.mode, args.split)
    b = bounds(len(rows), args.shards)
    records = []
    for i in range(len(b)):
        records += sorted(shard_set(args, i, args.shards).records(), key=lambda r: r["row_index"])
    idx = [r["row_index"] for r in records]
    if idx != list(range(len(rows))):
        missing = sorted(set(range(len(rows))) - set(idx))
        raise SystemExit(f"{len(records)} records for {len(rows)} rows; missing {len(missing)} "
                         f"(first {missing[:5]}), duplicates {len(idx) - len(set(idx))}")
    rules = {json.dumps(r["generation_stop_rule"], sort_keys=True) for r in records}
    if len(rules) != 1:
        raise SystemExit("shards were measured under different stop rules")
    checkpoint, epoch, _, _ = arm_checkpoint(args, context)
    want = {"mode": args.mode, "model": args.model, "split": args.split, "arm": args.arm,
            "epoch": epoch, "checkpoint": str(checkpoint)}
    for r in records:
        diff = {k: r.get(k) for k, v in want.items() if r.get(k) != v}
        if diff:
            raise SystemExit(f"row {r['row_index']} does not match the requested arm {want}: {diff}")
    out = results_store.ResultSet(FAMILY, args.arm, subdir=f"{args.mode}/{args.model}")
    # written once: the meta file is written last, so it marks a completed merge. a records file without
    # one is an interrupted earlier merge and is replaced. the records are written in one pass to a
    # temporary file and renamed into place (appending them one at a time re-reads the file per record,
    # which is quadratic in the 67k-75k rows of a sealed or reverse-auction arm)
    if out.meta_path.exists():
        raise SystemExit(f"{out.meta_path} exists; a merged latency result is written once")
    if out.path.exists():
        print(f"replacing an incomplete earlier merge at {out.path} (it has no meta file)", flush=True)
    import os
    out.dir.mkdir(parents=True, exist_ok=True)
    tmp = out.path.with_name(out.path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out.path)
    timed = [r for r in records if not r["warmup"]]
    summary = {
        "rows": len(records), "rows_timed": len(timed), "rows_warmup_excluded": len(records) - len(timed),
        "latency_ms": _stats([r["ms_per_decision"] for r in timed]),
        "input_tokens": _stats([r["input_tokens"] for r in records]),
        "output_tokens": _stats([r["output_tokens"] for r in records]),
        "total_tokens": _stats([r["total_tokens"] for r in records]),
        "output_tokens_per_second": _stats([r["output_tokens_per_second"] for r in timed]),
        "cap_hit_share_pct": round(100 * sum(r["hit_token_cap"] for r in records) / len(records), 4),
        "stopped_on_stop_id_share_pct": round(100 * sum(r.get("stopped_on_stop_id", False) for r in records)
                                              / len(records), 4),
        "input_to_output_ratio": round(sum(r["input_tokens"] for r in records)
                                       / max(1, sum(r["output_tokens"] for r in records)), 4),
        "nodes": sorted({f"{r['node']['host']}/job{r['node']['slurm_job_id']}/gcd{r['node']['gcd']}" for r in records}),
    }
    # csv through a temporary file and a rename, before the meta file
    final_csv = out.csv_path
    out.csv_path = final_csv.with_name(final_csv.name + ".tmp")
    out.write_csv()
    out.csv_path.replace(final_csv)
    out.csv_path = final_csv
    first = records[0]
    out.write_meta({"mode": args.mode, "model": args.model, "arm": args.arm, "epoch": first["epoch"],
                    "split": args.split, "shards": len(b), "checkpoint": first["checkpoint"],
                    "selection": first.get("selection"), "generation_stop_rule": first["generation_stop_rule"],
                    "measurement": "one decision at a time, batch size 1, whole split; latency statistics exclude "
                                   f"the first {args.warmup} rows of every shard (and the first row after a resume); "
                                   "token statistics include every row; output tokens counted from generated ids "
                                   "including the stop token",
                    "execution": context.as_dict(), "summary": summary})
    print(json.dumps(summary, indent=2))
    print(f"records {out.path}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    context = execution.require(args.backend, args.profile, "inference_profile",
                                root=args.profile_root or None)
    from src.dataset import context as dataset_context
    if dataset_context.resolve().version != "v5":
        raise SystemExit(f"v5 inference profiling needs {dataset_context.ENV_VAR}=v5")
    if args.action != "single":
        if not args.arm or not args.stage_dir:
            raise SystemExit("shard and merge need --arm and --stage-dir")
        if args.dry_run:
            print(json.dumps({"execution": context.as_dict(), "action": args.action, "arm": args.arm,
                              "split": args.split, "shard": args.shard, "shards": args.shards}, indent=2))
            return 0
        return shard(args, context) if args.action == "shard" else merge(args, context)

    spec = model_registry.model(args.model)
    # the adapter path wins and is passed through byte for byte; only the base checkpoint moves
    # with --base-layout, and a base directory carries no hub revision to pin
    base_layout = getattr(args, "base_layout", model_registry.BASE_LAYOUT_DEFAULT)
    base_is_directory = spec["local"] or base_layout == "profile"
    base_checkpoint = (str(context.base_model_dir(args.model)) if base_layout == "profile"
                       else spec["checkpoint"])
    checkpoint = args.adapter or base_checkpoint
    rows = select_rows(args.mode, args.split, args.rows)
    results = results_store.ResultSet("latency_profile", run_name(args), root=results_store.RESULTS_ROOT / "_scratch")
    done = results.done_keys()
    print(f"{run_name(args)}: {len(rows)} rows from {args.mode}/{args.split}, "
          f"{len(done)} already recorded, backend {context.backend.name}, profile {context.profile.name}")
    if args.dry_run:
        print(json.dumps({"execution": context.as_dict(), "checkpoint": checkpoint,
                          "would_measure": len(rows) - len(done)}, indent=2))
        return 0

    provenance = results_store.provenance(args.model, args.mode, args.epoch, checkpoint,
                                          backend=context.backend.name, profile=context.profile.name)
    from unsloth import FastLanguageModel  # imported only when a model is really loaded
    window = model_registry.MODES[args.mode]["window"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint, revision=None if args.adapter or base_is_directory else spec["revision"],
        max_seq_length=window, load_in_4bit=False, dtype=None)
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token
    stops, stop_info = stop_rule.stop_ids(model, tokenizer, args.model, args.mode)

    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except Exception:
        sync = lambda: None  # noqa: E731

    written = 0
    for index, row in enumerate(rows):
        key = results_store.record_key(args.model, args.mode, args.epoch, args.split, index)
        if key in done:
            continue
        measured = time_one(model, tokenizer, text_tok, row, args.max_new_tokens, sync, stops)
        record = {"record_key": key, "row_index": index, "row_id": row.get("order_id"),
                  "stage": row.get("stage"), "warmup": index < args.warmup,
                  "window": window, "max_new_tokens": args.max_new_tokens,
                  **measured, **provenance}
        written += results.append(record)
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"  {index + 1}/{len(rows)} timed, last {measured['ms_per_decision']} ms, "
                  f"{measured['output_tokens']} output tokens", flush=True)

    records = results.records()
    summary = summarise(records, args.warmup)
    results.write_meta({"run": run_name(args), "family": FAMILY, "split": args.split,
                        "rows_requested": args.rows, "warmup": args.warmup,
                        "checkpoint": checkpoint, "execution": context.as_dict(),
                        "input_length_reference": "reports/v5/phase6_token_lengths.json",
                        "measurement": "single example, batch size 1, not a training step and not eval throughput",
                        "provenance": provenance, "summary": summary})
    csv_path = results.write_csv()
    print(f"\nwrote {written} new record(s), {len(records)} total")
    print(json.dumps(summary, indent=2))
    print(f"records {results.path}\ncsv     {csv_path}\nmeta    {results.meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
