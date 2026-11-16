"""v5 validation evaluation split across GPUs, for runs that outlast a job's wall time

evaluate.py is not modified by this file and remains the only path for the TEST split. this module
exists because one validation arm (68k to 75k rows) takes 6 to 12 hours on one GCD, longer than the
3-hour limit of the partition that has capacity. it splits the GENERATION, the expensive part,
across jobs, and leaves everything that decides a number to evaluate.py's own functions:

  shard  generate outputs for one contiguous slice of the rows and store them. slices start on
         multiples of the generation batch (16), so every generation batch is the same 16 rows,
         padded the same way, as in a single-pass run. generate_all builds each prompt from that
         row's system and user text only and decodes greedily, so identical batches give
         identical outputs
  merge  load every row, concatenate the shard outputs in order, and score and summarize the
         whole set ONCE with the exact functions evaluate.main calls, then write the result
         through evaluate.guard, so the write-once protection of the canonical file still holds

the result is the same record evaluate.main writes, plus one field, sharded_generation, naming the
shards and their hashes, so a sharded number is never mistaken for a single-pass one.

the test split is accepted for DESCRIPTIVE arms only (evaluate_descriptive.py): the epochs selection
did not choose, read after selection.json exists. base and the selected epoch are refused on test
here exactly as evaluate_descriptive refuses them; their one-shot reads stay on the single-pass
evaluate.py and its authorize_test / authorize_base_test gates, which this module never calls.

generation uses the v5 stop rule (stop_rule.py), identical in the single-pass and sharded paths

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.evaluate_sharded shard \
      --mode sealed --run R --epoch 3 --adapter <dir> --shard 0/12 --stage-dir <dir>
  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.evaluate_sharded merge \
      --mode sealed --run R --epoch 3 --adapter <dir> --shards 12 --stage-dir <dir>
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from src.experiments.v5 import epoch_selection, evaluate, evaluate_descriptive, stop_rule, training_data

GEN_BATCH = 16   # eval_finetune.GEN_BATCH and eval_reverse_auction.GEN_BATCH; checked at run time


def _args_ns(a):
    """the namespace evaluate.guard and evaluate.resolve_checkpoint expect"""
    return SimpleNamespace(mode=a.mode, split=a.split, run=a.run, epoch=a.epoch, adapter=a.adapter,
                           base_model=a.base_model, base_layout=a.base_layout, base_root=a.base_root,
                           model_name=a.model_name, tag=a.tag, max_seq_length=a.max_seq_length,
                           smoke_rows=a.smoke_rows)


def _check_batch():
    from src.experiments import eval_finetune, eval_reverse_auction
    if eval_finetune.GEN_BATCH != GEN_BATCH or eval_reverse_auction.GEN_BATCH != GEN_BATCH:
        raise SystemExit(f"generation batch is no longer {GEN_BATCH}; shard boundaries would no longer "
                         f"reproduce single-pass batches")


def bounds(n_rows, n_shards):
    """contiguous slices whose starts are all multiples of the generation batch"""
    per = math.ceil(n_rows / n_shards / GEN_BATCH) * GEN_BATCH
    out = [(i * per, min((i + 1) * per, n_rows)) for i in range(n_shards)]
    return [(a, b) for a, b in out if a < b]


def _inputs(mode, rows):
    """exactly the objects evaluate.main hands to generate_all, per mode"""
    if mode == "sealed":
        return [{"id": f"{r['order_id']}/{r['agent_id']}", "stage": r["stage"], "system": r["system"],
                 "user": r["user"], "label": json.loads(r["completion"])} for r in rows]
    if mode == "reverse_auction":
        from src.experiments.eval_reverse_auction import prompt_priority
        for r in rows:
            r["priority"] = prompt_priority(r["user"])
        return rows
    return [{"system": r["system"], "user": r["user"]} for r in rows]


def shard(a):
    _check_batch()
    i, n = (int(x) for x in a.shard.split("/"))
    if a.split == "test":
        # the descriptive preconditions, checked before any generation: selection exists, not the
        # base arm, not the selected epoch, the ranked adapter for this epoch
        evaluate_descriptive.authorize_descriptive(a.run, a.epoch, a.adapter, allow_existing=bool(a.smoke_rows))
    rows = evaluate.load_rows(a.mode, a.split, a.smoke_rows)
    b = bounds(len(rows), n)
    if i >= len(b):
        print(f"shard {i}/{n}: empty, nothing to do")
        return 0
    lo, hi = b[i]
    stage = Path(a.stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    out = stage / f"shard-{i:03d}-of-{n:03d}.json"
    if out.exists():
        print(f"{out} exists; a shard is generated once")
        return 0
    ns = _args_ns(a)
    window = a.max_seq_length or training_data.V5_WINDOWS[evaluate.MODE_DATASET[a.mode]]
    training_data.check_window(evaluate.MODE_DATASET[a.mode], window)
    checkpoint, revision = evaluate.resolve_checkpoint(ns)
    sl = rows[lo:hi]
    print(f"{a.run} {a.mode} {a.split} shard {i}/{n}: rows [{lo}, {hi}) of {len(rows)} -> {out}", flush=True)
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=checkpoint, revision=revision,
                                                         max_seq_length=window, load_in_4bit=False, dtype=None)
    FastLanguageModel.for_inference(model)
    if a.mode == "reverse_auction":
        from src.experiments.eval_reverse_auction import generate_all
    else:
        from src.experiments.eval_finetune import generate_all
    model_name, _ = stop_rule.cell_of(a.run, a.model_name)
    stops, stop_info = stop_rule.stop_ids(model, tokenizer, model_name, a.mode)
    outputs = generate_all(model, tokenizer, _inputs(a.mode, sl), stop_ids=stops)
    rec = {"run": a.run, "mode": a.mode, "split": a.split, "epoch": a.epoch, "adapter": a.adapter,
           "shard": i, "shards": n, "generation_stop_rule": stop_info,
           "start": lo, "end": hi, "total_rows": len(rows), "smoke_rows": a.smoke_rows, "window": window,
           "checkpoint": str(checkpoint), "identities": [evaluate.identity(r) for r in sl], "outputs": outputs,
           "generated_at": datetime.now().isoformat(timespec="seconds")}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    tmp.replace(out)            # a killed job never leaves a half-written shard behind
    print(f"written {out}")
    return 0


def merge(a):
    _check_batch()
    rows = evaluate.load_rows(a.mode, a.split, a.smoke_rows)
    identities = [evaluate.identity(r) for r in rows]
    stage = Path(a.stage_dir)
    b = bounds(len(rows), a.shards)
    outputs, shard_meta, stop_infos = [], [], []
    for i, (lo, hi) in enumerate(b):
        p = stage / f"shard-{i:03d}-of-{a.shards:03d}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}; merge needs every shard")
        s = json.loads(p.read_text(encoding="utf-8"))
        for field, want in (("run", a.run), ("mode", a.mode), ("split", a.split), ("epoch", a.epoch), ("adapter", a.adapter),
                            ("start", lo), ("end", hi), ("total_rows", len(rows)), ("smoke_rows", a.smoke_rows)):
            if s[field] != want:
                raise SystemExit(f"{p}: {field} is {s[field]!r}, expected {want!r}")
        if s["identities"] != identities[lo:hi]:
            raise SystemExit(f"{p}: rows do not line up with the split at [{lo}, {hi})")
        if len(s["outputs"]) != hi - lo:
            raise SystemExit(f"{p}: {len(s['outputs'])} outputs for {hi - lo} rows")
        outputs.extend(s["outputs"])
        stop_infos.append(s["generation_stop_rule"])
        shard_meta.append({"shard": i, "start": lo, "end": hi, "file": str(p),
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                           "generated_at": s["generated_at"]})
    if len(outputs) != len(rows):
        raise SystemExit(f"{len(outputs)} outputs for {len(rows)} rows")

    if any(json.dumps(x, sort_keys=True) != json.dumps(stop_infos[0], sort_keys=True) for x in stop_infos):
        raise SystemExit("shards were generated under different stop rules")
    ns = _args_ns(a)
    if a.split == "test":
        out_path, authorization = evaluate_descriptive.resolve_output(a)   # descriptive gate
        canonical = evaluate.verify_test_digest(a.mode, identities) if not a.smoke_rows else None
    else:
        out_path, authorization = evaluate.guard(ns)      # the write-once validation gate, unchanged
        canonical = None
    window = a.max_seq_length or training_data.V5_WINDOWS[evaluate.MODE_DATASET[a.mode]]
    # from here down: the same calls, in the same order, as evaluate.main after generation
    if a.mode == "sealed":
        from src.experiments.eval_finetune import score, summarize
        scored = score(_inputs("sealed", rows), outputs)
        summary = summarize(f"{a.split} sealed", scored)
    elif a.mode == "reverse_auction":
        from src.experiments.eval_reverse_auction import score, summarize
        _inputs("reverse_auction", rows)
        scored = score(rows, outputs)
        summary = summarize(a.tag or a.run, a.adapter, scored)
    else:
        summary, scored = evaluate.score_supervisor(rows, outputs)
    sharded = {"shards": len(b), "batch": GEN_BATCH, "module": "src/experiments/v5/evaluate_sharded.py",
               "detail": shard_meta}
    if a.split == "test":
        result = evaluate_descriptive.build_result(a, rows, identities, canonical, window, authorization, summary,
                                                   scored, stop_infos[0], {"sharded_generation": sharded})
        evaluate_descriptive.write(out_path, result)
    else:
        result = {
            "dataset_version": "v5", "mode": a.mode, "split": "validation", "run": a.run, "epoch": a.epoch,
            "arm": "base" if evaluate.is_base(ns) else "tuned",
            "adapter": a.adapter, "tag": a.tag, "smoke": bool(a.smoke_rows), "rows": len(rows),
            "canonical_test": None,
            "row_ids_sha256": epoch_selection.row_ids_digest(identities), "window": window,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"), "authorization": authorization,
            "summary": summary, "selection_metrics": epoch_selection.selection_metrics(a.mode, summary),
            "generation_stop_rule": stop_infos[0],
            "rows_detail": scored,
            "sharded_generation": sharded,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"merged {len(b)} shard(s), {len(rows)} rows -> {out_path}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("shard", "merge"))
    ap.add_argument("--mode", required=True, choices=epoch_selection.MODES)
    ap.add_argument("--split", default="validation", choices=("validation", "test"),
                    help="validation, or test for a DESCRIPTIVE arm only (never base, never the selected epoch)")
    ap.add_argument("--smoke-out", default="", help="where a smoke descriptive merge is written (test split only)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", required=True, type=int)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--base-layout", default="registry")
    ap.add_argument("--base-root", default="")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-seq-length", type=int, default=0)
    ap.add_argument("--smoke-rows", type=int, default=0)
    ap.add_argument("--shard", default="", help="i/n, for the shard action")
    ap.add_argument("--shards", type=int, default=0, help="n, for the merge action")
    ap.add_argument("--stage-dir", required=True)
    a = ap.parse_args(argv)
    if a.split == "test" and (a.epoch == 0 or not a.adapter):
        raise SystemExit("evaluate_sharded refuses the base arm on test: its one-shot read is evaluate.py's alone")
    return a


def main(argv=None):
    from src.dataset import context as dataset_context
    if dataset_context.resolve().version != "v5":
        raise SystemExit(f"v5 evaluation needs {dataset_context.ENV_VAR}=v5")
    a = parse_args(argv)
    return shard(a) if a.action == "shard" else merge(a)


if __name__ == "__main__":
    sys.exit(main())
