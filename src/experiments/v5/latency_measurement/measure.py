"""full test split latency with time to first token, one request at a time

reproduces the official profile (src/experiments/v5/inference_profile.py, action shard): the same loader,
shard counts and bounds, load path, generation settings, stop rule and timed region, run by run.sbatch as
eight processes per node, one per gcd. the only addition is a first-token hook, a stopping criterion that
never stops (the same hook as adapter_merge_diagnostic/run_diagnostic.py).

per request it appends token counts and timings to records/<mode>/<model>/<arm>-shard-<i>-of-<n>.jsonl;
the generated text is never stored. prompts are loaded with every other field dropped, so no label is
kept. resumable: a finished shard is skipped and a started one continues after its last request, which
is then flagged warm-up, as in the official profile.

the 3,353 shards are split once into K disjoint parts (logs/parts.json: longest estimated shard first onto
the least loaded part); one slurm job runs one part. each gcd slot calls --claim again and again: a call
takes the next unfinished shard of its part under an exclusive lock in logs/locks/, measures it in that
fresh process and exits (exit status 3 when nothing is left).

  PYTHONDONTWRITEBYTECODE=1 python measure.py --make-parts [--parts 8]   write logs/parts.json once
  ... --list [--parts 8] [--nodes 8]          estimate per part: gcd-hours, wall time, node-hours, idle tail
  ... --claim --part i/K   (or --pilot)       measure the next free shard of a part (or pilot slice)
  ... --clear-stale --part i/K --live "ids"   remove this part's locks whose job is not among the live ids
  ... --status                                progress of all parts, slurm jobs and a rough eta
  ... --dry-run                               load the prompts and check them against the official records
"""
import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "config.py").exists())
sys.path.insert(0, str(REPO))

from src.experiments.v5 import execution, model_registry, stop_rule  # noqa: E402  (read only)
from src.experiments.v5.evaluate import load_rows  # noqa: E402  (the official profile's loader)

MODES = ("sealed", "reverse_auction", "supervisor")
MODELS = ("gemma", "granite", "ministral", "phi", "qwen", "smollm3")
ARMS = ("base", "selected")
WARMUP, MAX_NEW_TOKENS, LOAD_S = 3, 200, 30
KEEP = ("system", "user", "order_id")
OFFICIAL = REPO / "reports/v5/results/latency"
SELECTION = REPO / "reports/v5/selection"
PARTS = HERE / "logs" / "parts.json"
LOCKS = HERE / "logs" / "locks"
CHECK = HERE / "repeat_check"
PILOT = {"mode": "sealed", "model": "gemma", "rows": 400, "procs_per_arm": 4, "hook_check": 10}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--shard", default="", help="i/n, n as in the official records")
    ap.add_argument("--rows", default="", help="pilot: lo:hi, a slice of the split instead of a shard")
    ap.add_argument("--hook-check", type=int, default=0, help="pilot: rows timed with and without the hook")
    ap.add_argument("--part", default="", help="i/K, one part of logs/parts.json")
    ap.add_argument("--parts", type=int, default=8, help="K for --make-parts and --list")
    ap.add_argument("--nodes", type=int, default=8, help="nodes per part for --list")
    ap.add_argument("--live", default="", help="--clear-stale: the job ids squeue lists")
    ap.add_argument("--variant", default="", choices=("", "B", "C"),
                    help="repeat check only: B as the main run, C with the first-token hook off; off by default")
    for flag in ("--make-parts", "--list", "--claim", "--clear-stale", "--status", "--pilot", "--dry-run",
                 "--make-check", "--check"):
        ap.add_argument(flag, action="store_true")
    return ap.parse_args(argv)


def official(mode, model, arm):
    return json.loads((OFFICIAL / mode / model / f"{arm}.meta.json").read_text())


def bounds(n_rows, n_shards):
    # copied from inference_profile.bounds
    per = -(-n_rows // n_shards)
    return [(i * per, min((i + 1) * per, n_rows)) for i in range(n_shards) if i * per < n_rows]


def prompts(mode):
    # the official loader; everything except the prompt and the row id is dropped at load
    return [{k: r.get(k) for k in KEEP} for r in load_rows(mode, "test")]


def checkpoint(mode, model, arm, context):
    """the official arm_checkpoint with --base-layout profile: (path, epoch)"""
    if arm == "selected":
        chosen = json.loads((SELECTION / mode / model / "selection.json").read_text())["chosen"]
        return chosen["adapter"], chosen["epoch"]
    return str(context.base_model_dir(model)), 0


def out_path(mode, model, arm, shard=None, rows=None, root=None):
    root = root or HERE / "records"
    if rows:
        return root / "pilot" / mode / model / f"{arm}-rows-{rows[0]:05d}-{rows[1]:05d}.jsonl"
    i, n = shard
    return root / mode / model / f"{arm}-shard-{i:03d}-of-{n:03d}.jsonl"


def read(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # a line cut by a killed job; the request is simply timed again
    return out


def all_tasks():
    """every shard of the 36 arms with its estimate (rows x official mean latency + model load), longest first"""
    out = []
    for mode in MODES:
        for model in MODELS:
            for arm in ARMS:
                meta = official(mode, model, arm)
                n, rows, ms = meta["shards"], meta["summary"]["rows"], meta["summary"]["latency_ms"]["mean"]
                out += [{"line": f"--mode {mode} --model {model} --arm {arm} --shard {i}/{n}", "mode": mode,
                         "rows": hi - lo, "est_s": round((hi - lo) * ms / 1000 + LOAD_S, 1)}
                        for i, (lo, hi) in enumerate(bounds(rows, n))]
    return sorted(out, key=lambda t: (-t["est_s"], t["line"]))


def pilot_tasks():
    n = PILOT["rows"] // PILOT["procs_per_arm"]
    return [{"line": f"--mode {PILOT['mode']} --model {PILOT['model']} --arm {a} --rows {q * n}:{(q + 1) * n} "
                     f"--hook-check {PILOT['hook_check']}", "mode": PILOT["mode"], "rows": n, "est_s": 0}
            for a in ARMS for q in range(PILOT["procs_per_arm"])]


def split(tasks, k):
    """longest first onto the least loaded part (lowest index on ties); deterministic"""
    load, parts = [0.0] * k, [[] for _ in range(k)]
    for t in tasks:
        j = min(range(k), key=lambda j: (load[j], j))
        parts[j].append(t)
        load[j] += t["est_s"]
    return parts


def check_split(parts):
    """every shard exactly once over all parts"""
    lines = [t["line"] for p in parts for t in p]
    want = {t["line"] for t in all_tasks()}
    if len(lines) != len(set(lines)) or set(lines) != want:
        raise SystemExit(f"parts do not cover the shards exactly once: {len(lines)} entries, {len(set(lines))} distinct, "
                         f"{len(want)} shards, {len(want - set(lines))} missing")
    return len(lines)


def read_parts(k=None):
    doc = json.loads(PARTS.read_text()) if PARTS.exists() else None
    if doc and k is not None and doc["parts"] != k:
        raise SystemExit(f"{PARTS.name} holds {doc['parts']} parts; refusing {k}")
    return doc


def make_parts(k):
    doc = {"parts": k, "rule": "longest estimated shard first onto the least loaded part; estimate = rows x official "
                               "mean latency + 30 s model load", "tasks": split(all_tasks(), k)}
    n = check_split(doc["tasks"])
    old = read_parts(k)
    if old is not None:
        if old != doc:
            raise SystemExit(f"{PARTS.name} differs from a fresh split of the official shards; refusing")
        print(f"{PARTS.name} already holds these {k} parts ({n} shards)")
        return 0
    with open(PARTS, "x") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {PARTS.relative_to(HERE)}: {k} parts, {n} shards, each exactly once")
    return 0


def make_check():
    """fix the repeat-check shards once: the first 3 finished sealed smollm3 base shards and the first finished
    sealed qwen fine-tuned shard of the main records, in shard order"""
    want, picked = (("smollm3", "base", 3), ("qwen", "selected", 1)), []
    for model, arm, k in want:
        n = official("sealed", model, arm)["shards"]
        lines = [f"--mode sealed --model {model} --arm {arm} --shard {i}/{n}" for i in range(n)]
        picked += [x for x in lines if progress(parse_args(x.split())).finished][:k]
    if (CHECK / "shards.json").exists():
        old = json.loads((CHECK / "shards.json").read_text())
        print(f"repeat_check/shards.json already holds {old}")
        return 0
    CHECK.mkdir(exist_ok=True)
    with open(CHECK / "shards.json", "x") as f:
        json.dump(picked, f, indent=1)
    print(f"wrote repeat_check/shards.json: {picked}")
    return 0


def part_tasks(a):
    """the tasks a run may take: one part of logs/parts.json, the pilot slices, the repeat check, or (before any
    split) every shard"""
    if a.pilot:
        return pilot_tasks()
    if a.check:
        lines = json.loads((CHECK / "shards.json").read_text())
        return [{"line": f"{x} --variant {v}", "mode": "sealed", "rows": 0, "est_s": 0} for v in ("B", "C") for x in lines]
    if not a.part:
        doc = read_parts()
        if doc:
            raise SystemExit(f"{PARTS.name} exists: give --part i/{doc['parts']} (the single-job run is replaced)")
        return all_tasks()
    i, k = (int(x) for x in a.part.split("/"))
    doc = read_parts(k)
    if doc is None:
        raise SystemExit(f"no {PARTS.name} yet: run submit_all.sh or measure.py --make-parts --parts {k} first")
    if not 0 <= i < k:
        raise SystemExit(f"part {i} is not in 0..{k - 1}")
    return doc["tasks"][i]


def record_path(t):
    root = CHECK / t.variant if t.variant else None
    if t.rows:
        return out_path(t.mode, t.model, t.arm, rows=tuple(int(x) for x in t.rows.split(":")), root=root)
    return out_path(t.mode, t.model, t.arm, shard=tuple(int(x) for x in t.shard.split("/")), root=root)


def lock_path(t):
    if t.variant:
        return CHECK / "locks" / ("-".join(record_path(t).relative_to(CHECK).with_suffix("").parts) + ".lock")
    return LOCKS / ("-".join(record_path(t).relative_to(HERE / "records").with_suffix("").parts) + ".lock")


def progress(a):
    """bounds, record file and what is done of one task, from its records only (the official shard count guard)"""
    meta = official(a.mode, a.model, a.arm)
    if a.rows:
        lo, hi = (int(x) for x in a.rows.split(":"))
    else:
        i, n = (int(x) for x in a.shard.split("/"))
        if n != meta["shards"]:
            raise SystemExit(f"shards {n} differ from the official {meta['shards']}")
        lo, hi = bounds(meta["summary"]["rows"], n)[i]
    path = record_path(a)
    recs = read(path)
    done = {r["row_index"] for r in recs if r.get("kind") == "request"}
    checked = {(r["row_index"], r["hook"]) for r in recs if r.get("kind") == "hook_check"}
    check_rows = list(range(lo + WARMUP, min(hi, lo + WARMUP + a.hook_check)))
    finished = len(done) >= hi - lo and all((k, h) in checked for k in check_rows for h in (True, False))
    return SimpleNamespace(lo=lo, hi=hi, path=path, done=done, checked=checked, check_rows=check_rows,
                           finished=finished, meta=meta)


def estimate(a):
    """per part on --nodes nodes, from the official estimates of the shards still owed: each free gcd takes the next
    shard, longest first, as the claim loop does"""
    doc = read_parts(a.parts)
    parts = doc["tasks"] if doc else split(all_tasks(), a.parts)
    check_split(parts)
    print(f"{a.parts} part(s) on {a.nodes} node(s) each ({'logs/parts.json' if doc else 'not yet written'})")
    print(f"{'part':>4} {'shards':>7} {'gcd-h':>8} {'wall h':>7} {'node-h':>7} {'idle node-h':>11} {'tail h':>7}")
    tot = [0, 0.0, 0.0, 0.0, 0.0]
    for j, part in enumerate(parts):
        free, owed = [0.0] * (8 * a.nodes), 0
        for task in part:
            p = progress(parse_args(task["line"].split()))
            if p.finished:
                continue
            owed += 1
            s = min(range(len(free)), key=free.__getitem__)
            free[s] += (task["est_s"] - LOAD_S) * (1 - len(p.done) / task["rows"]) + LOAD_S
        work, wall = sum(free) / 3600, max(free) / 3600
        idle, tail = wall * a.nodes - work / 8, wall - min(free) / 3600
        print(f"{j:>4} {owed:>7} {work:>8.1f} {wall:>7.2f} {wall * a.nodes:>7.1f} {idle:>11.1f} {tail:>7.2f}")
        tot = [tot[0] + owed, tot[1] + work, max(tot[2], wall), tot[3] + wall * a.nodes, tot[4] + idle]
    limit = min(48, math.ceil(2 * tot[2]))
    print(f" all {tot[0]:>7} {tot[1]:>8.1f} {tot[2]:>7.2f} {tot[3]:>7.1f} {tot[4]:>11.1f}")
    print(f"warm-up rows add about 0.8 %; suggested --time={limit:02d}:00:00 (about twice the longest wall, "
          f"standard-g maximum 48 h)")
    return 0


def claim(a, context):
    """take the next unfinished task of this part under an exclusive lock, measure it and exit; 3 when none is left"""
    tasks = part_tasks(a)
    me = (f"{os.environ.get('SLURM_JOB_ID', 'local')} {socket.gethostname()} {os.getpid()} "
          f"gcd{os.environ.get('ROCR_VISIBLE_DEVICES')}")
    for task in tasks:
        t = parse_args(task["line"].split())
        if progress(t).finished:
            continue
        lock = lock_path(t)
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, (me + "\n").encode())
        os.close(fd)
        if progress(t).finished:  # finished between the check and the lock
            lock.unlink()
            continue
        print(f"claimed {task['line']} ({me})", flush=True)
        # an error keeps the lock, so no other slot of this job retries the shard; the next job clears it
        measure(t, context)
        lock.unlink()
        return 0
    print("nothing left to claim", flush=True)
    return 3


def clear_stale(a):
    """remove this part's locks whose job is not among the live ids, or is this job (at its start no slot runs yet)"""
    live = a.live.split()
    if not live:
        print("no answer from squeue: no lock cleared")
        return 0
    me, n = os.environ.get("SLURM_JOB_ID"), 0
    # a lock written after the squeue snapshot belongs to a job that started since, so it is never stale here
    snap = float(os.environ.get("LFT_SNAPSHOT", "inf")) - 30
    for task in part_tasks(a):
        lock = lock_path(parse_args(task["line"].split()))
        try:
            holder = (lock.read_text().split() or ["?"])[0]
            if lock.stat().st_mtime > snap:
                continue
        except FileNotFoundError:
            continue
        if holder not in live or holder == me:
            lock.unlink(missing_ok=True)
            n += 1
            print(f"cleared {lock.name} (job {holder})")
    print(f"{n} stale lock(s) cleared")
    return 0


def status(a):
    """progress of all parts from records/ and squeue only; totals come from logs/parts.json"""
    doc = read_parts()
    if doc is None:
        raise SystemExit(f"no {PARTS.name} yet: nothing has been split or submitted")
    by_mode = defaultdict(lambda: [0, 0, 0, 0])
    left, per_part = defaultdict(int), []
    for part in doc["tasks"]:
        rows = got_rows = 0
        for task in part:
            t = parse_args(task["line"].split())
            p = record_path(t)
            got = min(task["rows"], p.read_bytes().count(b'"kind": "request"')) if p.exists() else 0
            m = by_mode[task["mode"]]
            m[0], m[1], m[2], m[3] = m[0] + 1, m[1] + (got >= task["rows"]), m[2] + task["rows"], m[3] + got
            left[(t.mode, t.model, t.arm)] += task["rows"] - got
            rows, got_rows = rows + task["rows"], got_rows + got
        per_part.append(100 * got_rows / rows)
    print(f"latency full test: {doc['parts']} parts\n{'mode':<16} {'shards done/total':>18} {'rows done/total':>22} {'done':>7}")
    tot = [0, 0, 0, 0]
    for mode in MODES:
        m = by_mode[mode]
        tot = [x + y for x, y in zip(tot, m)]
        print(f"{mode:<16} {m[1]:>8}/{m[0]:<9} {m[3]:>10}/{m[2]:<11} {100 * m[3] / m[2]:>6.1f} %")
    print(f"{'all':<16} {tot[1]:>8}/{tot[0]:<9} {tot[3]:>10}/{tot[2]:<11} {100 * tot[3] / tot[2]:>6.1f} %")
    print("parts, % of rows: " + "  ".join(f"{j}: {v:.1f}" for j, v in enumerate(per_part)))
    pilot = list((HERE / "records" / "pilot").rglob("*.jsonl"))
    if pilot:
        data = b"".join(p.read_bytes() for p in pilot)
        req, hook = data.count(b'"kind": "request"'), data.count(b'"kind": "hook_check"')
        print(f"pilot: {req} of {PILOT['rows'] * len(ARMS)} requests, {hook} hook-check requests")
    try:
        q = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T|%M|%D|%R"],
                           capture_output=True, text=True, timeout=30).stdout
        jobs = [j.split("|") for j in q.splitlines() if j.split("|")[1].startswith("lft-")]
    except (OSError, subprocess.SubprocessError, IndexError):
        jobs = None
    print("slurm jobs of this run: " + ("squeue unavailable" if jobs is None else f"{len(jobs)}"))
    for j in jobs or []:
        print(f"  {j[0]:<10} {j[1]:<18} {j[2]:<10} {j[3]:>11}  {j[4]:>3} node(s)  {j[5]}")
    # rough eta: rows left x observed mean latency per arm (its newest record file), over the gcds running now
    ms = {}
    for mode, model, arm in left:
        files = sorted((HERE / "records" / mode / model).glob(f"{arm}-shard-*.jsonl"), key=lambda p: p.stat().st_mtime)
        vals = [r["total_ms"] for r in read(files[-1]) if r.get("kind") == "request" and not r["warmup"]] if files else []
        if vals:
            ms[(mode, model, arm)] = statistics.mean(vals)
    gcds = 8 * sum(int(j[4]) for j in jobs or [] if j[2] == "RUNNING" and j[1].startswith("lft-part"))
    rows_left = sum(left.values())
    if not rows_left:
        print("eta: nothing left")
    elif ms and gcds:
        fill = statistics.mean(ms.values())
        hours = sum(n * ms.get(k, fill) for k, n in left.items()) / gcds / 3.6e6
        print(f"rough eta: {hours:.1f} h for {rows_left} rows left on the {gcds} gcds running now "
              f"(observed latency of {len(ms)} arm(s), their mean for the rest)")
    else:
        print(f"rough eta: not yet ({rows_left} rows left; needs running part jobs and recorded requests)")
    return 0


def dry_run(a):
    """prompts as the run would load them, checked against the official records (row count and row ids)"""
    for mode in MODES if not a.mode else (a.mode,):
        rows = prompts(mode)
        keys = sorted({k for r in rows for k in r})
        meta = official(mode, a.model or "gemma", a.arm or "base")
        ids = []
        with open(OFFICIAL / mode / (a.model or "gemma") / f"{a.arm or 'base'}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                ids.append((r["row_index"], r["row_id"]))
        same = len(ids) == len(rows) and all(i == k and rid == rows[k]["order_id"] for k, (i, rid) in enumerate(sorted(ids)))
        print(f"{mode}: {len(rows)} prompts loaded, fields kept {keys}; official rows {meta['summary']['rows']}, "
              f"row ids identical in order: {same}")
    return 0


def make_hook():
    # copied from run_diagnostic.make_hook: time of the first generated token, never stops generation
    import torch
    from transformers import StoppingCriteria

    class FirstToken(StoppingCriteria):
        def __init__(self):
            self.t, self.calls = None, 0

        def __call__(self, input_ids, scores, **kw):
            self.calls += 1
            if self.t is None:
                torch.cuda.synchronize()
                self.t = time.perf_counter()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    return FirstToken()


def time_one(model, tokenizer, text_tok, row, stops, hook=True):
    """the official timed region (generate between two synchronisations) plus the first-token hook"""
    import torch
    from transformers import StoppingCriteriaList
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = text_tok(prompt, return_tensors="pt").to(model.device)
    n_in = int(encoded["input_ids"].shape[1])
    h = make_hook() if hook else None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=text_tok.pad_token_id,
                             **({"eos_token_id": list(stops)} if stops else {}),
                             **({"stopping_criteria": StoppingCriteriaList([h])} if hook else {}))
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    # batch size 1 never pads: every generated id counts, the stop token included
    ids = [int(t) for t in out[0][n_in:].tolist()]
    n_out = len(ids)
    ttft = (h.t - t0) if h is not None and h.t is not None else None
    stop_set = set(stops or stop_rule.generation_stop_ids(model))
    return {"input_tokens": n_in, "output_tokens": n_out, "total_ms": round(total * 1000, 3),
            "ttft_ms": round(ttft * 1000, 3) if ttft is not None else None,
            "tpot_ms": round((total - ttft) * 1000 / (n_out - 1), 4) if ttft is not None and n_out > 1 else None,
            "hook_calls": h.calls if h is not None else None, "hit_cap": n_out >= MAX_NEW_TOKENS,
            "stopped_on_stop_id": bool(ids) and ids[-1] in stop_set}


def load(ckpt, mode, model_name):
    """the official load path; (model, tokenizer, text tokenizer, stop ids, session fields)"""
    from unsloth import FastLanguageModel
    import torch
    window = model_registry.MODES[mode]["window"]
    t = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=ckpt, revision=None, max_seq_length=window,
                                                         load_in_4bit=False, dtype=None)
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token
    stops, stop_info = stop_rule.stop_ids(model, tokenizer, model_name, mode)
    load_s = round(time.perf_counter() - t, 1)
    import peft
    import transformers
    import unsloth
    info = {"load_s": load_s, "model_class": type(model).__name__,
            "attention": getattr(model.config, "_attn_implementation", None), "window": window,
            "stop_rule": stop_info, "device": torch.cuda.get_device_name(0),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__, "unsloth": getattr(unsloth, "__version__", "?")}}
    return model, tokenizer, text_tok, stops, info


def measure(a, context):
    p = progress(a)
    if p.finished:
        print(f"{p.path.relative_to(HERE)}: complete")
        return 0
    rows, meta, lo, hi, path, done, checked = prompts(a.mode), p.meta, p.lo, p.hi, p.path, p.done, p.checked
    if not a.rows and len(rows) != meta["summary"]["rows"]:
        raise SystemExit(f"rows {len(rows)} differ from the official {meta['summary']['rows']}")
    shard = None if a.rows else tuple(int(x) for x in a.shard.split("/"))
    ckpt, epoch = checkpoint(a.mode, a.model, a.arm, context)
    if ckpt != meta["checkpoint"] or epoch != meta["epoch"]:
        raise SystemExit(f"checkpoint {ckpt} epoch {epoch} differs from the official {meta['checkpoint']} {meta['epoch']}")
    model, tokenizer, text_tok, stops, info = load(ckpt, a.mode, a.model)
    node = {"host": socket.gethostname(), "gcd": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "job": os.environ.get("SLURM_JOB_ID")}
    base = {"mode": a.mode, "model": a.model, "arm": a.arm, "epoch": epoch, "shard": shard[0] if shard else None,
            "shards": shard[1] if shard else None, "node": node["host"], "gcd": node["gcd"], "job": node["job"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")  # end a line cut by a killed job

    def write(rec):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    write({"kind": "session", **base, "checkpoint": ckpt, "rows": [lo, hi], "done_before": len(done), **info})
    print(f"{a.mode} {a.model} {a.arm} epoch {epoch} rows [{lo}, {hi}): {len(done)} done, loaded in {info['load_s']} s, "
          f"{info['model_class']}, node {node}", flush=True)
    first = True
    for index in range(lo, hi):
        if index in done:
            continue
        m = time_one(model, tokenizer, text_tok, rows[index], stops, hook=a.variant != "C")
        # warm-up: the first rows of the shard and the first row after any resume
        warm = (index - lo) < WARMUP or first
        first = False
        write({"kind": "request", **base, "row_index": index, "row_id": rows[index]["order_id"], "warmup": warm, **m})
        if (index - lo + 1) % 50 == 0:
            print(f"  {index - lo + 1}/{hi - lo} timed, last {m['total_ms']:.0f} ms", flush=True)
    for k, index in enumerate(p.check_rows):
        # hook on and off on the same row, the order alternating
        for hook in ((True, False) if k % 2 == 0 else (False, True)):
            if (index, hook) not in checked:
                write({"kind": "hook_check", **base, "row_index": index, "hook": hook,
                       **time_one(model, tokenizer, text_tok, rows[index], stops, hook=hook)})
    print(f"{path.relative_to(HERE)}: done", flush=True)
    return 0


def main(argv=None):
    a = parse_args(argv)
    if a.make_parts:
        return make_parts(a.parts)
    if a.make_check:
        return make_check()
    if a.list:
        return estimate(a)
    if a.status:
        return status(a)
    if a.dry_run:
        return dry_run(a)
    if a.clear_stale:
        return clear_stale(a)
    context = execution.require("rocm", "lumi", "latency_full_test")
    if a.claim:
        return claim(a, context)
    if read_parts():
        raise SystemExit(f"{PARTS.name} exists: run parts through run.sbatch --part i/K (single runs are replaced)")
    if not (a.mode and a.model and a.arm and (a.shard or a.rows)):
        raise SystemExit("give --mode, --model, --arm and --shard i/n (or --rows lo:hi)")
    return measure(a, context)


if __name__ == "__main__":
    sys.exit(main())
