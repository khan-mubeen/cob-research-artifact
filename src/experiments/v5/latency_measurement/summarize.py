"""summary of the full test split latency run; rerunnable on partial results

reads records/<mode>/<model>/<arm>-shard-*.jsonl and writes into summary/:
  arms.csv, arms.md                    per arm: rows, tokens (all rows), ttft, tpot and total latency
                                       (warm-up rows excluded), warm-up mean and warm-up/steady ratio
  cells.csv, cells.md                  per cell, base against fine-tuned on the rows steady in both arms, and
                                       the exact decomposition of the latency difference
  ttft_vs_input.csv, ttft_vs_input.md  per arm: least squares fit of ttft on input tokens, ttft share of total
  repro.csv, repro.md                  per arm: new against official latency and output tokens, same rows
  table1..table4 (.md, .tex)           the paper tables
  summary.md                           status and at most five conclusion lines
with --repeat-check it compares the repeat-check shards (repeat_check/B and C) with the first run and the official
records and writes repeat_check/check.csv and check.md, exiting with status 1 when a shard fails.
with --pilot it reads records/pilot/ and writes pilot.csv and pilot.md instead; with --pilot --check it also exits
with status 1 when a pilot check fails (output tokens more than 1 % off the official records on the same rows, the
hook not firing on every request, a ttft identity error beyond rounding, or missing rows).

with --export DIR it writes the per-request records as one gzip csv per arm (counts and timings only), and
with --publish DIR --out DIR it rebuilds the published tables (latency_breakdown_*) from those csv files.
Add --skip-original-profile when the original profiling JSONL files are unavailable; only the
original-profile comparison is omitted, and all aggregation rules stay the same.

  PYTHONDONTWRITEBYTECODE=1 python3.11 summarize.py [--pilot [--check]]
  PYTHONDONTWRITEBYTECODE=1 python3.11 summarize.py --export reports/v5/evaluation/latency_records
  PYTHONDONTWRITEBYTECODE=1 python3.11 summarize.py --publish reports/v5/evaluation/latency_records --out DIR
"""
import argparse
import csv
import gzip
import io
import json
import math
import statistics
from collections import namedtuple
from types import SimpleNamespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "config.py").exists())
OFFICIAL = REPO / "reports/v5/results/latency"
OUT = HERE / "summary"
PREFIX = ""  # latency_breakdown_ when the published tables are written
RECORD_COLS = ["row_index", "row_id", "input_tokens", "output_tokens", "total_ms", "ttft_ms", "tpot_ms", "warmup"]
MODES = ("sealed", "reverse_auction", "supervisor")
MODELS = ("gemma", "granite", "ministral", "phi", "qwen", "smollm3")
ARMS = ("base", "selected")
NAME = {"gemma": "Gemma", "granite": "Granite", "ministral": "Ministral", "phi": "Phi", "qwen": "Qwen",
        "smollm3": "SmolLM3"}
TITLE = {"sealed": "Sealed bid", "reverse_auction": "Reverse auction", "supervisor": "Supervisor"}
FLAG_PCT = 3.0  # latency difference that is reported, never a stop
OUT_LIMIT_PCT = {"smollm3": 3.0, "qwen": 3.0}  # output-token limit per model against the official records; others 0.5
PILOT_ROWS = 400  # per arm, as measure.PILOT
LABEL = ("Whole test split, one request at a time, adapter attached and not merged; latency statistics exclude warm-up "
         "rows except in Table 3, and token means use every row.")
Req = namedtuple("Req", "warm n_in n_out ttft tpot total row_id cap calls")

# paper tables: (stem, header, source table, [(field, arm or None, decimals)], notes)
PAPER = [
    ("table1_tokens", ["Mode", "Model", "Input", "Output base", "Output fine-tuned"], "arms",
     [("in_tokens", "base", 1), ("out_tokens", "base", 1), ("out_tokens", "selected", 1)],
     ["Mean tokens per request over every row; output tokens include the stop token."]),
    ("table2_times", ["Mode", "Model", "TTFT base", "TTFT fine-tuned", "TPOT base", "TPOT fine-tuned",
                      "Latency base", "Latency fine-tuned", "TPOT ratio"], "cells",
     [("ttft_base", None, 1), ("ttft_ft", None, 1), ("tpot_base", None, 1), ("tpot_ft", None, 1),
      ("total_base", None, 1), ("total_ft", None, 1), ("tpot_ratio", None, 2)],
     ["Times in ms. TPOT is the mean over requests of (total - TTFT)/(output tokens - 1); ratio is fine-tuned/base."]),
    ("table3_warm_up", ["Mode", "Model", "Warm-up base", "Warm-up fine-tuned", "Warm-up/steady base",
                        "Warm-up/steady fine-tuned", "Warm-up rows base", "Warm-up rows fine-tuned"], "arms",
     [("warmup_mean", "base", 1), ("warmup_mean", "selected", 1), ("warmup_ratio", "base", 2),
      ("warmup_ratio", "selected", 2), ("warmup_rows", "base", 0), ("warmup_rows", "selected", 0)],
     ["Mean total latency in ms of the warm-up rows (the first three rows of every shard and the first row",
      "after any resume), and its ratio to the steady mean."]),
    ("table4_decomposition", ["Mode", "Model", "Latency difference", "TTFT part", "Output length effect",
                              "Per token effect"], "cells",
     [("difference", None, 1), ("ttft_part", None, 1), ("length_effect", None, 1), ("per_token_effect", None, 1)],
     ["Fine-tuned minus base, ms: difference = TTFT part + output length effect + per token effect, with",
      "T = (L - F)/(n - 1) as a ratio of means; the terms may differ from the sum in the last digit by rounding."]),
]


def read_jsonl(path):
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def load(folder, pattern):
    """({row_index: Req}, [hook-check records], {stop ids of the sessions}) for the files matching pattern"""
    reqs, hook, stops = {}, [], set()
    for p in sorted(folder.glob(pattern)):
        for r in read_jsonl(p):
            if r.get("kind") == "request":
                reqs.setdefault(r["row_index"], Req(r["warmup"], r["input_tokens"], r["output_tokens"], r["ttft_ms"],
                                                    r["tpot_ms"], r["total_ms"], r["row_id"], r["hit_cap"],
                                                    r["hook_calls"]))
            elif r.get("kind") == "hook_check":
                hook.append(r)
            elif r.get("kind") == "session":
                stops.add(json.dumps(r["stop_rule"].get("stop_ids_used")))
    return reqs, hook, stops


def load_published(folder, mode, model, arm):
    """{row_index: Req} from a published records file, <mode>_<model>_<arm>.csv.gz"""
    reqs = {}
    with gzip.open(Path(folder) / f"{mode}_{model}_{arm}.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            n = int(r["output_tokens"])
            reqs[int(r["row_index"])] = Req(r["warmup"] == "1", int(r["input_tokens"]), n, float(r["ttft_ms"]),
                                           float(r["tpot_ms"]) if r["tpot_ms"] else None, float(r["total_ms"]),
                                           r["row_id"], n >= 200, None)
    return reqs


def export(folder):
    """one gzip csv per arm from the run's records, rows in order; fixed gzip header, so the bytes are reproducible"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for model in MODELS:
            for arm in ARMS:
                reqs, _, _ = load(HERE / "records" / mode / model, f"{arm}-shard-*.jsonl")
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\n")
                w.writerow(RECORD_COLS)
                for i in sorted(reqs):
                    q = reqs[i]
                    w.writerow([i, q.row_id, q.n_in, q.n_out, q.total, q.ttft, "" if q.tpot is None else q.tpot,
                                int(q.warm)])
                with open(folder / f"{mode}_{model}_{arm}.csv.gz", "wb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as g:
                        g.write(buf.getvalue().encode())
                print(f"{mode} {model} {arm}: {len(reqs)} rows")
    return 0


def official_rows(mode, model, arm, want):
    """{row_index: (warmup, ms, output tokens, row id, input tokens)} from the official records, for the rows in want"""
    out = {}
    with open(OFFICIAL / mode / model / f"{arm}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["row_index"] in want:
                out[r["row_index"]] = (r["warmup"], r["ms_per_decision"], r["output_tokens"], r["row_id"], r["input_tokens"])
    return out


def meta(mode, model, arm):
    return json.loads((OFFICIAL / mode / model / f"{arm}.meta.json").read_text())


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def stats(v):
    s = sorted(x for x in v if x is not None)
    if not s:
        return {"mean": None, "std": None, "median": None, "p95": None}
    return {"mean": statistics.mean(s), "std": statistics.stdev(s) if len(s) > 1 else 0.0,
            "median": statistics.median(s), "p95": s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]}


def rnd(x, d):
    return None if x is None else round(x, d)


def pct(new, old):
    return None if new is None or not old else 100 * (new - old) / old


def write_pair(stem, rows, cols, out=None, md=True):
    """csv and markdown with the same cells"""
    out, stem = out or OUT, PREFIX + stem
    with open(out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if not md:
        return
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in cols) + " |" for r in rows]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n")


def paper_rows(spec, tables):
    """formatted cells of one paper table, in the row order of the earlier paper table"""
    _, head, source, fields, _ = spec
    src = tables[source]
    out = []
    # unrounded values (_raw) where kept, so a shown digit is never rounded twice
    for mode in MODES:
        for model in MODELS:
            if source == "arms":
                if not all((mode, model, a) in src for a in ARMS):
                    continue
                vals = [src[(mode, model, arm)].get("_raw", src[(mode, model, arm)])[f] for f, arm, _ in fields]
            else:
                if (mode, model) not in src:
                    continue
                vals = [src[(mode, model)].get("_raw", src[(mode, model)])[f] for f, _, _ in fields]
            out.append([TITLE[mode], NAME[model]] + ["" if v is None or v == "" else f"{float(v):.{d}f}"
                                                     for v, (_, _, d) in zip(vals, fields)])
    return out


def write_paper(spec, table, extra):
    stem, head, _, _, notes = spec
    stem = PREFIX + stem
    notes = [LABEL] + notes + extra
    md = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
    md += ["| " + " | ".join(r) + " |" for r in table]
    (OUT / f"{stem}.md").write_text("\n".join(md + [""] + notes) + "\n")
    tex = ["\\begin{tabular}{ll" + "r" * (len(head) - 2) + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in table]
    tex += ["\\bottomrule"] + [f"\\multicolumn{{{len(head)}}}{{l}}{{\\footnotesize {n}}} \\\\" for n in notes]
    (OUT / f"{stem}.tex").write_text("\n".join(tex + ["\\end{tabular}"]) + "\n")


def arm_row(mode, model, arm, reqs, status):
    steady = [r for r in reqs.values() if not r.warm]
    warm = [r.total for r in reqs.values() if r.warm]
    row = {"mode": mode, "model": model, "arm": arm, "status": status, "rows": len(reqs), "rows_steady": len(steady),
           "in_tokens": rnd(mean(r.n_in for r in reqs.values()), 4),
           "out_tokens": rnd(mean(r.n_out for r in reqs.values()), 4)}
    for k in ("ttft", "tpot", "total"):
        for s, v in stats(getattr(r, k) for r in steady).items():
            row[f"{k}_{s}"] = rnd(v, 3 if k == "tpot" else 2)
    row["warmup_rows"] = len(warm)
    steady_mean = mean(r.total for r in steady)
    row["warmup_mean"] = rnd(mean(warm), 2)
    row["warmup_ratio"] = rnd(mean(warm) / steady_mean, 3) if warm and steady_mean else None
    row["cap_hits"] = sum(r.cap for r in reqs.values())
    # unrounded, for the paper tables only (the csv writer ignores this key)
    row["_raw"] = {"in_tokens": mean(r.n_in for r in reqs.values()), "out_tokens": mean(r.n_out for r in reqs.values()),
                   "warmup_mean": mean(warm), "warmup_rows": len(warm),
                   "warmup_ratio": mean(warm) / steady_mean if warm and steady_mean else None}
    return row


def fit_row(mode, model, arm, reqs):
    pts = [(r.n_in, r.ttft) for r in reqs.values() if not r.warm and r.ttft is not None]
    row = {"mode": mode, "model": model, "arm": arm, "rows_steady": len(pts)}
    if len(pts) < 3:
        return row
    mx, my = mean(p[0] for p in pts), mean(p[1] for p in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    syy = sum((y - my) ** 2 for _, y in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx if sxx else None
    total = mean(r.total for r in reqs.values() if not r.warm and r.ttft is not None)
    row.update({"in_min": min(p[0] for p in pts), "in_max": max(p[0] for p in pts),
                "slope_ms_per_1k": rnd(slope * 1000, 2) if slope is not None else None,
                "intercept_ms": rnd(my - slope * mx, 2) if slope is not None else None,
                "r2": rnd(sxy * sxy / (sxx * syy), 3) if sxx and syy else None,
                "ttft_share_pct": rnd(100 * my / total, 2)})
    return row


def repro_row(mode, model, arm, reqs, m, stops=None):
    off = official_rows(mode, model, arm, set(reqs))
    both = [i for i in reqs if i in off]
    steady = [i for i in both if not reqs[i].warm and not off[i][0]]
    lat_new, lat_off = mean(reqs[i].total for i in steady), mean(off[i][1] for i in steady)
    out_new, out_off = mean(reqs[i].n_out for i in both), mean(off[i][2] for i in both)
    d_lat, d_out = pct(lat_new, lat_off), pct(out_new, out_off)
    # each source's own steady mean, as each one publishes it
    own_new = mean(q.total for q in reqs.values() if not q.warm)
    own_off = mean(off[i][1] for i in both if not off[i][0])
    # the rule: same inputs and stop ids as the official run, output tokens within the model's limit
    in_mis = sum(reqs[i].n_in != off[i][4] for i in both)
    stop_ok = stops is None or stops == {json.dumps(m["generation_stop_rule"]["stop_ids_used"])}
    limit = OUT_LIMIT_PCT.get(model, 0.5)
    ok = bool(both) and in_mis == 0 and stop_ok and d_out is not None and abs(d_out) <= limit
    return {"mode": mode, "model": model, "arm": arm, "rows_new": len(reqs), "rows_official": m["summary"]["rows"],
            "row_id_mismatches": sum(reqs[i].row_id != off[i][3] for i in both), "rows_compared": len(both),
            "rows_compared_latency": len(steady), "latency_new": rnd(lat_new, 2), "latency_official": rnd(lat_off, 2),
            "latency_diff_pct": rnd(d_lat, 2), "out_new": rnd(out_new, 3), "out_official": rnd(out_off, 3),
            "out_diff_pct": rnd(d_out, 3), "in_mismatches": in_mis, "stop_ids_match": stop_ok,
            "out_limit_pct": limit, "rule": "pass" if ok else "fail",
            "out_rows_differ": sum(reqs[i].n_out != off[i][2] for i in both),
            "out_rows_differ_pct": rnd(100 * sum(reqs[i].n_out != off[i][2] for i in both) / len(both), 2) if both else None,
            "latency_original_own": rnd(own_off, 2), "latency_new_own": rnd(own_new, 2),
            "latency_own_diff_pct": rnd(pct(own_new, own_off), 2),
            "flag": "yes" if d_lat is not None and abs(d_lat) > FLAG_PCT else "",
            # unrounded, for the comparison table (the csv writer ignores this key)
            "_raw": {"out_official": out_off, "out_new": out_new, "latency_original_own": own_off,
                     "latency_new_own": own_new}}


def cell_row(mode, model, base, ft, epoch):
    common = [i for i in base if i in ft and not base[i].warm and not ft[i].warm]
    if not common:
        return None
    g = {}
    for k, d in (("b", base), ("f", ft)):
        g[k] = {"n": mean(d[i].n_out for i in common), "F": mean(d[i].ttft for i in common),
                "L": mean(d[i].total for i in common), "tpot": mean(d[i].tpot for i in common),
                "in": mean(d[i].n_in for i in common)}
        g[k]["T"] = (g[k]["L"] - g[k]["F"]) / (g[k]["n"] - 1)
    b, f = g["b"], g["f"]
    diff = f["L"] - b["L"]
    terms = (f["F"] - b["F"], (f["n"] - b["n"]) * b["T"], (f["n"] - 1) * (f["T"] - b["T"]))
    return {"mode": mode, "model": model, "epoch": epoch, "rows_common": len(common),
            "in_base": rnd(b["in"], 2), "in_ft": rnd(f["in"], 2),
            "out_base": rnd(b["n"], 3), "out_ft": rnd(f["n"], 3), "out_ratio": rnd(f["n"] / b["n"], 3),
            "ttft_base": rnd(b["F"], 2), "ttft_ft": rnd(f["F"], 2), "ttft_ratio": rnd(f["F"] / b["F"], 3),
            "tpot_base": rnd(b["tpot"], 3), "tpot_ft": rnd(f["tpot"], 3), "tpot_ratio": rnd(f["tpot"] / b["tpot"], 3),
            "total_base": rnd(b["L"], 2), "total_ft": rnd(f["L"], 2), "total_ratio": rnd(f["L"] / b["L"], 3),
            "T_base": rnd(b["T"], 3), "T_ft": rnd(f["T"], 3), "difference": rnd(diff, 2),
            "ttft_part": rnd(terms[0], 2), "length_effect": rnd(terms[1], 2), "per_token_effect": rnd(terms[2], 2),
            "residual": round(diff - sum(terms), 6) + 0.0,  # float noise is far below 1e-6; + 0.0 drops a negative zero
            # unrounded, for the paper tables only (the csv writer ignores this key)
            "_raw": {"ttft_base": b["F"], "ttft_ft": f["F"], "tpot_base": b["tpot"], "tpot_ft": f["tpot"],
                     "total_base": b["L"], "total_ft": f["L"], "tpot_ratio": f["tpot"] / b["tpot"], "difference": diff,
                     "ttft_part": terms[0], "length_effect": terms[1], "per_token_effect": terms[2]}}


ARM_COLS = ["mode", "model", "arm", "status", "rows", "rows_steady", "in_tokens", "out_tokens"] + [
    f"{k}_{s}" for k in ("ttft", "tpot", "total") for s in ("mean", "std", "median", "p95")] + [
    "warmup_rows", "warmup_mean", "warmup_ratio", "cap_hits"]
CELL_COLS = ["mode", "model", "epoch", "rows_common", "in_base", "in_ft", "out_base", "out_ft", "out_ratio",
             "ttft_base", "ttft_ft", "ttft_ratio", "tpot_base", "tpot_ft", "tpot_ratio", "total_base", "total_ft",
             "total_ratio", "T_base", "T_ft", "difference", "ttft_part", "length_effect", "per_token_effect", "residual"]
FIT_COLS = ["mode", "model", "arm", "rows_steady", "in_min", "in_max", "slope_ms_per_1k", "intercept_ms", "r2",
            "ttft_share_pct"]
REPRO_COLS = ["mode", "model", "arm", "rows_new", "rows_official", "row_id_mismatches", "rows_compared",
              "rows_compared_latency", "latency_new", "latency_official", "latency_diff_pct", "out_new", "out_official",
              "out_diff_pct", "in_mismatches", "stop_ids_match", "out_limit_pct", "rule", "out_rows_differ_pct", "flag"]
PILOT_COLS = ["arm", "rows", "out_new", "out_official", "out_diff_pct", "latency_new", "latency_official",
              "latency_diff_pct", "max_identity_error_ms", "identity_within_rounding", "tpot_mean_reconstruction_pct", "hook_fired_every_request",
              "hook_calls_equal_output_tokens", "hook_pairs", "hook_on_ms", "hook_off_ms", "hook_overhead_pct"]


def pilot(check=False):
    rows = []
    for arm in ARMS:
        reqs, hook, _ = load(HERE / "records" / "pilot" / "sealed" / "gemma", f"{arm}-rows-*.jsonl")
        if not reqs:
            continue
        r = repro_row("sealed", "gemma", arm, reqs, meta("sealed", "gemma", arm))
        steady = [q for q in reqs.values() if not q.warm and q.tpot is not None]
        # ttft + (n - 1) x tpot = total up to rounding: ttft and total kept to 0.001 ms, tpot to 0.0001 ms
        err = [(abs(q.ttft + (q.n_out - 1) * q.tpot - q.total), 0.001 + 0.00005 * (q.n_out - 1) + 1e-9)
               for q in reqs.values() if q.ttft is not None and q.tpot is not None]
        ident = max(e for e, _ in err) if err else None
        recon = pct(mean(q.ttft for q in steady) + (mean(q.n_out for q in steady) - 1) * mean(q.tpot for q in steady),
                    mean(q.total for q in steady)) if steady else None
        pairs = {}
        for h in hook:
            pairs.setdefault(h["row_index"], {})[h["hook"]] = h["total_ms"]
        pairs = [p for p in pairs.values() if True in p and False in p]
        on, off = mean(p[True] for p in pairs), mean(p[False] for p in pairs)
        fired = all(q.ttft is not None and q.calls >= 1 for q in reqs.values())
        per_token = all(q.calls == q.n_out for q in reqs.values()) and all(
            h["hook_calls"] == h["output_tokens"] for h in hook if h["hook"])
        rows.append({"arm": arm, "rows": len(reqs), "out_new": r["out_new"], "out_official": r["out_official"],
                     "out_diff_pct": r["out_diff_pct"], "latency_new": r["latency_new"],
                     "latency_official": r["latency_official"], "latency_diff_pct": r["latency_diff_pct"],
                     "max_identity_error_ms": rnd(ident, 4), "identity_within_rounding": bool(err) and all(
                         e <= tol for e, tol in err), "tpot_mean_reconstruction_pct": rnd(recon, 3),
                     "hook_fired_every_request": fired, "hook_calls_equal_output_tokens": per_token, "hook_pairs": len(pairs), "hook_on_ms": rnd(on, 2),
                     "hook_off_ms": rnd(off, 2), "hook_overhead_pct": rnd(pct(on, off), 3)})
    OUT.mkdir(exist_ok=True)
    write_pair("pilot", rows, PILOT_COLS)
    fails = [f"{a}: no records" for a in ARMS if a not in {r["arm"] for r in rows}]
    for r in rows:
        if r["rows"] < PILOT_ROWS:
            fails.append(f"{r['arm']}: {r['rows']} of {PILOT_ROWS} rows")
        if r["out_diff_pct"] is None or abs(r["out_diff_pct"]) > 1:
            fails.append(f"{r['arm']}: output tokens {r['out_diff_pct']} % off the official records (limit 1 %)")
        if not r["hook_fired_every_request"]:
            fails.append(f"{r['arm']}: the hook did not fire on every request")
        if not r["identity_within_rounding"]:
            fails.append(f"{r['arm']}: ttft identity error {r['max_identity_error_ms']} ms exceeds rounding")
    verdict = "Pilot check: " + ("fail: " + "; ".join(fails) if fails else "pass") + "."
    with open(OUT / "pilot.md", "a") as f:
        f.write("\n" + verdict + "\n")
    print(verdict)
    return 1 if check and fails else 0


CHECK_COLS = ["shard", "rows"] + [f"{k}_{x}" for x in ("AO", "BA", "CA", "BO", "CO") for k in ("share_pct", "mean_diff")] + [
    "latency_B", "latency_C", "pass", "label"]


def repeat_check():
    """per check shard: share of rows with a different output-token count and mean signed difference for A vs O,
    B vs A, C vs A, B vs O and C vs O (A first run, B repeat as the main run, C repeat with the hook off, O official)"""
    root, rows = HERE / "repeat_check", []
    for line in json.loads((root / "shards.json").read_text()):
        a = line.split()
        mode, model, arm, (i, n) = a[1], a[3], a[5], (int(x) for x in a[7].split("/"))
        name = f"{mode}/{model}/{arm}-shard-{i:03d}-of-{n:03d}.jsonl"
        got = {k: load(d / Path(name).parent, Path(name).name)[0] for k, d in
               (("A", HERE / "records"), ("B", root / "B"), ("C", root / "C"))}
        off = official_rows(mode, model, arm, set(got["A"]))
        got["O"] = {k: SimpleNamespace(n_out=v[2]) for k, v in off.items()}
        r = {"shard": f"{mode} {model} {arm} {i}/{n}", "rows": len(got["B"])}
        for x, y in ("AO", "BA", "CA", "BO", "CO"):
            both = [k for k in got[x] if k in got[y]]
            d = [got[x][k].n_out - got[y][k].n_out for k in both]
            r[f"share_pct_{x}{y}"] = rnd(100 * sum(v != 0 for v in d) / len(d), 2) if d else None
            r[f"mean_diff_{x}{y}"] = rnd(mean(d), 3) if d else None
        r["latency_B"] = rnd(mean(q.total for q in got["B"].values() if not q.warm), 1)
        r["latency_C"] = rnd(mean(q.total for q in got["C"].values() if not q.warm), 1)
        ao, ba, ca = r["share_pct_AO"], r["share_pct_BA"], r["share_pct_CA"]
        complete = len(got["B"]) == len(got["A"]) == len(got["C"]) and None not in (ao, ba, ca)
        r["pass"] = complete and ca <= max(1.5 * ba, 0.5)
        r["label"] = ("incomplete" if not complete else "no difference to official" if ao == 0 else
                      "run-to-run variation" if ba >= 0.5 * ao else
                      "stable here, differs from official (software or environment)" if ba <= 0.1 * ao else "mixed")
        rows.append(r)
    write_pair("check", rows, CHECK_COLS, out=root)
    ok = bool(rows) and all(r["pass"] for r in rows)
    print(f"repeat check: {'PASS' if ok else 'FAIL'} ({sum(bool(r['pass']) for r in rows)} of {len(rows)} shards)")
    return 0 if ok else 1


VS_COLS = ["mode", "model", "arm", "rows", "out_original", "out_new", "out_rows_differ", "out_rows_differ_pct", "latency_original_ms",
           "latency_new_ms", "latency_diff_pct"]


def collect(records=None, compare_original=True):
    """per arm and per cell, from the run's records or (records=folder) from the published csv files"""
    arms, cells, fits, repros, status = [], [], [], [], {}
    n_req = n_all = 0
    for mode in MODES:
        for model in MODELS:
            data, epoch = {}, None
            for arm in ARMS:
                m = meta(mode, model, arm)
                n_all += m["summary"]["rows"]
                if records:
                    reqs, stops = load_published(records, mode, model, arm), None
                else:
                    reqs, _, stops = load(HERE / "records" / mode / model, f"{arm}-shard-*.jsonl")
                n_req += len(reqs)
                st = "complete" if len(reqs) == m["summary"]["rows"] else "partial" if reqs else "pending"
                status[(mode, model, arm)] = st
                if not reqs:
                    continue
                data[arm] = reqs
                epoch = m["epoch"] if arm == "selected" else epoch
                arms.append(arm_row(mode, model, arm, reqs, st))
                fits.append(fit_row(mode, model, arm, reqs))
                if compare_original:
                    repros.append(repro_row(mode, model, arm, reqs, m, stops))
            if len(data) == 2:
                c = cell_row(mode, model, data["base"], data["selected"], epoch)
                if c:
                    cells.append(c)
                    if abs(c["residual"]) > 1e-6:
                        print(f"decomposition does not sum for {mode} {model}: residual {c['residual']}")
    return arms, cells, fits, repros, status, n_req, n_all


def tables_of(arms, cells, status):
    tables = {"arms": {(r["mode"], r["model"], r["arm"]): r for r in arms},
              "cells": {(r["mode"], r["model"]): r for r in cells}}
    done = sum(v == "complete" for v in status.values())
    return tables, [] if done == len(status) else [f"Partial results: {done} of {len(status)} arms complete."]


def publish(records, out, compare_original=True):
    """the published tables from the published records: latency_breakdown_{arms,cells}.csv, the four paper tables
    (md, tex) and the comparison with the original profiling records (md, csv)"""
    global OUT, PREFIX
    OUT, PREFIX = Path(out), "latency_breakdown_"
    OUT.mkdir(parents=True, exist_ok=True)
    if not compare_original and any(OUT.glob("latency_breakdown_vs_original_profile.*")):
        raise SystemExit("use a separate output folder: it already contains an original-profile comparison")
    arms, cells, _, repros, status, _, _ = collect(records, compare_original=compare_original)
    write_pair("arms", arms, ARM_COLS, md=False)
    write_pair("cells", cells, CELL_COLS, md=False)
    tables, extra = tables_of(arms, cells, status)
    for spec in PAPER:
        write_paper(spec, paper_rows(spec, tables), extra)
    vs = [{"mode": r["mode"], "model": r["model"], "arm": r["arm"], "rows": r["rows_new"],
           "out_original": rnd(r["_raw"]["out_official"], 2), "out_new": rnd(r["_raw"]["out_new"], 2),
           "out_rows_differ": r["out_rows_differ"], "out_rows_differ_pct": r["out_rows_differ_pct"],
           "latency_original_ms": rnd(r["_raw"]["latency_original_own"], 1),
           "latency_new_ms": rnd(r["_raw"]["latency_new_own"], 1), "latency_diff_pct": r["latency_own_diff_pct"]}
          for r in repros]
    if compare_original:
        write_pair("vs_original_profile", vs, VS_COLS)
    else:
        print("skipped the original-profile comparison; no original profiling records were read")
    print(f"published {len(arms)} arms, {len(cells)} cells -> {OUT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--check", action="store_true", help="with --pilot: exit 1 when a pilot check fails")
    ap.add_argument("--repeat-check", action="store_true")
    ap.add_argument("--export", default="", help="write the per-request records as gzip csv files into this folder")
    ap.add_argument("--publish", default="", help="folder of published records; writes the published tables to --out")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-original-profile", action="store_true",
                    help="with --publish: omit the comparison requiring the original profiling JSONL files")
    a = ap.parse_args()
    if a.skip_original_profile and not a.publish:
        ap.error("--skip-original-profile requires --publish")
    if a.repeat_check:
        return repeat_check()
    if a.pilot:
        return pilot(a.check)
    if a.export:
        return export(a.export)
    if a.publish:
        return publish(a.publish, a.out or OUT, compare_original=not a.skip_original_profile)
    OUT.mkdir(exist_ok=True)
    arms, cells, fits, repros, status, n_req, n_all = collect()
    write_pair("arms", arms, ARM_COLS)
    write_pair("cells", cells, CELL_COLS)
    write_pair("ttft_vs_input", fits, FIT_COLS)
    write_pair("repro", repros, REPRO_COLS)
    tables, extra = tables_of(arms, cells, status)
    for spec in PAPER:
        write_paper(spec, paper_rows(spec, tables), extra)
    counts = {s: sum(v == s for v in status.values()) for s in ("complete", "partial", "pending")}
    md = ["# Full test split latency: summary", "",
          f"Arms: {counts['complete']} complete, {counts['partial']} partial, {counts['pending']} pending; "
          f"{n_req} of {n_all} requests recorded. Tables: arms, cells, ttft_vs_input, repro, table1 to table4.", ""]
    if cells:
        dom = {"TTFT part": 0, "output length effect": 0, "per token effect": 0}
        for c in cells:
            k = max(zip(("TTFT part", "output length effect", "per token effect"),
                        (c["ttft_part"], c["length_effect"], c["per_token_effect"])), key=lambda t: abs(t[1]))[0]
            dom[k] += 1
        rng = lambda k: f"{min(c[k] for c in cells):.2f} to {max(c[k] for c in cells):.2f}"  # noqa: E731
        bid = [f["ttft_share_pct"] for f in fits if f["mode"] != "supervisor" and f.get("ttft_share_pct") is not None]
        sup = [f["ttft_share_pct"] for f in fits if f["mode"] == "supervisor" and f.get("ttft_share_pct") is not None]
        md += ["## Conclusions (data in this folder only)", "",
               f"- Fine-tuned/base latency is {rng('total_ratio')} over {len(cells)} cells. The largest term is "
               + ", ".join(f"the {k} in {v} cell{'s' if v != 1 else ''}" for k, v in dom.items() if v)
               + f"; the three terms sum to the difference (largest residual "
               f"{max(abs(c['residual']) for c in cells):.1g} ms).",
               f"- Fine-tuned/base ratios: time per output token {rng('tpot_ratio')}, output tokens {rng('out_ratio')}, "
               f"TTFT {rng('ttft_ratio')}."]
        if bid or sup:
            md.append("- TTFT share of total latency: " + "; ".join(
                f"{lab} {min(v):.1f} to {max(v):.1f} %" for lab, v in (("bidding modes", bid), ("supervisor", sup)) if v)
                + " (fits of TTFT on input tokens in ttft_vs_input).")
        lat = [abs(r["latency_diff_pct"]) for r in repros if r["latency_diff_pct"] is not None]
        out = [abs(r["out_diff_pct"]) for r in repros if r["out_diff_pct"] is not None]
        md.append(f"- Against the official records on the same rows: latency within {max(lat):.2f} %, output tokens "
                  f"within {max(out):.3f} %; output-token rule passed by {sum(r['rule'] == 'pass' for r in repros)} of "
                  f"{len(repros)} arms; {sum(r['flag'] == 'yes' for r in repros)} arm(s) differ in latency by more than "
                  f"{FLAG_PCT:.0f} %." if lat and out else "- No rows comparable with the official records yet.")
        md.append("- Not shown: deployment latency or other runtimes. This is one implementation (Unsloth, bf16, "
                  "adapter attached and not merged, one request at a time, eight processes per node), and the "
                  "decomposition is an accounting identity, not a cause.")
    else:
        md.append("No cell has both arms measured yet.")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print(f"{len(arms)} arm(s), {len(cells)} cell(s) summarised; {counts}; {n_req} of {n_all} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
