"""v5 evaluation report: metric charts, latency charts and metric tables, drawn from the result files

reads only what exists on disk at the moment it runs, never a cached or typed-in status:
  training    models/training/<mode>/<model>/<model>-<mode>-v5-r01/  (TRAINING_DONE, epoch-N, checkpoints)
  evaluation  reports/v5/selection/<mode>/<model>/*.json               (the authoritative records)
  provenance  reports/v5/results/records/<mode>/<model>/records.jsonl  (commit, corpus, hardware)
  latency     reports/v5/evaluation/latency_records/<mode>_<model>_<arm>.csv.gz  (per-request counts and
              timings: tokens, provenance latency rows) and tables/latency_breakdown_arms.csv (latency charts,
              radar latency axis), both built from the same records

nothing is estimated or interpolated. a test number is drawn only from a test result file; a validation
number never stands in for it.

metrics come from the pipeline's own records. sealed and reverse-auction summaries are shaped
differently (sealed has no overall block, reverse-auction's by_stage has no offer deviation), so
overall and macro figures are aggregated from each file's per-row records with the pipeline's own
definitions; that aggregation reproduces the pipeline's per-stage summary values exactly (re-checked
on every file this script reads).

writes, under --out (default reports/v5/evaluation):
  <mode>/{validation,test}/<metric>.{png,pdf}   decision metric, price or cost deviation, completion-time
                                                 deviation; base and epochs 1-3, the selected epoch ringed
  <mode>/latency.{png,pdf}, <mode>/radar.{png,pdf}  latency per decision, radar scores
  tables/metrics.md, tables/completion_time_deviation.md, tables/radar_values.md
  tables/tokens_summary.{md,tex}, tables/tokens.{md,tex}  token means, read from tables/tokens.csv
  tables/latency_breakdown_*                      from latency_records (latency_measurement/summarize.py)
tokens.csv, completion_time_deviation.csv and provenance.csv are written only with --csv-out, into a
folder where none of them exists yet, so the published copies are never overwritten; with
--csv-out <out>/tables a run into an empty folder rebuilds every published file. a split whose label
file is absent gets no trivial baseline line.

for a public checkout without the original profiling JSONL files, use --skip-original-profile
with a separate --out folder. This omits the original-profile comparison, not the latency tables.

  PYTHONPATH=. python -m src.experiments.v5.render_report [--out reports/v5/evaluation] [--csv-out DIR]
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from config import PROJECT_ROOT

REPO = Path(PROJECT_ROOT)
TRAIN = REPO / "models" / "training"
SEL = REPO / "reports" / "v5" / "selection"
RECORDS = REPO / "reports" / "v5" / "results" / "records"
LAT = REPO / "reports" / "v5" / "results" / "latency"
LREC = REPO / "reports" / "v5" / "evaluation" / "latency_records"
OUT = REPO / "reports" / "v5" / "evaluation"

MODELS = ("gemma", "granite", "ministral", "phi", "qwen", "smollm3")
MODES = ("sealed", "reverse_auction", "supervisor")
STAGES = ("material", "tailoring", "logistics")
MODE_TITLE = {"sealed": "Sealed-bid", "reverse_auction": "Reverse auction", "supervisor": "Supervisor"}

# colours are fixed on every chart (Okabe-Ito, colourblind-safe); the rule-based reference (the label
# generator, i.e. ground truth) is a black dashed line and the selected epoch is ringed
ARM_COLOR = {"base": "#999999", 1: "#E69F00", 2: "#009E73", 3: "#0072B2"}
ARM_LABEL = {"base": "Base model", 1: "Epoch 1", 2: "Epoch 2", 3: "Epoch 3"}
ARM_MARKER = {"base": "D", 1: "o", 2: "o", 3: "o"}
MODEL_COLOR = {"gemma": "#56B4E9", "granite": "#D55E00", "ministral": "#CC79A7", "phi": "#F0E442",
               "qwen": "#0072B2", "smollm3": "#009E73"}
DODGE = {"base": 0.27, 1: 0.09, 2: -0.09, 3: -0.27}
EDGE = "#222222"
INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"

# the chart font: static Source Sans 3 files (SIL Open Font License) bundled next to this script, regular (400)
# for tick labels and ring numbers, semibold (600) for model names, axis titles, legends and radar axis titles
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FAMILY = "Source Sans 3"
FONT_FILES = {400: "SourceSans3-Regular.ttf", 600: "SourceSans3-Semibold.ttf"}
WEIGHT = "semibold"


def use_bundled_fonts():
    """register the bundled files; stop when one is missing or a weight would resolve to any other file"""
    from matplotlib import font_manager
    for name in FONT_FILES.values():
        if not (FONT_DIR / name).is_file():
            raise SystemExit(f"bundled font missing: {FONT_DIR / name}")
        font_manager.fontManager.addfont(str(FONT_DIR / name))
    for weight, name in FONT_FILES.items():
        got = font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY, weight=weight),
                                    fallback_to_default=False, rebuild_if_missing=False)
        if Path(got).resolve() != (FONT_DIR / name).resolve():
            raise SystemExit(f"{FONT_FAMILY} weight {weight} resolves to {got}, not the bundled {name}")


use_bundled_fonts()
# a glyph missing from the bundled font is an error, never a silent fallback
warnings.filterwarnings("error", message=r"Glyph .* missing")

# text sizes in points on the canvas (dot plots and latency charts drawn 10 in wide, radars 5 in wide). dot plots and
# radars: the largest sizes at which no text overlaps or is cut and the dot-plot legend keeps two rows; latency
# charts share the dot-plot sizes. a saved file is cut to the ink plus SAVE_PAD, so it is narrower than its canvas:
# placed at 7.0 in (radars 3.4 in), text prints at more than 0.7 (0.68) of these sizes
LABEL_PT, TICK_PT, RADAR_PT, RADAR_TICK_PT = 16, 15, 14, 13

plt.rcParams.update({
    "font.family": FONT_FAMILY, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "font.size": TICK_PT, "axes.labelsize": LABEL_PT, "axes.labelweight": WEIGHT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": LABEL_PT,
    "axes.linewidth": 2.0, "xtick.major.width": 2.0, "ytick.major.width": 2.0, "xtick.major.size": 6,
    "ytick.major.size": 6, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embedded TrueType fonts
})

# (key, chart stem, axis label, higher is better, reference value of the rule-based ground truth)
METRICS = {
    "sealed": [
        ("accuracy_pct", "decision_agreement", "Decision agreement (%)", True, 100.0),
        ("macro_mean_abs_offer_dev_pct", "price_deviation", "Absolute price deviation (%)", False, 0.0),
        ("macro_mean_abs_days_dev", "completion_time_deviation", "Absolute completion-time deviation (days)", False, 0.0),
    ],
    "reverse_auction": [
        ("accuracy_pct", "decision_agreement", "Decision agreement (%)", True, 100.0),
        ("macro_mean_abs_offer_dev_pct", "price_deviation", "Absolute price deviation (%)", False, 0.0),
        ("macro_mean_abs_days_dev", "completion_time_deviation", "Absolute completion-time deviation (days)", False, 0.0),
    ],
    "supervisor": [
        ("accuracy_pct", "selection_accuracy", "Selection accuracy (%)", True, 100.0),
        ("mean_abs_cost_dev_pct", "cost_deviation", "Absolute cost deviation (%)", False, 0.0),
        ("mean_abs_days_dev_pct", "completion_time_deviation", "Absolute completion-time deviation (%)", False, 0.0),
    ],
}


def always_bid_baselines():
    """{(mode, split): value} from tables/always_bid_baseline.csv, or None when the table is missing"""
    path = OUT / "tables" / "always_bid_baseline.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return {(r["mode"], r["split"]): float(r["value_percent"]) for r in csv.DictReader(f)}


def baselines_for():
    """the always-BID line per mode and split: the table first, the split's label file only for what it lacks"""
    table = always_bid_baselines() or {}
    out = {}
    for mode in ("sealed", "reverse_auction"):
        for split in ("validation", "test"):
            b = table[(mode, split)] if (mode, split) in table else trivial_baseline(mode, split)[0]
            if b is not None:
                out[(mode, split)] = b
    return out


def trivial_baseline(mode, split):
    """the score a model with no skill gets, from the labels themselves

    sealed / reverse auction: always answering the majority decision, as MACRO agreement (the
    chart's x axis and the selection key). supervisor: picking a candidate uniformly at random.
    computed from the split's own rows every run, so it can never drift from the data"""
    import re
    from collections import Counter
    root = REPO / "data" / "finetune" / "v5"
    pre = "" if mode == "sealed" else "reverse_auction/"
    files = [root / "supervisor" / f"selection_{split}.jsonl"] if mode == "supervisor" else \
        [root / f"{pre}{st}_{split}.jsonl" for st in STAGES]
    if not all(f.exists() for f in files):
        return None, "label file absent"
    if mode == "supervisor":
        cand = re.compile(r"candidate (\d+): ")
        rows = [json.loads(l) for l in open(root / "supervisor" / f"selection_{split}.jsonl")]
        return 100 * sum(1 / len(cand.findall(r["user"])) for r in rows) / len(rows), "uniform random pick"
    shares = []
    for st in STAGES:
        c = Counter(json.loads(json.loads(l)["completion"])["decision"] for l in open(root / f"{pre}{st}_{split}.jsonl"))
        shares.append(100 * max(c.values()) / sum(c.values()))
    return sum(shares) / len(shares), "always the majority decision (BID)"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_name(model, mode):
    return f"{model}-{mode}-v5-r01"


def cell_dir(run):
    """reports/v5/selection/<mode>/<model> for <model>-<mode>-v5-r01"""
    model, mode = run.removesuffix("-v5-r01").split("-", 1)
    return SEL / mode / model


def cell_records(run):
    model, mode = run.removesuffix("-v5-r01").split("-", 1)
    return RECORDS / mode / model / "records.jsonl"


# ---------------------------------------------------------------- training state
def training_state(model, mode):
    d = TRAIN / mode / model / run_name(model, mode)
    avail = [e for e in (1, 2, 3) if (d / f"epoch-{e}" / "adapter_model.safetensors").exists()]
    done = (d / "TRAINING_DONE").exists() and avail == [1, 2, 3]
    cks = sorted(d.glob("checkpoints/checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    step = max_steps = None
    if cks:
        st = cks[-1] / "trainer_state.json"
        if st.exists():
            s = json.loads(st.read_text())
            step, max_steps = s.get("global_step"), s.get("max_steps")
    return {"model": model, "mode": mode, "run": run_name(model, mode), "run_dir": str(d.relative_to(REPO)),
            "done": done, "available": avail, "step": step, "max_steps": max_steps,
            "pct": round(100 * step / max_steps, 1) if step and max_steps else None}


def status_text(c):
    if c["done"]:
        return "DONE"
    if c["step"] is None or c["max_steps"] is None:
        return "training state absent"  # e.g. a copy without the git-ignored models/ folder
    return (f"in progress: epoch {max(c['available']) if c['available'] else 0} of 3 saved, "
            f"step {c['step']:,}/{c['max_steps']:,} ({c['pct']}%)")


# ---------------------------------------------------------------- metrics
def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def row_metrics(rows):
    """the pipeline's own per-row values, aggregated with its own definitions"""
    g = lambda r, k: r.get(k)  # noqa: E731  absent means not applicable, exactly as None does
    return {
        "decision_agreement_pct": _mean([100.0 * bool(g(r, "decision_matched")) for r in rows]),
        "json_valid_pct": _mean([100.0 * bool(g(r, "json_valid")) for r in rows]),
        "mean_abs_offer_dev_pct": _mean([abs(g(r, "offer_pct_dev")) if g(r, "offer_pct_dev") is not None else None
                                         for r in rows]),
        "mean_abs_days_dev": _mean([g(r, "days_abs_dev") for r in rows]),
        "mean_days_dev_pct": _mean([g(r, "days_pct_dev") for r in rows]),
        # NO_BID is the positive class: most rows are BID, so recall and precision on NO_BID show what
        # decision agreement hides. an answer that does not parse counts as not NO_BID
        "nobid_recall_pct": _nobid(rows, "recall"),
        "nobid_precision_pct": _nobid(rows, "precision"),
        "n": len(rows),
    }


def _nobid(rows, kind):
    if not rows or "label_decision" not in rows[0]:
        return None
    tp = sum(1 for r in rows if r.get("label_decision") == "NO_BID" and r.get("model_decision") == "NO_BID")
    denom = sum(1 for r in rows if (r.get("label_decision") if kind == "recall" else r.get("model_decision")) == "NO_BID")
    return round(100.0 * tp / denom, 4) if denom else None


def fidelity_check(result):
    """re-verify, on this very file, that per-stage aggregation equals the pipeline's summary"""
    s, mode = result["summary"], result["mode"]
    for st in STAGES:
        mine = row_metrics([r for r in result["rows_detail"] if r.get("stage") == st])
        ref = s[st] if mode == "sealed" else s["by_stage"][st]
        for k, v in mine.items():
            if k == "n":
                continue
            refv = ref.get("decision_agreement_pct", ref.get("accuracy_pct")) if k == "decision_agreement_pct" else ref.get(k)
            if refv is not None and v is not None and abs(round(v, 2) - refv) > 0.011:
                raise SystemExit(f"fidelity check failed on {result['run']} {st} {k}: {v} vs {refv}")


def metrics(result):
    mode = result["mode"]
    if mode == "supervisor":
        s = result["summary"]
        return {"accuracy_pct": s["selection_accuracy_pct"], "json_valid_pct": s["json_valid_pct"],
                "mean_regret_pct": s["mean_regret_pct"], "mean_cost_dev_pct": s["mean_cost_dev_pct"],
                "mean_abs_cost_dev_pct": s["mean_abs_cost_dev_pct"], "mean_days_dev_pct": s["mean_days_dev_pct"],
                "mean_abs_days_dev_pct": s["mean_abs_days_dev_pct"],
                "feasibility_violation_pct": s["feasibility_violation_pct"], "n": s["n"]}
    fidelity_check(result)
    rows = result["rows_detail"]
    overall = row_metrics(rows)
    per = {st: row_metrics([r for r in rows if r.get("stage") == st]) for st in STAGES}
    macro = {k: _mean([per[st][k] for st in STAGES]) for k in overall if k != "n"}
    out = {"accuracy_pct": result["selection_metrics"]["parts"].get(
        "macro_decision_agreement_pct", result["selection_metrics"]["parts"].get("macro_accuracy_pct")),
        "n": overall["n"]}
    for k in ("decision_agreement_pct", "json_valid_pct", "mean_abs_offer_dev_pct", "mean_abs_days_dev", "mean_days_dev_pct",
              "nobid_recall_pct", "nobid_precision_pct"):
        out[f"overall_{k}"] = overall[k]
        out[f"macro_{k}"] = macro[k]
    return out


# ---------------------------------------------------------------- evaluation records
def model_path(p):
    """a recorded adapter path from its models/ part on, so no host path is published"""
    return "models/" + p.split("/models/", 1)[1] if p and "/models/" in p else p


def provenance_for(run, split, arm, epoch):
    p = cell_records(run)
    if not p.exists():
        return None
    key = f"{run}/{split}/{arm}/{epoch}"
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("record_key") == key:
            return r
    return None


def load_arm(run, fname, arm_label):
    p = cell_dir(run) / fname
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if r.get("smoke"):
        return None          # a smoke result is never reported
    prov = provenance_for(run, r["split"], "descriptive" if r.get("descriptive") else r["arm"], r["epoch"])
    return {"arm": arm_label, "epoch": r["epoch"], "split": r["split"], "file": str(p.relative_to(REPO)),
            "file_sha256": sha256(p), "rows": r["rows"], "row_ids_sha256": r["row_ids_sha256"],
            "checkpoint": model_path(r["adapter"]) or f"base:{run.split('-')[0]}", "window": r["window"],
            "evaluated_at": r["evaluated_at"], "metrics": metrics(r), "authorization": r.get("authorization"),
            "descriptive": bool(r.get("descriptive")), "generation_stop_rule": r.get("generation_stop_rule"),
            "git_commit": prov.get("git_commit") if prov else None,
            "source_committed": prov.get("source_committed") if prov else None,
            "corpus_manifest_sha256": prov.get("corpus_manifest_sha256") if prov else None,
            "hardware_host": (prov.get("hardware") or {}).get("host") if prov else None,
            "slurm_job_id": (prov.get("hardware") or {}).get("slurm_job_id") if prov else None,
            "provenance_record": "present" if prov else "MISSING"}


def cell(model, mode):
    t = training_state(model, mode)
    run = t["run"]
    val = {}
    b = load_arm(run, "validation_base.json", "base")
    if b:
        val["base"] = b
    for e in (1, 2, 3):
        a = load_arm(run, f"validation_epoch{e}.json", e)
        if a:
            val[e] = a
    sel_p = cell_dir(run) / "selection.json"
    selection = json.loads(sel_p.read_text()) if sel_p.exists() else None
    test = {}
    bt = load_arm(run, "test_base.json", "base")
    if bt:
        test["base"] = bt
    tt = load_arm(run, "test_selected.json", "selected")
    if tt:
        test["selected"] = tt
        test[tt["epoch"]] = tt                    # the selected epoch's one-shot read, by epoch
    for e in (1, 2, 3):
        d = load_arm(run, f"test_descriptive_epoch{e}.json", e)
        if d:
            test[e] = d                           # descriptive, not used for selection
    return {**t, "val": val, "selection": selection,
            "selection_sha256": sha256(sel_p) if sel_p.exists() else None, "test": test,
            "latency_full": latency_full(model, mode)}


def _stats(values):
    """the per-arm statistics of the latency profile: n, sum, mean, std, median, p95, min, max"""
    s = sorted(values)
    return {"n": len(s), "sum": round(sum(s), 4), "mean": round(statistics.mean(s), 4),
            "std": round(statistics.stdev(s), 4) if len(s) > 1 else 0.0, "median": round(statistics.median(s), 4),
            "p95": round(s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)], 4), "min": round(s[0], 4),
            "max": round(s[-1], 4)}


def records_summary(rows):
    """one arm from its per-request rows: latency and tokens per second over the steady rows, tokens over all"""
    steady = [r for r in rows if r["warmup"] == "0"]
    n_in = [int(r["input_tokens"]) for r in rows]
    n_out = [int(r["output_tokens"]) for r in rows]
    return {"rows": len(rows), "rows_timed": len(steady), "rows_warmup_excluded": len(rows) - len(steady),
            "latency_ms": _stats([float(r["total_ms"]) for r in steady]), "input_tokens": _stats(n_in),
            "output_tokens": _stats(n_out), "total_tokens": _stats([a + b for a, b in zip(n_in, n_out)]),
            "output_tokens_per_second": _stats([round(int(r["output_tokens"]) / (float(r["total_ms"]) / 1000), 2)
                                                for r in steady]),
            "cap_hit_share_pct": round(100 * sum(n >= 200 for n in n_out) / len(rows), 4),
            "input_to_output_ratio": round(sum(n_in) / max(1, sum(n_out)), 4)}


def latency_full(model, mode):
    """whole test split, batch size 1, base and the selected epoch, from the published per-request records
    (evaluation/latency_records/<mode>_<model>_<arm>.csv.gz)"""
    out = {}
    chosen = json.loads((SEL / mode / model / "selection.json").read_text())["chosen"]
    for arm in ("base", "selected"):
        p = LREC / f"{mode}_{model}_{arm}.csv.gz"
        if p.exists():
            with gzip.open(p, "rt", newline="") as f:
                rows = list(csv.DictReader(f))
            ckpt = f"models/base/{model}" if arm == "base" else "models/" + chosen["adapter"].split("/models/", 1)[1]
            out[arm] = {"summary": records_summary(rows), "epoch": chosen["epoch"] if arm == "selected" else 0,
                        "checkpoint": ckpt, "file": str(p.relative_to(REPO)), "file_sha256": sha256(p)}
    return out


# ---------------------------------------------------------------- selection margin
# per mode, the per-row binary field behind each selection key (None: continuous or not per-row)
KEY_ROW_FIELD = {"sealed": ["decision_matched", "json_valid", None],
                 "reverse_auction": ["decision_matched", None, "offer_exact"],
                 "supervisor": ["correct", "valid_json", None]}


def selection_margin(c):
    """is the chosen epoch distinguishable from the runner-up on the key that actually decided?

    the deciding key is the first of the pre-registered ordered keys on which chosen and runner-up
    differ. where that key has a per-row binary field, an exact two-sided McNemar test is run on it
    over the identical validation rows (checked by row-id digest); rows are pooled across stages
    while the key itself is a stage macro, so the test is a close proxy, not the key's own statistic.
    reported only, never used to change a selection"""
    from math import comb
    if not c["selection"]:
        return None
    rank = c["selection"]["ranking"]
    if len(rank) < 2:
        return None
    r0, r1 = rank[0], rank[1]
    d = next((i for i, (x, y) in enumerate(zip(r0["key"], r1["key"])) if x != y), None)
    out = {"chosen": r0["epoch"], "runner_up": r1["epoch"],
           "decided_by": "exact tie: lower epoch" if d is None else c["selection"]["rule"]["key_order"][d],
           "key_gap": None if d is None else r0["key"][d] - r1["key"][d],
           "chosen_only": None, "runner_up_only": None, "p": None}
    field = None if d is None else KEY_ROW_FIELD[c["mode"]][d]
    if field is None:
        return out
    a = json.loads((cell_dir(c["run"]) / f"validation_epoch{r0['epoch']}.json").read_text())
    b = json.loads((cell_dir(c["run"]) / f"validation_epoch{r1['epoch']}.json").read_text())
    if a["row_ids_sha256"] != b["row_ids_sha256"]:
        return out
    x = sum(1 for p, q in zip(a["rows_detail"], b["rows_detail"]) if p.get(field) and not q.get(field))
    y = sum(1 for p, q in zip(a["rows_detail"], b["rows_detail"]) if q.get(field) and not p.get(field))
    n = x + y
    pv = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(x, y) + 1)) / 2 ** n) if n else 1.0
    out.update(chosen_only=x, runner_up_only=y, p=pv, field=field)
    return out


def margin_note(c):
    """'selection not statistically distinguishable (p = ...)' when the chosen-vs-runner-up McNemar p >= 0.05"""
    mg = selection_margin(c)
    if mg and mg["p"] is not None and mg["p"] >= 0.05:
        return f"selection not statistically distinguishable from epoch {mg['runner_up']} (p = {mg['p']:.3g})"
    return ""


# ---------------------------------------------------------------- charts
PLOT_H, STRIP_W = 3.1, 2.2  # dot plots: model rows and zoom strip, in inches (strip 0.71 of the rows)
TOP_IN, LEGEND_GAP = 0.12, 0.12  # space above the plots, and between the axis title and the legend under them (in)
LAT_PLOT_H, LAT_RIGHT = 3.98, 0.2  # latency charts: plot height and right margin (in)
RING_PAD = (18.2 / 2 + 1.0 + 2.0) / 72  # ring radius, half its line and a gap (in)
BAND_FILL, BAND_EDGE, LINK = "#eef1f5", "#555555", "#8a8a8a"
DISPLAY = {"gemma": "Gemma", "granite": "Granite", "ministral": "Ministral", "phi": "Phi", "qwen": "Qwen",
           "smollm3": "SmolLM3"}
LEGEND_FONT = {"weight": WEIGHT, "size": LABEL_PT}
SAVE_PAD = 0.03  # white margin around the ink on every side of a saved chart (in)


def ink_px(fig, dpi, vector=False):
    """first and last ink row, first and last ink column of the chart drawn at dpi, and the raster height; with vector,
    every text is laid out with unhinted metrics and drawn from its glyph outlines, and nothing is snapped to the pixel
    grid, as in the pdf. stops when the ink reaches the canvas edge, where it could be cut"""
    import matplotlib.patheffects as pe
    from matplotlib.text import Text
    kept = [(a, a.get_snap(), a.get_path_effects()) for a in fig.findobj()] if vector else []
    for a, _, effects in kept:
        a.set_snap(False)
        if isinstance(a, Text) and not effects:
            a.set_path_effects([pe.Normal()])
    fig.set_dpi(dpi)
    with matplotlib.rc_context({"text.hinting": "no_hinting"} if vector else {}):
        fig.canvas.draw()
    ink = (np.asarray(fig.canvas.buffer_rgba())[..., :3] != 255).any(axis=2)
    for a, snap, effects in kept:
        a.set_snap(snap)
        a.set_path_effects(effects)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    h, w = ink.shape
    if rows[0] == 0 or cols[0] == 0 or rows[-1] == h - 1 or cols[-1] == w - 1:
        raise SystemExit("chart ink reaches the canvas edge")
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]), h


def pdf_bbox(fig):
    """the pdf's saved area in inches: its ink plus SAVE_PAD, measured on a 600 dpi raster drawn as the pdf draws it.
    the png's text is hinted and its rotated text resampled, so its ink can lie a few pixels off the outlines"""
    r0, r1, c0, c1, h = ink_px(fig, 600, vector=True)
    return Bbox([[c0 / 600 - SAVE_PAD, (h - r1 - 1) / 600 - SAVE_PAD], [(c1 + 1) / 600 + SAVE_PAD, (h - r0) / 600 + SAVE_PAD]])


def ink_bbox(fig):
    """the png's saved area in inches: the chart's ink on the 300 dpi raster plus SAVE_PAD, on whole pixels, so the
    saved png is the drawn raster cut out unchanged"""
    r0, r1, c0, c1, h = ink_px(fig, 300)
    pad = round(SAVE_PAD * 300)

    def inch(px):
        """px / 300, nudged so that it gives px again at 300 dpi"""
        x = px / 300
        while x * 300 < px:
            x = math.nextafter(x, math.inf)
        while x * 300 > px:
            x = math.nextafter(x, -math.inf)
        return x
    # agg flips path rows about the whole canvas height and text rows about the exact one, which need not be whole:
    # the crop starts on a whole pixel and its height keeps that fraction (at least 1e-6 px, so the pixel count is
    # not rounded down), so paths and text both move by whole pixels
    bottom = h - r1 - 1 - pad
    top = bottom + r1 - r0 + 1 + 2 * pad + max(fig.bbox.height - h, 1e-6)
    return Bbox([[inch(c0 - pad), inch(bottom)], [(c1 + 1 + pad + 1e-6) / 300, top / 300]])


def save(fig, folder, stem):
    """png at 300 dpi and a vector pdf, each cut to its own ink plus SAVE_PAD on every side"""
    folder.mkdir(parents=True, exist_ok=True)
    boxes = {"png": ink_bbox(fig), "pdf": pdf_bbox(fig)}
    paths = []
    for ext, box in boxes.items():
        p = folder / f"{stem}.{ext}"
        # no creation date in the pdf, so a rerun gives the same file
        fig.savefig(p, dpi=300, bbox_inches=box, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def text_width(s, size, weight="normal"):
    """ink width of a one-line label in inches, in the chart font"""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=FONT_FAMILY, weight=weight)).get_extents().width / 72


def drawn_labels(axis):
    """the tick labels an axis draws: visible and inside its view limits"""
    lo, hi = sorted(axis.get_view_interval())
    i = 0 if axis.axis_name == "x" else 1
    return [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text() and lo - 1e-9 <= t.get_position()[i] <= hi + 1e-9]


def handle_inset(handle, handlelength):
    """inches from the left edge of a legend handle's box to its visible mark: a line fills the box, a lone
    marker sits in its middle"""
    from matplotlib.markers import MarkerStyle
    if handle.get_linestyle() not in ("None", "", " ") or handle.get_marker() in ("None", "", None):
        return 0.0
    m = MarkerStyle(handle.get_marker())
    width = m.get_path().transformed(m.get_transform()).get_extents().width * handle.get_markersize()
    return (handlelength * LABEL_PT - width - handle.get_markeredgewidth()) / 2 / 72


def stack_below(fig, axes, legs, top_in, plot_h, xc, frame=None):
    """the canvas height that fits the tick labels, the axis title and the legend under the plots, measured on the
    drawn figure; the legend sits LEGEND_GAP under the lowest of those labels. with a frame, the legends are rows
    stacked inside it, each centred on xc by its visible marks (from the first handle's mark to the ink end of the
    last label), and the frame keeps a fixed padding around the widest row"""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = min(a.get_window_extent(r).y0 for a in axes)
    texts = [t for a in axes for t in drawn_labels(a.xaxis) + [a.xaxis.label] if t.get_visible() and t.get_text()]
    low = (y0 - min(t.get_window_extent(r).y0 for t in texts)) / fig.dpi
    boxes = [leg.get_window_extent(r) for leg in legs]
    pad, space = (0.45 * LABEL_PT / 72, 0.35 * LABEL_PT / 72) if frame else (0.0, 0.0)  # frame padding, row spacing
    block = sum(b.height / fig.dpi for b in boxes) + (len(legs) - 1) * space + 2 * pad
    bottom = low + LEGEND_GAP + block + 0.06
    h = top_in + plot_h + bottom
    fig.set_size_inches(fig.get_figwidth(), h)
    for a in axes:
        p = a.get_position()
        a.set_position([p.x0, bottom / h, p.width, plot_h / h])
    w, top = fig.get_figwidth(), bottom - low - LEGEND_GAP
    y, widths = top - pad, []
    for leg, b in zip(legs, boxes):
        inset = trail = 0.0
        if frame:
            last = leg.get_texts()[-1]
            ink = TextPath((0, 0), last.get_text(), size=last.get_fontsize(),
                           prop=FontProperties(family=FONT_FAMILY, weight=last.get_fontweight())).get_extents()
            inset = handle_inset(leg.legend_handles[0], leg.handlelength)
            trail = last.get_window_extent(r).width / fig.dpi - ink.x1 / 72
        leg.set_bbox_to_anchor((xc - (inset - trail) / 2 / w, y / h), transform=fig.transFigure)
        widths.append(b.width / fig.dpi - inset - trail)
        y -= b.height / fig.dpi + space
    if frame:
        fw = max(widths) + 2 * pad
        frame.set_bounds(xc - fw / 2 / w, (top - block) / h, fw / w, block / h)


def arms_of(c, split):
    arms = c["val"] if split == "validation" else c["test"]
    return {k: arms[k] for k in ("base", 1, 2, 3) if k in arms}


def nice_step(x):
    """the smallest of 1, 2, 2.5 or 5 x 10^k that is at least x"""
    k = math.floor(math.log10(x))
    return next(float(f"{m * 10 ** k:.12g}") for m in (1, 2, 2.5, 5, 10) if m * 10 ** k >= x * (1 - 1e-9))


def decimals(step):
    return next(d for d in range(10) if abs(round(step, d) - step) < 1e-12)


def ticks(lo, hi, step):
    return [round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)]


def thin(tk, step, width_in, span):
    """every k-th tick, both ends kept, so that labels at TICK_PT never touch on an axis width_in wide"""
    from matplotlib.textpath import TextPath
    labels = [f"{t:.{decimals(step)}f}" for t in tk]
    need = max(TextPath((0, 0), s, size=TICK_PT).get_extents().width for s in labels) / 72 + 0.1
    n = len(tk) - 1
    k = next((k for k in range(1, n + 1) if n % k == 0 and k * step / span * width_in >= need), n)
    return tk[::k], labels[::k]


def chart_limits(cells, key, higher, ref):
    """(main end, main step, zoom range, zoom step), shared by validation and test

    taken from the plotted values of both splits and rounded outward to clean numbers: percent metrics
    run from 0 to 100 and zoom on the tuned epochs up to 100; absolute deviations run from 0 and zoom
    on the tuned epochs from 0"""
    vals = {k: [a["metrics"].get(key) for c in cells.values() for split in ("validation", "test")
                for arm, a in arms_of(c, split).items() if k == "all" or arm != "base"] for k in ("all", "tuned")}
    vals = {k: [x for x in v if x is not None] for k, v in vals.items()}
    tuned = vals["tuned"] or vals["all"]
    if higher:
        step = nice_step(max(ref - min(tuned), 1e-6) / 4)
        return 100.0, 20.0, (round((math.ceil(min(tuned) / step - 1e-9) - 1) * step, 10), ref), step
    step = nice_step(max(max(tuned), 1e-6) / 4)
    main_step = nice_step(max(vals["all"]) / 5)
    main_end = round((math.floor(max(vals["all"]) / main_step + 1e-9) + 1) * main_step, 10)
    return main_end, main_step, (ref, round((math.floor(max(tuned) / step + 1e-9) + 1) * step, 10)), step


def metric_chart(mode, split, cells, key, stem, xlabel, higher, ref, baselines, out):
    """the main plot and, beside the zoomed band, a zoom strip with the same model rows: right of the main
    plot for agreement and accuracy (the cluster is at 100), left of it for deviations (the cluster is at 0)"""
    from matplotlib.patches import ConnectionPatch, Rectangle
    # the always-BID baseline, for the sealed and reverse-auction decision metric only
    bl = baselines.get((mode, split)) if key == "accuracy_pct" else None
    main_end, main_step, (z0, z1), zstep = chart_limits(cells, key, higher, ref)
    real = (0.0, 100.0 if higher else main_end)
    # margins from the label widths, so that larger text is never cut: the model names on the left, the outer
    # tick labels at both ends, and between the plots the two facing end labels (the strip's end tick sits
    # RING_PAD inside the strip, the main plot's end tick on its edge or just inside)
    lab_main, lab_zoom = (lambda v: f"{v:.{decimals(main_step)}f}"), (lambda v: f"{v:.{decimals(zstep)}f}")
    half = {s: text_width(s, TICK_PT) / 2 for s in (lab_main(real[0]), lab_main(real[1]), lab_zoom(z0), lab_zoom(z1))}
    names = max(text_width(DISPLAY[m], LABEL_PT, WEIGHT) for m in MODELS)
    names += (plt.rcParams["ytick.major.size"] + plt.rcParams["ytick.major.pad"]) / 72 + 0.1
    if higher:
        left_in = max(names, half[lab_main(real[0])] + 0.06)
        right_in = max(0.3, half[lab_zoom(z1)] - RING_PAD + 0.06)
        facing = half[lab_main(real[1])] + half[lab_zoom(z0)]
    else:
        left_in = max(names, half[lab_zoom(z0)] - RING_PAD + 0.06)
        right_in = max(0.3, half[lab_main(real[1])] + 0.06)
        facing = half[lab_zoom(z1)] + half[lab_main(real[0])]
    gap_in = max(0.5, facing + 0.12)
    fig = plt.figure(figsize=(10, TOP_IN + PLOT_H + 1.5))
    left, bottom, height, gap = left_in / 10, 1.5 / (TOP_IN + PLOT_H + 1.5), PLOT_H / (TOP_IN + PLOT_H + 1.5), gap_in / 10
    w_strip = STRIP_W / 10
    w_main = (10 - left_in - right_in - gap_in - STRIP_W) / 10
    width = w_main + gap + w_strip
    if higher:
        ax = fig.add_axes([left, bottom, w_main, height])
        zs = fig.add_axes([left + w_main + gap, bottom, w_strip, height], sharey=ax)
    else:
        zs = fig.add_axes([left, bottom, w_strip, height])
        ax = fig.add_axes([left + width - w_main, bottom, w_main, height], sharey=zs)
    labelled, unlabelled = (ax, zs) if higher else (zs, ax)
    ys = {m: len(MODELS) - 1 - i for i, m in enumerate(MODELS)}
    pending, points, partial = [], [], {}
    for m in MODELS:
        c = cells[m]
        arms = arms_of(c, split)
        chosen = c["selection"]["chosen"]["epoch"] if c["selection"] else None
        if not arms:
            pending.append(m)
            continue
        missing = [k for k in ("base", 1, 2, 3) if k not in arms]
        if missing:
            partial[m] = "pending: " + ", ".join("base" if k == "base" else f"epoch {k}" for k in missing)
        for k, a in arms.items():
            v = a["metrics"].get(key)
            if v is not None:
                points.append((m, k, v, k == chosen))
    # padding only past 0 or 100, so no marker or reference line is cut and no marker moves; on the strip it is a
    # ring radius plus a small gap, so a ring at the edge stays inside the strip and clear of the tick marks
    zpad = RING_PAD * (z1 - z0) / (STRIP_W - RING_PAD)
    lims = {ax: (0.0, 102.0) if higher else (-0.02 * main_end, main_end),
            zs: (z0, z1 + zpad) if higher else (z0 - zpad, z1)}
    for a, (lo, hi) in lims.items():
        a.set_xlim(lo, hi)
        for m, k, v, sel in points:
            if not lo <= v <= hi:
                continue          # the strip shows only what lies in its range
            y = ys[m] + DODGE[k]
            a.scatter([v], [y], s=95, marker=ARM_MARKER[k], color=ARM_COLOR[k], edgecolor=EDGE, lw=0.9, zorder=4,
                      clip_on=False)
            if sel:
                a.scatter([v], [y], s=330, marker="o", facecolor="none", edgecolor="black", lw=2.0, zorder=5,
                          clip_on=False)
        a.axvline(ref, color="black", ls="--", lw=2.0, zorder=2)
        if bl is not None and lo <= bl <= hi:
            a.axvline(bl, color="black", ls=":", lw=2.0, zorder=2)
        for m in MODELS:
            a.axhline(ys[m] - 0.5, color="#dddddd", lw=0.8, zorder=0)
    ax.set_ylim(-0.9, len(MODELS) - 0.1)
    ax.set_xticks(*thin(ticks(*real, main_step), main_step, w_main * 10, lims[ax][1] - lims[ax][0]))
    ax.spines["bottom"].set_bounds(*real)
    zs.set_xticks(*thin(ticks(z0, z1, zstep), zstep, w_strip * 10, lims[zs][1] - lims[zs][0]))
    zs.spines["bottom"].set_bounds(z0, z1)
    labelled.set_yticks([ys[m] for m in MODELS], [DISPLAY[m] for m in MODELS], fontsize=LABEL_PT, fontweight=WEIGHT)
    unlabelled.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel(xlabel)
    for m in pending:
        ax.text(0.02 * real[1], ys[m], "pending", va="center", fontsize=TICK_PT, style="italic", color="#555555")
    for m, text in partial.items():
        ax.text(0.02 * real[1], ys[m] - 0.32, text, va="center", fontsize=TICK_PT, style="italic", color="#555555")
    # the band marks the zoomed range at its exact width; the strip is tinted with the same fill and two
    # connectors run from the band's corners to the strip's nearest corners, outside the model rows
    ax.add_patch(Rectangle((z0, -0.5), z1 - z0, len(MODELS), facecolor=BAND_FILL, edgecolor=BAND_EDGE, lw=1.0,
                           zorder=1))
    zs.set_facecolor(BAND_FILL)
    edge_x, corner_x = (z1, 0.0) if higher else (z0, 1.0)
    for band_y, corner_y in ((len(MODELS) - 0.5, 1.0), (-0.5, 0.0)):
        fig.add_artist(ConnectionPatch(xyA=(edge_x, band_y), coordsA=ax.transData, xyB=(corner_x, corner_y),
                                       coordsB=zs.transAxes, color=LINK, lw=0.9, zorder=6))
    handles = [Line2D([], [], marker=ARM_MARKER[k], ls="", color=ARM_COLOR[k], markeredgecolor=EDGE, markersize=10,
                      label=ARM_LABEL[k]) for k in ("base", 1, 2, 3)]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black", markersize=15,
                          markeredgewidth=2, label="Selected on validation"))
    handles.append(Line2D([], [], color="black", ls="--", lw=2, label="Rule-based ground truth"))
    if bl is not None and lims[ax][0] <= bl <= lims[ax][1]:
        handles.append(Line2D([], [], color="black", ls=":", lw=2, label="Always-BID baseline"))
    # two rows built independently (base and epochs, then the rest), so the gaps inside a row are equal; each row
    # centred under the plots, both in one thin frame at a fixed gap under them
    legs = [fig.legend(handles=row, ncol=len(row), loc="upper center", frameon=False, prop=LEGEND_FONT, handlelength=1.3,
                       handletextpad=0.4, columnspacing=0.8, borderpad=0, borderaxespad=0) for row in (handles[:4], handles[4:])]
    frame = fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor="white", edgecolor="#9a9a9a",
                                     lw=0.6, alpha=0.8, zorder=1))
    stack_below(fig, [ax, zs], legs, TOP_IN, PLOT_H, left + width / 2, frame)
    return save(fig, out / mode / split, stem)


def latency_arms(out):
    """mean and standard deviation of the latency per decision (ms, warm-up rows excluded) per mode, model and arm,
    read from tables/latency_breakdown_arms.csv"""
    with open(Path(out) / "tables" / "latency_breakdown_arms.csv", newline="") as f:
        return {(r["mode"], r["model"], r["arm"]): {"mean": float(r["total_mean"]), "std": float(r["total_std"])}
                for r in csv.DictReader(f)}


def latency_chart(mode, cells, out):
    """models on x, mean latency per decision, base gray beside the fine-tuned bar, std whiskers"""
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.08, 0.18, 1 - 0.08 - LAT_RIGHT / 10, LAT_PLOT_H / 5])
    width = 0.38
    have = False
    lat = latency_arms(out)
    for i, m in enumerate(MODELS):
        for j, arm in enumerate(("base", "selected")):
            s = lat.get((mode, m, arm))
            x = i + (j - 0.5) * width
            if not s:
                ax.text(x, 0, "pending", rotation=90, ha="center", va="bottom", fontsize=TICK_PT, color="#555555")
                continue
            have = True
            color = ARM_COLOR["base"] if arm == "base" else MODEL_COLOR[m]
            ax.bar(x, s["mean"] / 1000, width=width * 0.94, color=color, edgecolor=EDGE, lw=1.2, zorder=3)
            ax.errorbar(x, s["mean"] / 1000, yerr=s["std"] / 1000, color="black", capsize=3, elinewidth=1.2,
                        capthick=1.2, zorder=4)
    ax.set_xticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS], fontsize=LABEL_PT, fontweight=WEIGHT)
    ax.set_ylabel("Latency per decision (s)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR["base"], ec=EDGE, label="Base model"),
               plt.Rectangle((0, 0), 1, 1, color="#cccccc", ec=EDGE, hatch="", label="Fine-tuned (model colour)"),
               Line2D([], [], color="black", lw=1.2, label="± 1 standard deviation")]
    leg = fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False, prop=LEGEND_FONT, columnspacing=1.2,
                     handletextpad=0.5, borderaxespad=0)
    if not have:
        ax.set_ylim(0, 1)
    # the left margin from the drawn y labels, then the height that fits the model names and the legend
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ylabels = drawn_labels(ax.yaxis) + [ax.yaxis.label]
    left = (ax.get_window_extent(r).x0 - min(t.get_window_extent(r).x0 for t in ylabels)) / fig.dpi + 0.08
    p = ax.get_position()
    ax.set_position([left / 10, p.y0, (10 - left - LAT_RIGHT) / 10, p.height])
    stack_below(fig, [ax], [leg], TOP_IN, LAT_PLOT_H, (left + (10 - left - LAT_RIGHT) / 2) / 10)
    return save(fig, out / mode, "latency")


# ---------------------------------------------------------------- radar
RADAR_AXES = ("accuracy", "price", "days", "latency", "tokens")
# every radar axis title, here only; each reads "more is better", like the scores. "\n" breaks a title
RADAR_TITLES = {"agreement": "Decision\nagreement", "selection": "Selection\naccuracy", "price": "Price\naccuracy",
                "cost": "Cost\naccuracy", "days": "Time accuracy", "latency": "Speed", "tokens": "Token\nefficiency"}
# titles clockwise from the top: radius of the anchor on the axis, offset from it (pt), alignment. the side titles
# sit just above the end of their axis, clear of the ring numbers, and do not widen the chart
RADAR_TITLE_AT = [(11.3, (0, 0), "center", "bottom"), (10, (4, 4), "left", "bottom"),
                  (12.6, (0, 0), "center", "top"), (12.6, (0, 0), "center", "top"), (10, (-4, 4), "right", "bottom")]
RADAR_LW, RADAR_EDGE_LW = 1.8, 2.4  # polygon line width (pt); dark edge under light colours


def radar_titles(mode):
    """the five axis titles of a mode, in RADAR_AXES order"""
    sup = mode == "supervisor"
    return [RADAR_TITLES[k] for k in ("selection" if sup else "agreement", "cost" if sup else "price", "days",
                                      "latency", "tokens")]


def clean_bound(x, up):
    """x rounded up (or down) to a multiple of half its power of ten: 311.6 -> 350, 148.8 -> 150, 8.41 -> 8.5"""
    unit = 10 ** math.floor(math.log10(x)) / 2
    k = math.ceil(x / unit - 1e-9) if up else math.floor(x / unit + 1e-9)
    return round(k * unit, 10)


def contrast_on_white(hex_color):
    """wcag contrast ratio of a colour against white"""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in matplotlib.colors.to_rgb(hex_color)]
    return 1.05 / (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] + 0.05)


def radar_scores(cells_by_mode, out):
    """per mode: values and 0-10 scores (outward is better) of the six fine-tuned selected epochs

    validation ranks on the three quality axes only; test gives the plotted values on all five. each split's
    references come from all twelve arms of the mode (base and selected): the worst value, rounded up for a
    deviation, latency or token count (score 10 x (1 - value / reference)) and rounded down to a floor for
    accuracy (score 10 x (value - floor) / (100 - floor)). base arms set the scale only and are never drawn.
    returns None (and says why) when a validation value is missing or a deviation is negative"""
    tokens = {}
    with open(Path(out) / "tables" / "tokens.csv", newline="") as f:
        for r in csv.DictReader(f):
            tokens[(r["mode"], r["model"], r["arm"])] = float(r["output mean"])
    lat = latency_arms(out)
    data = {}
    for mode in MODES:
        keys = [k for k, *_ in METRICS[mode]]
        raw = {}
        for split in ("validation", "test"):
            for m in MODELS:
                c = cells_by_mode[mode][m]
                chosen = c["selection"]["chosen"]["epoch"]
                arms = {"base": c["val"].get("base"), "selected": c["val"].get(chosen)} if split == "validation" \
                    else {"base": c["test"].get("base"), "selected": c["test"].get("selected")}
                for arm, a in arms.items():
                    v = {ax: (a or {}).get("metrics", {}).get(k) for ax, k in zip(RADAR_AXES, keys)}
                    if split == "test":
                        v["latency"] = lat[(mode, m, arm)]["mean"] / 1000
                        v["tokens"] = tokens[(MODE_TITLE[mode], m, "base" if arm == "base" else "fine-tuned")]
                    if any(x is None for x in v.values()):
                        print(f"radar skipped: {mode} {m} {split} {arm} has a missing value {v}")
                        return None
                    if any(v[ax] < 0 for ax in v if ax != "accuracy"):
                        print(f"radar skipped: {mode} {m} {split} {arm} has a negative value {v}")
                        return None
                    raw[(split, m, arm)] = v
        refs, scores = {}, {}
        for split, axes in (("validation", RADAR_AXES[:3]), ("test", RADAR_AXES)):
            arms = [raw[(split, m, arm)] for m in MODELS for arm in ("base", "selected")]
            refs[split] = {ax: clean_bound(min(a[ax] for a in arms), up=False) if ax == "accuracy"
                           else clean_bound(max(a[ax] for a in arms), up=True) for ax in axes}
            scores[split] = {m: {ax: 10 * (raw[(split, m, "selected")][ax] - refs[split][ax]) / (100 - refs[split][ax])
                                 if ax == "accuracy" else 10 * (1 - raw[(split, m, "selected")][ax] / refs[split][ax])
                                 for ax in axes} for m in MODELS}
        mean = {m: sum(scores["validation"][m].values()) / 3 for m in MODELS}
        rank = sorted(MODELS, key=lambda m: (-mean[m], -scores["validation"][m]["accuracy"], DISPLAY[m]))
        data[mode] = {"raw": raw, "refs": refs, "scores": scores, "mean": mean, "rank": rank, "top": rank[:3]}
    return data


# the ring numbers sit on one radial line shared by the three radars, each centred on its ring. the line is the one
# with the most room from the polygon lines (measured on all lines 0.1 degree apart): just below the price or cost
# axis, in the gap where the outlines of all drawn models run together
RING_LABEL_ANGLE = 83.7  # degrees clockwise from the top
RING_LABEL_HALO = 1.5  # white outline width (pt) around the ring numbers


def ring_ink(lab):
    """ink box (x0, y0, x1, y1) of a ring number in points, from its text origin on the baseline"""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    e = TextPath((0, 0), str(lab), size=RADAR_TICK_PT, prop=FontProperties(family=FONT_FAMILY)).get_extents()
    return e.x0, e.y0, e.x1, e.y1


def box_gap(cx, cy, hw, hh, p, q):
    """distance from the box centred at (cx, cy) with half sizes hw, hh to the segment pq (0 when they meet). the
    distance to a box is convex along the segment, so a ternary search finds its minimum"""
    def at(t):
        x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        return math.hypot(max(abs(x - cx) - hw, 0), max(abs(y - cy) - hh, 0))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (lo, b) if at(a) <= at(b) else (a, hi)
    return at((lo + hi) / 2)


def ring_label_spot(data):
    """the ring-number line shared by the three radars (radians) and its room in inches: the ink box of each number,
    centred on its ring and widened by the white outline, against every polygon line of the three charts; stops when
    a number or its outline would touch a line"""
    angles = [i * 2 * math.pi / 5 for i in range(5)]
    rad = 0.56 * 5 / 2
    t = math.radians(RING_LABEL_ANGLE)

    def xy(r, a):
        return r / 10 * rad * math.sin(a), r / 10 * rad * math.cos(a)
    gap = 9.0
    for lab in range(0, 11, 2):
        x0, y0, x1, y1 = ring_ink(lab)
        hw, hh = ((x1 - x0) / 2 + RING_LABEL_HALO / 2) / 72, ((y1 - y0) / 2 + RING_LABEL_HALO / 2) / 72
        cx, cy = xy(lab, t)
        for d in data.values():
            for m in d["top"]:
                rs = [d["scores"]["test"][m][axn] for axn in RADAR_AXES]
                half = (RADAR_EDGE_LW if "path_effects" in radar_line(m) else RADAR_LW) / 2 / 72
                for i in range(5):
                    p, q = xy(rs[i], angles[i]), xy(rs[(i + 1) % 5], angles[(i + 1) % 5])
                    gap = min(gap, box_gap(cx, cy, hw, hh, p, q) - half)
    if gap <= 0:
        raise SystemExit(f"radar ring numbers would touch a polygon line: room {gap:.4f} in at {RING_LABEL_ANGLE} degrees")
    return t, gap


def radar_line(m):
    """line style of a model on the radar: a thin line, with a dark edge under colours that are light on white"""
    import matplotlib.patheffects as pe
    if contrast_on_white(MODEL_COLOR[m]) < 3:
        return {"lw": RADAR_LW - 0.2, "path_effects": [pe.Stroke(linewidth=RADAR_EDGE_LW, foreground="#3a3a3a"), pe.Normal()]}
    return {"lw": RADAR_LW}


def radar_chart(mode, d, out, spot):
    """test scores of the three top-ranked fine-tuned models on five axes, clockwise from the top"""
    import matplotlib.patheffects as pe
    from matplotlib.transforms import offset_copy
    fig = plt.figure(figsize=(5, 4.65))
    ax = fig.add_axes([0.22, 0.2258, 0.56, 0.56 * 5 / 4.65], projection="polar")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    angles = [i * 2 * math.pi / 5 for i in range(5)]
    circle = [i * 2 * math.pi / 360 for i in range(361)]
    for r in range(1, 11):
        ax.plot(circle, [r] * len(circle), color="#9a9a9a" if r == 10 else "#e0e0e0", lw=1.1 if r == 10 else 0.7,
                zorder=0)
    for a in angles:
        ax.plot([a, a], [0, 10], color="#d5d5d5", lw=0.8, zorder=0)
    polys = {m: [d["scores"]["test"][m][axn] for axn in RADAR_AXES] for m in MODELS if m in d["top"]}
    area = {m: sum(r[i] * r[(i + 1) % 5] for i in range(5)) * math.sin(2 * math.pi / 5) / 2 for m, r in polys.items()}
    # the largest polygon is drawn first, so the smaller ones stay on top
    for k, m in enumerate(sorted(polys, key=lambda m: -area[m])):
        r = polys[m]
        ax.plot(angles + angles[:1], r + r[:1], color=MODEL_COLOR[m], label=DISPLAY[m], zorder=3 + k / 10,
                clip_on=False, **radar_line(m))
        ax.fill(angles + angles[:1], r + r[:1], color=MODEL_COLOR[m], alpha=0.12, zorder=2 + k / 10)
    # ring numbers on the line shared by all three charts, the ink of each centred on its ring, with a white outline
    # over the shaded area
    for lab in range(0, 11, 2):
        x0, y0, x1, y1 = ring_ink(lab)
        ax.text(spot[0], lab, str(lab), fontsize=RADAR_TICK_PT, color="#555555", ha="left", va="baseline", zorder=4,
                transform=offset_copy(ax.transData, fig, -(x0 + x1) / 2, -(y0 + y1) / 2, units="points"),
                path_effects=[pe.withStroke(linewidth=RING_LABEL_HALO, foreground="white")])
    # each title at its offset from the outer end of its axis
    titles = [ax.annotate(title, (a, r), xytext=off, textcoords="offset points", ha=ha, va=va, annotation_clip=False,
                          fontsize=RADAR_PT, fontweight=WEIGHT)
              for a, title, (r, off, ha, va) in zip(angles, radar_titles(mode), RADAR_TITLE_AT)]
    handles = [Line2D([], [], color=MODEL_COLOR[m], label=DISPLAY[m], **radar_line(m)) for m in MODELS if m in d["top"]]
    leg = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, prop={"weight": WEIGHT, "size": RADAR_PT},
                     handlelength=1.6, columnspacing=1.0, borderaxespad=0)
    # the legend box hangs LEGEND_GAP under the lowest title box, the gap the dot plots keep under their axis title
    fig.canvas.draw()
    low = min(t.get_window_extent().y0 for t in titles) / fig.dpi
    leg.set_bbox_to_anchor((0.5, (low - LEGEND_GAP) / fig.get_figheight()), transform=fig.transFigure)
    return save(fig, out / mode, "radar")


def render_charts(cells_by_mode, baselines, out):
    """the 21 metric and latency charts and the three radar charts; writes no table"""
    written = []
    for mode in MODES:
        for split in ("validation", "test"):
            for key, stem, xlabel, higher, ref in METRICS[mode]:
                written += metric_chart(mode, split, cells_by_mode[mode], key, stem, xlabel, higher, ref, baselines,
                                        out)
        written += latency_chart(mode, cells_by_mode[mode], out)
    data = radar_scores(cells_by_mode, out)
    if data:
        spot = ring_label_spot(data)
        for mode in MODES:
            written += radar_chart(mode, data[mode], out, spot)
    return written, data


# ---------------------------------------------------------------- tables
def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _fmt(v, nd=2):
    return "-" if v is None else (f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}")


def md_cell(text):
    """a literal pipe inside a markdown table cell must be escaped"""
    return str(text).replace("|", "\\|")


def arm_order(arms):
    """base first, then epochs 1-3; the 'selected' alias is never listed twice"""
    return [(k, arms[k]) for k in ("base", 1, 2, 3) if k in arms]


def arm_label(split, k, a, chosen):
    if k == "base":
        return "base" if split == "validation" else "base (one-shot)"
    if split == "validation":
        return f"epoch {k}" + (" (selected)" if k == chosen else "")
    return f"epoch {k} (selected, one-shot)" if k == chosen else f"epoch {k} (descriptive, not used for selection)"


def write_tables(cells_by_mode, out):
    """tables/metrics.md, and the provenance rows behind every number in it"""
    lines, prov_rows = [], []
    for mode in MODES:
        lines.append(f"\n## {MODE_TITLE[mode]}\n")
        if mode == "supervisor":
            cols = ["accuracy_pct", "json_valid_pct", "mean_regret_pct", "mean_abs_cost_dev_pct",
                    "mean_cost_dev_pct", "mean_abs_days_dev_pct", "mean_days_dev_pct", "feasibility_violation_pct"]
            head = ["sel. acc %", "JSON valid %", "mean regret %", "|cost dev| %", "cost dev %",
                    "|days dev| %", "days dev %", "infeasible %"]
        else:
            cols = ["accuracy_pct", "overall_decision_agreement_pct", "overall_json_valid_pct",
                    "overall_nobid_recall_pct", "overall_nobid_precision_pct",
                    "macro_mean_abs_offer_dev_pct", "overall_mean_abs_offer_dev_pct",
                    "macro_mean_abs_days_dev", "macro_mean_days_dev_pct"]
            head = ["macro agree %", "overall agree %", "JSON valid %", "NO_BID recall %", "NO_BID precision %",
                    "macro |price dev| %", "overall |price dev| %", "macro |days dev| (d)", "macro |days dev| %"]
            lines.append("price and days deviations are absolute: |model - rule-based| relative to the rule-based "
                         "value, never signed\n")
        for split in ("validation", "test"):
            lines.append(f"\n### {split}\n")
            if split == "test":
                lines.append("base and the selected epoch are the one-shot test reads; the other epochs are "
                             "descriptive, read after selection and not used for selection\n")
            lines.append("| model | arm | rows | " + " | ".join(md_cell(h) for h in head) + " | note |")
            lines.append("|" + "---|" * (4 + len(head)))
            for m in MODELS:
                c = cells_by_mode[mode][m]
                arms = c["val"] if split == "validation" else c["test"]
                chosen = c["selection"]["chosen"]["epoch"] if c["selection"] else None
                note = margin_note(c)
                if not arms:
                    lines.append(f"| {m} | - | - | " + " | ".join("-" for _ in head) + " | pending |")
                    continue
                for k, a in arm_order(arms):
                    arm = arm_label(split, k, a, chosen)
                    lines.append(f"| {m} | {arm} | {a['rows']:,} | "
                                 + " | ".join(fmt(a["metrics"].get(col)) for col in cols)
                                 + f" | {note if k == chosen else ''} |")
                    prov_rows.append({"mode": mode, "model": m, "split": split, "arm": arm, "run": c["run"],
                                      "training_run_dir": c["run_dir"], "training_status": status_text(c),
                                      "checkpoint": a["checkpoint"], "epoch": a["epoch"], "rows": a["rows"],
                                      "row_ids_sha256": a["row_ids_sha256"], "window": a["window"],
                                      "result_file": a["file"], "result_sha256": a["file_sha256"],
                                      "selection_sha256": c["selection_sha256"], "evaluated_at": a["evaluated_at"],
                                      "descriptive": a.get("descriptive", False),
                                      "generation_stop_rule_added": (a.get("generation_stop_rule") or {}).get("added"),
                                      "git_commit": a["git_commit"], "source_committed": a["source_committed"],
                                      "corpus_manifest_sha256": a["corpus_manifest_sha256"],
                                      "slurm_job_id": a["slurm_job_id"], "provenance_record": a["provenance_record"],
                                      **{f"metric.{col}": a["metrics"].get(col) for col in cols}})
    lines.append("\n## Selection margin: chosen epoch vs runner-up (validation, exact McNemar)\n")
    lines.append("the pre-registered rule is applied as written; this only reports whether its choice is "
                 "statistically distinguishable on the key that decided it: the first of the ordered keys on which "
                 "chosen and runner-up differ, tested on that key's per-row field (rows pooled across stages, while "
                 "the key is a stage macro: a close proxy)\n")
    lines.append("| mode | model | chosen | runner-up | decided by | gap on that key | chosen-only | runner-up-only | p (exact McNemar) | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for mode in MODES:
        for m in MODELS:
            mg = selection_margin(cells_by_mode[mode][m])
            if not mg:
                continue
            gap = "" if mg["key_gap"] is None else f"{mg['key_gap']:.4f}"
            if mg["p"] is None:
                test = "| n/a | n/a | n/a (continuous key) | |"
            else:
                test = (f"| {mg['chosen_only']} | {mg['runner_up_only']} | {mg['p']:.3g} ({mg['field']}) | "
                        f"{'selection not statistically distinguishable' if mg['p'] >= 0.05 else 'distinguishable at 0.05'} |")
            lines.append(f"| {mode} | {m} | epoch {mg['chosen']} | epoch {mg['runner_up']} | {md_cell(mg['decided_by'])} | {gap} {test}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "metrics.md").write_text("# v5 evaluation metrics\n" + "\n".join(lines) + "\n")
    return prov_rows


def latency_rows(cells_by_mode):
    """one provenance row per measured latency arm"""
    rows = []
    for mode in MODES:
        for m in MODELS:
            for arm, L in (cells_by_mode[mode][m].get("latency_full") or {}).items():
                s = L["summary"]
                rows.append({"mode": mode, "model": m, "split": "test", "arm": f"latency {arm}",
                             "run": run_name(m, mode), "checkpoint": L["checkpoint"], "epoch": L["epoch"],
                             "rows": s["rows"], "result_file": L["file"], "result_sha256": L["file_sha256"],
                             "metric.latency_mean_ms": s["latency_ms"]["mean"], "metric.latency_std_ms": s["latency_ms"]["std"],
                             "metric.output_tokens_mean": s["output_tokens"]["mean"],
                             "metric.rows_warmup_excluded": s["rows_warmup_excluded"]})
    return rows


def token_rows(cells_by_mode):
    """per mode and model, base and fine-tuned: tokens (sum, mean, median, p95), output tokens/s,
    cap share, input-to-output ratio"""
    cols = [("output_tokens", "sum"), ("output_tokens", "mean"), ("output_tokens", "median"), ("output_tokens", "p95"),
            ("input_tokens", "sum"), ("input_tokens", "mean"), ("input_tokens", "median"), ("input_tokens", "p95"),
            ("total_tokens", "sum"), ("total_tokens", "mean"), ("total_tokens", "median"), ("total_tokens", "p95")]
    head = ["mode", "model", "arm", "rows"] + [f"{a.split('_')[0]} {b}" for a, b in cols] + \
        ["output tok/s (mean)", "cap hit %", "input:output"]
    rows = []
    for mode in MODES:
        for m in MODELS:
            L = cells_by_mode[mode][m].get("latency_full") or {}
            for arm, label in (("base", "base"), ("selected", "fine-tuned")):
                s = (L.get(arm) or {}).get("summary")
                if not s:
                    rows.append([MODE_TITLE[mode], m, label, "pending"] + ["-"] * (len(head) - 4))
                    continue
                rows.append([MODE_TITLE[mode], m, label, s["rows"]] +
                            [(s.get(a) or {}).get(b) for a, b in cols] +
                            [(s.get("output_tokens_per_second") or {}).get("mean"), s.get("cap_hit_share_pct"),
                             s.get("input_to_output_ratio")])
    return head, rows


def tex_escape(text):
    """latex special characters in plain text"""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in str(text))


def output_caps():
    """{(mode, model, arm): max new tokens} from tables/latency_output_cap.csv. where an original profiling record
    is present, the cap in its first line is compared, and a difference is only printed, never an error"""
    with open(OUT / "tables" / "latency_output_cap.csv", newline="") as f:
        caps = {(r["mode"], r["model"], r["arm"]): int(r["max_new_tokens"]) for r in csv.DictReader(f)}
    for (mode, m, arm), cap in caps.items():
        try:
            p = LAT / mode / m / f"{arm}.jsonl"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    raw = json.loads(f.readline())["max_new_tokens"]
                if raw != cap:
                    print(f"warning: {mode} {m} {arm}: output cap {cap} in latency_output_cap.csv, {raw} in {p.name}")
        except Exception as e:
            print(f"warning: output cap of {mode} {m} {arm} not compared with the original record: {e}")
    return caps


def token_summary(out):
    """tables/tokens_summary.{md,tex}: per-decision means read from tables/tokens.csv, one row per mode and model"""
    tdir = Path(out) / "tables"
    with open(tdir / "tokens.csv", newline="") as f:
        by = {}
        for r in csv.DictReader(f):
            by.setdefault((r["mode"], r["model"]), {})[r["arm"]] = r
    cap = output_caps()
    caps = {cap[(mode, m, arm)] for mode in MODES for m in MODELS for arm in ("base", "selected")}
    if len(caps) != 1:
        raise SystemExit(f"the latency arms used different output caps: {sorted(caps)}")
    head = ["Mode", "Model", "Input (mean)", "Output base", "Output fine-tuned", "Total base", "Total fine-tuned"]
    rows, capped = [], []
    name = {"Sealed-bid": "Sealed bid"}
    for (mode, model), arms in by.items():
        b, t = arms["base"], arms["fine-tuned"]
        if float(b["input mean"]) != float(t["input mean"]):
            raise SystemExit(f"input mean differs between the arms of {mode} {model}: {b['input mean']} vs {t['input mean']}")
        rows.append([name.get(mode, mode), model] + [f"{float(x):,.1f}" for x in (b["input mean"], b["output mean"], t["output mean"],
                                                                  b["total mean"], t["total mean"])])
        capped += [f"{name.get(mode, mode)} {model} {label} ({float(r['cap hit %']):.1f} %)"
                   for label, r in (("base", b), ("fine-tuned", t)) if float(r["cap hit %"]) >= 1.0]
    caption = ("Mean tokens per decision on the whole test split at batch size 1, for the base model and the "
               "selected fine-tuned epoch. The input is the same prompt for both arms.")
    note = (f"Output is capped at {caps.pop()} new tokens. Arms that hit the cap on at least 1 % of decisions: "
            + ("; ".join(capped) if capped else "none") + ".")
    md = ["# Tokens per decision (summary)\n", caption + "\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(md_cell(x) for x in r) + " |" for r in rows]
    rounding = "Values are rounded independently, so a total can differ from input plus output by 0.1."
    (tdir / "tokens_summary.md").write_text("\n".join(md + ["", md_cell(note), "", rounding]) + "\n")
    tex = ["% generated by src/experiments/v5/render_report.py from tables/tokens.csv", "\\begin{table}[t]",
           "\\centering", "\\small", "\\caption{" + tex_escape(caption) + "}", "\\label{tab:tokens_summary}",
           "\\begin{tabular}{llrrrrr}", "\\toprule", " & ".join(tex_escape(h) for h in head) + " \\\\", "\\midrule"]
    tex += [" & ".join(tex_escape(x) for x in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip", "{\\footnotesize " + tex_escape(note) + "\\par " + tex_escape(rounding) + "\\par}",
            "\\end{table}"]
    (tdir / "tokens_summary.tex").write_text("\n".join(tex) + "\n")


TOKENS_CAPTION = ("Tokens per decision on the whole test split at batch size 1 (base arm and the selected fine-tuned epoch). Output tokens are counted from the generated token ids and include the stop token. Generation stops on the model's generation stop ids. The chat template's end-of-turn token <|im_end|> (248046) is added as a stop id only for qwen reverse_auction; the five other models already stop on their end-of-turn token, and the other qwen cells continue past it by one newline before stopping on <|endoftext|> (248044). The cap is 200 new tokens.")


def token_tables(out):
    """tables/tokens.{md,tex}: every column of tables/tokens.csv per arm (md) and the main ones (tex)"""
    tdir = Path(out) / "tables"
    with open(tdir / "tokens.csv", newline="") as f:
        rows = list(csv.reader(f))
    head, rows = rows[0], rows[1:]

    def num(v):
        return f"{int(v):,}" if v.lstrip("-").isdigit() else f"{float(v):,.2f}"
    cells = [r[:3] + [num(v) for v in r[3:]] for r in rows]
    md = ["# Tokens per decision", "", TOKENS_CAPTION, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(c) + " |" for c in cells]
    (tdir / "tokens.md").write_text("\n".join(md) + "\n")
    keep = ["mode", "model", "arm", "rows", "output sum", "output mean", "output median", "output p95", "input mean",
            "total mean", "output tok/s (mean)", "cap hit %", "input:output"]
    show = ["mode", "model", "arm", "rows", "\\textbf{out sum}", "\\textbf{out mean}", "\\textbf{out median}",
            "\\textbf{out p95}", "in mean", "total mean", "\\textbf{out tok/s}", "cap \\%", "in:out"]
    idx = [head.index(k) for k in keep]
    tex = ["% generated by src/experiments/v5/render_report.py from tables/tokens.csv", "\\begin{table}[t]",
           "\\centering", "\\small", "\\caption{" + tex_escape(TOKENS_CAPTION).replace("<|im\\_end|>",
                                                                                   "\\texttt{<|im\\_end|>}") + "}",
           "\\label{tab:tokens}", "\\begin{tabular}{lllrrrrrrrrrr}", "\\toprule", " & ".join(show) + " \\\\",
           "\\midrule"]
    tex += [" & ".join(c[i] for i in idx) + " \\\\" for c in cells]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (tdir / "tokens.tex").write_text("\n".join(tex) + "\n")


def radar_values(cells_by_mode, data, out):
    """tables/radar_values.md: every radar value and score on both splits, cross-checked against metrics.md,
    completion_time_deviation.csv, latency_breakdown_arms.csv, the latency meta files (output tokens) and
    tokens.csv; stops on any difference"""
    tdir = Path(out) / "tables"
    lat = latency_arms(out)
    md = {}
    mode = split = None
    titles = {v: k for k, v in MODE_TITLE.items()}
    for line in (tdir / "metrics.md").read_text().splitlines():
        if line.startswith("## "):
            mode = titles.get(line[3:].strip())
        elif line.startswith("### "):
            split = line[4:].strip()
        elif line.startswith("| ") and mode and split in ("validation", "test") and not line.startswith("| model"):
            c = [x.strip() for x in line.strip("|").split(" | ")]
            arm = "base" if c[1].startswith("base") else ("selected" if "(selected" in c[1] else None)
            if arm:
                md[(mode, split, c[0], arm)] = c
    with open(tdir / "completion_time_deviation.csv", newline="") as f:
        days = {(r["mode"], r["split"], r["model"], "base" if r["arm"] == "base" else "selected"): r
                for r in csv.DictReader(f) if r["arm"] == "base" or "(selected)" in r["arm"]}
    problems = []
    for mode, d in data.items():
        sup = mode == "supervisor"
        for (split, m, arm), v in d["raw"].items():
            row = md[(mode, split, m, arm)]
            for ax, col in (("accuracy", 3), ("price", 6 if sup else 8)):
                if abs(round(v[ax], 2) - float(row[col])) > 0.0051:
                    problems.append(f"{mode} {split} {m} {arm} {ax}: {v[ax]} vs metrics.md {row[col]}")
            dc = days[(mode, split, m, arm)]
            ref = float(dc["mean days deviation (%)"] if sup else dc["mean |days deviation| (days)"])
            if abs(v["days"] - ref) > 5e-5:
                problems.append(f"{mode} {split} {m} {arm} days: {v['days']} vs csv {ref}")
            if split == "test":
                meta = cells_by_mode[mode][m]["latency_full"][arm]["summary"]
                if abs(v["latency"] - lat[(mode, m, arm)]["mean"] / 1000) > 1e-12:
                    problems.append(f"{mode} {m} {arm} latency differs from latency_breakdown_arms.csv")
                if abs(v["tokens"] - meta["output_tokens"]["mean"]) > 5e-5:
                    problems.append(f"{mode} {m} {arm} output tokens: tokens.csv {v['tokens']} vs latency_records "
                                    f"{meta['output_tokens']['mean']}")
    if problems:
        raise SystemExit("radar values disagree with the tables:\n  " + "\n  ".join(problems))
    unit = {"accuracy": "%", "price": "%", "days": "days", "latency": "s", "tokens": "tokens"}
    name = {"accuracy": "accuracy", "price": "price deviation", "days": "completion-time deviation",
            "latency": "latency", "tokens": "output tokens"}
    lines = ["# Radar values\n",
             "Six fine-tuned models per mode, each at its epoch selected on validation. Scores run from 0 to 10 and "
             "outward is better on every axis: accuracy scores 10 x (value - floor) / (100 - floor); price or cost "
             "deviation, completion-time deviation, latency and output tokens score 10 x (1 - value / reference). "
             "Each split's floor and references are the worst value among all twelve arms of the mode (six models, "
             "base and selected), rounded down (floor) or up (references) to a multiple of half a power of ten. "
             "The ranking uses the mean of the three validation quality scores, ties broken by the accuracy score, "
             "then alphabetically; the three top-ranked models are plotted with their test scores. Latency and "
             "output tokens were measured on the test split only.\n"]
    for mode, d in data.items():
        sup = mode == "supervisor"
        unit["days"] = "%" if sup else "days"
        name["accuracy"] = "selection accuracy" if sup else "decision agreement"
        name["price"] = "cost deviation" if sup else "price deviation"
        # values carry the metric's name, scores the axis title
        score = {ax: title.replace("\n", " ").lower() for ax, title in zip(RADAR_AXES, radar_titles(mode))}
        lines.append(f"\n## {MODE_TITLE[mode]}\n")
        for split, axes in (("validation", RADAR_AXES[:3]), ("test", RADAR_AXES)):
            refs = ", ".join(f"{name[ax]} {'floor' if ax == 'accuracy' else 'reference'} "
                             f"{d['refs'][split][ax]:g} {unit[ax]}" for ax in axes)
            lines.append(f"\n### {split}\n\n{refs}\n")
            head = ["model", "epoch"] + [f"{name[ax]} ({unit[ax]})" for ax in axes] + [f"{score[ax]} score" for ax in axes]
            head += ["mean score", "rank"] if split == "validation" else ["plotted"]
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|---|" + "---:|" * (len(head) - 1))
            for m in d["rank"]:
                v, sc = d["raw"][(split, m, "selected")], d["scores"][split][m]
                cells = [DISPLAY[m], str(cells_by_mode[mode][m]["selection"]["chosen"]["epoch"])]
                cells += [f"{v[ax]:.4f}" for ax in axes] + [f"{sc[ax]:.3f}" for ax in axes]
                cells += [f"{d['mean'][m]:.3f}", str(d["rank"].index(m) + 1)] if split == "validation" \
                    else ["yes" if m in d["top"] else "no"]
                lines.append("| " + " | ".join(cells) + " |")
    (tdir / "radar_values.md").write_text("\n".join(lines) + "\n")


def days_table(cells_by_mode, out):
    """tables/completion_time_deviation.md; returns the csv header and rows"""
    head = ["mode", "split", "model", "arm", "mean |days deviation| (days)", "mean days deviation (%)"]
    md_head = ["mode", "split", "model", "arm", "mean |days deviation| (days)", "mean |days deviation| (%)"]
    rows = []
    for mode in MODES:
        for split in ("validation", "test"):
            for m in MODELS:
                c = cells_by_mode[mode][m]
                chosen = c["selection"]["chosen"]["epoch"] if c["selection"] else None
                for k, a in arms_of(c, split).items():
                    mt = a["metrics"]
                    arm = "base" if k == "base" else f"epoch {k}" + (" (selected)" if k == chosen else
                                                                    (" (descriptive)" if split == "test" else ""))
                    if mode == "supervisor":
                        rows.append([mode, split, m, arm, None, mt.get("mean_abs_days_dev_pct")])
                    else:
                        rows.append([mode, split, m, arm, mt.get("macro_mean_abs_days_dev"), mt.get("macro_mean_days_dev_pct")])
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    md = ["# Completion-time deviation\n",
          "Sealed-bid and reverse auction: macro mean absolute deviation of the estimated completion days from the "
          "rule-based value, in days and in percent of the rule-based value. Supervisor: mean absolute deviation "
          "of the chosen plan's total days from the optimal plan's, in percent. The rule-based reference is 0.\n",
          "| " + " | ".join(md_cell(h) for h in md_head) + " |", "|" + "---|" * len(md_head)]
    md += ["| " + " | ".join(_fmt(x, 4) if isinstance(x, float) else ("-" if x is None else str(x)) for x in r) + " |"
           for r in rows]
    (tdir / "completion_time_deviation.md").write_text("\n".join(md) + "\n")
    return head, rows


def write_csvs(csv_out, tables):
    """only into a folder where none of the files exists yet, so a published csv is never overwritten"""
    clash = [name for name in tables if (csv_out / name).exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite {', '.join(clash)} in {csv_out}")
    csv_out.mkdir(parents=True, exist_ok=True)
    for name, (head, rows, as_dicts) in tables.items():
        with open(csv_out / name, "w", newline="") as f:
            if as_dicts:
                w = csv.DictWriter(f, fieldnames=head)
                w.writeheader()
            else:
                w = csv.writer(f)
                w.writerow(head)
            w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--csv-out", default="", help="folder for tokens.csv, completion_time_deviation.csv and "
                                                  "provenance.csv; none of them may exist there yet")
    ap.add_argument("--skip-original-profile", action="store_true",
                    help="omit the original-profile comparison; requires a separate --out folder")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.skip_original_profile and out.resolve() == OUT.resolve():
        ap.error("--skip-original-profile requires --out outside reports/v5/evaluation")
    out.mkdir(parents=True, exist_ok=True)
    from src.experiments.v5.latency_measurement import summarize as latency_tables
    cells_by_mode = {mode: {m: cell(m, mode) for m in MODELS} for mode in MODES}
    baselines = baselines_for()
    # the latency tables come first: the latency charts read latency_breakdown_arms.csv
    latency_tables.publish(LREC, out / "tables", compare_original=not args.skip_original_profile)
    prov = write_tables(cells_by_mode, out)
    days_head, days_rows = days_table(cells_by_mode, out)
    if args.csv_out:
        prov += latency_rows(cells_by_mode)
        prov_head = sorted({k for r in prov for k in r}, key=lambda k: (k.startswith("metric."), k))
        write_csvs(Path(args.csv_out), {"tokens.csv": (*token_rows(cells_by_mode), False),
                                        "completion_time_deviation.csv": (days_head, days_rows, False),
                                        "provenance.csv": (prov_head, prov, True)})
    written, radar = render_charts(cells_by_mode, baselines, out)
    if radar:
        radar_values(cells_by_mode, radar, out)
    token_summary(out)
    token_tables(out)
    print(f"wrote {len(written)} chart files and the tables -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
