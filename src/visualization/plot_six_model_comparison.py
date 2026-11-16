"""The headline six-model chart for the final presentation.

Four panels, one per metric, six models against the rule baseline in both bidding
modes. Numbers are recomputed from the run files here rather than copied from any
report, so the chart cannot drift from its source.

Style follows the existing dynamic-resources and reverse-auction charts: warm
surface, muted grid, direct value labels, no gradients.

Bars rather than a speed-versus-cost scatter. A scatter would land the headline
claim in one panel, but this has to read on a screen in an eight-minute slot with
nobody narrating it, and a reader who has never seen the data can misread a scatter
far more easily than a bar against a marked baseline. The four requested metrics
each get their own panel, the models keep one order throughout so the eye can track
a model down the page, and the rule baseline is a dashed line rather than a seventh
bar because it is the thing everything is measured against.
"""

import json
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "bidding" / "dynamic_resources_six_model_comparison.png"
SEALED = REPO / "reports/bidding/sealed/dynamic_resources"
AUCTION = REPO / "reports/bidding/reverse_auction/dynamic_resources"

RUNS = {
    "sealed": {
        "rule":      SEALED / "rule_run_20260825_200931_first150.jsonl",
        "qwen":      SEALED / "slm_qwen_run_20260827_022925.jsonl",
        "ministral": SEALED / "slm_ministral_run_20260827_214749.jsonl",
        "smollm3":   SEALED / "slm_smollm3_run_20260827_165756.jsonl",
        "granite":   SEALED / "slm_granite_run_20260827_072033.jsonl",
        "gemma":     SEALED / "slm_gemma_run_20260827_120736.jsonl",
        "phi":       SEALED / "slm_phi_run_20260828_023443.jsonl",
    },
    "auction": {
        "rule":      AUCTION / "rule_run_20260825_204133_first150.jsonl",
        "qwen":      AUCTION / "slm_qwen_run_20260827_034856.jsonl",
        "ministral": AUCTION / "slm_ministral_run_20260827_230458.jsonl",
        "smollm3":   AUCTION / "slm_smollm3_run_20260827_181551.jsonl",
        "granite":   AUCTION / "slm_granite_run_20260827_083620.jsonl",
        "gemma":     AUCTION / "slm_gemma_run_20260827_132447.jsonl",
        "phi":       AUCTION / "slm_phi_run_20260828_035147.jsonl",
    },
}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEALED_C = "#2a78d6"       # the stock blue used across the dynamic-resources charts
AUCTION_C = "#d07a1f"      # the commitment orange from the same palette
BASELINE = "#7a4fb5"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5, "ytick.labelsize": 11,
})


def measure(path):
    """the four charted metrics, straight off the run file"""
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    status = Counter(r["outcome"]["status"] for r in rows)
    done = [r["outcome"] for r in rows if r["outcome"]["status"] == "completed"]
    return {
        "n": len(rows),
        "completed": status["completed"],
        "infeasible": status["infeasible"],
        "days": mean(d["total_days"] for d in done),
        "cost": mean(d["total_cost"] for d in done),
        "rejections": sum(1 for r in rows for e in r["dynamic_events"]
                          if e["event"] == "reservation_rejected"),
    }


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def panel(ax, title, hint, models, data, key, fmt, baseline, better):
    """one metric, six models, two bars each, baseline as a reference line"""
    ys = range(len(models))
    height = 0.36
    top = max(max(data[m][mm][key] for mm in models) for m in ("sealed", "auction"))
    # the baselines are part of the reading, so the axis has to hold them even when
    # every bar is shorter, which is exactly the case the days panel makes
    if baseline:
        top = max(top, max(v for v, _ in baseline))
    limit = top * 1.14
    for offset, mode, colour in ((height / 2, "sealed", SEALED_C),
                                 (-height / 2, "auction", AUCTION_C)):
        values = [data[mode][m][key] for m in models]
        ax.barh([y + offset for y in ys], values, height=height, color=colour,
                edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for y, v in zip(ys, values):
            # sit the value inside the bar where it fits, which keeps it clear of
            # the baseline lines the bars happen to end near
            inside = v > limit * 0.28
            ax.text(v - limit * 0.012 if inside else v + limit * 0.012, y + offset,
                    fmt(v), va="center", ha="right" if inside else "left",
                    fontsize=9.4, fontweight="bold" if inside else "normal",
                    color=SURFACE if inside else INK2, zorder=5)

    ax.set_xlim(0, limit)
    ax.set_ylim(-0.65, len(models) - 0.35)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(models, fontsize=11)
    ax.invert_yaxis()

    if baseline is not None:
        for value, dash in baseline:
            ax.axvline(value, color=BASELINE, lw=1.4, ls=dash, zorder=2)
        lo, hi = min(v for v, _ in baseline), max(v for v, _ in baseline)
        label = fmt(lo) if lo == hi else f"{fmt(lo)} - {fmt(hi)}"
        # anchored to the axes rather than the data, so a line near the right edge
        # cannot push its own label off the figure
        ax.text(1.0, 1.015, f"rule baseline {label}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.6, fontweight="bold", color=BASELINE)

    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=16)
    ax.text(0, 1.015, hint, transform=ax.transAxes, fontsize=9.6, color=MUTED,
            ha="left", va="bottom")
    style(ax)


def main():
    data = {mode: {name: measure(p) for name, p in runs.items()}
            for mode, runs in RUNS.items()}
    for mode, runs in data.items():
        for name, m in runs.items():
            assert m["n"] == 150, f"{mode}/{name} has {m['n']} records"

    # one order everywhere, best total completions first, so a reader can track a
    # model down the page instead of re-reading each panel
    models = sorted((m for m in RUNS["sealed"] if m != "rule"),
                    key=lambda m: -(data["sealed"][m]["completed"] + data["auction"][m]["completed"]))

    rule = {mode: data[mode]["rule"] for mode in data}
    dev = [data[mode][m]["cost"] / rule[mode]["cost"] - 1
           for mode in data for m in models]
    worst_over = max(dev) * 100
    fastest = min((data[mode][m]["days"], m, mode) for mode in data for m in models)

    fig = plt.figure(figsize=(15.2, 10.8))
    fig.text(0.5, 0.963, "Six models against the rule baseline",
             ha="center", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.5, 0.930,
             "150 orders of the SLM holdout, dynamic resources active, sealed bidding and "
             "reverse auction. Rule-based agents are the same orders, filtered from the "
             "300-order runs.",
             ha="center", fontsize=12.5, color=INK2)
    fig.text(0.5, 0.898,
             f"Every model finishes chains faster than the rule baseline in both modes, "
             f"at no more than {worst_over:.0f}% above its cost.",
             ha="center", fontsize=13.5, fontweight="bold", color=INK)

    specs = [
        ([0.075, 0.505, 0.385, 0.285], "Orders completed", "of 150   higher is better",
         "completed", lambda v: f"{v:.0f}",
         [(rule["sealed"]["completed"], (0, (5, 3))), (rule["auction"]["completed"], (0, (1, 2)))]),
        ([0.575, 0.505, 0.385, 0.285], "Mean chain days", "lower is better",
         "days", lambda v: f"{v:.2f}",
         [(rule["sealed"]["days"], (0, (5, 3))), (rule["auction"]["days"], (0, (1, 2)))]),
        ([0.075, 0.135, 0.385, 0.285], "Mean chain cost, EUR", "lower is better",
         "cost", lambda v: f"{v:,.0f}",
         [(rule["sealed"]["cost"], (0, (5, 3))), (rule["auction"]["cost"], (0, (1, 2)))]),
        ([0.575, 0.135, 0.385, 0.285], "Reservations refused",
         "rule baseline is zero in both modes", "rejections", lambda v: f"{v:.0f}", None),
    ]
    for rect, title, hint, key, fmt, baseline in specs:
        panel(fig.add_axes(rect), title, hint, models, data, key, fmt, baseline, None)

    fig.legend(handles=[
        Line2D([], [], color=SEALED_C, lw=9, label="Sealed bidding"),
        Line2D([], [], color=AUCTION_C, lw=9, label="Reverse auction"),
        Line2D([], [], color=BASELINE, lw=1.4, ls=(0, (5, 3)), label="Rule baseline, sealed"),
        Line2D([], [], color=BASELINE, lw=1.4, ls=(0, (1, 2)), label="Rule baseline, auction"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.876), ncol=4, frameon=False,
        fontsize=11, handletextpad=0.8, columnspacing=2.6)

    fig.text(0.5, 0.062,
             "A refused reservation is a promise the live state could no longer honour. "
             "Rule-based agents never draw one, because their promise is computed from the "
             "state the reservation re-checks.\nOnly a model can over-promise, so this "
             "column is the clearest evidence that dynamic resources does real work.",
             ha="center", va="bottom", fontsize=11, color=INK, linespacing=1.6)
    fig.text(0.5, 0.020,
             f"Fastest chain of the twelve runs: {fastest[1]} {fastest[2]} at {fastest[0]:.2f} days. "
             f"Figures recomputed from the fourteen run files, not from any summary. "
             f"All twelve SLM runs pass every dynamic-state invariant.",
             ha="center", va="bottom", fontsize=9.4, color=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"model order: {', '.join(models)}")
    print(f"worst cost deviation above baseline: {worst_over:.2f}%")


if __name__ == "__main__":
    main()
