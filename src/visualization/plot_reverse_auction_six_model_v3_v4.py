"""Create the six-model v3-against-v4 comparison chart.

All six stage models, each trained once on v3 and once on the v4 worded dataset,
same protocol throughout. The question this answers is whether the minimum-price
rounding fix carried in v4 resolves the at_floor gate for every model or only for
Qwen.

Distinct from reverse_auction_v3_vs_v4.png, which is the Qwen-only three-arm
wording ablation. This file is the six-model version comparison.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

REPO = Path(__file__).resolve().parents[2]
EPOCH = REPO / "reports" / "reverse_auction" / "epoch-1"
OUT = REPO / "reports" / "reverse_auction"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OLD, NEW, REF = "#8a8880", "#2a78d6", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5, "ytick.labelsize": 10.5,
})

# (model key, display name); v3 arms carry the -fixed suffix from the re-scoring
MODELS = [
    ("qwen", "Qwen3.5-4B"),
    ("granite", "Granite-4.1-3b"),
    ("phi", "Phi-4-mini"),
    ("ministral", "Ministral-3-3B"),
    ("gemma", "Gemma-4-E2B"),
    ("smollm3", "SmolLM3-3B"),
]


def load(version, tag):
    """(gate result, evaluation summary) for one arm."""
    hits = sorted((EPOCH / version / "gates").glob(f"eval_{tag}_*_gates.json"))
    if not hits:
        raise FileNotFoundError(f"no gate file for {tag} under {EPOCH / version / 'gates'}")
    with hits[-1].open(encoding="utf-8") as handle:
        gates = json.load(handle)
    with (REPO / gates["eval_result"]).open(encoding="utf-8") as handle:
        return gates, json.load(handle)["summary"]


def arms(model):
    """(v3, v4) for one model, each as (gates, summary)."""
    return load("v3", f"ra-e1-{model}-fixed"), load("v4", f"ra-e1-{model}")


def style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=6)


def dumbbell(ax, rows, xlim, fmt="{:.1f}", target=None):
    """one v3-to-v4 row per model, both ends labelled"""
    span = xlim[1] - xlim[0]
    pad = span * 0.022
    halo = [path_effects.withStroke(linewidth=3.0, foreground=SURFACE)]
    for i, (before, after) in enumerate(rows):
        ax.plot([before, after], [i, i], color=AXIS, lw=1.7, zorder=1,
                solid_capstyle="round")
        if abs(after - before) > span * 0.01:
            ax.add_patch(FancyArrowPatch((before, i), (after, i), arrowstyle="-|>",
                                         mutation_scale=10, color=MUTED, lw=0, zorder=2))
        ax.plot([before], [i], "o", ms=8, color=OLD, mec=SURFACE, mew=1.4, zorder=3)
        ax.plot([after], [i], "o", ms=8, color=NEW, mec=SURFACE, mew=1.4, zorder=4)
        left, right = (before, after) if before <= after else (after, before)
        lcol, rcol = (OLD, NEW) if before <= after else (NEW, OLD)
        ax.text(left - pad, i, fmt.format(left), ha="right", va="center", fontsize=9.5,
                fontweight="bold", color=lcol, zorder=6, path_effects=halo)
        if abs(after - before) > 1e-9:
            ax.text(right + pad, i, fmt.format(right), ha="left", va="center",
                    fontsize=9.5, fontweight="bold", color=rcol, zorder=6,
                    path_effects=halo)
    if target is not None:
        ax.axvline(target, color=REF, lw=2.0, zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for _, name in MODELS])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    style_axis(ax)


def save(fig, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def chart():
    data = {m: arms(m) for m, _ in MODELS}

    fig = plt.figure(figsize=(15.0, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34,
                          left=0.115, right=0.975, top=0.800, bottom=0.115)

    fig.text(0.035, 0.945,
             "Does the minimum-price fix hold for every model, or only for Qwen?",
             ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.035, 0.902,
             "All six stage models, trained once on v3 and once on the v4 worded dataset. "
             "Same protocol, same 1,296 held-out prompts.",
             ha="left", fontsize=11, color=INK2)

    def panel(pos, letter, title, note, values, xlim, fmt="{:.1f}", target=None, xlabel=""):
        ax = fig.add_subplot(gs[pos])
        dumbbell(ax, values, xlim, fmt=fmt, target=target)
        ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold",
                     fontsize=12.5, pad=20, color=INK)
        ax.text(0, 1.035, note, transform=ax.transAxes, fontsize=9.5, color=MUTED,
                va="bottom", style="italic")
        ax.set_xlabel(xlabel, fontsize=9.5)
        return ax

    at_floor = [(data[m][0][1]["by_floor_case"]["floor_boundary:at_floor"]["accuracy_pct"],
                 data[m][1][1]["by_floor_case"]["floor_boundary:at_floor"]["accuracy_pct"])
                for m, _ in MODELS]
    panel((0, 0), "A", "Offer exactly at the minimum price",
          "the gate the rounding fix was meant to repair, 132 prompts",
          at_floor, (-8, 118), fmt="{:.1f}", target=100,
          xlabel="Answers matching the rule (%)")

    gates = [(sum(1 for g in data[m][0][0]["gates"].values() if g["status"] == "PASS"),
              sum(1 for g in data[m][1][0]["gates"].values() if g["status"] == "PASS"))
             for m, _ in MODELS]
    ax = panel((0, 1), "B", "Release gates cleared, out of 11",
               "the eleven bars agreed in advance",
               gates, (-0.8, 12.8), fmt="{:.0f}", target=11, xlabel="Gates passed")
    ax.set_xticks([0, 2, 4, 6, 8, 10])

    overall = [(data[m][0][1]["overall"]["accuracy_pct"],
                data[m][1][1]["overall"]["accuracy_pct"]) for m, _ in MODELS]
    panel((1, 0), "C", "Agrees with the rule overall",
          "all 1,296 held-out prompts",
          overall, (86, 106), fmt="{:.1f}", target=100,
          xlabel="Answers matching the rule (%)")

    offer = [(data[m][0][1]["overall"]["offer_exact_match_pct"],
              data[m][1][1]["overall"]["offer_exact_match_pct"]) for m, _ in MODELS]
    panel((1, 1), "D", "Names the exact price the rule names",
          "to the cent, on every prompt where a bid was due",
          offer, (57, 75), fmt="{:.1f}", xlabel="Exact price matches (%)")

    handles = [
        Line2D([], [], marker="o", ls="", ms=9, mfc=OLD, mec="none", label="v3"),
        Line2D([], [], marker="o", ls="", ms=9, mfc=NEW, mec="none", label="v4 worded"),
        Line2D([], [], color=REF, lw=2.2, label="Correct answer / all gates"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.032), columnspacing=2.8)
    fig.text(0.035, 0.010,
             "v3's training data showed the minimum price a cent too low in half the floor "
             "examples, so its “one cent below” cases read as “exactly at”. v4 corrected that. "
             "Sources: reports/reverse_auction/epoch-1/v3 and /v4.",
             ha="left", va="bottom", fontsize=9, color=MUTED)
    return save(fig, "reverse_auction_six_model_v3_vs_v4.png")


if __name__ == "__main__":
    chart()
