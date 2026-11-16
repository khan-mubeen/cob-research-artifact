"""Plot the epoch 1 and epoch 2 results for the six stage bidding models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

if __package__:
    from .extract_chart_data import load_chart_data
else:
    from extract_chart_data import load_chart_data


REPO = Path(__file__).resolve().parents[2]
D = load_chart_data()["stage"]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
E1, E2, REF, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"

ORDER = ["qwen", "granite", "phi", "gemma", "ministral", "smollm3"]
LABELS = [D[k]["label"] for k in ORDER]
STAGES = [("material", "Material"), ("tailoring", "Tailoring"), ("logistics", "Logistics")]
CLASSES = [("above_infeasible", "Above the line, should refuse"),
           ("below_feasible", "Below the line, should bid"),
           ("exact_feasible", "Exactly on the line, should bid")]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.labelsize": 9, "ytick.labelsize": 10,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)


def dumbbell(
    ax,
    v1,
    v2,
    ylab,
    fmt,
    xlim,
    pad,
    show_y=True,
    err1=None,
    err2=None,
    labels1=None,
    labels2=None,
    label_offsets=None,
):
    for i, (a, b) in enumerate(zip(v1, v2)):
        ax.plot([a, b], [i, i], color=AXIS, lw=1.6, zorder=1, solid_capstyle="round")
        if abs(b - a) > (xlim[1] - xlim[0]) * 0.012:
            ax.add_patch(FancyArrowPatch((a, i), (b, i), arrowstyle="-|>", mutation_scale=9,
                                         color=MUTED, lw=0, zorder=2))
        if err1 is not None:
            ax.errorbar(
                a,
                i,
                xerr=[[min(a, err1[i])], [err1[i]]],
                fmt="none",
                ecolor=E1,
                elinewidth=1.25,
                capsize=3,
                alpha=0.72,
                zorder=2,
            )
        if err2 is not None:
            ax.errorbar(
                b,
                i,
                xerr=[[min(b, err2[i])], [err2[i]]],
                fmt="none",
                ecolor=E2,
                elinewidth=1.25,
                capsize=3,
                alpha=0.72,
                zorder=2,
            )
        ax.plot([a], [i], "o", ms=8, color=E1, zorder=3, mec=SURFACE, mew=1.6)
        ax.plot([b], [i], "o", ms=8, color=E2, zorder=4, mec=SURFACE, mew=1.6)
        lo, hi = (a, b) if a <= b else (b, a)
        clo, chi = (E1, E2) if a <= b else (E2, E1)
        label_a = labels1[i] if labels1 is not None else fmt.format(a)
        label_b = labels2[i] if labels2 is not None else fmt.format(b)
        if label_offsets is not None:
            ax.annotate(
                label_a,
                (a, i),
                xytext=(0, label_offsets[0]),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.5,
                color=E1,
                fontweight="bold",
            )
            ax.annotate(
                label_b,
                (b, i),
                xytext=(0, label_offsets[1]),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.5,
                color=E2,
                fontweight="bold",
            )
        else:
            label_lo, label_hi = (label_a, label_b) if a <= b else (label_b, label_a)
            ax.text(lo - pad, i, label_lo, va="center", ha="right", fontsize=8.5,
                    color=clo, fontweight="bold")
            ax.text(hi + pad, i, label_hi, va="center", ha="left", fontsize=8.5,
                    color=chi, fontweight="bold")
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab if show_y else [""] * len(ylab))
    ax.set_ylim(len(ylab) - 0.45, -0.65)
    ax.set_xlim(*xlim)
    style(ax)


def note(ax, txt, color, xy=(0.985, 0.045), ha="right"):
    ax.text(*xy, txt, transform=ax.transAxes, ha=ha, va="bottom", fontsize=8.6,
            color=color, fontweight="bold")


fig = plt.figure(figsize=(17.5, 24))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1],
                      hspace=0.42, wspace=0.16, left=0.072, right=0.985, top=0.925, bottom=0.085)

fig.text(0.5, 0.978, "Stage bidding models: what a second training epoch changed",
         ha="center", fontsize=22, fontweight="bold", color=INK)
fig.text(0.5, 0.958,
         "Every model improved on the everyday task. None improved at the decision boundary. "
         "Blue dot is epoch 1, orange dot is epoch 2.",
         ha="center", fontsize=13, color=INK2)

# Row 1: decision agreement per stage
for j, (skey, sname) in enumerate(STAGES):
    ax = fig.add_subplot(gs[0, j])
    v1 = [D[k]["epoch_1"]["decisions"][skey] for k in ORDER]
    v2 = [D[k]["epoch_2"]["decisions"][skey] for k in ORDER]
    dumbbell(ax, v1, v2, LABELS, "{:.1f}", (96.9, 101.4), 0.13, show_y=(j == 0))
    ax.axvline(100, color=REF, lw=1.4, zorder=1)
    ax.set_title(f"{'ABC'[j]}. Everyday decision agreement, {sname.lower()} stage",
                 loc="left", fontweight="bold", pad=9, fontsize=11.5, color=INK)
    ax.set_xlabel("Agreement with the rule (%)", fontsize=9)
    ax.set_xticks([97, 98, 99, 100, 101])
    if j == 0:
        note(ax, "green line = rule-based, 100%", REF)

# Row 2: pricing error (wide) + JSON validity
ax = fig.add_subplot(gs[1, 0:2])
stats1 = [D[k]["epoch_1"]["offer_dev_stats"]["pooled"] for k in ORDER]
stats2 = [D[k]["epoch_2"]["offer_dev_stats"]["pooled"] for k in ORDER]
v1 = [stats["mean"] for stats in stats1]
v2 = [stats["mean"] for stats in stats2]
std1 = [stats["std"] for stats in stats1]
std2 = [stats["std"] for stats in stats2]
dumbbell(
    ax,
    v1,
    v2,
    LABELS,
    "{:.2f}",
    (-0.12, 5.75),
    0.07,
    err1=std1,
    err2=std2,
    labels1=[f"{mean:.2f} (SD {std:.2f})" for mean, std in zip(v1, std1)],
    labels2=[f"{mean:.2f} (SD {std:.2f})" for mean, std in zip(v2, std2)],
    label_offsets=(11, -11),
)
ax.axvline(0, color=REF, lw=1.4, zorder=1)
ax.set_title("D. Pricing error across all scorable offer rows (lower is better)",
             loc="left", fontweight="bold", pad=9, fontsize=11.5, color=INK)
ax.set_xlabel("Mean absolute offer deviation with sample SD (%)", fontsize=9)
ax.text(0.985, 0.16, "Base-model mean deviations were 25% to 91%, outside this scale.\n"
                     "Epoch 2 reduced the pooled mean for all six models.\n"
                     "Error bars show sample SD and stop at 0%.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.2, color=INK2,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f3ef", ec=AXIS, lw=0.7))

ax = fig.add_subplot(gs[1, 2])
v1 = [D[k]["epoch_1"]["json_valid"] for k in ORDER]
v2 = [D[k]["epoch_2"]["json_valid"] for k in ORDER]
dumbbell(ax, v1, v2, LABELS, "{:.2f}", (98.82, 100.42), 0.05, show_y=False)
ax.axvline(100, color=REF, lw=1.4, zorder=1)
ax.set_title("E. Output format validity", loc="left", fontweight="bold", pad=9,
             fontsize=11.5, color=INK)
ax.set_xlabel("Valid JSON (%), axis starts at 98.8", fontsize=9)
ax.set_xticks([99.0, 99.5, 100.0])
note(ax, "100% is required", REF, xy=(0.015, 0.045), ha="left")

# Row 3: baseline six (wide) + pooled boundary score
ax = fig.add_subplot(gs[2, 0:2])
v1 = [int(D[k]["epoch_1"]["baseline_six"].split("/")[0]) for k in ORDER]
v2 = [int(D[k]["epoch_2"]["baseline_six"].split("/")[0]) for k in ORDER]
dumbbell(ax, v1, v2, LABELS, "{:.0f}", (-0.7, 7.6), 0.15)
for i in range(len(ORDER)):
    ax.plot([6], [i], "D", ms=8.5, color=REF, mec=SURFACE, mew=1.5, zorder=5)
ax.set_title("F. The six hand-built refusal cases, before and after the safety guard",
             loc="left", fontweight="bold", pad=9, fontsize=11.5, color=INK)
ax.set_xlabel("Correct refusals out of 6", fontsize=9)
ax.set_xticks(range(0, 8))
note(ax, "green diamond = 6 of 6 once Guard B is applied, in both epochs, for every model", REF,
     xy=(0.018, 0.93), ha="left")

ax = fig.add_subplot(gs[2, 2])
v1 = [D[k]["epoch_1"]["robustness_total"] for k in ORDER]
v2 = [D[k]["epoch_2"]["robustness_total"] for k in ORDER]
dumbbell(ax, v1, v2, LABELS, "{:.0f}", (5.5, 24.5), 0.45, show_y=False)
ax.axvspan(12, 13, color=E2, alpha=0.13, zorder=0)
ax.set_title("G. Boundary score, all case types pooled", loc="left", fontweight="bold",
             pad=9, fontsize=11.5, color=INK)
ax.set_xlabel("Correct out of 36", fontsize=9)
ax.set_xticks([6, 9, 12, 15, 18, 21, 24])
note(ax, "shaded band = 12 to 13, where all six ended up", E2, xy=(0.985, 0.045))

# Row 4: boundary score by case type
for j, (ckey, cname) in enumerate(CLASSES):
    ax = fig.add_subplot(gs[3, j])
    v1 = [D[k]["epoch_1"]["robustness"][ckey] for k in ORDER]
    v2 = [D[k]["epoch_2"]["robustness"][ckey] for k in ORDER]
    dumbbell(ax, v1, v2, LABELS, "{:.0f}", (-1.1, 15.2), 0.32, show_y=(j == 0))
    ax.axvline(9, color=CRIT, lw=1.6, zorder=1)
    ax.set_title(f"{'HIJ'[j]}. Boundary test: {cname}", loc="left", fontweight="bold",
                 pad=9, fontsize=11.5, color=INK)
    ax.set_xlabel("Correct out of 12", fontsize=9)
    ax.set_xticks([0, 3, 6, 9, 12])
    if j == 0:
        note(ax, "red line = pass mark, 9 of 12", CRIT, xy=(0.018, 0.045), ha="left")

handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=E1, mec=SURFACE, label="Epoch 1"),
           Line2D([], [], marker="o", ls="", ms=9, mfc=E2, mec=SURFACE, label="Epoch 2"),
           Line2D([], [], color=REF, lw=2.4, label="Rule-based reference"),
           Line2D([], [], marker="D", ls="", ms=8.5, mfc=REF, mec=SURFACE, label="With Guard B"),
           Line2D([], [], color=CRIT, lw=2.4, label="Pass mark, 9 of 12 in every case type")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11.5,
           handletextpad=0.7, columnspacing=2.4, bbox_to_anchor=(0.5, 0.030))
fig.text(0.5, 0.010,
         "Drawn from evaluation files already on disk (reports/epoch-1/production and "
         "reports/epoch-2/production). Nothing was retrained or re-scored. "
         "Both epochs are scored on the same 1,626 rows per model.",
         ha="center", fontsize=9.5, color=MUTED)

out = REPO / "reports" / "epoch-2" / "production" / "epoch2_stage_models_comparison.png"
fig.savefig(out, dpi=135, facecolor=SURFACE)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
