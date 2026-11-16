"""Plot the supervisor model comparison across the three training states."""

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
S = load_chart_data()["supervisor"]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
E1, E2, REF = "#2a78d6", "#eb6834", "#1baf7a"
BASE = "#b3b1a9"

ARMS = [("base", "Untrained\nbase model", BASE), ("epoch_1", "1 epoch", E1), ("epoch_2", "2 epochs", E2)]
STRATS = [("urgent", "Urgent\n(fastest wins)"), ("normal", "Normal\n(balanced blend)"),
          ("flexible", "Flexible\n(cheapest wins)")]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})


def style(ax, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    getattr(ax, f"{axis}axis").grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    other = "x" if axis == "y" else "y"
    getattr(ax, f"{other}axis").grid(False)


fig = plt.figure(figsize=(17.5, 11.4))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.26,
                      left=0.065, right=0.982, top=0.875, bottom=0.140)

fig.text(0.5, 0.966, "Supervisor model: picking the best supply chain from a shortlist",
         ha="center", fontsize=22, fontweight="bold", color=INK)
fig.text(0.5, 0.930,
         "A separate model from the six stage bidders. It reads up to 12 candidate chains and picks one. "
         "Here a second epoch clearly helped.",
         ha="center", fontsize=13, color=INK2)

# A. selection accuracy
ax = fig.add_subplot(gs[0, 0])
vals = [S[a]["accuracy"] for a, _, _ in ARMS]
cols = [c for _, _, c in ARMS]
bars = ax.bar(range(3), vals, color=cols, width=0.6, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color=cols[i])
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in ARMS])
ax.set_ylim(0, 116); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Correct pick (%)", fontsize=10)
ax.set_title("A. How often it picks the same chain as the rule", loc="left",
             fontweight="bold", fontsize=12, pad=10, color=INK)
style(ax)

# B. mean regret, log scale
ax = fig.add_subplot(gs[0, 1])
vals = [S[a]["mean_regret"] for a, _, _ in ARMS]
ax.bar(range(3), vals, color=cols, width=0.6, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v * 1.35, f"{v:.3f}%", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color=cols[i])
ax.set_yscale("log"); ax.set_ylim(0.012, 90)
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in ARMS])
ax.set_ylabel("Mean regret (%), log scale", fontsize=10)
ax.set_title("B. How much worse its pick is when it differs", loc="left",
             fontweight="bold", fontsize=12, pad=10, color=INK)
ax.text(0.63, 0.52, "Regret is 0 when the chain it picks\nis just as good as the rule's own.\n\n"
                    "Log scale. The untrained model is\nroughly 560 times worse than 2 epochs.",
        transform=ax.transAxes, ha="center", va="center", fontsize=8.8, color=INK2,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f3ef", ec=AXIS, lw=0.7))
style(ax)

# C. zero-regret share
ax = fig.add_subplot(gs[0, 2])
vals = [S[a]["zero_regret"] for a, _, _ in ARMS]
ax.bar(range(3), vals, color=cols, width=0.6, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color=cols[i])
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in ARMS])
ax.set_ylim(0, 116); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Share of picks with zero regret (%)", fontsize=10)
ax.set_title("C. Picks that are as good as the rule's own choice", loc="left",
             fontweight="bold", fontsize=12, pad=10, color=INK)
style(ax)

# D. per-strategy accuracy, 1 vs 2 epochs
ax = fig.add_subplot(gs[1, 0])
v1 = [S["epoch_1"]["by_strategy"][k]["accuracy"] for k, _ in STRATS]
v2 = [S["epoch_2"]["by_strategy"][k]["accuracy"] for k, _ in STRATS]
for i, (a, b) in enumerate(zip(v1, v2)):
    ax.plot([a, b], [i, i], color=AXIS, lw=1.8, zorder=1, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((a, i), (b, i), arrowstyle="-|>", mutation_scale=11,
                                 color=MUTED, lw=0, zorder=2))
    ax.plot([a], [i], "o", ms=10, color=E1, mec=SURFACE, mew=1.7, zorder=3)
    ax.plot([b], [i], "o", ms=10, color=E2, mec=SURFACE, mew=1.7, zorder=4)
    ax.text(a - 0.3, i, f"{a:.2f}", ha="right", va="center", fontsize=10,
            fontweight="bold", color=E1)
    ax.text(b + 0.3, i, f"{b:.2f}", ha="left", va="center", fontsize=10,
            fontweight="bold", color=E2)
ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in STRATS])
ax.set_ylim(2.5, -0.5); ax.set_xlim(87.5, 102.5); ax.set_xticks([90, 95, 100])
ax.set_xlabel("Correct pick (%)", fontsize=10)
ax.set_title("D. Accuracy by order priority: 2 epochs wins all three", loc="left",
             fontweight="bold", fontsize=12, pad=10, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0); ax.set_axisbelow(True)

# E. per-strategy regret
ax = fig.add_subplot(gs[1, 1:3])
w = 0.34
r1 = [S["epoch_1"]["by_strategy"][k]["regret"] for k, _ in STRATS]
r2 = [S["epoch_2"]["by_strategy"][k]["regret"] for k, _ in STRATS]
x = range(3)
ax.bar([i - w / 2 - 0.012 for i in x], r1, width=w, color=E1, zorder=3, label="1 epoch")
ax.bar([i + w / 2 + 0.012 for i in x], r2, width=w, color=E2, zorder=3, label="2 epochs")
for i in x:
    ax.text(i - w / 2 - 0.012, r1[i] + 0.003, f"{r1[i]:.3f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=E1)
    lab = "0.000" if r2[i] == 0 else f"{r2[i]:.3f}"
    ax.text(i + w / 2 + 0.012, r2[i] + 0.003, lab, ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=E2)
ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in STRATS])
ax.set_ylim(0, 0.098)
ax.set_ylabel("Mean regret (%)", fontsize=10)
ax.set_title("E. Regret by order priority (lower is better)", loc="left",
             fontweight="bold", fontsize=12, pad=10, color=INK)
ax.legend(frameon=False, fontsize=10, loc="upper right")
style(ax)

handles = [Line2D([], [], marker="s", ls="", ms=11, mfc=BASE, mec="none",
                  label=f"Untrained base model (scored on {S['base']['n']} orders)"),
           Line2D([], [], marker="s", ls="", ms=11, mfc=E1, mec="none",
                  label=f"1 epoch (scored on {S['epoch_1']['n']:,} orders)"),
           Line2D([], [], marker="s", ls="", ms=11, mfc=E2, mec="none",
                  label=f"2 epochs (scored on {S['epoch_2']['n']:,} orders)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11.5,
           handletextpad=0.7, columnspacing=2.6, bbox_to_anchor=(0.5, 0.048))
fig.text(0.5, 0.022,
         "Drawn from evaluation files already on disk: reports/supervisor/epoch-1, "
         "reports/supervisor/epoch-2, and reports/supervisor/smoke for the untrained baseline. "
         "Nothing was retrained or re-scored.",
         ha="center", fontsize=9.2, color=MUTED)
fig.text(0.5, 0.006,
         "The untrained baseline was only ever scored on the smaller 150-order sample, so read it as a "
         "floor rather than a like-for-like number. All three produced valid JSON 100% of the time.",
         ha="center", fontsize=9.2, color=MUTED)

out = REPO / "reports" / "supervisor" / "supervisor_model_comparison.png"
fig.savefig(out, dpi=135, facecolor=SURFACE)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
