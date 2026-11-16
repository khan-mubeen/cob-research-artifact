"""Create the three-way reverse-auction comparison chart.

Two panels: the eleven release gates as a table, and a short accuracy plot
carrying only the slices the gate table does not already state. Three arms, all
Qwen, all one epoch on the same protocol:

  v3             bare symbols, training data still showing the floor a cent low
  v4 worded      operators also written in words, floor display corrected
  v4 control     the same corrected display, bare symbols again

v4 changed the wording and the floor display together. The control holds the
correction and reverts the wording, so the two effects separate.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
EPOCH = REPO / "reports" / "reverse_auction" / "epoch-1"
OUT = REPO / "reports" / "reverse_auction"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# v3 is the de-emphasised prior version, so it takes a neutral stepped to clear
# 3:1 on the surface; the two v4 arms are categorical slots 1 and 2, which clear
# the all-pairs CVD and normal-vision floors against it
OLD, WORDED, CONTROL, MISS = "#8a8880", "#2a78d6", "#eb6834", "#d03b3b"
BAND = "#f4f3ef"

# (folder, gate-file tag, short label, colour, marker size). arms land on the same
# value on most slices, so they are drawn concentrically, largest first: every arm
# in a coincident stack stays visible as its own ring
ARMS = [
    ("v3", "ra-e1-qwen-fixed", "v3", OLD, 12.0),
    ("v4", "ra-e1-qwen", "v4 worded", WORDED, 8.0),
    ("v4-control-bare-symbols", "ra-e1-qwen-control", "v4 control", CONTROL, 4.6),
]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5, "ytick.labelsize": 10,
})

GATE_ROWS = [
    ("3_floor_above_floor", "Offer above the minimum price", "90%"),
    ("2_floor_at_floor", "Offer exactly at the minimum price", "90%"),
    ("1_floor_one_cent_below_floor", "Offer one cent below the minimum", "90%"),
    ("4_bid_rate_priority_urgent", "Bid-rate gap, urgent orders", "10 pts"),
    ("4_bid_rate_priority_normal", "Bid-rate gap, normal orders", "10 pts"),
    ("4_bid_rate_priority_flexible", "Bid-rate gap, flexible orders", "10 pts"),
    ("5_bid_rate_stage_material", "Bid-rate gap, material stage", "10 pts"),
    ("5_bid_rate_stage_tailoring", "Bid-rate gap, tailoring stage", "10 pts"),
    ("5_bid_rate_stage_logistics", "Bid-rate gap, logistics stage", "10 pts"),
    ("6_json_validity", "Valid JSON", "99%"),
    ("7_days_no_regression", "Delivery-time figure vs untrained", "no drop"),
]

# (summary section, key inside it, row label); a None section is a group heading.
# only slices the gate table does not already carry: the headline number, and the
# round split, where v3 degrades as the price walks down toward the floor while
# both v4 arms stay flat. the priority, stage and floor-case splits were dropped
# because the gate table states them, and the rest moved by less than a point
SLICE_ROWS = [
    ("overall", None, "All 1,296 prompts"),
    (None, None, "By auction round"),
    ("by_round", "round_1", "Round 1"),
    ("by_round", "round_2", "Round 2"),
    ("by_round", "round_3", "Round 3"),
]


def load_arm(version, tag):
    """(gate result, evaluation summary) for one arm."""
    hits = sorted((EPOCH / version / "gates").glob(f"eval_{tag}_*_gates.json"))
    if not hits:
        raise FileNotFoundError(f"no gate file for {tag} under {EPOCH / version / 'gates'}")
    with hits[-1].open(encoding="utf-8") as handle:
        gates = json.load(handle)
    with (REPO / gates["eval_result"]).open(encoding="utf-8") as handle:
        return gates, json.load(handle)["summary"]


def section(summary, name, key):
    """One slice out of an evaluation summary, whether or not it is nested."""
    return summary[name] if key is None else summary[name][key]


def save(fig, filename):
    """Save one chart to its established report path."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def gate_table(ax, arms, data):
    """The eleven gates, one numeric column per arm, a miss called out in words."""
    ax.set_xlim(0, 1)
    ax.set_ylim(len(GATE_ROWS) + 0.85, -1.9)
    ax.axis("off")

    x_label, x_bar = 0.0, 0.425
    columns = [0.585, 0.788, 0.985]

    ax.text(x_label, -1.42, "Release gate", ha="left", va="center", fontsize=9.5, color=MUTED)
    ax.text(x_bar, -1.42, "bar", ha="right", va="center", fontsize=9.5, color=MUTED)
    for x, (_, _, label, colour, _size) in zip(columns, arms):
        head, _, tail = label.partition(" ")
        ax.text(x, -1.42, head, ha="right", va="center", fontsize=11.5,
                fontweight="bold", color=colour)
        if tail:
            ax.text(x, -0.92, tail, ha="right", va="center", fontsize=9.5, color=colour)
    ax.plot([0, 1], [-0.62, -0.62], color=AXIS, lw=1.0)

    for i, (key, label, bar) in enumerate(GATE_ROWS):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, i - 0.42), 1, 0.84, facecolor=BAND,
                                   edgecolor="none", zorder=0))
        ax.text(x_label, i, label, ha="left", va="center", fontsize=10.5, color=INK, zorder=2)
        ax.text(x_bar, i, bar, ha="right", va="center", fontsize=9.5, color=MUTED, zorder=2)
        for x, (gates, _), (_, _, _, colour, _size) in zip(columns, data, arms):
            gate = gates["gates"][key]
            missed = gate["status"] != "PASS"
            ax.text(x, i, f"{gate['measured']:g}", ha="right", va="center", fontsize=10.5,
                    color=MISS if missed else colour,
                    fontweight="bold" if missed else "normal", zorder=2)
            # never colour alone: the miss is named
            if missed:
                ax.text(x - 0.075, i, "miss", ha="right", va="center", fontsize=9,
                        color=MISS, fontweight="bold", zorder=2)

    rule = len(GATE_ROWS) - 0.5
    ax.plot([0, 1], [rule, rule], color=AXIS, lw=1.0)
    for x, (gates, _), (_, _, _, colour, _size) in zip(columns, data, arms):
        ax.text(x, rule + 0.45, f"{gates['counts']['PASS']} of 11", ha="right",
                va="center", fontsize=10.5, color=colour, fontweight="bold")


def slice_plot(ax, arms, data):
    """Holdout accuracy on the few slices the gate table does not already carry."""
    summaries = [summary for _, summary in data]
    marks = [(colour, size) for _, _, _, colour, size in arms]

    labels, rows, headings = [], [], set()
    for name, key, label in SLICE_ROWS:
        labels.append(label)
        if name is None:
            headings.add(len(rows))
            rows.append(None)
        else:
            rows.append([section(s, name, key)["accuracy_pct"] for s in summaries])

    xlim = (88, 102.4)
    span = xlim[1] - xlim[0]
    halo = [path_effects.withStroke(linewidth=3.0, foreground=SURFACE)]
    for i, values in enumerate(rows):
        if values is None:
            continue
        lo, hi = min(values), max(values)
        ax.plot([lo, hi], [i, i], color=OLD, lw=1.6, alpha=0.55, zorder=1,
                solid_capstyle="round")
        # largest first, so a smaller arm landing on the same value stays visible
        for value, (colour, size) in sorted(zip(values, marks),
                                            key=lambda pair: pair[1][1], reverse=True):
            ax.plot([value], [i], "o", ms=size, color=colour, mec=SURFACE, mew=1.1,
                    zorder=3 + (12 - size))
        # few enough rows now that every value can be read off directly
        ax.text(lo - span * 0.022, i, f"{lo:.1f}", ha="right", va="center",
                fontsize=9.5, color=INK2, zorder=6, path_effects=halo)
        if hi - lo > 0.05:
            ax.text(hi + span * 0.022, i, f"{hi:.1f}", ha="left", va="center",
                    fontsize=9.5, fontweight="bold", color=INK2, zorder=6,
                    path_effects=halo)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    for i, tick in enumerate(ax.get_yticklabels()):
        if i in headings:
            tick.set_fontsize(9.5)
            tick.set_color(MUTED)
        else:
            tick.set_fontsize(10.5)
            tick.set_color(INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xticks([90, 95, 100])
    ax.set_xlabel("Answers matching the rule (%)", fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=6)


def chart():
    """The gates and the slices, three arms side by side."""
    data = [load_arm(version, tag) for version, tag, _, _, _ in ARMS]

    fig = plt.figure(figsize=(14.2, 7.6))
    fig.text(0.035, 0.940, "Qwen on the reverse auction: was it the wording, or the data fix?",
             ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.035, 0.897,
             "Three arms, same model, same settings, same 1,296 held-out prompts. "
             "Only the training data differs.",
             ha="left", fontsize=11, color=INK2)

    ax_gate = fig.add_axes([0.035, 0.150, 0.505, 0.670])
    gate_table(ax_gate, ARMS, data)

    # the slice panel is deliberately short: it carries only what the gate table
    # cannot, so it is top-aligned and the space below it is left empty
    ax_slice = fig.add_axes([0.650, 0.512, 0.322, 0.248])
    slice_plot(ax_slice, ARMS, data)

    handles = [Line2D([], [], marker="o", ls="", ms=size, mfc=colour, mec="none", label=label)
               for _, _, label, colour, size in ARMS]
    # title on its own line above the legend, the two collide side by side
    ax_slice.legend(handles=handles, loc="lower left", frameon=False, fontsize=10,
                    ncol=3, columnspacing=1.4, handletextpad=0.4,
                    bbox_to_anchor=(-0.015, 1.02))
    ax_slice.text(0, 1.34, "Where the gain shows up", transform=ax_slice.transAxes,
                  ha="left", va="bottom", fontsize=11.5, fontweight="bold", color=INK)
    ax_slice.text(0, -0.62, "v3 slips further the deeper the round runs, as the price walks\n"
                            "down toward the floor. Both v4 arms stay flat.",
                  transform=ax_slice.transAxes, ha="left", va="top",
                  fontsize=9.5, color=MUTED, linespacing=1.6)

    fig.text(0.035, 0.045,
             "v3's training data showed the minimum price a cent too low in half the floor "
             "examples, so its “one cent below” cases read as “exactly at”. "
             "v4 corrected that and\nalso wrote the formula operators in words. v4 control "
             "keeps the correction with bare symbols, so it differs from v4 worded in "
             "wording alone.",
             ha="left", va="bottom", fontsize=9.5, color=MUTED, linespacing=1.7)
    return save(fig, "reverse_auction_v3_vs_v4.png")


if __name__ == "__main__":
    chart()
