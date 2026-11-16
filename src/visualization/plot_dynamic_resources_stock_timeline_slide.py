"""Create a concise slide version of the dynamic material-stock example.

The figure is read directly from the saved rule-based sealed-bidding run. It
focuses on M11's polyester stock because that sequence contains both supported
restock triggers without needing a second example.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "reports" / "bidding" / "sealed" / "dynamic_resources"
RUN_FILE = RUN_DIR / "rule_run_20260825_200931.jsonl"
SUMMARY_FILE = RUN_DIR / "rule_run_20260825_200931_summary.json"
OUTPUT = RUN_DIR / "dynamic_resources_stock_timeline_slide.png"

FOCUS_AGENT = "M11"
FOCUS_MATERIAL = "polyester"

WHITE = "#ffffff"
INK = "#243447"
MUTED = "#626b72"
GRID = "#e3e5e4"
AXIS = "#bfc4c1"
BLUE = "#287fd1"
ORANGE = "#eb6834"
TEAL = "#1597a6"
GREEN = "#43a984"
ZERO = "#c85a5a"
NEGATIVE_BAND = "#fdf2f1"


plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.9,
})


def parse_time(value):
    return datetime.fromisoformat(value)


def load_example():
    """Load and validate the saved M11/polyester event sequence."""
    records = [
        json.loads(line)
        for line in RUN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    events = []
    for record in records:
        for event in record["dynamic_events"]:
            if (
                event.get("agent_id") == FOCUS_AGENT
                and event.get("material") == FOCUS_MATERIAL
                and event["event"] in {
                    "stock_reserved",
                    "restock_triggered",
                    "restock_landed",
                }
            ):
                events.append(event)

    reserves = [event for event in events if event["event"] == "stock_reserved"]
    triggers = [event for event in events if event["event"] == "restock_triggered"]
    landings = [event for event in events if event["event"] == "restock_landed"]
    trigger_types = {event["trigger"] for event in triggers}
    if trigger_types != {"commitment", "threshold"} or len(landings) != 2:
        raise ValueError("saved example no longer contains both restock cycles")

    return summary, reserves, triggers, landings


def build_sequence(summary, reserves, landings):
    """Build an evenly spaced sequence of recorded stock changes."""
    start = parse_time(summary["dynamic_state"]["sim_clock_from"])
    starting_kg = reserves[0]["starting_kg"]
    moves = sorted(
        [*reserves, *landings],
        key=lambda event: parse_time(event["sim_now"]),
    )
    second_landing = max(parse_time(event["sim_now"]) for event in landings)
    moves = [
        event
        for event in moves
        if parse_time(event["sim_now"]) <= second_landing
    ]
    values = [starting_kg]
    values.extend(event["available_kg_after"] for event in moves)
    labels = [f"Start\n{start:%d %b}"]
    labels.extend(parse_time(event["sim_now"]).strftime("%d %b") for event in moves)
    event_x = {event["sim_now"]: index + 1 for index, event in enumerate(moves)}
    return values, labels, event_x, moves, starting_kg


def draw():
    summary, reserves, triggers, landings = load_example()
    values, labels, event_x, moves, starting_kg = build_sequence(
        summary, reserves, landings
    )
    threshold_kg = reserves[0]["restock_threshold_kg"]

    fig, ax = plt.subplots(figsize=(12.8, 5.8), facecolor=WHITE)
    fig.subplots_adjust(left=0.105, right=0.965, top=0.92, bottom=0.22)

    ax.axhspan(-450, 0, color=NEGATIVE_BAND, zorder=0)
    ax.axhline(0, color=ZERO, lw=1.25, zorder=2)
    ax.axhline(
        threshold_kg,
        color=TEAL,
        lw=1.45,
        ls=(0, (5, 4)),
        alpha=0.9,
        zorder=2,
    )

    xs = list(range(len(values)))
    line_xs = [*xs, xs[-1] + 0.35]
    line_ys = [*values, values[-1]]
    ax.step(
        line_xs,
        line_ys,
        where="post",
        color=BLUE,
        lw=3.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )

    plotted_reserves = [event for event in moves if event["event"] == "stock_reserved"]
    for event in plotted_reserves:
        ax.plot(
            event_x[event["sim_now"]],
            event["available_kg_after"],
            "o",
            ms=7,
            mfc=WHITE,
            mec=BLUE,
            mew=2.0,
            zorder=6,
        )

    trigger_style = {
        "commitment": (ORANGE, "s", "Commitment trigger"),
        "threshold": (TEAL, "D", "Threshold trigger"),
    }
    for event in triggers:
        colour, marker, label = trigger_style[event["trigger"]]
        x = event_x[event["sim_now"]]
        y = event["available_kg"]
        ax.plot(x, y, marker=marker, ms=10, mfc=colour, mec=WHITE,
                mew=1.5, zorder=8)
        if event["trigger"] == "commitment":
            offset, align, valign = ((16, 15), "left", "bottom")
        else:
            offset, align, valign = ((-10, 18), "right", "bottom")
        ax.annotate(
            f"{label}\n{y:,.0f} kg",
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va=valign,
            fontsize=11.5,
            fontweight="bold",
            color=colour,
            linespacing=1.25,
            zorder=9,
        )

    for event in landings:
        x = event_x[event["sim_now"]]
        y = event["available_kg_after"]
        ax.plot(x, y, "^", ms=11, mfc=GREEN, mec=WHITE, mew=1.5, zorder=8)
        ax.annotate(
            f"+{event['amount_kg']:,.0f} kg",
            xy=(x, y),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color=GREEN,
            zorder=9,
        )

    ax.annotate(
        f"{starting_kg:,.0f} kg",
        xy=(0, starting_kg),
        xytext=(7, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=MUTED,
    )
    ax.text(
        0.20,
        threshold_kg + 38,
        f"20% threshold  ·  {threshold_kg:,.0f} kg",
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=TEAL,
    )
    fig.text(
        0.105,
        0.955,
        "M11  ·  polyester",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )

    ax.set_xlim(-0.25, xs[-1] + 0.35)
    ax.set_ylim(-450, 1700)
    ax.set_yticks([-400, 0, 400, 800, 1200, 1600])
    ax.set_ylabel("Available to promise (kg)", fontsize=12.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.85, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.text(
        0.535,
        0.145,
        "Events shown in sequence  ·  2026",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=3.0, label="Available to promise"),
            Line2D([], [], marker="o", ls="", ms=7, mfc=WHITE, mec=BLUE,
                   mew=2.0, label="Accepted order"),
            Line2D([], [], marker="^", ls="", ms=10, mfc=GREEN, mec="none",
                   label="Restock arrives"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=12,
        columnspacing=2.8,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, facecolor=WHITE)
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    path = draw()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
