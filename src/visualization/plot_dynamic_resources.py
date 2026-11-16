"""Create the two presentation charts for a dynamic-resources batch run.

Drawn straight from a run under reports/bidding/sealed/dynamic_resources/,
nothing is rerun and no number is recomputed from the agent CSVs. Pass a run stem
to pick one, otherwise the newest rule-based sealed run is used. Style follows the
reverse-auction charts already in reports/reverse_auction/: warm surface, muted
grid, direct labels, no gradients.

Both charts are built the same way, for a room seeing this for the first time:
the picture carries the shape of the story and nothing else, and every exact
figure sits in a plain table beneath it. Nothing is dropped for the sake of a
clean picture, it is moved to where it can be read.

A restock that lands can drop straight back under its threshold and re-trigger,
and that follow-on event is filed against an order whose record was already
written. Landings carry triggered_at and triggered_by, so such a trigger is
recovered from the landing it produced rather than by rerunning the batch. The
summary chart footer says so when it happened.
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "reports" / "bidding" / "sealed" / "dynamic_resources"
# the newest rule-based sealed run unless one is named on the command line
RUN_STEM = None

# the agent and material the timeline chart follows. None picks the pair with
# the most restock cycles in the run, which is the clearest single example
FOCUS_AGENT, FOCUS_MATERIAL = None, None

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STOCK = "#2a78d6"          # the stock line
COMMITMENT = "#d07a1f"     # trigger B, commitment based
THRESHOLD = "#7a4fb5"      # trigger A, proactive 20% threshold
LANDED = "#1baf7a"         # a restock arriving
CRIT = "#d03b3b"           # the zero line
NEGBAND = "#faeeee"
ROWTINT = "#f4f3ef"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5, "ytick.labelsize": 10,
})


def style_axis(ax, axis="y"):
    """Recessive grid on one axis, no box."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    getattr(ax, f"{axis}axis").grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    other = "x" if axis == "y" else "y"
    getattr(ax, f"{other}axis").grid(False)


def save(fig, filename):
    """Save one chart next to the run it is drawn from."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / filename
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def resolve_stem(argv):
    """the run to draw: the one named on the command line, else the newest
    rule-based sealed run in the directory"""
    if len(argv) > 1:
        return argv[1].removesuffix(".jsonl")
    runs = sorted(RUN_DIR.glob("rule_run_*.jsonl"))
    if not runs:
        raise SystemExit(f"no rule_run_*.jsonl in {RUN_DIR}, pass a run stem")
    return runs[-1].stem


def pick_focus(records):
    """the agent and material the timeline follows

    the smallest pair that still fires both triggers. over 300 orders the busiest
    supplier runs to a hundred-odd events, which no single readable timeline
    holds; a pair with one cycle of each rule tells the whole story and every
    number stays on the page. chart 2 carries the run-wide picture"""
    events = Counter()
    triggers = defaultdict(set)
    for record in records:
        for event in record["dynamic_events"]:
            if event.get("stage") != "material":
                continue
            key = (event["agent_id"], event["material"])
            events[key] += 1
            if event["event"] == "restock_triggered":
                triggers[key].add(event["trigger"])
    restocked = {k for k, rules in triggers.items() if rules}
    if not restocked:
        raise SystemExit("no restocks in this run, nothing to plot")
    both = {k for k in restocked if len(triggers[k]) > 1} or restocked
    return min(both, key=lambda k: (events[k], k))


def load_records():
    with (RUN_DIR / f"{RUN_STEM}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_summary():
    with (RUN_DIR / f"{RUN_STEM}_summary.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def ts(value):
    return datetime.fromisoformat(value)


def kg(value):
    """Kilograms as recorded in the run file, trailing zeros trimmed.

    a true minus sign, so a negative reads the same in the table as in the
    change column beside it"""
    return f"{value:,.2f}".rstrip("0").rstrip(".").replace("-", "−")


def table_axes(fig, rect, rows, columns):
    """A bare axes used as a table surface.

    x runs 0 to 1 across the given rectangle, y is one unit per row counting
    down from the header, so every cell is placed in the same coordinates."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(rows + 0.6, -1.35)
    ax.axis("off")
    for label, x, align in columns:
        ax.text(x, -0.92, label, ha=align, va="center", fontsize=9.1,
                fontweight="bold", color=MUTED)
    ax.plot([0, 1], [-0.42, -0.42], color=AXIS, lw=0.9, clip_on=False)
    return ax


def zebra(ax, index, height=0.92):
    """A whisper of tint on alternate rows, enough to track across a wide row."""
    if index % 2:
        ax.axhspan(index - height / 2, index + height / 2, color=ROWTINT, zorder=0)


# ── chart 1, one agent's stock over the simulated clock ──────────────────────

def focus_events(records):
    """Every recorded state change for the focus agent and material, in order."""
    out = []
    for record in records:
        for event in record["dynamic_events"]:
            if (event.get("agent_id") == FOCUS_AGENT
                    and event.get("material") == FOCUS_MATERIAL):
                out.append((record["order_id"], event))
    return out


def timeline_rows(events):
    """Every event as one table row, in clock order, nothing summarised away."""
    rows = []
    for order_id, event in events:
        when = ts(event["sim_now"])
        if event["event"] == "stock_reserved":
            rows.append({
                "when": when, "order": event["order_id"],
                "what": "Order accepted, stock reserved",
                "change": f"−{kg(event['required_kg'])}",
                "after": event["available_kg_after"], "colour": INK2,
            })
        elif event["event"] == "restock_triggered":
            rule = "Commitment" if event["trigger"] == "commitment" else "Threshold"
            rows.append({
                "when": when, "order": event["order_id"],
                "what": f"{rule} trigger, {kg(event['amount_kg'])} kg ordered, "
                        f"due {ts(event['eta']):%d %b %H:%M}",
                "change": "—", "after": None,
                "colour": COMMITMENT if event["trigger"] == "commitment" else THRESHOLD,
            })
        elif event["event"] == "restock_landed":
            rows.append({
                "when": when, "order": event["observed_on_order_id"],
                "what": "Restock arrives",
                "change": f"+{kg(event['amount_kg'])}",
                "after": event["available_kg_after"], "colour": LANDED,
            })
    return sorted(rows, key=lambda r: r["when"])


def stock_timeline(records, summary):
    events = focus_events(records)
    reserves = [(o, e) for o, e in events if e["event"] == "stock_reserved"]
    triggers = [(o, e) for o, e in events if e["event"] == "restock_triggered"]
    landings = [(o, e) for o, e in events if e["event"] == "restock_landed"]

    starting = reserves[0][1]["starting_kg"]
    threshold = reserves[0][1]["restock_threshold_kg"]
    clock_from = ts(summary["dynamic_state"]["sim_clock_from"])
    clock_to = ts(summary["dynamic_state"]["sim_clock_to"])

    moves = sorted((e for _, e in events
                    if e["event"] in ("stock_reserved", "restock_landed")),
                   key=lambda e: ts(e["sim_now"]))
    steps = [(clock_from, starting)]
    steps += [(ts(e["sim_now"]), e["available_kg_after"]) for e in moves]
    steps.append((clock_to, steps[-1][1]))
    xs = [x for x, _ in steps]
    ys = [y for _, y in steps]
    trough, peak, ending = min(ys), max(ys), ys[-1]

    rows = timeline_rows(events)

    fig = plt.figure(figsize=(15.0, 10.9))
    ax = fig.add_axes([0.075, 0.475, 0.900, 0.400])

    fig.text(0.5, 0.962, "One supplier's stock as orders are accepted",
             ha="center", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.5, 0.928,
             f"{FOCUS_AGENT}, {FOCUS_MATERIAL}. Stock falls as orders are won, and is "
             f"refilled by two separate restock rules.",
             ha="center", fontsize=13, color=INK2)

    lo, hi = -1080, 1820
    ax.axhspan(lo, 0, color=NEGBAND, zorder=0)
    ax.axhline(0, color=CRIT, lw=1.2, zorder=2)
    ax.axhline(threshold, color=THRESHOLD, lw=1.5, ls=(0, (5, 3)), zorder=2)
    ax.step(xs, ys, where="post", color=STOCK, lw=2.6, zorder=4, solid_joinstyle="miter")

    for _, event in reserves:
        ax.plot([ts(event["sim_now"])], [event["available_kg_after"]], "o",
                ms=6, mfc=SURFACE, mec=STOCK, mew=1.9, zorder=6)

    # the two reference lines each carry one short label, at the margin
    ax.text(clock_to, threshold + 55, f"20% threshold   {kg(threshold)} kg",
            ha="right", va="bottom", fontsize=10, fontweight="bold", color=THRESHOLD)
    ax.text(clock_from, -880,
            "below zero: already promised, and covered by a restock on its way",
            ha="left", va="center", fontsize=10, color=CRIT)

    # four values on the line, the shape of the whole story. the peak is found in
    # the data rather than fixed, so a longer or shorter batch still labels itself
    peak_at = next(x for x, y in steps if y == peak)
    ax.annotate(f"{kg(starting)} kg", xy=(clock_from, starting), xytext=(4, 11),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10.5, color=INK2)
    ax.annotate(f"{kg(peak)} kg", xy=(peak_at, peak),
                xytext=(8, 12), textcoords="offset points", ha="left", va="bottom",
                fontsize=10.5, fontweight="bold", color=INK2)
    ax.annotate(f"{kg(ending)} kg", xy=(clock_to, ending), xytext=(-4, 11),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=10.5, color=INK2)

    # each trigger is named at its own marker, and the commitment one carries the
    # trough value with it, since that dip is what set it off
    TRIGGER_PLACE = {"commitment": (0, -14, "center", "top"),
                     "threshold": (-13, -6, "right", "top")}
    for _, event in triggers:
        x, y = ts(event["sim_now"]), event["available_kg"]
        rule = event["trigger"]
        colour = COMMITMENT if rule == "commitment" else THRESHOLD
        dx, dy, ha, va = TRIGGER_PLACE[rule]
        label = ("Commitment trigger" + "\n" + f"{kg(trough)} kg"
                 if rule == "commitment" else "Threshold trigger")
        ax.plot([x], [y], "s" if rule == "commitment" else "D",
                ms=10, mfc=colour, mec=SURFACE, mew=1.4, zorder=8)
        ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=10.5, fontweight="bold", color=colour,
                    zorder=8, linespacing=1.45)

    for _, event in landings:
        x, y = ts(event["sim_now"]), event["available_kg_after"]
        ax.plot([x], [y], "^", ms=11, mfc=LANDED, mec=SURFACE, mew=1.4, zorder=8)
        ax.annotate(f"+{kg(event['amount_kg'])} kg", xy=(x, y), xytext=(-10, 15),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=10.5, fontweight="bold", color=LANDED, zorder=8)

    ax.set_ylim(lo, hi)
    ax.set_xlim(clock_from, clock_to)
    # a 300-order batch walks months, a short slice walks weeks, so let the tick
    # spacing follow the span rather than fixing it at three days
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=11))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.set_ylabel("Available to promise (kg)", fontsize=10.5)
    style_axis(ax)

    ax.legend(handles=[
        Line2D([], [], color=STOCK, lw=2.6, label="Stock available to promise"),
        Line2D([], [], color=COMMITMENT, marker="s", ls="none", ms=9,
               label="Commitment trigger, the winning bid counted on the restock"),
        Line2D([], [], color=THRESHOLD, marker="D", ls="none", ms=8,
               label="Threshold trigger, stock at or under 20%"),
        Line2D([], [], color=LANDED, marker="^", ls="none", ms=10,
               label="Restock arrives"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.155), ncol=4, frameon=False,
        fontsize=10, handletextpad=0.7, columnspacing=2.0)

    # every figure behind the picture, in clock order
    columns = [("TIME", 0.000, "left"), ("ORDER", 0.148, "left"),
               ("WHAT HAPPENED", 0.268, "left"), ("CHANGE (KG)", 0.790, "right"),
               ("AVAILABLE AFTER (KG)", 1.000, "right")]
    tab = table_axes(fig, [0.075, 0.075, 0.900, 0.300], len(rows), columns)
    for i, row in enumerate(rows):
        zebra(tab, i)
        tab.text(0.000, i, f"{row['when']:%d %b  %H:%M}", ha="left", va="center",
                 fontsize=9.6, color=INK2)
        tab.text(0.148, i, row["order"], ha="left", va="center",
                 fontsize=9.6, color=INK2)
        tab.text(0.268, i, row["what"], ha="left", va="center",
                 fontsize=9.6, color=row["colour"],
                 fontweight="bold" if row["change"] == "—" else "normal")
        tab.text(0.790, i, row["change"], ha="right", va="center", fontsize=9.6,
                 color=row["colour"], fontweight="bold" if row["change"] != "—" else "normal")
        tab.text(1.000, i, "" if row["after"] is None else kg(row["after"]),
                 ha="right", va="center", fontsize=9.6, fontweight="bold", color=INK)

    fig.text(0.5, 0.036,
             f"Stock is never capped. The second restock takes it to {kg(peak)} kg, above the "
             f"{kg(starting)} kg it started with, because the schema carries no maximum.",
             ha="center", va="bottom", fontsize=11, color=INK)
    fig.text(0.5, 0.011,
             f"Read from {RUN_STEM}.jsonl. Sealed bidding, rule-based agents, "
             f"{len(records)}-order SLM holdout. Time is the simulated clock, which advances to "
             f"each order's own creation time.",
             ha="center", va="bottom", fontsize=9.4, color=MUTED)

    return save(fig, "dynamic_resources_stock_timeline.png")


# ── chart 2, did this happen anywhere else ───────────────────────────────────

def restock_table(records):
    """Per agent and material: triggers by rule, landings, and what is in flight.

    Triggers are reconciled against landings. A landing names the moment and the
    rule that produced it, so a trigger missing from the file is recovered from
    its own landing instead of by rerunning the batch."""
    recorded, landings, facts = defaultdict(list), defaultdict(list), defaultdict(dict)
    suppressed = defaultdict(int)
    for record in records:
        for event in record["dynamic_events"]:
            key = (event.get("agent_id"), event.get("material"))
            if event["event"] == "restock_triggered":
                recorded[key].append((event["sim_now"], event["trigger"]))
            elif event["event"] == "restock_landed":
                landings[key].append((event["triggered_at"], event["triggered_by"]))
            elif event["event"] == "restock_skipped_pending":
                suppressed[key] += 1
                continue
            else:
                continue
            facts[key]["amount_kg"] = event["amount_kg"]
            facts[key]["lead_days"] = event["restock_time_days"]

    rows = []
    for key in sorted(set(recorded) | set(landings)):
        triggers = list(recorded[key])
        recovered = [t for t in landings[key] if t not in triggers]
        triggers += recovered
        rows.append({
            "agent": key[0], "material": key[1],
            "commitment": sum(1 for _, rule in triggers if rule == "commitment"),
            "threshold": sum(1 for _, rule in triggers if rule == "threshold"),
            "triggers": len(triggers), "landed": len(landings[key]),
            "recovered": len(recovered), "suppressed": suppressed[key],
            "amount_kg": facts[key]["amount_kg"], "lead_days": facts[key]["lead_days"],
        })
    return rows


def restock_summary(records):
    rows = restock_table(records)
    total_c = sum(r["commitment"] for r in rows)
    total_t = sum(r["threshold"] for r in rows)
    total_l = sum(r["landed"] for r in rows)
    total_s = sum(r["suppressed"] for r in rows)
    total_x = total_c + total_t
    in_flight = total_x - total_l

    fig = plt.figure(figsize=(15.0, 8.7))

    fig.text(0.5, 0.958, "Every restock in the batch",
             ha="center", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.5, 0.923,
             f"{len(rows)} supplier-material pairs restocked, on both rules, not only the "
             f"supplier followed in the previous chart.",
             ha="center", fontsize=13, color=INK2)

    # the headline four, so the room has the answer before reading a single row
    tiles = [
        (f"{total_x}", "restocks triggered", INK),
        (f"{total_c} · {total_t}", "by commitment · by threshold", INK),
        (f"{total_l}", f"arrived inside the batch, {in_flight} still in flight", LANDED),
        (f"{total_s}", "blocked, one in flight at a time", MUTED),
    ]
    for i, (value, label, colour) in enumerate(tiles):
        x = 0.115 + i * 0.257
        fig.text(x, 0.822, value, ha="center", va="bottom", fontsize=32,
                 fontweight="bold", color=colour)
        fig.text(x, 0.797, label, ha="center", va="top", fontsize=10.5, color=INK2)

    # the rule that fired gets its own colour-matched sub-column rather than dots
    # plus words saying the same thing twice
    columns = [("SUPPLIER", 0.000, "left"), ("MATERIAL", 0.100, "left"),
               ("COMMITMENT TRIGGER", 0.225, "left"), ("THRESHOLD TRIGGER", 0.380, "left"),
               ("RESTOCK ORDER", 0.610, "right"), ("LEAD TIME", 0.735, "right"),
               ("ARRIVED", 0.865, "right"), ("BLOCKED", 1.000, "right")]
    tab = table_axes(fig, [0.075, 0.168, 0.900, 0.540], len(rows) + 1, columns)

    for i, row in enumerate(rows):
        zebra(tab, i)
        tab.text(0.000, i, row["agent"], ha="left", va="center", fontsize=10.5,
                 fontweight="bold", color=INK)
        tab.text(0.100, i, row["material"], ha="left", va="center",
                 fontsize=10.5, color=INK2)
        for count, x, colour in ((row["commitment"], 0.225, COMMITMENT),
                                 (row["threshold"], 0.380, THRESHOLD)):
            tab.text(x, i, str(count) if count else "—", ha="left", va="center",
                     fontsize=10.5, fontweight="bold" if count else "normal",
                     color=colour if count else MUTED)
        tab.text(0.610, i, f"{kg(row['amount_kg'])} kg", ha="right", va="center",
                 fontsize=10.5, color=INK2)
        tab.text(0.735, i, f"{row['lead_days']:g} days", ha="right", va="center",
                 fontsize=10.5, color=INK2)
        landed_all = row["landed"] == row["triggers"]
        tab.text(0.865, i, f"{row['landed']} of {row['triggers']}", ha="right", va="center",
                 fontsize=10.5, fontweight="bold", color=INK if landed_all else COMMITMENT)
        tab.text(1.000, i, str(row["suppressed"]) if row["suppressed"] else "—",
                 ha="right", va="center", fontsize=10.5,
                 fontweight="bold" if row["suppressed"] else "normal",
                 color=INK2 if row["suppressed"] else MUTED)

    total_row = len(rows)
    tab.plot([0, 1], [total_row - 0.52, total_row - 0.52], color=AXIS, lw=0.9,
             clip_on=False)
    tab.text(0.000, total_row, f"{len(rows)} pairs", ha="left", va="center",
             fontsize=10.5, fontweight="bold", color=INK)
    tab.text(0.225, total_row, str(total_c), ha="left", va="center",
             fontsize=10.5, fontweight="bold", color=COMMITMENT)
    tab.text(0.380, total_row, str(total_t), ha="left", va="center",
             fontsize=10.5, fontweight="bold", color=THRESHOLD)
    tab.text(0.865, total_row, f"{total_l} of {total_x}", ha="right", va="center",
             fontsize=10.5, fontweight="bold", color=INK)
    tab.text(1.000, total_row, str(total_s), ha="right", va="center",
             fontsize=10.5, fontweight="bold", color=INK)

    recovered = sum(r["recovered"] for r in rows)
    fig.text(0.5, 0.084,
             "Blocked means the rule fired again while that material already had a restock "
             "on its way, so no second order was placed.",
             ha="center", va="bottom", fontsize=11, color=INK)
    # a landing can drop straight back under its threshold and re-trigger, and that
    # follow-on is filed against an order record already written. say so only when
    # it actually happened in this run
    note = (f" {recovered} trigger(s) recovered from the landing they produced: a restock "
            f"that lands straight back under its threshold re-triggers, and that follow-on "
            f"is filed against an order whose record was already written.\n"
            f"Supplier stock was correct throughout, only the event attribution was "
            f"incomplete." if recovered else "")
    fig.text(0.5, 0.014,
             f"Read from {RUN_STEM}.jsonl. Sealed bidding, rule-based agents.{note}",
             ha="center", va="bottom", fontsize=9.4, color=MUTED, linespacing=1.55)

    return save(fig, "dynamic_resources_restock_summary.png")


def main(argv):
    global RUN_STEM, FOCUS_AGENT, FOCUS_MATERIAL
    RUN_STEM = resolve_stem(argv)
    records = load_records()
    summary = load_summary()
    FOCUS_AGENT, FOCUS_MATERIAL = pick_focus(records)
    print(f"run {RUN_STEM}, {len(records)} orders, focus {FOCUS_AGENT} {FOCUS_MATERIAL}")
    stock_timeline(records, summary)
    restock_summary(records)


if __name__ == "__main__":
    main(sys.argv)
