"""Create the five summary charts for the reverse-auction experiment.

Drawn from the v3 evaluation files after the floor-rounding fix, the arms tagged
`-fixed`. Style follows the epoch-2 and supervisor comparison charts already in
reports/: warm surface, dumbbell rows, direct value labels, muted grid.
"""

import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
EPOCH = REPO / "reports" / "reverse_auction" / "epoch-1"
OUT = REPO / "reports" / "reverse_auction"

# the dataset version these charts are drawn from; each version is its own folder
VERSION = "v3"
# evaluation arms re-run against the floor-rounding-corrected holdout
ARM = "fixed"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BASE, TRAINED, REF, CRIT = "#b3b1a9", "#2a78d6", "#1baf7a", "#d03b3b"
PASS_FILL, FAIL_FILL = "#e8f4ee", "#fbeaea"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5, "ytick.labelsize": 10,
})

MODELS = [
    ("qwen", "Qwen"),
    ("granite", "Granite"),
    ("phi", "Phi"),
    ("ministral", "Ministral"),
    ("gemma", "Gemma"),
    ("smollm3", "SmolLM3"),
]

# every technical term carries a plain-language gloss, so the label reads on its own
GATE_ROWS = [
    ("3_floor_above_floor", "Just above the minimum price", "should bid"),
    ("2_floor_at_floor", "Exactly at the minimum price", "should bid"),
    ("1_floor_one_cent_below_floor", "One cent below the minimum price", "should refuse"),
    ("4_bid_rate_priority_urgent", "Bids as often as the rule, urgent orders", "gap in points"),
    ("4_bid_rate_priority_normal", "Bids as often as the rule, normal orders", "gap in points"),
    ("4_bid_rate_priority_flexible", "Bids as often as the rule, flexible orders", "gap in points"),
    ("5_bid_rate_stage_material", "Bids as often as the rule, material stage", "gap in points"),
    ("5_bid_rate_stage_tailoring", "Bids as often as the rule, tailoring stage", "gap in points"),
    ("5_bid_rate_stage_logistics", "Bids as often as the rule, logistics stage", "gap in points"),
    ("6_json_validity", "Answers in valid JSON", "should be 100%"),
    ("7_days_no_regression", "Delivery-time figure not made worse", "vs untrained"),
]

FLOOR_CASES = [
    ("floor_boundary:above_floor", "Just ABOVE the minimum price", "the agent should bid", 100),
    ("floor_boundary:at_floor", "EXACTLY AT the minimum price", "the agent should bid", 100),
    ("floor_boundary:one_cent_below_floor", "One cent BELOW the minimum price",
     "the agent should refuse", 0),
]


def style_axis(ax, axis="y"):
    """Recessive grid on one axis, no box."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    getattr(ax, f"{axis}axis").grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    other = "x" if axis == "y" else "y"
    getattr(ax, f"{other}axis").grid(False)


def load_gates(tag):
    """Load one saved release-gate result, newest timestamp for that stem."""
    hits = sorted((EPOCH / VERSION / "gates").glob(f"eval_{tag}_*_gates.json"))
    if not hits:
        raise FileNotFoundError(f"no gate file for {tag} under {EPOCH / VERSION / 'gates'}")
    with hits[-1].open(encoding="utf-8") as handle:
        return json.load(handle)


def load_summary(gate_json):
    """Load the evaluation summary referenced by a gate file."""
    with (REPO / gate_json["eval_result"]).open(encoding="utf-8") as handle:
        return json.load(handle)["summary"]


def arms(model):
    """(base gate result, trained gate result) for one model."""
    return load_gates(f"ra-e1-{model}-base-{ARM}"), load_gates(f"ra-e1-{model}-{ARM}")


def read_log(name):
    """Read a training log written as either UTF-16 or UTF-8."""
    raw = (EPOCH / VERSION / "training-logs" / name).read_bytes()
    for encoding in ("utf-16", "utf-8"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "train_runtime" in text:
            return text
    return raw.decode("utf-8", errors="ignore")


def save(fig, filename):
    """Save one chart to its established report path."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def dumbbell(ax, rows, xlim, fmt="{:.1f}", pad=None):
    """One base-to-trained row per model, both ends labelled."""
    span = xlim[1] - xlim[0]
    pad = pad if pad is not None else span * 0.018
    for i, (before, after) in enumerate(rows):
        ax.plot([before, after], [i, i], color=AXIS, lw=1.7, zorder=1, solid_capstyle="round")
        if abs(after - before) > span * 0.004:
            ax.add_patch(FancyArrowPatch((before, i), (after, i), arrowstyle="-|>",
                                         mutation_scale=10, color=MUTED, lw=0, zorder=2))
        ax.plot([before], [i], "o", ms=8, color=BASE, mec=SURFACE, mew=1.5, zorder=3)
        ax.plot([after], [i], "o", ms=8, color=TRAINED, mec=SURFACE, mew=1.5, zorder=4)
        left, right = (before, after) if before <= after else (after, before)
        lcol, rcol = (BASE, TRAINED) if before <= after else (TRAINED, BASE)
        # a surface halo keeps a value legible where it lands on the target line
        halo = [path_effects.withStroke(linewidth=2.6, foreground=SURFACE)]
        ax.text(left - pad, i, fmt.format(left), ha="right", va="center", fontsize=9,
                fontweight="bold", color=lcol, zorder=6, path_effects=halo)
        ax.text(right + pad, i, fmt.format(right), ha="left", va="center", fontsize=9,
                fontweight="bold", color=rcol, zorder=6, path_effects=halo)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, label in MODELS])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    style_axis(ax, axis="x")


# ── 1. gate grid ──────────────────────────────────────────────────────────────

def chart_gates():
    """Pass or fail on every gate, untrained base against trained adapter."""
    data = {m: arms(m) for m, _ in MODELS}

    fig = plt.figure(figsize=(15.4, 9.2))
    ax = fig.add_axes([0.335, 0.115, 0.615, 0.700])
    ax.set_xlim(0, len(MODELS))
    ax.set_ylim(len(GATE_ROWS), 0)
    ax.axis("off")

    fig.text(0.5, 0.960, "Reverse auction: which release gates each model clears",
             ha="center", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.5, 0.925,
             "Eleven pre-set bars, scored on 1,296 held-out prompts per model. "
             "Every trained model clears at least ten.",
             ha="center", fontsize=12.5, color=INK2)

    for j, (model, label) in enumerate(MODELS):
        _, trained = data[model]
        ax.text(j + 0.5, -0.62, label, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(j + 0.5, -0.22, f"{trained['counts']['PASS']} of 11 pass",
                ha="center", va="bottom", fontsize=9.5, color=INK2)

    for i, (key, label, gloss) in enumerate(GATE_ROWS):
        ax.text(-0.05, i + 0.42, label, ha="right", va="center", fontsize=10.5, color=INK,
                transform=ax.transData)
        ax.text(-0.05, i + 0.74, gloss, ha="right", va="center", fontsize=8.8, color=MUTED,
                transform=ax.transData, style="italic")
        for j, (model, _) in enumerate(MODELS):
            base, trained = data[model]
            g = trained["gates"][key]
            passed = g["status"] == "PASS"
            ax.add_patch(Rectangle((j + 0.05, i + 0.10), 0.90, 0.80,
                                   facecolor=PASS_FILL if passed else FAIL_FILL,
                                   edgecolor=REF if passed else CRIT, lw=1.0, zorder=2))
            # status is never colour alone: a mark and the number accompany it
            ax.text(j + 0.34, i + 0.50, "✓" if passed else "✗",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color=REF if passed else CRIT, zorder=3)
            ax.text(j + 0.60, i + 0.50, f"{g['measured']:g}", ha="left", va="center",
                    fontsize=10, color=INK2, zorder=3)
            # untrained baseline as a slim chip on the cell's left edge
            b = base["gates"][key]
            ax.add_patch(Rectangle((j + 0.05, i + 0.10), 0.055, 0.80,
                                   facecolor=REF if b["status"] == "PASS" else CRIT,
                                   edgecolor="none", alpha=0.55, zorder=3))

    handles = [
        Line2D([], [], marker="s", ls="", ms=12, mfc=PASS_FILL, mec=REF, label="✓  meets the bar"),
        Line2D([], [], marker="s", ls="", ms=12, mfc=FAIL_FILL, mec=CRIT, label="✗  misses the bar"),
        Line2D([], [], marker="s", ls="", ms=12, mfc=REF, mec="none", alpha=0.55,
               label="slim edge bar = untrained model met it"),
        Line2D([], [], marker="s", ls="", ms=12, mfc=CRIT, mec="none", alpha=0.55,
               label="slim edge bar = untrained model missed it"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.062), columnspacing=2.4, handletextpad=0.7)
    fig.text(0.5, 0.012,
             "Number in each cell is the trained model's measured value. Bars: the three "
             "minimum-price rows need 90% correct, the bidding-frequency rows need to be within\n"
             "10 points of how often the rule bids, valid JSON needs 99%, and the delivery-time "
             "figure must not fall below the untrained model.",
             ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.6)
    return save(fig, "reverse_auction_gates.png")


# ── 2. floor behaviour ────────────────────────────────────────────────────────

def chart_floor():
    """How often each model bids on the three minimum-price tests."""
    fig = plt.figure(figsize=(15.4, 6.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.075, right=0.982, top=0.735, bottom=0.215)

    fig.text(0.5, 0.955, "Behaviour right at the agent's minimum price",
             ha="center", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.5, 0.905,
             "An agent must never sell below its own minimum price. Each panel is one test: "
             "the green line is the correct answer.",
             ha="center", fontsize=12.5, color=INK2)

    for panel, (key, title, gloss, target) in enumerate(FLOOR_CASES):
        ax = fig.add_subplot(gs[0, panel])
        rows = []
        for model, _ in MODELS:
            b, t = arms(model)
            rows.append((load_summary(b)["by_floor_case"][key]["model_bid_rate_pct"],
                         load_summary(t)["by_floor_case"][key]["model_bid_rate_pct"]))
        dumbbell(ax, rows, (-14, 118), fmt="{:.0f}")
        ax.axvline(target, color=REF, lw=2.4, zorder=5)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Prompts answered with a bid (%)", fontsize=10)
        ax.set_title(f"{'ABCDEF'[panel]}. {title}", loc="left", fontweight="bold",
                     fontsize=12, pad=16, color=INK)
        ax.text(0, 1.015, gloss, transform=ax.transAxes, fontsize=10, color=REF,
                fontweight="bold", va="bottom")

    handles = [
        Line2D([], [], marker="o", ls="", ms=9, mfc=BASE, mec="none", label="Untrained model"),
        Line2D([], [], marker="o", ls="", ms=9, mfc=TRAINED, mec="none", label="After training"),
        Line2D([], [], color=REF, lw=2.4, label="Correct answer"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.075), columnspacing=2.6)
    fig.text(0.5, 0.015,
             "Training fixes the refusal case almost completely. What remains is the middle "
             "panel: five of six models treat an offer that exactly equals\ntheir minimum price "
             "as if it were below it, and refuse. 132 held-out prompts per panel per model.",
             ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.6)
    return save(fig, "reverse_auction_floor_bid_rate.png")


# ── 3. sealed vs auction cost ─────────────────────────────────────────────────

def load_runs():
    """Load the saved sealed and reverse-auction live run records."""
    runs = []
    for path in sorted(glob.glob(str(REPO / "reports" / "bidding" / "*" / "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                decrements = {t["platform_minimum_decrement"]
                              for t in record.get("stage_traces", [])}
                runs.append({
                    "order_id": record["order_id"], "mode": record["mode"],
                    "budget": record["budget"], "status": record["outcome"]["status"],
                    "cost": record["outcome"].get("total_cost"),
                    "decrement": sorted(decrements)[0] if decrements else None,
                })
    return runs


def chart_cost():
    """What the auction saved against sealed bidding on the one order with both."""
    runs = [r for r in load_runs()
            if r["order_id"] == "ORD-00001" and r["status"] == "completed"]
    runs.sort(key=lambda r: (0 if r["mode"] == "sealed" else 1, r["decrement"] or 0))

    labels, colours, values = [], [], []
    for r in runs:
        if r["mode"] == "sealed":
            labels.append("Sealed bidding\none offer each, no rounds")
            colours.append(BASE)
        else:
            labels.append(f"Reverse auction\nup to 3 rounds, EUR {r['decrement']:g} step")
            colours.append(TRAINED)
        values.append(r["cost"])
    budget = runs[0]["budget"]
    sealed_cost = values[0]

    fig = plt.figure(figsize=(11.6, 7.4))
    ax = fig.add_axes([0.115, 0.225, 0.845, 0.560])

    fig.text(0.5, 0.955, "What the auction saves on a single order",
             ha="center", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.5, 0.905,
             "ORD-00001, an urgent order, run end to end through the live pipeline in each mode.",
             ha="center", fontsize=12.5, color=INK2)

    ax.bar(range(len(values)), values, width=0.52, color=colours, zorder=3)
    for i, (v, c) in enumerate(zip(values, colours)):
        ax.text(i, v + budget * 0.015, f"EUR {v:,.2f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=c if i else INK2)
        if i:
            ax.text(i, v / 2, f"EUR {sealed_cost - v:,.0f}\nsaved", ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=SURFACE)

    ax.axhline(budget, color=CRIT, lw=1.8, ls=(0, (5, 3)), zorder=4)
    ax.text(-0.42, budget + budget * 0.012, f"order budget  EUR {budget:,.2f}",
            ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=CRIT)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, budget * 1.16)
    ax.set_ylabel("Total cost of the winning supply chain (EUR)", fontsize=11)
    style_axis(ax)

    fig.text(0.5, 0.095,
             "Each round takes one fixed step off the best price, across three stages, so the "
             "saving scales with the step the platform sets.",
             ha="center", va="bottom", fontsize=11, color=INK)
    fig.text(0.5, 0.020,
             "Rule-based agents, not the trained models. Only this order has both modes saved "
             "on disk; ORD-00002's sealed run is recorded as over budget,\nbut its auction "
             "counterpart is no longer on disk and is not reconstructed here.",
             ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.6)
    return save(fig, "reverse_auction_vs_sealed_cost.png")


# ── 4. training summary ───────────────────────────────────────────────────────

TRAIN_LOGS = {
    "qwen": ("Qwen3.5-4B", "qwen_ra_train.log"),
    "granite": ("Granite-4.1-3b", "granite_ra_train.log"),
    "phi": ("Phi-4-mini", "phi_ra_train.log"),
    "ministral": ("Ministral-3-3B", "ministral_ra_train.log"),
    "gemma": ("Gemma-4-E2B", "gemma_ra_train.log"),
    "smollm3": ("SmolLM3-3B", "smollm3_ra_train.log"),
}


def training_rows():
    """Extract the figures used in the summary table."""
    rows = []
    for model, _ in MODELS:
        name, log_name = TRAIN_LOGS[model]
        text = read_log(log_name)
        runtime = re.search(r"train_runtime: ([\d.]+)s", text)
        losses = re.findall(r"'train_loss': '([\d.e+-]+)'", text)
        vram = re.search(r"peak vram: ([\d.]+) GiB allocated, ([\d.]+) GiB reserved", text)
        examples = re.search(r"training examples this run: ([\d,]+)", text)
        measured = re.search(r"over (\d+) measured steps", text)
        step = re.search(r"per-step wall time: median ([\d.]+)s", text)
        if not all((runtime, losses, vram, examples, measured, step)):
            raise ValueError(f"could not parse training values from {log_name}")
        seconds = float(runtime.group(1))
        rows.append([
            name,
            f"{int(examples.group(1).replace(',', '')):,}",
            f"{int(measured.group(1)) + 1:,}",
            f"{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} m",
            f"{float(step.group(1)):.2f} s",
            f"{float(losses[-1]):.5f}",
            f"{vram.group(2)} GiB",
        ])
    return rows


def chart_training():
    """The basic facts for all six completed training runs."""
    rows = training_rows()
    headers = ["Model", "Training rows", "Steps", "Time", "Per step",
               "Final loss", "Peak memory"]

    fig = plt.figure(figsize=(13.6, 6.2))
    ax = fig.add_axes([0.035, 0.185, 0.935, 0.620])
    ax.set_xlim(0, len(headers))
    ax.set_ylim(len(rows) + 0.1, -1.0)
    ax.axis("off")

    fig.text(0.5, 0.945, "Reverse-auction training runs, one epoch each",
             ha="center", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.5, 0.893,
             "All six models trained back to back on identical data and settings.",
             ha="center", fontsize=12.5, color=INK2)

    widths = [1.45, 1.0, 0.75, 0.9, 0.8, 0.95, 1.0]
    total = sum(widths)
    edges = [sum(widths[:i]) / total * len(headers) for i in range(len(widths) + 1)]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(headers))]

    for j, header in enumerate(headers):
        ha = "left" if j == 0 else "center"
        x = edges[0] + 0.06 if j == 0 else centres[j]
        ax.text(x, -0.55, header, ha=ha, va="center", fontsize=10.5,
                fontweight="bold", color=INK2)
    ax.plot([edges[0], edges[-1]], [-0.18, -0.18], color=AXIS, lw=1.1)

    for i, row in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((edges[0], i - 0.02), edges[-1] - edges[0], 0.86,
                                   facecolor="#f4f3ef", edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((edges[0] + 0.03, i + 0.10), 0.075, 0.62,
                               facecolor=TRAINED, edgecolor="none", zorder=2))
        for j, value in enumerate(row):
            ha = "left" if j == 0 else "center"
            x = edges[0] + 0.20 if j == 0 else centres[j]
            ax.text(x, i + 0.41, value, ha=ha, va="center", fontsize=11.5,
                    color=INK if j == 0 else INK2,
                    fontweight="bold" if j == 0 else "normal", zorder=3)

    fig.text(0.5, 0.105,
             "Identical protocol throughout: 48,446 training rows, one epoch, batch 8 with "
             "gradient accumulation 2, 1,024-token window, seed 42.",
             ha="center", va="bottom", fontsize=11, color=INK)
    fig.text(0.5, 0.022,
             "Peak memory is what the run reserved; the machine has 23.9 GiB, so four of the "
             "six spilled into system memory and Ministral by a wide margin.\n"
             "About 43 hours of training in total. Source: "
             "reports/reverse_auction/epoch-1/v3/training-logs.",
             ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.6)
    return save(fig, "reverse_auction_training_summary.png")


# ── 5. six-model comparison ───────────────────────────────────────────────────

def chart_models_comparison():
    """All six models side by side, untrained against trained."""
    stats = {}
    for model, _ in MODELS:
        b, t = arms(model)
        stats[model] = (b, t, load_summary(b), load_summary(t))

    fig = plt.figure(figsize=(16.6, 12.4))
    gs = fig.add_gridspec(3, 3, hspace=0.72, wspace=0.30,
                          left=0.070, right=0.980, top=0.870, bottom=0.115)

    fig.text(0.5, 0.965, "Reverse-auction bidding: all six models, before and after training",
             ha="center", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.5, 0.930,
             "Grey dot is the untrained model, blue dot is the same model after training on the "
             "auction rules. Every model improves on every measure.",
             ha="center", fontsize=13, color=INK2)

    def panel(pos, letter, title, note, values, xlim, fmt="{:.1f}", target=None,
              target_label=None, xlabel=""):
        # target_label is unused: the shared legend names the green line once,
        # and per-panel labels collided with the panel titles
        ax = fig.add_subplot(gs[pos])
        dumbbell(ax, values, xlim, fmt=fmt)
        if target is not None:
            ax.axvline(target, color=REF, lw=2.2, zorder=5)
        ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold",
                     fontsize=12.5, pad=18, color=INK)
        ax.text(0, 1.045, note, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom", style="italic")
        ax.set_xlabel(xlabel, fontsize=9.5)
        return ax

    overall = [(stats[m][2]["overall"]["accuracy_pct"],
                stats[m][3]["overall"]["accuracy_pct"]) for m, _ in MODELS]
    panel((0, 0), "A", "Agrees with the rule overall",
          "share of all 1,296 prompts answered exactly as the rule would",
          overall, (20, 112), target=100, target_label="rule", xlabel="Agreement (%)")

    for idx, (key, title, gloss, target) in enumerate(FLOOR_CASES):
        vals = [(stats[m][2]["by_floor_case"][key]["model_bid_rate_pct"],
                 stats[m][3]["by_floor_case"][key]["model_bid_rate_pct"]) for m, _ in MODELS]
        panel((0, 1) if idx == 0 else ((0, 2) if idx == 1 else (1, 0)),
              "BCD"[idx], title, gloss + ", so the correct rate is "
              + ("100%" if target else "0%"),
              vals, (-14, 118), fmt="{:.0f}", target=target,
              target_label="correct", xlabel="Prompts answered with a bid (%)")

    offer = [(stats[m][2]["overall"]["offer_exact_match_pct"],
              stats[m][3]["overall"]["offer_exact_match_pct"]) for m, _ in MODELS]
    panel((1, 1), "E", "Names the exact price the rule names",
          "to the cent, on every prompt where a bid was due",
          offer, (-8, 88), target=None, xlabel="Exact price matches (%)")

    js = [(stats[m][2]["overall"]["json_valid_pct"],
           stats[m][3]["overall"]["json_valid_pct"]) for m, _ in MODELS]
    panel((1, 2), "F", "Answers in the required format",
          "valid JSON, which the surrounding system needs to read the bid",
          js, (98.6, 100.35), fmt="{:.2f}", target=100, target_label="required",
          xlabel="Valid JSON (%)")

    days = [(stats[m][2]["overall"]["days_exact_match_pct"],
             stats[m][3]["overall"]["days_exact_match_pct"]) for m, _ in MODELS]
    panel((2, 0), "G", "Reports the delivery time correctly",
          "a field carried through from the opening round, it must not degrade",
          days, (10, 112), target=None, xlabel="Correct delivery-time figure (%)")

    gates_passed = [(sum(1 for g in stats[m][0]["gates"].values() if g["status"] == "PASS"),
                     sum(1 for g in stats[m][1]["gates"].values() if g["status"] == "PASS"))
                    for m, _ in MODELS]
    ax = panel((2, 1), "H", "Release gates cleared, out of 11",
               "the eleven pre-set bars this project agreed in advance",
               gates_passed, (-1.4, 12.6), fmt="{:.0f}", target=11,
               target_label="all 11", xlabel="Gates passed")
    ax.set_xticks([0, 2, 4, 6, 8, 10])

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ax.text(0, 0.94, "What this shows", fontsize=12.5, fontweight="bold",
            color=INK, va="top", transform=ax.transAxes)
    ax.text(0, 0.80,
            "Training works. Every model goes from broadly\n"
            "ignoring the rules to following them on more than\n"
            "94% of prompts, with valid output every time.\n\n"
            "One weakness is left, panel C. When the price the\n"
            "agent is asked to offer is exactly its own minimum,\n"
            "five of six refuse instead of bidding. They handle\n"
            "clearly-above and clearly-below correctly, so this\n"
            "is the boundary itself, not the idea of a floor.\n\n"
            "Granite is the only model that clears all eleven\n"
            "gates.",
            fontsize=10.5, color=INK2, va="top", transform=ax.transAxes, linespacing=1.62)

    handles = [
        Line2D([], [], marker="o", ls="", ms=10, mfc=BASE, mec="none", label="Untrained model"),
        Line2D([], [], marker="o", ls="", ms=10, mfc=TRAINED, mec="none", label="After training"),
        Line2D([], [], color=REF, lw=2.4, label="Correct answer / required level"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.045), columnspacing=3.0)
    fig.text(0.5, 0.014,
             "Drawn from evaluation files already on disk, reports/reverse_auction/epoch-1/v3, "
             "scored after the minimum-price rounding fix. Nothing was retrained or re-scored "
             "for this chart.",
             ha="center", va="bottom", fontsize=9.5, color=MUTED)
    return save(fig, "reverse_auction_models_comparison.png")


if __name__ == "__main__":
    chart_gates()
    chart_floor()
    chart_cost()
    chart_training()
    chart_models_comparison()

