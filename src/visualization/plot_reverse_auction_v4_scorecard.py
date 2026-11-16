"""Create the v4 gate scorecard: six models against all eleven release gates.

Current-state scorecard, v4 only. The v3-against-v4 movement is already covered by
reverse_auction_six_model_v3_vs_v4.png; this one answers "where does each model
stand now".

Each cell carries the trained model's measured value and its pass/fail. The thin
strip on the cell's left edge is the same gate for the untrained base model, so a
reader can see what training actually moved.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
EPOCH = REPO / "reports" / "reverse_auction" / "epoch-1"
OUT = REPO / "reports" / "reverse_auction"
VERSION = "v4"

SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
PASS_EDGE, FAIL_EDGE = "#1baf7a", "#d03b3b"
PASS_FILL, FAIL_FILL = "#e8f4ee", "#fbeaea"
NOTE_INK = "#2a78d6"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
})

MODELS = [
    ("qwen", "Qwen3.5-4B"),
    ("granite", "Granite-4.1-3b"),
    ("phi", "Phi-4-mini"),
    ("ministral", "Ministral-3-3B"),
    ("gemma", "Gemma-4-E2B"),
    ("smollm3", "SmolLM3-3B"),
]

# (gate key, plain-language row label, the bar it must clear)
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

# the two failures that are evaluation artifacts rather than model defects; each
# is footnoted on the chart so an unexplained red cell cannot mislead a reader
CAVEATS = {
    ("gemma", "6_json_validity"): "a",
    ("phi", "7_days_no_regression"): "b",
}
FOOTNOTES = [
    ("a", "Gemma, valid JSON: not malformed output. On 29 of 1,296 prompts it narrates its "
          "constraint checks first and exhausts the 200-token generation budget before "
          "reaching the JSON.\n     Its v3 arm did this on 0 of 1,296. Deliberately not "
          "re-scored at a larger budget, which would require re-running every arm for "
          "comparability."),
    ("b", "Phi, delivery-time: the two rates are not measured over the same rows. The untrained "
          "arm bids on 40 of 1,296 prompts and gets all 40 right; the trained arm bids on 873 "
          "and gets 708 right.\n     The gate compares those rates directly. This is a limitation "
          "of the gate definition, left unchanged because altering it would invalidate every "
          "earlier result."),
]


def load(version, tag):
    hits = sorted((EPOCH / version / "gates").glob(f"eval_{tag}_*_gates.json"))
    if not hits:
        raise FileNotFoundError(f"no gate file for {tag} under {EPOCH / version / 'gates'}")
    with hits[-1].open(encoding="utf-8") as handle:
        return json.load(handle)


def chart():
    trained = {m: load(VERSION, f"ra-e1-{m}") for m, _ in MODELS}
    base = {m: load(VERSION, f"ra-e1-{m}-base") for m, _ in MODELS}

    fig = plt.figure(figsize=(15.6, 10.2))
    ax = fig.add_axes([0.335, 0.212, 0.630, 0.638])
    ax.set_xlim(0, len(MODELS))
    ax.set_ylim(len(GATE_ROWS), -1.35)
    ax.axis("off")

    fig.text(0.035, 0.950, "Reverse auction: where all six models stand on the eleven gates",
             ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.035, 0.912,
             "Trained on the v4 worded dataset, scored on 1,296 held-out prompts each. "
             "Number in each cell is the measured value.",
             ha="left", fontsize=11, color=INK2)

    # column headers
    for j, (model, label) in enumerate(MODELS):
        ax.text(j + 0.5, -1.02, label, ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=INK)
        passed = trained[model]["counts"]["PASS"]
        ax.text(j + 0.5, -0.55, f"{passed} of 11", ha="center", va="bottom",
                fontsize=10, color=PASS_EDGE if passed == 11 else INK2,
                fontweight="bold" if passed == 11 else "normal")

    for i, (key, label, bar) in enumerate(GATE_ROWS):
        ax.text(-0.055, i + 0.40, label, ha="right", va="center", fontsize=10.5,
                color=INK, transform=ax.transData)
        ax.text(-0.055, i + 0.73, bar, ha="right", va="center", fontsize=8.8,
                color=MUTED, transform=ax.transData)
        for j, (model, _) in enumerate(MODELS):
            gate = trained[model]["gates"][key]
            ok = gate["status"] == "PASS"
            ax.add_patch(Rectangle((j + 0.05, i + 0.10), 0.90, 0.80,
                                   facecolor=PASS_FILL if ok else FAIL_FILL,
                                   edgecolor=PASS_EDGE if ok else FAIL_EDGE,
                                   lw=1.0, zorder=2))
            # the untrained baseline for the same gate, as a strip on the left edge
            b = base[model]["gates"][key]
            b_ok = b["status"] == "PASS"
            ax.add_patch(Rectangle((j + 0.05, i + 0.10), 0.058, 0.80,
                                   facecolor=PASS_EDGE if b_ok else FAIL_EDGE,
                                   edgecolor="none", alpha=0.55, zorder=3))
            value = f"{gate['measured']:g}"
            ax.text(j + 0.52, i + 0.50, value, ha="center", va="center", fontsize=10.5,
                    color=INK if ok else FAIL_EDGE,
                    fontweight="normal" if ok else "bold", zorder=4)
            # fails are never colour alone
            if not ok:
                ax.text(j + 0.90, i + 0.50, "miss", ha="right", va="center", fontsize=8.2,
                        color=FAIL_EDGE, fontweight="bold", zorder=4)
            mark = CAVEATS.get((model, key))
            if mark:
                ax.text(j + 0.115, i + 0.24, mark, ha="left", va="center", fontsize=8.5,
                        color=NOTE_INK, fontweight="bold", zorder=5)

    handles = [
        Line2D([], [], marker="s", ls="", ms=12, mfc=PASS_FILL, mec=PASS_EDGE,
               label="meets the bar"),
        Line2D([], [], marker="s", ls="", ms=12, mfc=FAIL_FILL, mec=FAIL_EDGE,
               label="misses the bar"),
        Line2D([], [], marker="s", ls="", ms=12, mfc=PASS_EDGE, mec="none", alpha=0.55,
               label="left strip: untrained model met it"),
        Line2D([], [], marker="s", ls="", ms=12, mfc=FAIL_EDGE, mec="none", alpha=0.55,
               label="left strip: untrained model missed it"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.140), columnspacing=2.2, handletextpad=0.7)

    y = 0.104
    for mark, text in FOOTNOTES:
        fig.text(0.035, y, f"{mark}  {text}", ha="left", va="top", fontsize=8.8,
                 color=INK2, linespacing=1.6)
        y -= 0.052
    return save(fig)


def save(fig):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "reverse_auction_v4_gate_scorecard.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


if __name__ == "__main__":
    chart()
