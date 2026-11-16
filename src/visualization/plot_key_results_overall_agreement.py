"""Plot presentation charts for sealed and reverse-auction model results.

The sealed charts compare the base checkpoint with one-epoch and two-epoch
fine-tunes. Rule-based logic is the fixed reference: 100% decision agreement
and 0% offer deviation. Every value is recomputed from raw evaluation rows.

The reverse-auction charts compare the base checkpoint with the corrected v3
fine-tune and the v4 fine-tune on the same 1,296-row evaluation sequence.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO / "reports" / "bidding"

OUTPUTS = {
    "sealed_agreement": REPORT_DIR / "key_results_sealed_overall_agreement.png",
    "sealed_offer_deviation": REPORT_DIR / "key_results_sealed_offer_deviation.png",
    "reverse_auction_agreement": REPORT_DIR / "key_results_reverse_auction_overall_agreement.png",
    "reverse_auction_offer_deviation": REPORT_DIR / "key_results_reverse_auction_offer_deviation.png",
}

SEALED_RUNS = {
    "qwen": {
        "base": REPO / "reports/reference/base-evaluations/raw/eval_base-audit_20260718_235632.json",
        "epoch_1": REPO / "reports/epoch-1/production/evaluation/raw/eval_qwen-v2_20260720_025436.json",
        "epoch_2": REPO / "reports/epoch-2/production/evaluation/raw/eval_qwen-e2_20260728_032713.json",
    },
    "granite": {
        "base": REPO / "reports/reference/base-evaluations/raw/eval_granite-base_20260719_170737.json",
        "epoch_1": REPO / "reports/epoch-1/production/evaluation/raw/eval_granite-v2_20260720_031152.json",
        "epoch_2": REPO / "reports/epoch-2/production/evaluation/raw/eval_granite-e2_20260727_192603.json",
    },
    "phi": {
        "base": REPO / "reports/reference/base-evaluations/raw/eval_phi-base_20260720_195336.json",
        "epoch_1": REPO / "reports/epoch-1/production/evaluation/raw/eval_phi-v2_20260720_200758.json",
        "epoch_2": REPO / "reports/epoch-2/production/evaluation/raw/eval_phi-e2_20260727_225122.json",
    },
    "ministral": {
        "base": REPO / "reports/reference/base-evaluations/raw/eval_ministral-base_20260722_011608.json",
        "epoch_1": REPO / "reports/epoch-1/production/evaluation/raw/eval_ministral-v2_20260722_013418.json",
        "epoch_2": REPO / "reports/epoch-2/production/evaluation/raw/eval_ministral-e2_20260728_160341.json",
    },
    "gemma": {
        "base": REPO / "reports/reference/base-evaluations/raw/eval_gemma-base_20260721_010311.json",
        "epoch_1": REPO / "reports/epoch-1/production/evaluation/raw/eval_gemma-v2_20260721_012125.json",
        "epoch_2": REPO / "reports/epoch-2/production/evaluation/raw/eval_gemma-e2_20260728_093025.json",
    },
    "smollm3": {
        "base": REPO / "reports/reference/base-evaluations/raw/eval_smollm3-base_20260722_004118.json",
        "epoch_1": REPO / "reports/epoch-1/production/evaluation/raw/eval_smollm3-v2_20260722_005904.json",
        "epoch_2": REPO / "reports/epoch-2/production/evaluation/raw/eval_smollm3-e2_20260728_191351.json",
    },
}

AUCTION_RUNS = {
    "qwen": {
        "base": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-qwen-base_20260814_052759.json",
        "v3": REPO / "reports/reverse_auction/epoch-1/v3/evaluation/raw/eval_ra-e1-qwen-fixed_20260809_175115.json",
        "v4": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-qwen_20260814_053845.json",
    },
    "granite": {
        "base": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-granite-base_20260818_095357.json",
        "v3": REPO / "reports/reverse_auction/epoch-1/v3/evaluation/raw/eval_ra-e1-granite-fixed_20260809_181132.json",
        "v4": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-granite_20260818_101439.json",
    },
    "phi": {
        "base": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-phi-base_20260818_031642.json",
        "v3": REPO / "reports/reverse_auction/epoch-1/v3/evaluation/raw/eval_ra-e1-phi-fixed_20260809_182634.json",
        "v4": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-phi_20260818_032947.json",
    },
    "ministral": {
        "base": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-ministral-base_20260817_192449.json",
        "v3": REPO / "reports/reverse_auction/epoch-1/v3/evaluation/raw/eval_ra-e1-ministral-fixed_20260809_184546.json",
        "v4": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-ministral_20260817_193957.json",
    },
    "gemma": {
        "base": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-gemma-base_20260817_064909.json",
        "v3": REPO / "reports/reverse_auction/epoch-1/v3/evaluation/raw/eval_ra-e1-gemma-fixed_20260809_190701.json",
        "v4": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-gemma_20260817_071025.json",
    },
    "smollm3": {
        "base": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-smollm3-base_20260817_195848.json",
        "v3": REPO / "reports/reverse_auction/epoch-1/v3/evaluation/raw/eval_ra-e1-smollm3-fixed_20260809_192423.json",
        "v4": REPO / "reports/reverse_auction/epoch-1/v4/evaluation/raw/eval_ra-e1-smollm3_20260818_173333.json",
    },
}

MODELS = [
    ("qwen", "Qwen3.5-4B"),
    ("granite", "Granite-4.1-3B"),
    ("phi", "Phi-4-mini"),
    ("ministral", "Ministral-3-3B"),
    ("gemma", "Gemma-4-E2B"),
    ("smollm3", "SmolLM3-3B"),
]

SHORT_LABELS = {
    "qwen": "Qwen",
    "granite": "Granite",
    "phi": "Phi",
    "ministral": "Ministral",
    "gemma": "Gemma",
    "smollm3": "SmolLM3",
}

# Match reports/reverse_auction/reverse_auction_six_model_v3_vs_v4.png.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BASE = "#5f605c"
EPOCH_1 = "#287fd1"
EPOCH_2 = "#1769aa"
REFERENCE = "#43a984"

ARM_STYLE = {
    "base": {"label": "Base", "color": BASE, "marker": "o", "size": 8},
    "epoch_1": {"label": "Epoch 1", "color": EPOCH_1, "marker": "o", "size": 8},
    "epoch_2": {"label": "Epoch 2", "color": EPOCH_2, "marker": "o", "size": 9},
    "fine_tuned": {"label": "Fine-tuned", "color": EPOCH_2, "marker": "o", "size": 9},
    "v3": {"label": "v3 fine-tuned", "color": EPOCH_1, "marker": "o", "size": 8},
    "v4": {"label": "v4 fine-tuned", "color": "#eb6834", "marker": "o", "size": 9},
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 13,
})


def accuracy(rows):
    """Return raw decision agreement and the matching-row count."""
    matched = sum(row.get("decision_matched") is True for row in rows)
    return 100.0 * matched / len(rows), matched


def offer_deviation(rows):
    """Pool every scorable raw offer-deviation value across all stages."""
    values = [
        float(row["offer_pct_dev"])
        for row in rows
        if row.get("offer_pct_dev") is not None
    ]
    if not values:
        raise ValueError("evaluation contains no scorable offer-deviation rows")
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def sealed_case_key(row):
    return row["id"], row["stage"], row["label_decision"]


def auction_case_key(row):
    return (
        row["id"],
        row["stage"],
        row["priority"],
        row["category"],
        row["round_number"],
        row["withdrawn_in_earlier_round"],
        row["current_best_bid"],
        row["platform_minimum_decrement"],
        row["label_decision"],
    )


def load_arm(mode, path, expected_n):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if mode == "sealed":
        rows = payload["main_holdout"]["rows"]
        case_keys = [sealed_case_key(row) for row in rows]
    else:
        rows = payload["rows"]
        case_keys = [auction_case_key(row) for row in rows]
    if len(rows) != expected_n:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_n}")
    value, matched = accuracy(rows)
    return {
        "accuracy": value,
        "matched": matched,
        "n": len(rows),
        "offer_deviation": offer_deviation(rows),
        "case_keys": case_keys,
        "source": path,
    }


def load_mode(mode, runs, arms, expected_n):
    results = {}
    canonical_cases = None
    for model, _ in MODELS:
        results[model] = {}
        for arm in arms:
            result = load_arm(mode, runs[model][arm], expected_n)
            if canonical_cases is None:
                canonical_cases = result["case_keys"]
            if result["case_keys"] != canonical_cases:
                raise ValueError(f"{mode}/{model}/{arm} uses a different holdout sequence")
            results[model][arm] = result
    return results


def load_results():
    return {
        "sealed": load_mode(
            "sealed", SEALED_RUNS, ("base", "epoch_1", "epoch_2"), 1500
        ),
        "reverse_auction": load_mode(
            "reverse_auction", AUCTION_RUNS, ("base", "v3", "v4"), 1296
        ),
    }


def style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_axisbelow(True)


def add_legend(fig, arms, reference_label):
    handles = [
        Line2D(
            [], [], marker=ARM_STYLE[arm]["marker"], ls="",
            ms=ARM_STYLE[arm]["size"], mfc=ARM_STYLE[arm]["color"],
            mec="none", label=ARM_STYLE[arm]["label"],
        )
        for arm in arms
    ]
    handles.append(Line2D([], [], color=REFERENCE, lw=1.5, label=reference_label))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=len(handles),
        frameon=False,
        fontsize=11,
        columnspacing=2.4,
    )


def arm_offsets(arms):
    if len(arms) == 3:
        return {"base": 0.0, "epoch_1": -0.13, "epoch_2": 0.13}
    return {arm: 0.0 for arm in arms}


def draw_series(
    ax,
    results,
    arms,
    value_key,
    value_format,
    span,
    trained_label_x=None,
):
    """Draw one connected model row with restrained offsets for close points."""
    offsets = arm_offsets(arms)
    halo = [path_effects.withStroke(linewidth=3.2, foreground=SURFACE)]
    pad = span * 0.018

    for y, (model, _) in enumerate(MODELS):
        values = []
        ys = []
        for arm in arms:
            record = results[model][arm]
            value = (
                record["accuracy"]
                if value_key == "accuracy"
                else record["offer_deviation"]["mean"]
            )
            values.append(value)
            ys.append(y + offsets[arm])

        ax.plot(
            values,
            ys,
            color=AXIS,
            lw=1.8,
            solid_capstyle="round",
            zorder=1,
        )

        for arm, value, point_y in zip(arms, values, ys):
            style = ARM_STYLE[arm]
            ax.plot(
                value,
                point_y,
                marker=style["marker"],
                ms=style["size"],
                color=style["color"],
                mec=SURFACE,
                mew=1.4,
                zorder=4,
            )

        base = values[0]
        ax.text(
            base - pad,
            ys[0],
            value_format.format(base),
            va="center",
            ha="right",
            fontsize=11,
            color=ARM_STYLE[arms[0]]["color"],
            path_effects=halo,
            zorder=6,
        )

        if len(arms) == 2:
            final = values[1]
            ax.text(
                final + pad,
                ys[1],
                value_format.format(final),
                va="center",
                ha="left",
                fontsize=11.5,
                fontweight="bold",
                color=ARM_STYLE[arms[1]]["color"],
                path_effects=halo,
                zorder=6,
            )
        else:
            ax.text(
                trained_label_x,
                ys[1],
                value_format.format(values[1]),
                ha="left",
                va="center",
                fontsize=10.5,
                color=ARM_STYLE[arms[1]]["color"],
                path_effects=halo,
                zorder=6,
            )
            ax.text(
                trained_label_x,
                ys[2],
                value_format.format(values[2]),
                ha="left",
                va="center",
                fontsize=10.8,
                fontweight="bold",
                color=ARM_STYLE[arms[2]]["color"],
                path_effects=halo,
                zorder=6,
            )


def prepare_figure(title):
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    fig.subplots_adjust(left=0.235, right=0.94, top=0.86, bottom=0.22)
    fig.text(
        0.055,
        0.935,
        title,
        fontsize=15,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([label for _, label in MODELS])
    ax.set_ylim(len(MODELS) - 0.5, -0.5)
    style_axis(ax)
    return fig, ax


def draw_sealed_agreement(results):
    """Show the base range and a readable view of the two trained arms."""
    chart_surface = "#ffffff"
    fig = plt.figure(figsize=(12.8, 5.8), facecolor=chart_surface)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.18, 1.0),
        left=0.145,
        right=0.965,
        top=0.96,
        bottom=0.205,
        wspace=0.052,
    )
    base_ax = fig.add_subplot(grid[0, 0])
    tuned_ax = fig.add_subplot(grid[0, 1], sharey=base_ax)
    halo = [path_effects.withStroke(linewidth=3.4, foreground=chart_surface)]

    for y, (model, _) in enumerate(MODELS):
        base = results[model]["base"]["accuracy"]
        epoch_1 = results[model]["epoch_1"]["accuracy"]
        epoch_2 = results[model]["epoch_2"]["accuracy"]

        # Plain lines, without directional arrowheads, join the three arms.
        base_ax.plot([base, 98.15], [y, y], color=AXIS, lw=1.9,
                     solid_capstyle="round", zorder=1)
        tuned_ax.plot([98.80, epoch_2], [y, y], color=AXIS, lw=1.9,
                      solid_capstyle="round", zorder=1)

        base_ax.plot(base, y, "o", ms=9, color=BASE, mec=chart_surface,
                     mew=1.5, zorder=4)
        tuned_ax.plot(epoch_1, y, "o", ms=9.5, color=EPOCH_1,
                      mec=chart_surface, mew=1.5, zorder=4)
        tuned_ax.plot(epoch_2, y, "o", ms=10, color="#eb6834",
                      mec=chart_surface, mew=1.5, zorder=5)

        base_ax.text(base - 1.4, y, f"{base:.1f}", va="center", ha="right",
                     fontsize=12, fontweight="bold", color=BASE,
                     path_effects=halo, zorder=6)
        tuned_ax.text(epoch_1 - 0.06, y, f"{epoch_1:.1f}", va="center",
                      ha="right", fontsize=12, fontweight="bold",
                      color=EPOCH_1, path_effects=halo, zorder=6)
        tuned_ax.text(epoch_2 + 0.06, y, f"{epoch_2:.1f}", va="center",
                      ha="left", fontsize=12, fontweight="bold",
                      color="#eb6834", path_effects=halo, zorder=6)

    for ax in (base_ax, tuned_ax):
        ax.set_facecolor(chart_surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.85, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=12)

    base_ax.set_xlim(20, 98.15)
    base_ax.set_xticks([20, 40, 60, 80])
    base_ax.set_yticks(range(len(MODELS)))
    base_ax.set_yticklabels(
        [SHORT_LABELS[model] for model, _ in MODELS],
        fontsize=15,
        fontweight="bold",
        color="#2f3437",
    )
    base_ax.tick_params(axis="y", length=0, pad=11)

    tuned_ax.set_xlim(98.80, 100.35)
    tuned_ax.set_xticks([99.0, 99.5, 100.0])
    tuned_ax.tick_params(axis="y", left=False, labelleft=False)
    tuned_ax.axvline(100, color=REFERENCE, lw=1.6, alpha=0.9, zorder=3)
    tuned_ax.text(99.98, len(MODELS) - 0.58, "rule-based, 100%",
                  ha="right", va="bottom", fontsize=10.5,
                  fontweight="bold", color=REFERENCE)

    # The axis marks disclose the compressed interval between both views.
    break_size = 0.012
    base_ax.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, break_size),
        transform=base_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    tuned_ax.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=tuned_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )

    fig.text(0.56, 0.135, "Answers matching the rule (%)", fontsize=12.5,
             color=INK2, ha="center", va="center")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=9, mfc=BASE,
                   mec="none", label="Base"),
            Line2D([], [], marker="o", ls="", ms=9.5, mfc=EPOCH_1,
                   mec="none", label="Epoch 1"),
            Line2D([], [], marker="o", ls="", ms=10, mfc="#eb6834",
                   mec="none", label="Epoch 2"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        columnspacing=2.8,
    )

    output = OUTPUTS["sealed_agreement"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor=chart_surface)
    plt.close(fig)
    return output


def draw_agreement(mode, results):
    if mode == "sealed":
        return draw_sealed_agreement(results)
    return draw_reverse_auction_agreement(results)


def draw_reverse_auction_agreement(results):
    """Compare reverse-auction Base, corrected v3, and v4 agreement."""
    chart_surface = "#ffffff"
    fig = plt.figure(figsize=(12.8, 5.8), facecolor=chart_surface)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.18),
        left=0.145,
        right=0.965,
        top=0.96,
        bottom=0.205,
        wspace=0.052,
    )
    base_ax = fig.add_subplot(grid[0, 0])
    tuned_ax = fig.add_subplot(grid[0, 1], sharey=base_ax)
    halo = [path_effects.withStroke(linewidth=3.4, foreground=chart_surface)]

    for y, (model, _) in enumerate(MODELS):
        base = results[model]["base"]["accuracy"]
        v3 = results[model]["v3"]["accuracy"]
        v4 = results[model]["v4"]["accuracy"]

        base_ax.plot([base, 71.0], [y, y], color=AXIS, lw=1.9,
                     solid_capstyle="round", zorder=1)
        tuned_ax.plot([93.5, v4], [y, y], color=AXIS, lw=1.9,
                      solid_capstyle="round", zorder=1)

        base_ax.plot(base, y, "o", ms=9, color=BASE, mec=chart_surface,
                     mew=1.5, zorder=4)
        tuned_ax.plot(v3, y, "o", ms=9.5, color=EPOCH_1,
                      mec=chart_surface, mew=1.5, zorder=4)
        tuned_ax.plot(v4, y, "o", ms=10, color="#eb6834",
                      mec=chart_surface, mew=1.5, zorder=5)

        base_ax.text(base - 0.8, y, f"{base:.1f}", va="center", ha="right",
                     fontsize=12, fontweight="bold", color=BASE,
                     path_effects=halo, zorder=6)
        tuned_ax.text(v3 - 0.12, y, f"{v3:.1f}", va="center", ha="right",
                      fontsize=12, fontweight="bold", color=EPOCH_1,
                      path_effects=halo, zorder=6)
        tuned_ax.text(v4 + 0.12, y, f"{v4:.1f}", va="center", ha="left",
                      fontsize=12, fontweight="bold", color="#eb6834",
                      path_effects=halo, zorder=6)

    for ax in (base_ax, tuned_ax):
        ax.set_facecolor(chart_surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.85, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=12)

    base_ax.set_xlim(30, 71.0)
    base_ax.set_xticks([30, 40, 50, 60, 70])
    base_ax.set_yticks(range(len(MODELS)))
    base_ax.set_yticklabels(
        [SHORT_LABELS[model] for model, _ in MODELS],
        fontsize=15,
        fontweight="bold",
        color="#2f3437",
    )
    base_ax.tick_params(axis="y", length=0, pad=11)

    tuned_ax.set_xlim(93.5, 100.35)
    tuned_ax.set_xticks([94, 96, 98, 100])
    tuned_ax.tick_params(axis="y", left=False, labelleft=False)
    tuned_ax.axvline(100, color=REFERENCE, lw=1.6, alpha=0.9, zorder=3)
    tuned_ax.text(99.96, len(MODELS) - 0.58, "rule-based, 100%",
                  ha="right", va="bottom", fontsize=10.5,
                  fontweight="bold", color=REFERENCE)

    break_size = 0.012
    base_ax.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, break_size),
        transform=base_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    tuned_ax.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=tuned_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )

    fig.text(0.56, 0.135, "Answers matching the rule (%)", fontsize=12.5,
             color=INK2, ha="center", va="center")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=9, mfc=BASE,
                   mec="none", label="Base"),
            Line2D([], [], marker="o", ls="", ms=9.5, mfc=EPOCH_1,
                   mec="none", label="v3 fine-tuned"),
            Line2D([], [], marker="o", ls="", ms=10, mfc="#eb6834",
                   mec="none", label="v4 fine-tuned"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        columnspacing=2.8,
    )

    output = OUTPUTS["reverse_auction_agreement"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor=chart_surface)
    plt.close(fig)
    return output


def draw_sealed_offer_deviation(results):
    """Show trained pricing accuracy alongside the distant base-model range."""
    chart_surface = "#ffffff"
    fig = plt.figure(figsize=(12.8, 5.8), facecolor=chart_surface)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.18),
        left=0.145,
        right=0.965,
        top=0.96,
        bottom=0.205,
        wspace=0.052,
    )
    tuned_ax = fig.add_subplot(grid[0, 0])
    base_ax = fig.add_subplot(grid[0, 1])
    halo = [path_effects.withStroke(linewidth=3.4, foreground=chart_surface)]

    for y, (model, _) in enumerate(MODELS):
        base = results[model]["base"]["offer_deviation"]["mean"]
        epoch_1 = results[model]["epoch_1"]["offer_deviation"]["mean"]
        epoch_2 = results[model]["epoch_2"]["offer_deviation"]["mean"]

        # Plain lines, without directional arrowheads, join the three arms.
        tuned_ax.plot([epoch_2, 2.55], [y, y], color=AXIS, lw=1.9,
                      solid_capstyle="round", zorder=1)
        base_ax.plot([35.0, base], [y, y], color=AXIS, lw=1.9,
                     solid_capstyle="round", zorder=1)

        tuned_ax.plot(epoch_1, y, "o", ms=9.5, color=EPOCH_1,
                      mec=chart_surface, mew=1.5, zorder=4)
        tuned_ax.plot(epoch_2, y, "o", ms=10, color="#eb6834",
                      mec=chart_surface, mew=1.5, zorder=5)
        base_ax.plot(base, y, "o", ms=9, color=BASE, mec=chart_surface,
                     mew=1.5, zorder=4)

        tuned_ax.text(epoch_2 - 0.08, y, f"{epoch_2:.2f}", va="center",
                      ha="right", fontsize=12, fontweight="bold",
                      color="#eb6834", path_effects=halo, zorder=6)
        tuned_ax.text(epoch_1 + 0.08, y, f"{epoch_1:.2f}", va="center",
                      ha="left", fontsize=12, fontweight="bold",
                      color=EPOCH_1, path_effects=halo, zorder=6)
        base_ax.text(base - 0.9, y, f"{base:.2f}", va="center", ha="right",
                     fontsize=12, fontweight="bold", color=BASE,
                     path_effects=halo, zorder=6)

    for ax in (tuned_ax, base_ax):
        ax.set_facecolor(chart_surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.85, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=12)

    tuned_ax.set_xlim(-0.15, 2.55)
    tuned_ax.set_xticks([0, 1, 2])
    tuned_ax.set_yticks(range(len(MODELS)))
    tuned_ax.set_yticklabels(
        [SHORT_LABELS[model] for model, _ in MODELS],
        fontsize=15,
        fontweight="bold",
        color="#2f3437",
    )
    tuned_ax.tick_params(axis="y", length=0, pad=11)
    tuned_ax.axvline(0, color=REFERENCE, lw=1.6, alpha=0.9, zorder=3)
    tuned_ax.text(0.04, len(MODELS) - 0.58, "rule-based, 0%",
                  ha="left", va="bottom", fontsize=10.5,
                  fontweight="bold", color=REFERENCE)

    base_ax.set_xlim(35.0, 70.5)
    base_ax.set_xticks([40, 50, 60, 70])
    base_ax.tick_params(axis="y", left=False, labelleft=False)

    # The axis marks disclose the omitted interval between both views.
    break_size = 0.012
    tuned_ax.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, break_size),
        transform=tuned_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    base_ax.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=base_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )

    fig.text(0.56, 0.135, "Mean absolute offer deviation (%)", fontsize=12.5,
             color=INK2, ha="center", va="center")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=9, mfc=BASE,
                   mec="none", label="Base"),
            Line2D([], [], marker="o", ls="", ms=9.5, mfc=EPOCH_1,
                   mec="none", label="Epoch 1"),
            Line2D([], [], marker="o", ls="", ms=10, mfc="#eb6834",
                   mec="none", label="Epoch 2"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        columnspacing=2.8,
    )

    output = OUTPUTS["sealed_offer_deviation"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor=chart_surface)
    plt.close(fig)
    return output


def draw_reverse_auction_offer_deviation(results):
    """Compare raw-row offer deviation for Base, corrected v3, and v4."""
    chart_surface = "#ffffff"
    fig = plt.figure(figsize=(12.8, 5.8), facecolor=chart_surface)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(0.16, 2.0, 0.40, 1.0),
        left=0.145,
        right=0.965,
        top=0.96,
        bottom=0.205,
        wspace=0.060,
    )
    reference_ax = fig.add_subplot(grid[0, 0])
    cluster_ax = fig.add_subplot(grid[0, 1])
    granite_ax = fig.add_subplot(grid[0, 2])
    base_ax = fig.add_subplot(grid[0, 3])
    halo = [path_effects.withStroke(linewidth=3.4, foreground=chart_surface)]

    for y, (model, _) in enumerate(MODELS):
        base = results[model]["base"]["offer_deviation"]["mean"]
        v3 = results[model]["v3"]["offer_deviation"]["mean"]
        v4 = results[model]["v4"]["offer_deviation"]["mean"]

        # Straight segments retain the sealed chart's comparison structure.
        cluster_ax.plot([min(v3, v4), 0.092], [y, y], color=AXIS, lw=1.9,
                        solid_capstyle="round", zorder=1)
        granite_ax.plot([0.145, 0.175], [y, y], color=AXIS, lw=1.9,
                        solid_capstyle="round", zorder=1)
        base_ax.plot([5.0, base], [y, y], color=AXIS, lw=1.9,
                     solid_capstyle="round", zorder=1)

        v3_ax = cluster_ax if v3 < 0.10 else granite_ax
        v4_ax = cluster_ax if v4 < 0.10 else granite_ax
        v3_ax.plot(v3, y, "o", ms=9.5, color=EPOCH_1,
                   mec=chart_surface, mew=1.5, zorder=4)
        v4_ax.plot(v4, y, "o", ms=10, color="#eb6834",
                   mec=chart_surface, mew=1.5, zorder=5)
        base_ax.plot(base, y, "o", ms=9, color=BASE, mec=chart_surface,
                     mew=1.5, zorder=4)

        v3_ax.text(v3, y - 0.13, f"{v3:.3f}", va="bottom",
                   ha="center", fontsize=11.5, fontweight="bold",
                   color=EPOCH_1, path_effects=halo, zorder=6)
        v4_ax.text(v4, y + 0.13, f"{v4:.3f}", va="top",
                   ha="center", fontsize=11.5, fontweight="bold",
                   color="#eb6834", path_effects=halo, zorder=6)
        base_ax.text(base + 0.6, y, f"{base:.2f}", va="center", ha="left",
                     fontsize=12, fontweight="bold", color=BASE,
                     path_effects=halo, zorder=6)

    for ax in (reference_ax, cluster_ax, granite_ax, base_ax):
        ax.set_facecolor(chart_surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.85, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=12)

    reference_ax.set_xlim(-0.001, 0.004)
    reference_ax.set_xticks([0.00])
    reference_ax.set_yticks(range(len(MODELS)))
    reference_ax.set_yticklabels(
        [SHORT_LABELS[model] for model, _ in MODELS],
        fontsize=15,
        fontweight="bold",
        color="#2f3437",
    )
    reference_ax.tick_params(axis="y", length=0, pad=11)
    reference_ax.axvline(0, color=REFERENCE, lw=1.6, alpha=0.9, zorder=3)
    cluster_ax.set_xlim(0.028, 0.092)
    cluster_ax.set_xticks([0.03, 0.05, 0.07, 0.09])
    cluster_ax.tick_params(axis="y", left=False, labelleft=False)
    cluster_ax.text(0.029, 0.50, "rule-based, 0%",
                    ha="left", va="center", fontsize=10.5,
                    fontweight="bold", color=REFERENCE)

    granite_ax.set_xlim(0.145, 0.175)
    granite_ax.set_xticks([0.15, 0.17])
    granite_ax.tick_params(axis="y", left=False, labelleft=False)

    base_ax.set_xlim(5.0, 34.5)
    base_ax.set_xticks([10, 20, 30])
    base_ax.tick_params(axis="y", left=False, labelleft=False)

    break_size = 0.012
    reference_ax.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, break_size),
        transform=reference_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    cluster_ax.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=cluster_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    cluster_ax.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, break_size),
        transform=cluster_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    granite_ax.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=granite_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    granite_ax.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, break_size),
        transform=granite_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )
    base_ax.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=base_ax.transAxes,
        color=AXIS,
        lw=1.2,
        clip_on=False,
    )

    fig.text(0.56, 0.135, "Mean absolute offer deviation (%)", fontsize=12.5,
             color=INK2, ha="center", va="center")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=9, mfc=BASE,
                   mec="none", label="Base"),
            Line2D([], [], marker="o", ls="", ms=9.5, mfc=EPOCH_1,
                   mec="none", label="v3 fine-tuned"),
            Line2D([], [], marker="o", ls="", ms=10, mfc="#eb6834",
                   mec="none", label="v4 fine-tuned"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        columnspacing=2.8,
    )

    output = OUTPUTS["reverse_auction_offer_deviation"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor=chart_surface)
    plt.close(fig)
    return output


def print_sources(results, arms):
    for model, _ in MODELS:
        for arm in arms:
            result = results[model][arm]
            offer = ""
            if result["offer_deviation"] is not None:
                stats = result["offer_deviation"]
                offer = f" offer={stats['mean']:.4f}% (n={stats['n']})"
            print(
                f"  {model:9s} {arm:10s} {result['matched']:4d}/{result['n']} "
                f"= {result['accuracy']:.4f}%{offer}  "
                f"{result['source'].relative_to(REPO)}"
            )


def main():
    results = load_results()

    output = draw_agreement("sealed", results["sealed"])
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")
    output = draw_sealed_offer_deviation(results["sealed"])
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")
    print_sources(results["sealed"], ("base", "epoch_1", "epoch_2"))

    output = draw_agreement("reverse_auction", results["reverse_auction"])
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")
    output = draw_reverse_auction_offer_deviation(results["reverse_auction"])
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")
    print_sources(results["reverse_auction"], ("base", "v3", "v4"))


if __name__ == "__main__":
    main()
