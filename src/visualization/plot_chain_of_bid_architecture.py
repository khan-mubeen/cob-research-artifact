"""Render the Chain-of-Bid system architecture as SVG and transparent PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "reports" / "bidding"
SVG_OUTPUT = OUTPUT_DIR / "chain_of_bid_system_architecture.svg"
PNG_OUTPUT = OUTPUT_DIR / "chain_of_bid_system_architecture.png"

DARK_BLUE = "#0F4C81"
MEDIUM_BLUE = "#1769AA"
LIGHT_BLUE = "#EAF4F9"
HEADER_FILL = "#D7EBF5"
TEAL = "#1597A6"
TEXT = "#243447"
BORDER = "#AFC9D8"
WHITE = "#FFFFFF"

LINE_WIDTH = 1.75
ARROW_SCALE = 14
CORNER_RADIUS = 0.075

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "svg.fonttype": "none",
    "svg.hashsalt": "chain-of-bid-system-architecture",
})


def add_box(ax, x, y, width, height, label, *, fill=WHITE,
            edge=DARK_BLUE, fontsize=14, weight=600):
    """Draw a simple component box with a very small corner radius."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={CORNER_RADIUS}",
        linewidth=LINE_WIDTH,
        edgecolor=edge,
        facecolor=fill,
        antialiased=True,
        joinstyle="round",
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=fontsize,
        fontweight=weight,
        zorder=4,
    )


def add_arrow(ax, start, end, *, color=DARK_BLUE, dashed=False, zorder=2):
    """Draw a consistently styled process or return-message arrow."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ARROW_SCALE,
        linewidth=LINE_WIDTH,
        linestyle=(0, (4, 3)) if dashed else "solid",
        color=color,
        capstyle="round",
        joinstyle="round",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def add_routed_arrow(ax, points, *, color=DARK_BLUE):
    """Draw an orthogonal connector with one arrowhead at the destination."""
    route_x, route_y = zip(*points[:-1])
    ax.plot(
        route_x,
        route_y,
        color=color,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        zorder=1,
    )
    add_arrow(ax, points[-2], points[-1], color=color)


def add_orchestration_layer(ax):
    """Draw Supervisor and NATS as one compact orchestration component."""
    x, y, width, height = 4.45, 6.95, 7.10, 0.72
    layer = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={CORNER_RADIUS}",
        linewidth=LINE_WIDTH,
        edgecolor=DARK_BLUE,
        facecolor=LIGHT_BLUE,
        antialiased=True,
        joinstyle="round",
        zorder=3,
    )
    ax.add_patch(layer)
    ax.text(
        8.00,
        y + height / 2,
        "Supervisor + NATS Broker",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=17.5,
        fontweight=600,
        zorder=4,
    )

    # NATS publishes to and receives from the shared stage message bus.
    ax.plot(
        [2.86, 13.14],
        [6.18, 6.18],
        color=DARK_BLUE,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        zorder=1,
    )
    add_arrow(ax, (8.00, 6.95), (8.00, 6.18))


def add_stage_card(ax, x, title, agent_ids):
    """Draw one equal-sized stage table with four representative agents."""
    y, width, height = 2.72, 4.10, 2.48
    header_height = 0.58

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={CORNER_RADIUS}",
        linewidth=LINE_WIDTH,
        edgecolor=MEDIUM_BLUE,
        facecolor=WHITE,
        antialiased=True,
        joinstyle="round",
        zorder=3,
    )
    ax.add_patch(card)
    ax.add_patch(FancyBboxPatch(
        (x + 0.012, y + height - header_height),
        width - 0.024,
        header_height - 0.012,
        boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}",
        linewidth=0,
        facecolor=HEADER_FILL,
        antialiased=True,
        joinstyle="round",
        zorder=3.1,
    ))
    ax.add_patch(Rectangle(
        (x + 0.012, y + height - header_height),
        width - 0.024,
        header_height / 2,
        linewidth=0,
        facecolor=HEADER_FILL,
        antialiased=True,
        zorder=3.15,
    ))
    ax.plot(
        [x, x + width],
        [y + height - header_height, y + height - header_height],
        color=BORDER,
        linewidth=1.1,
        solid_capstyle="round",
        antialiased=True,
        zorder=3.2,
    )
    ax.text(
        x + width / 2,
        y + height - header_height / 2,
        title,
        ha="center",
        va="center",
        color=DARK_BLUE,
        fontsize=15.5,
        fontweight=600,
        zorder=4,
    )

    body_top = y + height - header_height
    row_height = (height - header_height) / len(agent_ids)
    for index, agent_id in enumerate(agent_ids):
        row_center = body_top - (index + 0.5) * row_height
        if index:
            divider_y = body_top - index * row_height
            ax.plot(
                [x + 0.34, x + width - 0.34],
                [divider_y, divider_y],
                color=BORDER,
                linewidth=0.85,
                solid_capstyle="round",
                antialiased=True,
                zorder=3.2,
            )
        ax.text(
            x + width / 2,
            row_center,
            agent_id,
            ha="center",
            va="center",
            color=TEXT,
            fontsize=13,
            fontweight=400,
            zorder=4,
        )


def add_stage_exchange(ax, center_x, number, stage_name):
    """Draw one RFQ path and its bid return beside the stage card."""
    bus_y = 6.18
    card_top = 5.20
    outbound_x = center_x - 0.34
    return_x = center_x + 0.34

    add_arrow(ax, (outbound_x, bus_y), (outbound_x, card_top))
    add_arrow(
        ax,
        (return_x, card_top),
        (return_x, bus_y),
        color=TEAL,
        dashed=True,
    )
    ax.text(
        outbound_x - 0.11,
        5.70,
        f"{number}. {stage_name} RFQ",
        ha="right",
        va="center",
        color=DARK_BLUE,
        fontsize=11.2,
        fontweight=600,
        zorder=4,
    )
    ax.text(
        return_x + 0.11,
        5.70,
        f"{number}.1 Bids",
        ha="left",
        va="center",
        color=TEAL,
        fontsize=11.2,
        fontweight=600,
        zorder=4,
    )


def render():
    """Build and export the architecture diagram."""
    fig = plt.figure(figsize=(16, 9), facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor("none")

    # Entry point and shared orchestration layer.
    add_box(ax, 6.50, 8.12, 3.00, 0.62, "Order Created")
    add_arrow(ax, (8.00, 8.12), (8.00, 7.67))
    add_orchestration_layer(ax)

    stages = [
        (1.15, 3.20, "Material Agents", "Material", ("M01", "M02", "M03", "...")),
        (5.95, 8.00, "Manufacturing Agents", "Manufacturing", ("T01", "T02", "T03", "...")),
        (10.75, 12.80, "Logistics Agents", "Logistics", ("L01", "L02", "L03", "...")),
    ]
    for number, (x, center_x, title, stage_name, agent_ids) in enumerate(stages, start=1):
        add_stage_card(ax, x, title, agent_ids)
        add_stage_exchange(ax, center_x, number, stage_name)

    # Sequential stage flow.
    add_arrow(ax, (5.27, 4.04), (5.91, 4.04))
    add_arrow(ax, (10.07, 4.04), (10.71, 4.04))

    # The Supervisor selects a chain and initiates the final response RFQ.
    add_box(ax, 1.20, 0.80, 4.00, 0.78, "Selected Supply Chain")
    add_box(
        ax,
        6.00,
        0.80,
        4.00,
        0.78,
        "Accepted / Rejected\nNotifications",
        fontsize=13.2,
    )
    add_box(ax, 10.80, 0.80, 4.00, 0.78, "Chain Log")
    add_routed_arrow(
        ax,
        ((4.45, 7.31), (0.72, 7.31), (0.72, 1.19), (1.20, 1.19)),
    )
    add_arrow(ax, (5.20, 1.19), (6.00, 1.19))
    add_arrow(ax, (10.00, 1.19), (10.80, 1.19))

    # The response RFQ is sent back to all participating agent groups.
    response_bus_y = 2.38
    add_arrow(ax, (8.00, 1.58), (8.00, response_bus_y))
    ax.plot(
        [3.20, 12.80],
        [response_bus_y, response_bus_y],
        color=DARK_BLUE,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        zorder=1,
    )
    for center_x in (3.20, 8.00, 12.80):
        add_arrow(ax, (center_x, response_bus_y), (center_x, 2.72))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        SVG_OUTPUT,
        format="svg",
        transparent=True,
        metadata={"Date": None},
    )
    fig.savefig(PNG_OUTPUT, format="png", dpi=300, transparent=True)
    plt.close(fig)
    print(f"wrote {SVG_OUTPUT}")
    print(f"wrote {PNG_OUTPUT}")


if __name__ == "__main__":
    render()
