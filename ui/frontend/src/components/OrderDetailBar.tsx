import { useState } from "react";
import type { OrderChain, RoundBid, Stage } from "../types";
import { STAGE_ORDER } from "../types";
import { chainTotal } from "../liveEvents";
import { reasonShort } from "../reasons";

interface Props {
    order: OrderChain | null;
}

const STAGE_LABELS: Record<Stage, string> = {
    material: "Material",
    tailoring: "Tailoring",
    logistics: "Logistics",
};

function statusText(bid: RoundBid): string {
    if (bid.verdict === "accepted") return "won";
    if (bid.verdict === "rejected") return "rejected";
    if (bid.model_decision === "NO_BID") return reasonShort(bid.decline_reason);
    return "—";
}

/** day counts come off the wire as floats and print as 7.8100000000000005 */
function days(value: number): string {
    return String(Math.round(value * 100) / 100);
}

/** the backend's summary is a clause, so give it a capital and a full stop */
function sentence(text: string): string {
    const trimmed = text.trim();
    const capped = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
    return /[.!?]$/.test(capped) ? capped : `${capped}.`;
}

export default function OrderDetailBar({ order }: Props) {
    // undefined follows the stage that is currently bidding; a click pins one
    // stage open or shut. the view is keyed by order id, so a new order comes
    // back to following on its own
    const [pick, setPick] = useState<Stage | null | undefined>(undefined);

    if (!order) {
        return (
            <div className="panel order-detail-bar">
                <p className="empty">Submit an order to inspect its detail.</p>
            </div>
        );
    }

    const presentStages = STAGE_ORDER.filter((s) => order.stages[s]);
    const total = chainTotal(order);
    const latest = presentStages[presentStages.length - 1] ?? null;
    const openStage = pick === undefined ? latest : pick;
    const openAuction = openStage ? order.stages[openStage] : null;
    const accepted = order.status === "completed";

    return (
        <div className="panel order-detail-bar">
            <div className="detail-row">
                <div className="detail-chip detail-chip-order">
                    <span className="detail-chip-title">{order.order_id}</span>
                    <span className="dim small">
                        {order.priority ?? "priority n/a"} · {order.config ?? "?"}
                    </span>
                </div>

                {presentStages.map((stage) => {
                    const auction = order.stages[stage]!;
                    const isOpen = openStage === stage;
                    return (
                        <div className={`detail-chip ${isOpen ? "detail-chip-open" : ""}`} key={stage}>
                            <div className="detail-chip-head">
                                <span className="detail-chip-title">{STAGE_LABELS[stage]}</span>
                                <button
                                    className="detail-info-btn"
                                    onClick={() => setPick(isOpen ? null : stage)}
                                    title={`Inspect ${STAGE_LABELS[stage]} data`}
                                    aria-label={`Inspect ${STAGE_LABELS[stage]} data`}
                                >
                                    ?
                                </button>
                            </div>
                            {auction.winner_agent_id ? (
                                <span className="dim small">
                                    {auction.winner_agent_id}
                                    {auction.closing_offer != null ? ` · €${auction.closing_offer.toFixed(2)}` : " · price unknown"}
                                </span>
                            ) : auction.leading_agent_id ? (
                                <span className="dim small tone-warn" title="cheapest bid so far, nothing reserved yet">
                                    {auction.leading_agent_id} leading
                                    {auction.closing_offer != null ? ` · €${auction.closing_offer.toFixed(2)}` : ""}
                                </span>
                            ) : (
                                <span className="dim small">{auction.closed ? "closed, no bids" : "bidding…"}</span>
                            )}
                        </div>
                    );
                })}

                <div className={`detail-chip detail-chip-total ${accepted ? "" : "detail-chip-total-none"}`}>
                    <span className="detail-chip-title">{accepted ? "Chain total" : "Chain"}</span>
                    <span className="dim small">
                        {total != null ? `€${total.toFixed(2)}` : order.status === "running" ? "still bidding" : "none accepted"}
                        {accepted && order.total_days != null ? ` · ${days(order.total_days)}d` : ""}
                    </span>
                </div>
            </div>

            {order.summary && <div className="notice notice-error">No chain was accepted. {sentence(order.summary)}</div>}

            {openAuction && openStage && (
                <div className="detail-expand">
                    <div className="detail-expand-head">
                        <span>
                            {STAGE_LABELS[openStage]}
                            {openAuction.platform_minimum_decrement != null && (
                                <span className="dim small"> · min decrement €{openAuction.platform_minimum_decrement.toFixed(2)}</span>
                            )}
                            {openAuction.closed_reason && <span className="dim small"> · closed: {openAuction.closed_reason}</span>}
                            {openAuction.total_reduction_eur != null && openAuction.total_reduction_eur > 0 && (
                                <span className="dim small"> · came down €{openAuction.total_reduction_eur.toFixed(2)}</span>
                            )}
                        </span>
                        <button className="detail-close-btn" onClick={() => setPick(null)} aria-label="Close">
                            ×
                        </button>
                    </div>
                    <div className="detail-expand-body">
                        <div className="detail-expand-pane">
                            <div className="detail-pane-title">Price by round</div>
                            <div className="panel-content scrollable">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Round</th>
                                            <th className="border-left">Announced</th>
                                            <th>Bids</th>
                                            <th>Leading</th>
                                            <th>Best</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {openAuction.rounds.map((round) => {
                                            const bidders = round.bids.filter((b) => b.model_decision === "BID");
                                            const best = round.best_bid_after_round
                                                ?? bidders.reduce<number | null>((low, b) => (b.offer_amount != null && (low == null || b.offer_amount < low) ? b.offer_amount : low), null);
                                            const leader = round.best_agent_id_after_round
                                                ?? bidders.find((b) => b.offer_amount === best)?.agent_id
                                                ?? null;
                                            return (
                                                <tr key={round.round_number}>
                                                    <td>R{round.round_number}</td>
                                                    <td className="border-left">
                                                        {round.announced_best_bid != null ? `€${round.announced_best_bid.toFixed(2)}` : "—"}
                                                    </td>
                                                    <td>{bidders.length}</td>
                                                    <td>{leader ?? "—"}</td>
                                                    <td className="cell-strong">{best != null ? `€${best.toFixed(2)}` : "—"}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div className="detail-expand-pane">
                            <div className="detail-pane-title">Every reply</div>
                            <div className="panel-content scrollable">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Round</th>
                                            <th className="border-left">Agent</th>
                                            <th>Offer</th>
                                            <th>Est. days</th>
                                            <th>Defect %</th>
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {openAuction.rounds.map((round) =>
                                            round.bids.map((bid, i) => (
                                                <tr key={`${round.round_number}-${bid.agent_id}`}>
                                                    {i === 0 ? <td rowSpan={round.bids.length}>R{round.round_number}</td> : null}
                                                    <td className="border-left">{bid.agent_id}</td>
                                                    <td>{bid.offer_amount != null ? `€${bid.offer_amount.toFixed(2)}` : "—"}</td>
                                                    <td>{bid.estimated_completion_days ?? "—"}</td>
                                                    <td>{bid.defect_probability != null ? `${(bid.defect_probability * 100).toFixed(1)}%` : "—"}</td>
                                                    <td>{statusText(bid)}</td>
                                                </tr>
                                            )),
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
