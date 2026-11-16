import type { OrderChain, RoundBid, Stage } from "../types";
import { STAGE_ORDER } from "../types";
import { chainTotal } from "../liveEvents";
import { reasonShort } from "../reasons";

interface Props {
    order: OrderChain | null;
    config: string;
}

const STAGE_LABELS: Record<Stage, string> = {
    material: "Material",
    tailoring: "Tailoring",
    logistics: "Logistics",
};

/** what is actually known about this bid once the order is over: the
 * supervisor's verdict, or the agent's own reason for staying out */
function statusOf(bid: RoundBid): { text: string; tone: string } {
    if (bid.verdict === "accepted") return { text: "won", tone: "matched" };
    if (bid.verdict === "rejected") return { text: "rejected", tone: "no-bid" };
    if (bid.model_decision === "NO_BID") return { text: reasonShort(bid.decline_reason), tone: "no-bid" };
    return { text: "standing", tone: "bid" };
}

/** the backend's summary is a clause, so give it a capital and a full stop */
function sentence(text: string): string {
    const trimmed = text.trim();
    const capped = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
    return /[.!?]$/.test(capped) ? capped : `${capped}.`;
}

export default function RoundTracker({ order, config }: Props) {
    if (!order) {
        return (
            <div className="panel fill round-tracker">
                <h3>Chain</h3>
                <p className="empty">
                    Submit an order on <span className="mono">{config}</span> to see its material → tailoring → logistics chain.
                </p>
            </div>
        );
    }

    const presentStages = STAGE_ORDER.filter((s) => order.stages[s]);
    const total = chainTotal(order);
    const totalRounds = presentStages.reduce((sum, s) => sum + (order.stages[s]?.rounds.length ?? 0), 0);
    const finished = order.status !== "running";

    return (
        <div className="panel fill round-tracker">
            <h3>
                {order.order_id}
                <span className="dim small"> · {order.config ?? config}</span>
            </h3>
            {!order.from_start && (
                <div className="notice notice-warn">
                    Joined this order after it had started. The server replays only its last 50 events, so the round
                    tables below can have gaps. Winners and totals are the backend&rsquo;s own figures.
                </div>
            )}
            <div className="stat-grid">
                <div className="stat-tile">
                    <div className="stat-value">{presentStages.length}/3</div>
                    <div className="stat-label">Stages</div>
                </div>
                <div className="stat-tile" title="Sum of rounds played across all stages -- each stage runs at most 3 rounds on its own">
                    <div className="stat-value">{totalRounds}</div>
                    <div className="stat-label">Rounds (Σ, ≤3 ea.)</div>
                </div>
                <div className="stat-tile">
                    <div className="stat-value">{total != null ? `€${total.toFixed(0)}` : "—"}</div>
                    <div className="stat-label">Chain total</div>
                </div>
            </div>
            <div className="panel-content scrollable">
                {presentStages.map((stage) => {
                    const auction = order.stages[stage]!;
                    return (
                        <div className="stage-block" key={stage}>
                            <div className="stage-block-title">{STAGE_LABELS[stage]}</div>
                            <table className="compact">
                                <thead>
                                    <tr>
                                        <th>Rnd</th>
                                        <th className="border-left">Agent</th>
                                        <th>Decision</th>
                                        <th>Offer</th>
                                        <th>Days</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {auction.rounds.map((round) =>
                                        round.bids.length === 0 ? (
                                            <tr key={`${round.round_number}-empty`}>
                                                <td>R{round.round_number}</td>
                                                <td className="border-left dim" colSpan={5}>
                                                    no replies
                                                </td>
                                            </tr>
                                        ) : (
                                            round.bids.map((bid, i) => {
                                                const state = statusOf(bid);
                                                return (
                                                    <tr key={`${round.round_number}-${bid.agent_id}`}>
                                                        {/* the announced best bid used to be repeated here; it is
                                                            the price-by-round table in the detail strip now */}
                                                        {i === 0 ? <td rowSpan={round.bids.length}>R{round.round_number}</td> : null}
                                                        <td className="border-left">{bid.agent_id}</td>
                                                        <td>
                                                            <span className={`badge ${bid.model_decision === "BID" ? "bid" : "no-bid"}`}>
                                                                {bid.model_decision ?? "—"}
                                                            </span>
                                                        </td>
                                                        <td>{bid.offer_amount != null ? `€${bid.offer_amount.toFixed(2)}` : "—"}</td>
                                                        <td>{bid.estimated_completion_days ?? "—"}</td>
                                                        <td className="cell-clip" title={state.text}>
                                                            <span className={`badge ${state.tone}`}>{state.text}</span>
                                                        </td>
                                                    </tr>
                                                );
                                            })
                                        ),
                                    )}
                                </tbody>
                            </table>
                            <div className="stage-block-footer">
                                {auction.winner_agent_id ? (
                                    <span className="badge matched" title="reserved: the whole chain was accepted">
                                        Winner {auction.winner_agent_id}
                                        {auction.closing_offer != null ? ` · €${auction.closing_offer.toFixed(2)}` : ""}
                                    </span>
                                ) : auction.leading_agent_id ? (
                                    <span
                                        className="badge no-winner"
                                        title="cheapest bid at this stage. nothing is reserved until the whole chain is accepted"
                                    >
                                        {finished ? "Led" : "Leading"} {auction.leading_agent_id}
                                        {auction.closing_offer != null ? ` · €${auction.closing_offer.toFixed(2)}` : ""}
                                        {finished ? ", not accepted" : ", not reserved"}
                                    </span>
                                ) : auction.closed ? (
                                    <span className="badge no-winner">Closed with no bids</span>
                                ) : (
                                    <span className="badge pending">Bidding…</span>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
            {finished && order.status !== "completed" && (
                <div className="round-tracker-footer">
                    <div className="notice notice-error">
                        No chain was accepted{order.summary ? `. ${sentence(order.summary)}` : ` (${order.status}).`}
                    </div>
                </div>
            )}
        </div>
    );
}
