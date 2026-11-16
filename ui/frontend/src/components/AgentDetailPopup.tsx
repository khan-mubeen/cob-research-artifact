import { useEffect, useRef, useState } from "react";
import type { GraphNodeClick } from "./GraphCanvas";
import type { DynamicState, OrderChain, RestockNote, RoundBid, Stage } from "../types";
import { STAGE_ORDER } from "../types";
import { API_BASE } from "../hooks/useLiveSession";
import { chainTotal } from "../liveEvents";
import { reasonPhrase } from "../reasons";

interface Props {
    node: GraphNodeClick;
    orderChain: OrderChain | null;
    config: string;
    // bumped by every resource event, so an open popup re-reads the state
    // instead of waiting to be clicked again
    resourceSeq: number;
    restocks: RestockNote[];
    anchor: { x: number; y: number };
    onClose: () => void;
}

const STAGE_LABELS: Record<Stage, string> = {
    material: "Material",
    tailoring: "Tailoring",
    logistics: "Logistics",
};

function pct(v: number | null | undefined): string {
    return v != null ? `${(v * 100).toFixed(1)}%` : "—";
}

function eur(v: number | null | undefined): string {
    return v != null ? `€${v.toFixed(2)}` : "—";
}

// day counts come off the wire as floats and print as 7.8100000000000005
function days(v: number | null | undefined): string {
    return v != null ? String(Math.round(v * 100) / 100) : "—";
}

const STATUS_LABEL: Record<OrderChain["status"], string> = {
    running: "Running",
    completed: "Completed",
    infeasible: "Infeasible",
    timeout: "Timed out",
    error: "Error",
};

function bidsFor(orderChain: OrderChain | null, stage: Stage | undefined, agentId: string): { round: number; bid: RoundBid }[] {
    if (!orderChain || !stage) return [];
    const auction = orderChain.stages[stage];
    if (!auction) return [];
    const out: { round: number; bid: RoundBid }[] = [];
    for (const round of auction.rounds) {
        const bid = round.bids.find((b) => b.agent_id === agentId);
        if (bid) out.push({ round: round.round_number, bid });
    }
    return out.reverse();
}

/** the agent's live stock or workload, not the csv figures the roster holds
 *
 * /api/state/{config} is the only place the running numbers exist -- the
 * roster is the starting snapshot and never moves */
function useDynamicState(config: string, enabled: boolean, version: number): DynamicState | null {
    const [state, setState] = useState<DynamicState | null>(null);
    useEffect(() => {
        if (!enabled) return;
        let live = true;
        fetch(`${API_BASE}/api/state/${config}`)
            .then((r) => (r.ok ? r.json() : null))
            .then((data: DynamicState | null) => live && setState(data))
            .catch(() => live && setState(null));
        return () => {
            live = false;
        };
        // version moves on every resource event, which is what makes this live
    }, [config, enabled, version]);
    return state;
}

export default function AgentDetailPopup({ node, orderChain, config, resourceSeq, restocks, anchor, onClose }: Props) {
    const ref = useRef<HTMLDivElement>(null);
    const [pos, setPos] = useState(anchor);

    const agent = node.agent;
    const stage = node.stage;
    const dynamic = useDynamicState(config, agent != null && (stage === "material" || stage === "tailoring"), resourceSeq);

    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const margin = 12;
        const x = Math.min(anchor.x, window.innerWidth - rect.width - margin);
        const y = Math.min(anchor.y, window.innerHeight - rect.height - margin);
        setPos({ x: Math.max(margin, x), y: Math.max(margin, y) });
    }, [anchor, dynamic]);

    useEffect(() => {
        function onKey(e: KeyboardEvent) {
            if (e.key === "Escape") onClose();
        }
        document.addEventListener("keydown", onKey);
        return () => document.removeEventListener("keydown", onKey);
    }, [onClose]);

    const bids = agent && stage ? bidsFor(orderChain, stage, agent.agent_id) : [];
    const auction = stage ? orderChain?.stages[stage] : undefined;
    const isWinner = agent != null && auction?.winner_agent_id === agent.agent_id;
    const isLeading = agent != null && !isWinner && auction?.leading_agent_id === agent.agent_id;
    const settled = orderChain != null && orderChain.status !== "running";

    const stock = agent && stage === "material"
        ? Object.entries(dynamic?.material?.snapshot ?? {}).filter(([key]) => key.startsWith(`${agent.agent_id}:`))
        : [];
    const load = agent && stage === "tailoring" ? dynamic?.tailoring?.snapshot?.[agent.agent_id] : undefined;

    return (
        <div className="agent-popup" ref={ref} style={{ left: pos.x, top: pos.y }}>
            <div className="agent-popup-head">
                <span className="agent-popup-title">
                    {node.kind === "agent" ? node.id : node.kind === "source" ? "Order" : "Supervisor"}
                    {stage && <span className="dim small"> · {STAGE_LABELS[stage]}</span>}
                </span>
                <button className="detail-close-btn" onClick={onClose} aria-label="Close">
                    ×
                </button>
            </div>

            {node.kind === "supervisor" && (
                <p className="dim small">Routes RFQs out to every eligible agent and collects their replies.</p>
            )}

            {node.kind === "source" && (
                orderChain ? (
                    <>
                        <dl className="agent-popup-facts">
                            <div>
                                <dt>Config</dt>
                                <dd>{orderChain.config ?? config}</dd>
                            </div>
                            <div>
                                <dt>Priority</dt>
                                <dd>{orderChain.priority ?? "—"}</dd>
                            </div>
                            <div>
                                <dt>Product</dt>
                                <dd>{orderChain.product_id ?? "—"}</dd>
                            </div>
                            <div>
                                <dt>Quantity</dt>
                                <dd>{orderChain.quantity ?? "—"}</dd>
                            </div>
                            <div>
                                <dt>Material</dt>
                                <dd>{orderChain.material_type ?? "—"}</dd>
                            </div>
                            <div>
                                <dt>Budget</dt>
                                <dd>{eur(orderChain.budget)}</dd>
                            </div>
                            <div>
                                <dt>Deadline</dt>
                                <dd>{orderChain.deadline ?? "—"}</dd>
                            </div>
                            <div>
                                <dt>Deliver to</dt>
                                <dd>{orderChain.deliver_to ?? "—"}</dd>
                            </div>
                            <div>
                                <dt>Max defect</dt>
                                <dd>{pct(orderChain.max_defect_rate)}</dd>
                            </div>
                            <div>
                                <dt>Min reputation</dt>
                                <dd>{orderChain.min_reputation_score != null ? orderChain.min_reputation_score.toFixed(2) : "—"}</dd>
                            </div>
                            {orderChain.scenario_type && (
                                <div>
                                    <dt>Scenario</dt>
                                    <dd>{orderChain.scenario_type}</dd>
                                </div>
                            )}
                            {!orderChain.from_start && (
                                <div>
                                    <dt>Joined</dt>
                                    <dd>mid-order</dd>
                                </div>
                            )}
                        </dl>

                        <div className="agent-popup-section-title">Progress</div>
                        <ul className="agent-popup-state">
                            <li>
                                <span>Status</span>
                                <span className={orderChain.status === "running" ? "" : orderChain.status === "completed" ? "tone-bid" : "tone-warn"}>
                                    {STATUS_LABEL[orderChain.status]}
                                </span>
                            </li>
                            <li>
                                <span>Stages reached</span>
                                <span>{STAGE_ORDER.filter((s) => orderChain.stages[s]).length}/3</span>
                            </li>
                            {chainTotal(orderChain) != null && (
                                <li>
                                    <span>Chain total</span>
                                    <span>
                                        {eur(chainTotal(orderChain))}
                                        {orderChain.status === "completed" && orderChain.total_days != null ? ` · ${days(orderChain.total_days)}d` : ""}
                                    </span>
                                </li>
                            )}
                        </ul>
                        {orderChain.summary && <p className="notice notice-error">{orderChain.summary}</p>}

                        {STAGE_ORDER.some((s) => orderChain.stages[s]) && (
                            <>
                                <div className="agent-popup-section-title">Chain so far</div>
                                <ul className="agent-popup-bids">
                                    {STAGE_ORDER.filter((s) => orderChain.stages[s]).map((s) => {
                                        const stageAuction = orderChain.stages[s]!;
                                        const who = stageAuction.winner_agent_id ?? stageAuction.leading_agent_id;
                                        return (
                                            <li key={s} className={stageAuction.winner_agent_id ? "tone-bid" : "tone-no-bid"}>
                                                {STAGE_LABELS[s]}:{" "}
                                                {who
                                                    ? `${who} · ${eur(stageAuction.closing_offer)}${stageAuction.winner_agent_id ? "" : " (leading, not reserved)"}`
                                                    : stageAuction.closed
                                                      ? "no bids"
                                                      : "bidding…"}
                                            </li>
                                        );
                                    })}
                                </ul>
                            </>
                        )}
                    </>
                ) : (
                    <p className="dim small">no order tracked yet</p>
                )
            )}

            {agent && (
                <>
                    <dl className="agent-popup-facts">
                        <div>
                            <dt>Location</dt>
                            <dd>{agent.location ?? "—"}</dd>
                        </div>
                        <div>
                            <dt>Reputation</dt>
                            <dd>{agent.reputation_score != null ? agent.reputation_score.toFixed(2) : "—"}</dd>
                        </div>
                        {agent.defect_probability != null && (
                            <div>
                                <dt>Defect rate</dt>
                                <dd>{pct(agent.defect_probability)}</dd>
                            </div>
                        )}
                        {agent.damage_probability != null && (
                            <div>
                                <dt>Damage rate</dt>
                                <dd>{pct(agent.damage_probability)}</dd>
                            </div>
                        )}
                        {agent.material_types && stock.length === 0 && (
                            <div>
                                <dt>Materials</dt>
                                <dd>{agent.material_types.replaceAll("|", ", ")}</dd>
                            </div>
                        )}
                        {agent.daily_capacity_units != null && !load && (
                            <div>
                                <dt>Capacity</dt>
                                <dd>{agent.daily_capacity_units} units/day</dd>
                            </div>
                        )}
                        {agent.product_types && (
                            <div>
                                <dt>Products</dt>
                                <dd>{agent.product_types.replaceAll("|", ", ")}</dd>
                            </div>
                        )}
                        {agent.service_regions && (
                            <div>
                                <dt>Regions</dt>
                                <dd>{agent.service_regions.replaceAll("|", ", ")}</dd>
                            </div>
                        )}
                    </dl>
                    {agent.specialty && <p className="agent-popup-specialty dim small">{agent.specialty}</p>}

                    {restocks.length > 0 && (
                        <div className="restock-log">
                            {restocks.map((r) => (
                                <div className={`restock-note restock-${r.status}`} key={`${r.status}-${r.at}`}>
                                    {r.status === "triggered" ? (
                                        <>
                                            Restock ordered: {r.amount_kg?.toFixed(0) ?? "?"} kg of {r.material}
                                            {r.days != null ? `, ${r.days} day lead time` : ""}
                                        </>
                                    ) : (
                                        <>
                                            Restock arrived: {r.amount_kg?.toFixed(0) ?? "?"} kg of {r.material}
                                            {r.available_after != null ? `, now holding ${r.available_after.toFixed(0)} kg` : ""}
                                        </>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    {stock.length > 0 && (
                        <>
                            <div className="agent-popup-section-title">Stock now</div>
                            <ul className="agent-popup-state">
                                {stock.map(([key, row]) => (
                                    <li key={key}>
                                        <span>{key.split(":")[1]}</span>
                                        <span className={row.available_kg <= row.restock_threshold_kg ? "tone-warn" : ""}>
                                            {row.available_kg.toFixed(1)} / {row.starting_kg.toFixed(0)} kg
                                            {row.restock_pending ? " · restocking" : ""}
                                            {row.restocks_landed > 0 ? ` · +${row.restocks_landed} landed` : ""}
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        </>
                    )}

                    {load && (
                        <>
                            <div className="agent-popup-section-title">Workload now</div>
                            <ul className="agent-popup-state">
                                <li>
                                    <span>units</span>
                                    <span className={load.current_workload_units > load.daily_capacity_units * 0.8 ? "tone-warn" : ""}>
                                        {load.current_workload_units.toFixed(0)} / {load.daily_capacity_units.toFixed(0)}
                                        {load.open_reservations > 0 ? ` · ${load.open_reservations} held` : ""}
                                    </span>
                                </li>
                            </ul>
                        </>
                    )}

                    <div className="agent-popup-section-title">This order</div>
                    {bids.length === 0 ? (
                        <p className="empty">No activity yet this stage.</p>
                    ) : (
                        <ul className="agent-popup-bids">
                            {bids.map(({ round, bid }) => (
                                <li key={round} className={bid.model_decision === "NO_BID" ? "tone-no-bid" : "tone-bid"}>
                                    R{round}: {bid.model_decision === "BID" ? (
                                        <>
                                            €{bid.offer_amount?.toFixed(2)} · {bid.estimated_completion_days}d
                                            {bid.defect_probability != null ? ` · defect ${pct(bid.defect_probability)}` : ""}
                                        </>
                                    ) : (
                                        <>stayed out, {reasonPhrase(bid.decline_reason)}</>
                                    )}
                                </li>
                            ))}
                        </ul>
                    )}
                    {isWinner && (
                        <span className="badge matched" title="this agent's stock or capacity is reserved for this order">
                            Stage winner, reserved
                        </span>
                    )}
                    {isLeading && (
                        <span
                            className="badge no-winner"
                            title="cheapest bid at this stage. nothing is reserved until the whole chain is accepted"
                        >
                            {settled ? "Led the bidding, chain not accepted" : "Leading bid, not reserved yet"}
                        </span>
                    )}
                </>
            )}
        </div>
    );
}
