import { useEffect, useRef } from "react";
import type { LiveEnvelope } from "../types";
import { reasonPhrase } from "../reasons";

interface Props {
    events: LiveEnvelope[];
    // the configuration the page is watching, so lines from any other one are
    // visibly somebody else's
    watching: string;
    // shut panels still receive and buffer, they just do not draw
    collapsed?: boolean;
}

interface LogLine {
    tone: "info" | "bid" | "no-bid" | "verdict" | "error" | "milestone" | "warn";
    text: string;
}

function fmt(v: unknown): string {
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
    if (v == null) return "?";
    return String(v);
}

function str(v: unknown): string | null {
    return typeof v === "string" ? v : null;
}

function eur(v: unknown): string {
    return typeof v === "number" ? `€${v.toFixed(2)}` : "€?";
}

function describe(envelope: LiveEnvelope): LogLine | null {
    const data = envelope.data ?? {};
    switch (envelope.type) {
        case "order_started":
            return { tone: "milestone", text: `${envelope.order_id} started: ${fmt(data.quantity)} units of ${fmt(data.material_type)}, budget ${eur(data.budget)}` };
        case "stage_opened":
            return { tone: "info", text: `${fmt(data.stage)} stage opened: ${fmt(data.roster_size)} agents asked, ${fmt(data.window_seconds)}s to reply, up to ${fmt(data.max_rounds)} round(s)` };
        case "rfq":
            return {
                tone: "info",
                text: `RFQ out to ${fmt(envelope.stage)}${envelope.round_number ? `, round ${envelope.round_number}` : ""}${
                    envelope.current_best_bid != null ? `, price to beat ${eur(envelope.current_best_bid)}` : ""
                }`,
            };
        case "bid":
            return {
                tone: "bid",
                text: `${fmt(data.agent_id)} bid ${eur(data.offer_amount)} for ${fmt(envelope.stage)}, ${fmt(
                    data.estimated_completion_days,
                )} days, defect ${typeof data.defect_probability === "number" ? `${(data.defect_probability * 100).toFixed(1)}%` : "unknown"}`,
            };
        case "no_bid":
            return { tone: "no-bid", text: `${fmt(data.agent_id)} stayed out of ${fmt(data.stage)}, it ${reasonPhrase(str(data.reason))}` };
        case "auction_trace": {
            const why = data.closed_reason === "no_improvement" ? "no round improved on it" : "it hit the round cap";
            const head = `${fmt(data.stage)} auction closed after ${fmt(data.rounds_run)} round(s), ${why}`;
            if (data.closing_best_bid == null) return { tone: "info", text: `${head}, and nobody bid` };
            return {
                tone: "info",
                text: `${head}. Price went ${eur(data.opening_best_bid)} to ${eur(data.closing_best_bid)}${
                    typeof data.total_reduction_eur === "number" && data.total_reduction_eur > 0
                        ? `, saving €${data.total_reduction_eur.toFixed(2)}`
                        : ""
                }`,
            };
        }
        case "stage_closed": {
            return data.best_agent_id
                ? {
                      tone: "milestone",
                      text: `${fmt(data.stage)} stage closed with ${fmt(data.best_agent_id)} leading at ${eur(
                          data.best_offer_amount,
                      )}, nothing reserved yet`,
                  }
                : { tone: "error", text: `${fmt(data.stage)} stage closed with no bids at all` };
        }
        case "verdict":
            return {
                tone: "verdict",
                text: data.verdict === "accepted"
                    ? `${fmt(envelope.agent_id ?? data.agent_id)} accepted at ${eur(data.agreed_price)}, its resources are now reserved`
                    : `${fmt(envelope.agent_id ?? data.agent_id)} rejected${data.reason ? `, ${fmt(data.reason)}` : ""}`,
            };
        case "state_change":
            return describeState(data);
        case "order_finished": {
            const outcome = (data.outcome as Record<string, unknown> | undefined) ?? {};
            if (data.status !== "completed") {
                // the plain-language explanation sits on data, beside outcome
                return {
                    tone: "error",
                    text: `${envelope.order_id} found no workable chain${data.summary ? `, ${fmt(data.summary)}` : ` (${fmt(data.status)})`}`,
                };
            }
            const chain = (outcome.chain as Record<string, string> | undefined) ?? {};
            const path = ["material", "tailoring", "logistics"].map((s) => chain[s]).filter(Boolean).join(" → ");
            return {
                tone: "milestone",
                text: `${envelope.order_id} completed via ${path} for ${eur(outcome.total_cost)} in ${fmt(outcome.total_days)} days`,
            };
        }
        case "mode_switching":
            // the broadcast names both ends of the switch, from and to
            return { tone: "info", text: `switching bidding mode from ${fmt(data.from)} to ${fmt(data.to)}, restarting every stack` };
        case "mode_switched":
            return { tone: "milestone", text: `bidding mode is now ${fmt(data.mode)}, the restart took ${fmt(data.seconds)}s` };
        case "mode_result":
            return data.switched === false
                ? { tone: "info", text: `mode switch skipped, ${fmt(data.reason)}` }
                : { tone: "milestone", text: `mode now ${fmt(data.mode)}` };
        case "reset_done":
        case "state_reset": {
            const configs = Object.keys(data);
            return { tone: "milestone", text: `state reset: ${configs.length > 0 ? configs.join(" and ") : "every config"} back to the csv baseline` };
        }
        case "accepted":
            return { tone: "info", text: `submission accepted as job ${fmt(data.job_id)} on ${fmt(data.config)}, ${fmt(data.queued)} in the queue` };
        case "order_queued":
            return { tone: "info", text: `${(envelope.order_ids ?? []).join(", ")} queued at position ${fmt(envelope.position)}` };
        case "failure":
            // a synthetic relay envelope: its text is top level, not in data
            return { tone: "error", text: `something failed on the backend: ${fmt(envelope.message)}` };
        case "error":
            return { tone: "error", text: `the server refused that request: ${fmt(envelope.message)}` };
        default:
            return null;
    }
}

/** state_change carries a different set of numbers per event kind */
function describeState(data: Record<string, unknown>): LogLine {
    const who = fmt(data.agent_id);
    switch (data.event) {
        case "stock_reserved":
            return { tone: "info", text: `${who} set aside ${fmt(data.required_kg)}kg of ${fmt(data.material)}, ${fmt(data.available_kg_after)}kg left` };
        case "restock_triggered":
            return {
                tone: "warn",
                text: `${who} dropped below its ${fmt(data.material)} threshold, so ${fmt(data.amount_kg)}kg was ordered and arrives in ${fmt(
                    data.restock_time_days,
                )} days`,
            };
        case "restock_landed":
            return {
                tone: "warn",
                text: `${who} took delivery of ${fmt(data.amount_kg)}kg of ${fmt(data.material)}, now holding ${fmt(data.available_kg_after)}kg`,
            };
        case "restock_skipped_pending":
            return { tone: "warn", text: `${who} is already waiting on a ${fmt(data.material)} restock, so no second one was ordered` };
        case "workload_reserved":
            return {
                tone: "info",
                text: `${who} booked ${fmt(data.reserved_units)} units, now at ${fmt(data.current_workload_units_after)} of ${fmt(
                    data.daily_capacity_units,
                )}`,
            };
        case "workload_released":
            return {
                tone: "info",
                text: `${who} freed up capacity, now at ${fmt(data.current_workload_units_after)} of ${fmt(data.daily_capacity_units)}`,
            };
        default:
            return { tone: "info", text: `${who} reported ${plainEvent(data.event)}` };
    }
}

function plainEvent(event: unknown): string {
    return typeof event === "string" ? event.replaceAll("_", " ") : "a state change";
}

export default function LiveEventLog({ events, watching, collapsed = false }: Props) {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const el = scrollRef.current;
        if (el && !collapsed) el.scrollTop = el.scrollHeight;
    }, [events.length, collapsed]);

    const lines = events
        .map((e, i) => ({ key: `${e.seq ?? `d${i}`}-${e.type}`, config: e.config ?? null, line: describe(e) }))
        .filter((x) => x.line);

    return (
        <div className="panel fill live-event-log">
            <h3>
                Agent activity <span className="dim small">· watching {watching}</span>
            </h3>
            <div className="panel-content scrollable" ref={scrollRef}>
                {lines.length === 0 ? (
                    <p className="empty">Waiting for activity…</p>
                ) : (
                    <ul className="event-log-list">
                        {lines.map(({ key, config, line }) => (
                            <li key={key} className={`event-log-line tone-${line!.tone} ${config && config !== watching ? "other-config" : ""}`}>
                                {config && <span className="event-config-badge">{config}</span>}
                                {line!.text}
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}
