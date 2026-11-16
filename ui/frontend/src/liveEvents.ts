import type { AuctionEvent, LiveEnvelope, OrderChain, RoundBid, Stage, StageAuction, StageRound } from "./types";
import { STAGE_ORDER } from "./types";

function str(v: unknown): string | null {
    return typeof v === "string" ? v : null;
}

function num(v: unknown): number | null {
    return typeof v === "number" ? v : null;
}

function bool(v: unknown): boolean | null {
    return typeof v === "boolean" ? v : null;
}

function rec(v: unknown): Record<string, unknown> {
    return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
}

// the trace says an agent went quiet, not why; these stand in until, or
// unless, that agent's own no_bid reason was seen on the wire
const SILENT_REASON: Record<string, string> = {
    never_entered: "never entered",
    withdrawn: "withdrawn",
};

function emptyStageAuction(): StageAuction {
    return { platform_minimum_decrement: null, rounds: [], leading_agent_id: null,
             winner_agent_id: null, closing_offer: null, closed: false };
}

export function emptyChain(orderId: string, config: string | null = null, fromStart = false): OrderChain {
    return {
        order_id: orderId,
        config,
        priority: null,
        product_id: null,
        quantity: null,
        material_type: null,
        budget: null,
        deadline: null,
        deliver_to: null,
        max_defect_rate: null,
        min_reputation_score: null,
        scenario_type: null,
        stages: {},
        from_start: fromStart,
        status: "running",
        total_cost: null,
        total_days: null,
        summary: null,
        agreed: {},
    };
}

function bidFromData(agentId: string, data: Record<string, unknown>): RoundBid {
    return {
        agent_id: agentId,
        model_decision: "BID",
        offer_amount: num(data.offer_amount),
        estimated_completion_days: num(data.estimated_completion_days),
        defect_probability: num(data.defect_probability),
        decline_reason: null,
        meets_decrement: bool(data.meets_decrement),
    };
}

function noBidFromData(agentId: string, data: Record<string, unknown>): RoundBid {
    return {
        agent_id: agentId,
        model_decision: "NO_BID",
        offer_amount: null,
        estimated_completion_days: null,
        defect_probability: null,
        decline_reason: str(data.reason),
    };
}

/** the stage an envelope is about: bid and rfq carry it hoisted, the
 * monitoring feeds carry it in the body */
function stageOf(envelope: LiveEnvelope): Stage | undefined {
    return envelope.stage ?? ((envelope.data ?? {}).stage as Stage | undefined);
}

export function envelopeToAuctionEvents(envelope: LiveEnvelope): AuctionEvent[] {
    const data = envelope.data ?? {};
    const seq = envelope.seq ?? Math.random();

    switch (envelope.type) {
        case "stage_opened": {
            const stage = stageOf(envelope);
            if (!stage) return [];
            return [{ id: `${seq}-stage-start`, type: "STAGE_START", stage, round_number: 0 }];
        }
        case "rfq": {
            const stage = stageOf(envelope);
            if (!stage) return [];
            const roundNumber = envelope.round_number ?? num(data.round_number) ?? 1;
            return [{
                id: `${seq}-rfq`,
                type: "RFQ_OPEN",
                stage,
                round_number: roundNumber,
                // the relay hoists the round context; the raw rfq body repeats it
                announced_best_bid: envelope.current_best_bid ?? num(data.current_best_bid),
            }];
        }
        case "bid": {
            const stage = stageOf(envelope);
            const agentId = str(data.agent_id);
            if (!stage || !agentId) return [];
            return [{
                id: `${seq}-bid-${agentId}`,
                type: "BID",
                stage,
                round_number: envelope.round_number ?? 1,
                agent_id: agentId,
                bid: bidFromData(agentId, data),
            }];
        }
        case "no_bid": {
            const stage = stageOf(envelope);
            const agentId = str(data.agent_id);
            if (!stage || !agentId) return [];
            return [{
                id: `${seq}-nobid-${agentId}`,
                type: "NO_BID",
                stage,
                round_number: num(data.round_number) ?? 1,
                agent_id: agentId,
                bid: noBidFromData(agentId, data),
            }];
        }
        case "stage_closed": {
            // the leader, not the winner: nothing is reserved until the whole
            // chain is accepted
            const stage = stageOf(envelope);
            if (!stage) return [];
            return [{
                id: `${seq}-stage-close`,
                type: "STAGE_CLOSE",
                stage,
                round_number: 0,
                leading_agent_id: str(data.best_agent_id),
                closing_offer: num(data.best_offer_amount),
                closed: true,
            }];
        }
        case "order_finished": {
            // the confirmed final winners
            const outcome = rec(data.outcome);
            const chain = (outcome.chain as Record<string, string> | undefined) ?? {};
            const reached = STAGE_ORDER.filter((s) => chain[s]);
            const lastStage = reached[reached.length - 1] ?? "logistics";
            const winnerEvents: AuctionEvent[] = reached.map((stage) => ({
                id: `${seq}-winner-${stage}`,
                type: "WINNER",
                stage,
                round_number: 0,
                winner_agent_id: chain[stage],
            }));
            return [...winnerEvents, { id: `${seq}-chain-complete`, type: "CHAIN_COMPLETE", stage: lastStage, round_number: 0 }];
        }
        default:
            return [];
    }
}

/** rebuild one stage from its auction trace
 *
 * the trace is the supervisor's own complete per-round record and it arrives
 * as one event, so a page that joined mid-order and missed the individual
 * bids still gets that stage back exactly rather than in pieces
 *
 * it records who bid, and who went quiet, but never why: the reason is the
 * agent's own and travels on the monitoring feed. so the declines this page
 * already saw are kept, and only the ones it never saw are filled in from the
 * trace's silent list */
function stageFromTrace(existing: StageAuction, trace: Record<string, unknown>): StageAuction {
    const rawRounds = Array.isArray(trace.rounds) ? (trace.rounds as Record<string, unknown>[]) : [];
    const rounds: StageRound[] = rawRounds.map((round) => {
        const number = num(round.round_number) ?? 0;
        const rawBids = Array.isArray(round.bids) ? (round.bids as Record<string, unknown>[]) : [];
        const rawSilent = Array.isArray(round.silent) ? (round.silent as Record<string, unknown>[]) : [];
        const bids = rawBids.map((b) => bidFromData(str(b.agent_id) ?? "?", b));
        const seen = new Set(bids.map((b) => b.agent_id));

        const known = existing.rounds.find((r) => r.round_number === number);
        for (const previous of known?.bids ?? []) {
            if (!seen.has(previous.agent_id)) {
                bids.push(previous);
                seen.add(previous.agent_id);
            }
        }
        for (const quiet of rawSilent) {
            const agentId = str(quiet.agent_id);
            if (!agentId || seen.has(agentId)) continue;
            bids.push(noBidFromData(agentId, { reason: SILENT_REASON[str(quiet.status) ?? ""] ?? "no reply" }));
            seen.add(agentId);
        }

        return {
            round_number: number,
            announced_best_bid: num(round.announced_best_bid),
            best_bid_after_round: num(round.best_bid_after_round),
            best_agent_id_after_round: str(round.best_agent_id_after_round),
            silent_count: rawSilent.length,
            bids,
        };
    });
    const entrants = Array.isArray(trace.entrant_agent_ids) ? trace.entrant_agent_ids.length : null;
    return {
        ...existing,
        platform_minimum_decrement: num(trace.platform_minimum_decrement) ?? existing.platform_minimum_decrement,
        rounds: rounds.length > 0 ? rounds : existing.rounds,
        leading_agent_id: str(trace.winner_agent_id) ?? existing.leading_agent_id,
        closing_offer: num(trace.winning_offer_amount) ?? existing.closing_offer,
        closed: true,
        rounds_run: num(trace.rounds_run),
        closed_reason: str(trace.closed_reason),
        opening_best_bid: num(trace.opening_best_bid),
        total_reduction_eur: num(trace.total_reduction_eur),
        roster_size: num(trace.roster_size),
        entrant_count: entrants,
    };
}

/** the accepted verdict is the winning price straight from the backend
 *
 * it matters on a reconnect: the verdicts all land in the last handful of
 * events, so a stage whose own bids fell outside the replay still has a
 * price here rather than a blank */
function applyVerdict(chain: OrderChain, envelope: LiveEnvelope): OrderChain {
    const data = envelope.data ?? {};
    const agentId = envelope.agent_id ?? str(data.agent_id);
    const verdict = str(data.verdict);
    if (!agentId || (verdict !== "accepted" && verdict !== "rejected")) return chain;
    const price = num(data.agreed_price);

    let next: OrderChain = verdict === "accepted" && price != null
        ? { ...chain, agreed: { ...chain.agreed, [agentId]: price } }
        : chain;
    for (const stage of STAGE_ORDER) {
        const auction = next.stages[stage];
        if (!auction) continue;
        if (!auction.rounds.some((r) => r.bids.some((b) => b.agent_id === agentId && b.model_decision === "BID"))) continue;
        // the verdict is per agent, per order. stamping it on a round this
        // agent sat out would read as the reason it sat out, which it is not
        const rounds = auction.rounds.map((r) => ({
            ...r,
            bids: r.bids.map((b) => (b.agent_id === agentId && b.model_decision === "BID" ? { ...b, verdict } : b)),
        }));
        const won = verdict === "accepted";
        next = {
            ...next,
            stages: {
                ...next.stages,
                [stage]: {
                    ...auction,
                    rounds,
                    winner_agent_id: won ? agentId : auction.winner_agent_id,
                    closing_offer: won && price != null ? price : auction.closing_offer,
                },
            },
        };
    }
    return next;
}

export function applyEnvelopeToChain(chain: OrderChain, envelope: LiveEnvelope): OrderChain {
    const data = envelope.data ?? {};

    if (envelope.type === "order_started") {
        return {
            ...chain,
            priority: str(data.priority) ?? chain.priority,
            product_id: str(data.product_id) ?? chain.product_id,
            quantity: num(data.quantity) ?? chain.quantity,
            material_type: str(data.material_type) ?? chain.material_type,
            budget: num(data.budget) ?? chain.budget,
            deadline: str(data.deadline) ?? chain.deadline,
            deliver_to: str(data.deliver_to) ?? chain.deliver_to,
            max_defect_rate: num(data.max_defect_rate) ?? chain.max_defect_rate,
            min_reputation_score: num(data.min_reputation_score) ?? chain.min_reputation_score,
            scenario_type: str(data.scenario_type) ?? chain.scenario_type,
            from_start: true,
        };
    }

    if (envelope.type === "rfq") {
        const stage = stageOf(envelope);
        if (!stage) return chain;
        const roundNumber = envelope.round_number ?? num(data.round_number) ?? 1;
        const existing = chain.stages[stage] ?? emptyStageAuction();
        if (existing.rounds.some((r) => r.round_number === roundNumber)) return chain;
        const best = envelope.current_best_bid ?? num(data.current_best_bid);
        const rounds = [...existing.rounds, { round_number: roundNumber, announced_best_bid: best, bids: [] }];
        const decrement = envelope.platform_minimum_decrement ?? num(data.platform_minimum_decrement);
        return { ...chain, stages: { ...chain.stages, [stage]: { ...existing, rounds, platform_minimum_decrement: decrement ?? existing.platform_minimum_decrement } } };
    }

    if (envelope.type === "bid" || envelope.type === "no_bid") {
        const stage = stageOf(envelope);
        const agentId = str(data.agent_id);
        if (!stage || !agentId) return chain;
        const roundNumber = (envelope.type === "bid" ? envelope.round_number : num(data.round_number)) ?? 1;
        const bid = envelope.type === "bid" ? bidFromData(agentId, data) : noBidFromData(agentId, data);

        const existing = chain.stages[stage] ?? emptyStageAuction();
        const idx = existing.rounds.findIndex((r) => r.round_number === roundNumber);
        const rounds =
            idx === -1
                ? [...existing.rounds, { round_number: roundNumber, announced_best_bid: null, bids: [bid] }]
                : existing.rounds.map((r, i) =>
                      i === idx ? { ...r, bids: [...r.bids.filter((b) => b.agent_id !== agentId), bid] } : r,
                  );
        return { ...chain, stages: { ...chain.stages, [stage]: { ...existing, rounds } } };
    }

    if (envelope.type === "auction_trace") {
        const stage = stageOf(envelope);
        if (!stage) return chain;
        const existing = chain.stages[stage] ?? emptyStageAuction();
        return { ...chain, stages: { ...chain.stages, [stage]: stageFromTrace(existing, data) } };
    }

    if (envelope.type === "stage_closed") {
        const stage = stageOf(envelope);
        if (!stage) return chain;
        const existing = chain.stages[stage] ?? emptyStageAuction();
        // this is the cheapest standing bid, in either mode. nothing is
        // reserved and no verdict is out yet, so it is a leader, not a winner
        return {
            ...chain,
            stages: {
                ...chain.stages,
                [stage]: {
                    ...existing,
                    closed: true,
                    leading_agent_id: str(data.best_agent_id) ?? existing.leading_agent_id,
                    closing_offer: num(data.best_offer_amount) ?? existing.closing_offer,
                },
            },
        };
    }

    if (envelope.type === "verdict") return applyVerdict(chain, envelope);

    if (envelope.type === "order_finished") {
        const outcome = rec(data.outcome);
        const winners = (outcome.chain as Record<string, string> | undefined) ?? {};
        let next: OrderChain = {
            ...chain,
            // the backend's own figures, not a sum of whatever bids this page
            // happened to see -- a page that joined mid-order would otherwise
            // total up an incomplete chain and be quietly wrong
            status: (str(data.status) as OrderChain["status"]) ?? "completed",
            total_cost: num(outcome.total_cost),
            total_days: num(outcome.total_days),
            summary: str(data.summary),
        };
        for (const stage of STAGE_ORDER) {
            const agentId = winners[stage];
            if (!agentId) continue;
            const existing = next.stages[stage] ?? emptyStageAuction();
            // the verdict's agreed price first: it is the backend's own figure
            // and it survives a replay that dropped this stage's bids
            const offer = next.agreed[agentId]
                ?? (existing.winner_agent_id === agentId ? existing.closing_offer : null)
                ?? winningBidAmount(existing, agentId);
            next = { ...next, stages: { ...next.stages, [stage]: { ...existing, closed: true, winner_agent_id: agentId, closing_offer: offer } } };
        }
        return next;
    }

    return chain;
}

function winningBidAmount(auction: StageAuction, agentId: string): number | null {
    for (let i = auction.rounds.length - 1; i >= 0; i--) {
        const bid = auction.rounds[i].bids.find((b) => b.agent_id === agentId);
        if (bid?.offer_amount != null) return bid.offer_amount;
    }
    return null;
}

/** what to show as the chain total
 *
 * the backend's total when it has reported one, otherwise the sum of the
 * stage prices, and nothing at all while a stage is still missing a price,
 * because a partial sum reads as a real number and is not one
 *
 * an order that ended without an accepted chain gets no number at all. summing
 * the leading bids would print a total that no chain ever cost: the supervisor
 * never accepted that combination, and on a model-backed config it was not
 * even the combination selection was looking at */
export function chainTotal(order: OrderChain): number | null {
    if (order.status !== "running" && order.status !== "completed") return null;
    if (order.total_cost != null) return order.total_cost;
    const stages = STAGE_ORDER.filter((s) => order.stages[s]);
    if (stages.length === 0) return null;
    if (!stages.every((s) => order.stages[s]?.closing_offer != null)) return null;
    return stages.reduce((sum, s) => sum + (order.stages[s]?.closing_offer ?? 0), 0);
}
