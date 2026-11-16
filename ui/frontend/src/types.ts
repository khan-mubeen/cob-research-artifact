export type Decision = "BID" | "NO_BID";
export type Stage = "material" | "tailoring" | "logistics";

// one agent's reply in one round, as the live stream actually reports it.
// the offline-evaluation fields (label, match, deviation, days match,
// withdrawn) are not here on purpose: nothing live carries a ground-truth
// label, so they belong to the Evaluation Dashboard and nowhere else
export interface RoundBid {
    agent_id: string;
    model_decision: Decision | null;
    offer_amount: number | null;
    estimated_completion_days: number | null;
    defect_probability: number | null;
    decline_reason?: string | null;
    // from order.verdict.* at the end of the order
    verdict?: "accepted" | "rejected" | null;
    // reverse auction only, from the trace: did this offer actually clear
    // the announced best bid by the platform minimum decrement
    meets_decrement?: boolean | null;
}

export interface StageRound {
    round_number: number;
    announced_best_bid: number | null;
    // where the stage stood once the round closed, from auction_trace
    best_bid_after_round?: number | null;
    best_agent_id_after_round?: string | null;
    silent_count?: number | null;
    bids: RoundBid[];
}

// one stage's auction (material, tailoring, or logistics) within an order's
// chain
export interface StageAuction {
    platform_minimum_decrement: number | null;
    rounds: StageRound[];
    // the cheapest standing bid when the stage closed. it is not a result: no
    // stock or capacity is reserved until a whole chain passes, and on a
    // model-backed config chain selection can pick a different agent entirely
    leading_agent_id: string | null;
    // set only from order_finished's accepted chain, which is the point the
    // reservations actually happened
    winner_agent_id: string | null;
    closing_offer: number | null;
    // a stage that closed without a winner is not the same thing as a stage
    // still running, and the graph has to be able to tell them apart
    closed: boolean;
    // the rest is the auction trace, reverse-auction mode only
    rounds_run?: number | null;
    closed_reason?: string | null;
    opening_best_bid?: number | null;
    total_reduction_eur?: number | null;
    roster_size?: number | null;
    entrant_count?: number | null;
}

export type OrderStatus = "running" | "completed" | "infeasible" | "timeout" | "error";

export interface OrderChain {
    order_id: string;
    config: string | null;
    priority: string | null;
    // the order's own specification, from order_started.data (the parsed csv
    // row) -- absent until order_started is actually seen, same as priority
    product_id: string | null;
    quantity: number | null;
    material_type: string | null;
    budget: number | null;
    deadline: string | null;
    deliver_to: string | null;
    max_defect_rate: number | null;
    min_reputation_score: number | null;
    scenario_type: string | null;
    stages: Partial<Record<Stage, StageAuction>>;
    // false when this page joined after the order had already started. the
    // server replays only its last 50 events, so the round tables can have
    // holes; the totals below are the backend's own figures either way
    from_start: boolean;
    status: OrderStatus;
    total_cost: number | null;
    total_days: number | null;
    // the plain-language reason an order did not complete, from
    // order_finished.data.summary
    summary: string | null;
    // agreed price per accepted agent, from order.verdict.*. those events land
    // in the last handful an order produces, so a page that joined late still
    // has the winning price for a stage whose own bids fell out of the replay
    agreed: Record<string, number>;
}


export const STAGE_ORDER: Stage[] = ["material", "tailoring", "logistics"];

export type AuctionEventType =
    | "STAGE_START"
    | "RFQ_OPEN"
    | "BID"
    | "NO_BID"
    | "STAGE_CLOSE"
    | "WINNER"
    | "CHAIN_COMPLETE";

export interface AuctionEvent {
    id: string;
    type: AuctionEventType;
    stage: Stage;
    round_number: number;
    agent_id?: string;
    bid?: RoundBid;
    announced_best_bid?: number | null;
    // STAGE_CLOSE: cheapest standing bid, nothing reserved yet
    leading_agent_id?: string | null;
    // WINNER: confirmed by the accepted chain
    winner_agent_id?: string | null;
    closing_offer?: number | null;
    // STAGE_CLOSE only: the stage is over, leader or not
    closed?: boolean;
}

export interface NodePosition {
    x: number;
    y: number;
}

export const SOURCE_ID = "source";
export const SUPERVISOR_ID = "supervisor";

export interface RosterAgent {
    agent_id: string;
    location: string | null;
    specialty: string | null;
    reputation_score: number | null;
    defect_probability?: number | null;
    material_types?: string | null;
    product_types?: string | null;
    daily_capacity_units?: number | null;
    current_workload_units?: number | null;
    damage_probability?: number | null;
    service_regions?: string | null;
}

export type Roster = Record<Stage, RosterAgent[]>;

export interface LiveEnvelope {
    seq?: number;
    type: string;
    config?: string;
    order_id?: string | null;
    at?: string;
    stage?: Stage;
    round_number?: number | null;
    job_id?: string;
    agent_id?: string;
    position?: number;
    // hoisted by the relay onto an auction rfq, see src/api/relay.py:classify
    current_best_bid?: number | null;
    platform_minimum_decrement?: number | null;
    // failure and error put their text at the top level, not inside data
    message?: string;
    source?: string;
    order_ids?: string[];
    data?: Record<string, unknown>;
}

export type ModelVariant = "base" | "tuned";

export interface ModelVariantAvailability {
    available: boolean;
    reason: string | null;
}

export interface ModelMenuEntry {
    model: string;
    label: string;
    variants: Partial<Record<ModelVariant, ModelVariantAvailability>>;
    active: boolean;
    reason: string | null;
}

export interface ConfigDescriptor {
    config: string;
    label: string;
    model: string | null;
    variant: ModelVariant | null;
    mode: string;
    port: number;
    gguf: string | null;
    window_seconds: number;
}

export interface QueueRunning {
    job_id: string;
    config: string;
    order_id: string;
}

export interface QueueStatus {
    queued: number;
    capacity: number;
    running: QueueRunning | null;
    completed_jobs: number;
}

export interface LiveStatus {
    mode: string;
    switching: boolean;
    models: ModelMenuEntry[];
    configs: ConfigDescriptor[];
    unavailable: Record<string, string>;
    stacks: Record<string, "running" | "dead">;
    queue: QueueStatus;
    session_log: string;
    clients: number;
    events_seen: number;
}

// ── dynamic resource state, GET /api/state/{config} ──────────────────
// only material and tailoring hold state; logistics has none

export interface MaterialStock {
    available_kg: number;
    starting_kg: number;
    restock_threshold_kg: number;
    restock_pending: boolean;
    restocks_triggered: number;
    restocks_landed: number;
}

export interface TailoringLoad {
    current_workload_units: number;
    starting_workload_units: number;
    daily_capacity_units: number;
    open_reservations: number;
    reservations: number;
    releases: number;
}

export interface StateSnapshot<T> {
    ok: boolean;
    stage: string;
    sim_now: string | null;
    snapshot: Record<string, T>;
}

// the last restock this page saw for one agent and material, from the
// order.dynamic_state feed. the snapshot says a restock is pending, it never
// says when it lands, so the eta is kept from the event that ordered it
export interface RestockNote {
    agent_id: string;
    material: string;
    status: "triggered" | "landed";
    amount_kg: number | null;
    days: number | null;
    available_after: number | null;
    at: number;
}

export interface DynamicState {
    // material rows are keyed "AGENT:material", tailoring rows by agent id
    material?: StateSnapshot<MaterialStock>;
    tailoring?: StateSnapshot<TailoringLoad>;
}

export type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

// ── dashboard ────────────────────────────────────────────────────────

export interface SuiteSummary {
    n: number;
    accuracy_pct?: number;
    decision_agreement_pct?: number;
    json_valid_pct?: number;
    label_bid_rate_pct?: number;
    model_bid_rate_pct?: number;
    bid_recall_pct?: number;
    nobid_recall_pct?: number;
    offer_exact_match_pct?: number;
    mean_abs_offer_dev_pct?: number | null;
    median_abs_offer_dev_pct?: number | null;
    days_exact_match_pct?: number | null;
}

export interface Gate {
    // the current gate json carries NOT_EVALUATED too: a measurement that was
    // never taken, which is not the same thing as one that failed
    status: "PASS" | "FAIL" | "NOT_EVALUATED";
    bar: string;
    measured: number;
    n: number;
    evidence: string;
}

export interface ReverseAuctionEntry {
    tag: string;
    model: string;
    generated_at: string;
    kind: "reverse_auction";
    summary: {
        overall: SuiteSummary;
        by_priority?: Record<string, SuiteSummary>;
        by_category?: Record<string, SuiteSummary>;
        [key: string]: unknown;
    };
    gates: Record<string, Gate>;
}

// each suite's summary is broken down by stage (or, for scale_diagnostic, by
// diagnostic category) -- there is no flat top-level SuiteSummary per suite
export interface Epoch2Entry {
    tag: string;
    model: string;
    generated_at: string;
    kind: "epoch2_production";
    suites: Record<string, Record<string, SuiteSummary>>;
    // the headline number people are actually shown -- lifted verbatim from
    // reports/epoch-2/production/comparison/{arch}_epoch1_vs_epoch2.json
    // (arms.epoch_2.main_holdout_decisions), not recomputed here, so this
    // dashboard can never disagree with that comparison report
    main_holdout_decisions: Record<Stage, number> | null;
}

export interface DashboardData {
    reverse_auction: ReverseAuctionEntry[];
    epoch2_production: Epoch2Entry[];
}
