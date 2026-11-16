import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods, type NodeObject } from "react-force-graph-2d";
import type { AuctionEvent, NodePosition, Roster, RosterAgent, Stage } from "../types";
import { SOURCE_ID, STAGE_ORDER, SUPERVISOR_ID } from "../types";

export interface GraphNodeClick {
    id: string;
    kind: NodeKind;
    stage?: Stage;
    agent?: RosterAgent;
}

interface Props {
    roster: Roster;
    events: AuctionEvent[];
    // whichever stage the order is running under. STAGE_CLOSE's leading bid
    // is real data either way (see stageLeaders below) -- this only gates
    // whether the graph *draws* it early. Reverse auction is a live
    // negotiation worth watching converge stage by stage; sealed is one
    // simultaneous round with nothing to track mid-stage, so the map stays
    // quiet there until order_finished names the real, reserved winner
    mode?: string | null;
    onNodeClick?: (node: GraphNodeClick, screenPos: { x: number; y: number }) => void;
    onBackgroundClick?: () => void;
}

const IDLE_STAGE = "rgba(148, 163, 184, 0.35)";
const ACTIVE_STAGE = "#38bdf8";
const CLOSED_STAGE = "#22c55e";
// a stage that closed with nobody winning: amber, because reading it as the
// same blue as a stage still taking bids is the difference between "wait for
// it" and "that stage is over and empty"
const EMPTY_STAGE = "#f59e0b";
const IDLE_AGENT = "#334155";
const IDLE_AGENT_RING = "rgba(148, 163, 184, 0.55)";
const BID_AGENT = "#8b5cf6";
const NO_BID_AGENT = "rgba(239, 68, 68, 0.6)";
const WINNER_AGENT = "#22c55e";
const WINNER_RING = "#4ade80";
// the cheapest bid at a stage, before any chain is accepted. amber rather than
// green because nothing is reserved yet and a model config can still pick
// somebody else for the final chain
const LEADER_RING = "#f59e0b";
const BASE_LINE = "rgba(148, 163, 184, 0.18)";
const REACHED_LINE = "rgba(56, 189, 248, 0.65)";
const ANSWERED_LINE = "rgba(139, 92, 246, 0.5)";

const STAGE_LABELS: Record<Stage, string> = {
    material: "Material",
    tailoring: "Tailoring",
    logistics: "Logistics",
};

// a fixed default composition: Order on top, Supervisor beneath it, the
// three stage clusters in a row beneath that -- both nodes are pinned
// (fx/fy) so this framing is stable on load, not left to chance the way a
// fully free physics layout would be. dragging still works normally: a
// drag just moves the pin to wherever it's dropped, same as any other node
const ORDER_POS: NodePosition = { x: 0, y: -265 };
const SUPERVISOR_POS: NodePosition = { x: 0, y: -155 };

// fixed pull-points for the per-stage clustering force -- agents belonging
// to the same stage are nudged toward their stage's point every tick, which
// is what keeps them visually grouped now that there's no hub node holding
// them together structurally
// one row at one height, left to right in the order the chain runs. the old
// arrangement put tailoring lower than its neighbours, which read as a shape
// rather than a sequence and pushed that cluster off the bottom of the panel
const CLUSTER_ANCHORS: Record<Stage, NodePosition> = {
    material: { x: -330, y: 10 },
    tailoring: { x: 0, y: 10 },
    logistics: { x: 330, y: 10 },
};

// stage order, said on the graph rather than left to be inferred
const STAGE_STEP: Record<Stage, number> = { material: 1, tailoring: 2, logistics: 3 };
const FLOW_ARROW = "rgba(148, 163, 184, 0.35)";
const CLUSTER_STRENGTH = 0.55;
const HULL_PADDING = 30;

type NodeKind = "source" | "supervisor" | "agent";

interface GNode {
    id: string;
    kind: NodeKind;
    stage?: Stage;
    agent?: RosterAgent;
}

interface GLink {
    source: string;
    target: string;
    stage?: Stage;
    forAgent?: string;
    // agent -> supervisor "reply" channel: not drawn as a visible line (see
    // linkWidth), it exists purely so a bid/no-bid packet has a link to
    // travel along in the agent-to-supervisor direction, opposite the
    // always-visible supervisor -> agent broadcast line
    reply?: boolean;
}

function tooltipFor(agent: RosterAgent): string {
    const parts = [agent.agent_id];
    if (agent.location) parts.push(agent.location);
    if (agent.specialty) parts.push(agent.specialty);
    if (agent.reputation_score != null) parts.push(`reputation ${agent.reputation_score.toFixed(2)}`);
    return parts.join(" · ");
}

/** monotone-chain convex hull, expanded outward from its centroid by
 * `padding` so the drawn boundary has breathing room around the dots */
function paddedHull(points: NodePosition[], padding: number): NodePosition[] {
    if (points.length < 3) return points;
    const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
    const cross = (o: NodePosition, a: NodePosition, b: NodePosition) =>
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    const lower: NodePosition[] = [];
    for (const p of pts) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
        lower.push(p);
    }
    const upper: NodePosition[] = [];
    for (let i = pts.length - 1; i >= 0; i--) {
        const p = pts[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
        upper.push(p);
    }
    upper.pop();
    lower.pop();
    const hull = lower.concat(upper);

    const cx = hull.reduce((s, p) => s + p.x, 0) / hull.length;
    const cy = hull.reduce((s, p) => s + p.y, 0) / hull.length;
    return hull.map((p) => {
        const dx = p.x - cx;
        const dy = p.y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        return { x: p.x + (dx / dist) * padding, y: p.y + (dy / dist) * padding };
    });
}

/** draws a padded hull as a smooth closed curve through the midpoints of
 * each edge, instead of straight segments -- a hull is only ever as smooth
 * as its sharpest corner, and dragging one node away from its cluster pulls
 * a hull corner into a hard spike; curving through midpoints turns that
 * into a soft bulge instead of a jagged point */
function traceSmoothHull(ctx: CanvasRenderingContext2D, hull: NodePosition[]) {
    const n = hull.length;
    const mid = (a: NodePosition, b: NodePosition) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
    const start = mid(hull[n - 1], hull[0]);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    for (let i = 0; i < n; i++) {
        const curr = hull[i];
        const next = hull[(i + 1) % n];
        const m = mid(curr, next);
        ctx.quadraticCurveTo(curr.x, curr.y, m.x, m.y);
    }
    ctx.closePath();
}

/** the container needs explicit pixel dimensions -- the canvas library
 * doesn't fill a percentage-sized parent on its own */
function useElementSize() {
    const ref = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState({ width: 0, height: 0 });
    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const observer = new ResizeObserver(([entry]) => {
            if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
        });
        observer.observe(el);
        return () => observer.disconnect();
    }, []);
    return [ref, size] as const;
}

/** the canvas paints its own background, so it has to read the same token the
 * panels do rather than carry a second copy of the colour */
function canvasBackground(): string {
    if (typeof document === "undefined") return "#070c17";
    return getComputedStyle(document.documentElement).getPropertyValue("--canvas-bg").trim() || "#070c17";
}

export default function GraphCanvas({ roster, events, mode, onNodeClick, onBackgroundClick }: Props) {
    const [containerRef, size] = useElementSize();
    const canvasBg = useMemo(() => canvasBackground(), []);
    // stageLeaders (below) still fills in truthfully for both modes -- this
    // only decides whether that leader gets drawn on the agent before the
    // chain is actually decided
    const showLeaders = mode === "reverse_auction";
    const fgRef = useRef<ForceGraphMethods<GNode, GLink>>(undefined);
    const hasZoomed = useRef(false);
    const processedCount = useRef(0);

    const [agentDecision, setAgentDecision] = useState<Record<string, "BID" | "NO_BID">>({});
    const [stageRound, setStageRound] = useState<Partial<Record<Stage, number>>>({});
    const [stageBest, setStageBest] = useState<Partial<Record<Stage, number | null>>>({});
    const [stageWinners, setStageWinners] = useState<Partial<Record<Stage, string | null>>>({});
    const [stageLeaders, setStageLeaders] = useState<Partial<Record<Stage, string | null>>>({});
    const [stageClosed, setStageClosed] = useState<Partial<Record<Stage, boolean>>>({});
    const [bidRounds, setBidRounds] = useState<Record<string, number>>({});
    const [broadcastingStage, setBroadcastingStage] = useState<Stage | null>(null);

    // typed as NodeObject<GNode> (not the plain GNode we construct here)
    // because react-force-graph mutates these exact objects in place, adding
    // x/y/vx/vy/fx/fy once the simulation starts -- other code below reads
    // those live positions off the same references
    const nodes = useMemo<NodeObject<GNode>[]>(() => {
        const list: NodeObject<GNode>[] = [
            { id: SOURCE_ID, kind: "source", x: ORDER_POS.x, y: ORDER_POS.y, fx: ORDER_POS.x, fy: ORDER_POS.y },
            { id: SUPERVISOR_ID, kind: "supervisor", x: SUPERVISOR_POS.x, y: SUPERVISOR_POS.y, fx: SUPERVISOR_POS.x, fy: SUPERVISOR_POS.y },
        ];
        for (const stage of STAGE_ORDER) {
            for (const agent of roster[stage] ?? []) list.push({ id: agent.agent_id, kind: "agent", stage, agent });
        }
        return list;
    }, [roster]);

    const nodesByStage = useMemo(() => {
        const map: Record<Stage, NodeObject<GNode>[]> = { material: [], tailoring: [], logistics: [] };
        for (const n of nodes) if (n.kind === "agent" && n.stage) map[n.stage].push(n);
        return map;
    }, [nodes]);

    // every agent connects straight to the supervisor -- there is no
    // per-stage hub node in between. each agent gets both the always-drawn
    // broadcast line (supervisor -> agent) and an invisible reply channel
    // (agent -> supervisor) that a bid/no-bid packet travels along
    const baseLinks = useMemo<GLink[]>(() => {
        const list: GLink[] = [{ source: SOURCE_ID, target: SUPERVISOR_ID }];
        for (const stage of STAGE_ORDER) {
            for (const agent of roster[stage] ?? []) {
                list.push({ source: SUPERVISOR_ID, target: agent.agent_id, stage, forAgent: agent.agent_id });
                list.push({ source: agent.agent_id, target: SUPERVISOR_ID, stage, forAgent: agent.agent_id, reply: true });
            }
        }
        return list;
    }, [roster]);

    const linksByAgent = useMemo(() => {
        const map: Record<string, { out: GLink; reply: GLink }> = {};
        for (const l of baseLinks) {
            if (!l.forAgent) continue;
            const entry = (map[l.forAgent] ??= {} as { out: GLink; reply: GLink });
            if (l.reply) entry.reply = l;
            else entry.out = l;
        }
        return map;
    }, [baseLinks]);

    // degree only counts the visible structural links -- the invisible
    // reply channels shouldn't inflate node size
    const degree = useMemo(() => {
        const d: Record<string, number> = {};
        for (const l of baseLinks) {
            if (l.reply) continue;
            d[l.source] = (d[l.source] ?? 0) + 1;
            d[l.target] = (d[l.target] ?? 0) + 1;
        }
        return d;
    }, [baseLinks]);

    // the order and the supervisor are one node each and would otherwise come
    // out the same size as an agent; an agent grows a little for every round it
    // has actually bid in, which is the round history the tables keep but the
    // single latest-decision colour cannot show
    const radiusFor = useCallback(
        (id: string) => {
            if (id === SOURCE_ID) return 9;
            if (id === SUPERVISOR_ID) return 3 + 2.4 * Math.sqrt(degree[id] ?? 1);
            return 5.5 + Math.min(bidRounds[id] ?? 0, 3) * 1.1;
        },
        [degree, bidRounds],
    );

    const graphData = useMemo(() => ({ nodes, links: baseLinks }), [nodes, baseLinks]);

    // register the custom per-stage clustering force once the graph engine
    // is mounted -- pulls each agent toward its stage's anchor point every
    // simulation tick, which is what keeps material/tailoring/logistics
    // visually grouped now that they aren't tied together by a hub node
    const engineMounted = size.width > 0;
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg) return; // ForceGraph2D doesn't mount until size.width > 0
        const clusterForce = (alpha: number) => {
            for (const stage of STAGE_ORDER) {
                const anchor = CLUSTER_ANCHORS[stage];
                for (const n of nodesByStage[stage]) {
                    if (n.x == null || n.y == null) continue;
                    n.vx = (n.vx ?? 0) + (anchor.x - n.x) * CLUSTER_STRENGTH * alpha;
                    n.vy = (n.vy ?? 0) + (anchor.y - n.y) * CLUSTER_STRENGTH * alpha;
                }
            }
        };
        fg.d3Force("cluster", clusterForce);

        // the library's default repulsion is tuned for a much sparser graph
        // -- at 39 same-stage nodes it easily overpowers the cluster pull
        // above and spreads everything into one shared ring instead of 3
        // separate groups, so it's weakened here
        const charge = fg.d3Force("charge");
        if (charge?.strength) charge.strength(-110);

        // every agent now links straight to the supervisor, and d3's default
        // link force pulls all of them to roughly the same short distance
        // from it regardless of stage -- that alone produces a single ring
        // around the supervisor no matter how strong the cluster force
        // above is, since it only controls angle, not radius. matching the
        // link's target distance to how far the cluster anchors actually
        // are lets both forces agree instead of fight, and keeps the whole
        // graph compact instead of stretched out with empty space
        const link = fg.d3Force("link");
        if (link?.distance) link.distance(190);
        // re-run once the graph engine actually mounts (size.width flips
        // from 0 to real), not just when the roster-derived data changes
    }, [nodesByStage, engineMounted]);

    useEffect(() => {
        const newEvents = events.slice(processedCount.current);
        processedCount.current = events.length;

        for (const evt of newEvents) {
            switch (evt.type) {
                case "STAGE_START": {
                    setStageRound((prev) => ({ ...prev, [evt.stage]: prev[evt.stage] ?? 0 }));
                    break;
                }
                case "RFQ_OPEN": {
                    setStageRound((prev) => ({ ...prev, [evt.stage]: evt.round_number }));
                    setStageBest((prev) => ({ ...prev, [evt.stage]: evt.announced_best_bid ?? null }));
                    setBroadcastingStage(evt.stage);
                    setTimeout(() => setBroadcastingStage((cur) => (cur === evt.stage ? null : cur)), 500);
                    // one packet per roster member, supervisor -> agent, for
                    // the RFQ actually going out to everyone in the stage
                    for (const n of nodesByStage[evt.stage]) {
                        const link = linksByAgent[n.id]?.out;
                        if (link) fgRef.current?.emitParticle(link);
                    }
                    break;
                }
                case "BID": {
                    if (!evt.agent_id) break;
                    setAgentDecision((prev) => ({ ...prev, [evt.agent_id!]: "BID" }));
                    setBidRounds((prev) => ({ ...prev, [evt.agent_id!]: (prev[evt.agent_id!] ?? 0) + 1 }));
                    const bidLink = linksByAgent[evt.agent_id]?.reply;
                    if (bidLink) fgRef.current?.emitParticle(bidLink);
                    break;
                }
                case "NO_BID": {
                    if (!evt.agent_id) break;
                    setAgentDecision((prev) => ({ ...prev, [evt.agent_id!]: "NO_BID" }));
                    const noBidLink = linksByAgent[evt.agent_id]?.reply;
                    if (noBidLink) fgRef.current?.emitParticle(noBidLink);
                    break;
                }
                case "STAGE_CLOSE": {
                    setStageClosed((prev) => ({ ...prev, [evt.stage]: true }));
                    // kept unconditionally (both modes): "no bids" vs "closed
                    // with a leader" still has to be told apart regardless of
                    // whether the map draws that leader early
                    setStageLeaders((prev) => ({ ...prev, [evt.stage]: evt.leading_agent_id ?? null }));
                    setStageBest((prev) => ({ ...prev, [evt.stage]: evt.closing_offer ?? prev[evt.stage] ?? null }));
                    if (showLeaders && evt.leading_agent_id) {
                        const link = linksByAgent[evt.leading_agent_id]?.reply;
                        if (link) fgRef.current?.emitParticle(link);
                    }
                    break;
                }
                case "WINNER": {
                    if (!evt.winner_agent_id) break;
                    setStageWinners((prev) => ({ ...prev, [evt.stage]: evt.winner_agent_id ?? null }));
                    const link = linksByAgent[evt.winner_agent_id]?.reply;
                    if (link) fgRef.current?.emitParticle(link);
                    break;
                }
                default:
                    break;
            }
        }
    }, [events, nodesByStage, linksByAgent, showLeaders]);

    const agentFill = useCallback(
        (agentId: string, stage: Stage) => {
            if (stageWinners[stage] === agentId) return WINNER_AGENT;
            if (showLeaders && stageLeaders[stage] === agentId) return LEADER_RING;
            const decision = agentDecision[agentId];
            if (decision === "BID") return BID_AGENT;
            if (decision === "NO_BID") return NO_BID_AGENT;
            return IDLE_AGENT;
        },
        [stageWinners, stageLeaders, showLeaders, agentDecision],
    );

    const stageReached = useCallback((stage: Stage) => stageRound[stage] != null, [stageRound]);
    const stageColor = useCallback(
        (stage: Stage) => {
            if (stageWinners[stage]) return CLOSED_STAGE;
            // a stage that closed with literally nobody bidding is a hard fact,
            // not a provisional pick -- worth flagging in either mode
            if (stageClosed[stage] && !stageLeaders[stage]) return EMPTY_STAGE;
            // closed with a leader, nothing reserved: reverse auction marks that
            // as a stage-of-its-own state; sealed reads as still "in progress"
            // until the real, reserved winner is known, same as everything else
            // this gate already hides on the agents themselves
            if (stageClosed[stage] && showLeaders) return EMPTY_STAGE;
            return stageReached(stage) ? ACTIVE_STAGE : IDLE_STAGE;
        },
        [stageWinners, stageClosed, stageLeaders, showLeaders, stageReached],
    );

    const nodeCanvasObject = useCallback(
        (node: NodeObject<GNode>, ctx: CanvasRenderingContext2D, globalScale: number) => {
            const x = node.x ?? 0;
            const y = node.y ?? 0;
            const r = radiusFor(node.id);
            const isWinner = node.kind === "agent" && node.stage != null && stageWinners[node.stage] === node.id;
            const isLeader = showLeaders && node.kind === "agent" && node.stage != null && !isWinner
                && stageLeaders[node.stage] === node.id;

            let fill = IDLE_AGENT;
            let label: string | undefined;

            if (node.kind === "source") {
                fill = "#cbd5e1";
                label = "Order";
            } else if (node.kind === "supervisor") {
                fill = "#8b5cf6";
                label = "Supervisor";
            } else if (node.kind === "agent" && node.stage) {
                fill = agentFill(node.id, node.stage);
            }

            if (isWinner || isLeader) {
                // a bold ring, not the same purple every other bidder gets.
                // green once the chain is accepted and the resources are
                // actually reserved, amber while it is only the cheapest bid
                ctx.beginPath();
                ctx.arc(x, y, r + 3.5, 0, 2 * Math.PI, false);
                ctx.strokeStyle = isWinner ? WINNER_RING : LEADER_RING;
                ctx.lineWidth = 2.5;
                ctx.stroke();
            }

            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI, false);
            ctx.fillStyle = fill;
            ctx.fill();
            // an agent that has not answered yet is drawn hollow, so the ones
            // that have actually replied are the ones the eye lands on
            if (node.kind === "agent" && fill === IDLE_AGENT) {
                ctx.strokeStyle = IDLE_AGENT_RING;
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            if (label) {
                const fontSize = 12 / globalScale;
                ctx.font = `${fontSize}px Inter, sans-serif`;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "#9ca3af";
                ctx.fillText(label, x + r + 5, y);
            }

            // the id of whichever agent is marked, written on the graph: the
            // alternative is clicking the node to find out who won
            if (isWinner || isLeader) {
                const fontSize = 11 / globalScale;
                ctx.font = `700 ${fontSize}px Inter, sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                const pad = 3 / globalScale;
                const text = node.id;
                const w = ctx.measureText(text).width;
                const top = y + r + 5 / globalScale;
                ctx.fillStyle = "rgba(7, 12, 23, 0.8)";
                ctx.fillRect(x - w / 2 - pad, top - pad / 2, w + pad * 2, fontSize + pad);
                ctx.fillStyle = isWinner ? WINNER_RING : LEADER_RING;
                ctx.fillText(text, x, top);
            }
        },
        [radiusFor, agentFill, stageWinners, stageLeaders, showLeaders],
    );

    const nodePointerAreaPaint = useCallback(
        (node: NodeObject<GNode>, color: string, ctx: CanvasRenderingContext2D) => {
            const r = Math.max(radiusFor(node.id), 6) + 4;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI, false);
            ctx.fill();
        },
        [radiusFor],
    );

    const nodeLabel = useCallback(
        (node: NodeObject<GNode>) => (node.kind === "agent" && node.agent ? tooltipFor(node.agent) : ""),
        [],
    );

    const linkColor = useCallback(
        (link: NodeObject<GLink>) => {
            if (link.reply) return "transparent";
            if (link.forAgent) {
                if (broadcastingStage === link.stage) return REACHED_LINE;
                if (agentDecision[link.forAgent]) return ANSWERED_LINE;
                return BASE_LINE;
            }
            return STAGE_ORDER.some(stageReached) ? REACHED_LINE : BASE_LINE;
        },
        [broadcastingStage, agentDecision, stageReached],
    );

    const linkWidth = useCallback(
        (link: NodeObject<GLink>) => {
            if (link.reply) return 0; // invisible -- exists only to carry reply packets
            if (link.forAgent) return broadcastingStage === link.stage || agentDecision[link.forAgent] ? 1.4 : 0.8;
            return 1.3;
        },
        [broadcastingStage, agentDecision],
    );

    // packet color: outbound (supervisor -> agent) RFQ packets are blue;
    // reply (agent -> supervisor) packets are purple/red for an ordinary
    // bid/no-bid, or a bold green for the agent that goes on to win
    const particleColor = useCallback(
        (link: NodeObject<GLink>) => {
            if (!link.reply) return REACHED_LINE;
            if (!link.forAgent) return BASE_LINE;
            if (link.stage && stageWinners[link.stage] === link.forAgent) return WINNER_RING;
            if (showLeaders && link.stage && stageLeaders[link.stage] === link.forAgent) return LEADER_RING;
            const decision = agentDecision[link.forAgent];
            if (decision === "BID") return BID_AGENT;
            if (decision === "NO_BID") return NO_BID_AGENT;
            return BASE_LINE;
        },
        [stageWinners, stageLeaders, showLeaders, agentDecision],
    );

    // draws the moving dashed cluster boundary + stage label every frame,
    // recomputed from each agent's live (possibly dragged) position
    const renderClusters = useCallback(
        (ctx: CanvasRenderingContext2D, globalScale: number) => {
            // a faint arrow from each stage to the next, so the row reads as a
            // sequence. purely decoration on the same pass as the hulls, it
            // touches nothing clickable
            const centres: Partial<Record<Stage, { x: number; y: number; r: number }>> = {};
            for (const stage of STAGE_ORDER) {
                const pts = nodesByStage[stage].filter((n) => n.x != null && n.y != null);
                if (pts.length === 0) continue;
                const cx = pts.reduce((s, n) => s + (n.x as number), 0) / pts.length;
                const cy = pts.reduce((s, n) => s + (n.y as number), 0) / pts.length;
                const r = Math.max(...pts.map((n) => Math.hypot((n.x as number) - cx, (n.y as number) - cy)));
                centres[stage] = { x: cx, y: cy, r: r + HULL_PADDING + 10 };
            }
            for (let i = 0; i < STAGE_ORDER.length - 1; i++) {
                const from = centres[STAGE_ORDER[i]];
                const to = centres[STAGE_ORDER[i + 1]];
                if (!from || !to) continue;
                const angle = Math.atan2(to.y - from.y, to.x - from.x);
                const x0 = from.x + Math.cos(angle) * from.r;
                const y0 = from.y + Math.sin(angle) * from.r;
                const x1 = to.x - Math.cos(angle) * to.r;
                const y1 = to.y - Math.sin(angle) * to.r;
                if (Math.hypot(x1 - x0, y1 - y0) < 24) continue;
                ctx.strokeStyle = FLOW_ARROW;
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.stroke();
                const head = 7;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x1 - Math.cos(angle - 0.4) * head, y1 - Math.sin(angle - 0.4) * head);
                ctx.moveTo(x1, y1);
                ctx.lineTo(x1 - Math.cos(angle + 0.4) * head, y1 - Math.sin(angle + 0.4) * head);
                ctx.stroke();
            }

            for (const stage of STAGE_ORDER) {
                const pts = nodesByStage[stage]
                    .filter((n) => n.x != null && n.y != null)
                    .map((n) => ({ x: n.x as number, y: n.y as number }));
                if (pts.length < 3) continue;

                const hull = paddedHull(pts, HULL_PADDING);
                const color = stageColor(stage);

                traceSmoothHull(ctx, hull);
                ctx.setLineDash([6, 5]);
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.3;
                ctx.stroke();
                ctx.setLineDash([]);

                const top = hull.reduce((a, b) => (b.y < a.y ? b : a));
                const round = stageRound[stage];
                const best = stageBest[stage];
                const parts = [`${STAGE_STEP[stage]}. ${STAGE_LABELS[stage]}`];
                if (round) parts.push(`R${round}`);
                if (best != null) parts.push(`€${best.toFixed(0)}`);
                if (stageClosed[stage] && !stageWinners[stage] && !stageLeaders[stage]) parts.push("no bids");
                else if (stageClosed[stage] && !stageWinners[stage] && showLeaders) parts.push("leading, not reserved");

                const fontSize = 13 / globalScale;
                ctx.font = `700 ${fontSize}px Inter, sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = "bottom";
                ctx.fillStyle = color;
                ctx.fillText(parts.join(" · "), top.x, top.y - 6 / globalScale);
            }
        },
        [nodesByStage, stageColor, stageRound, stageBest, stageClosed, stageWinners, stageLeaders, showLeaders],
    );

    const handleDragEnd = useCallback((node: NodeObject<GNode>) => {
        node.fx = node.x;
        node.fy = node.y;
    }, []);

    const handleNodeClick = useCallback(
        (node: NodeObject<GNode>, event: MouseEvent) => {
            onNodeClick?.({ id: node.id, kind: node.kind, stage: node.stage, agent: node.agent }, { x: event.clientX, y: event.clientY });
        },
        [onNodeClick],
    );

    const handleEngineStop = useCallback(() => {
        if (hasZoomed.current) return;
        hasZoomed.current = true;
        fgRef.current?.zoomToFit(500, 60);
    }, []);

    return (
        <div className="graph-canvas" ref={containerRef}>
            {size.width > 0 && (
                <ForceGraph2D<GNode, GLink>
                    ref={fgRef}
                    width={size.width}
                    height={size.height}
                    graphData={graphData}
                    backgroundColor={canvasBg}
                    nodeRelSize={4}
                    nodeCanvasObject={nodeCanvasObject}
                    nodeCanvasObjectMode={() => "replace"}
                    nodePointerAreaPaint={nodePointerAreaPaint}
                    nodeLabel={nodeLabel}
                    linkColor={linkColor}
                    linkWidth={linkWidth}
                    linkDirectionalParticleWidth={3.4}
                    linkDirectionalParticleColor={particleColor}
                    linkDirectionalParticleSpeed={0.0045}
                    onNodeDragEnd={handleDragEnd}
                    onNodeClick={handleNodeClick}
                    onBackgroundClick={onBackgroundClick}
                    onEngineStop={handleEngineStop}
                    onRenderFramePost={renderClusters}
                    cooldownTicks={200}
                    d3AlphaDecay={0.02}
                    d3VelocityDecay={0.35}
                />
            )}
        </div>
    );
}
