import { useEffect, useState } from "react";
import GraphCanvas from "./GraphCanvas";
import type { GraphNodeClick } from "./GraphCanvas";
import RoundTracker from "./RoundTracker";
import OrderDetailBar from "./OrderDetailBar";
import LiveControlPanel from "./LiveControlPanel";
import LiveEventLog from "./LiveEventLog";
import AgentDetailPopup from "./AgentDetailPopup";
import { useLiveSession, API_BASE } from "../hooks/useLiveSession";
import { useResizableColumn } from "../hooks/useResizableColumn";
import { useCollapsiblePanel } from "../hooks/useCollapsiblePanel";
import type { Roster } from "../types";

const EMPTY_ROSTER: Roster = { material: [], tailoring: [], logistics: [] };

interface Selection {
    node: GraphNodeClick;
    anchor: { x: number; y: number };
    // which order it was opened against, so it cannot outlive it
    orderId: string | null;
}

export default function LiveAuctionView() {
    const [roster, setRoster] = useState<Roster>(EMPTY_ROSTER);
    const [rosterError, setRosterError] = useState(false);
    // the configuration this page is watching. it lives here rather than in
    // the control panel because the session hook filters the shared stream by
    // it: without that, any other configuration's order takes the screen
    const [configPick, setConfigPick] = useState("rule");
    const [selected, setSelected] = useState<Selection | null>(null);

    const left = useResizableColumn("cob-left-w", 320, 240, 640, "left");
    const right = useResizableColumn("cob-right-w", 320, 240, 640, "right");
    const log = useCollapsiblePanel("cob-right-collapsed");

    const {
        config,
        connection,
        status,
        switching,
        events,
        auctionEvents,
        orderChain,
        trackedOrderId,
        submitSeq,
        submitError,
        resourceSeq,
        restocks,
        submit,
        submitFile,
        resetState,
        switchMode,
        clearLocal,
    } = useLiveSession(configPick);

    useEffect(() => {
        fetch(`${API_BASE}/api/agents`)
            .then((r) => {
                if (!r.ok) throw new Error(`${r.status}`);
                return r.json();
            })
            .then((data: Roster) => {
                setRoster(data);
                setRosterError(false);
            })
            .catch(() => {
                setRoster(EMPTY_ROSTER);
                setRosterError(true);
            });
    }, []);

    function changeConfig(next: string) {
        setConfigPick(next);
        clearLocal(); // the order on screen belongs to the configuration we just left
    }

    // a popup opened on the previous order must not stay up reading the next
    // one's bids
    const popup = selected && selected.orderId === trackedOrderId ? selected : null;
    const rightWidth = log.collapsed ? 0 : right.width;

    return (
        <div className="replay-view">
            <div className="col col-left" style={{ width: left.width, flex: `0 0 ${left.width}px` }}>
                <LiveControlPanel
                    connection={connection}
                    status={status}
                    switching={switching}
                    config={config}
                    submitError={submitError ?? (rosterError ? "could not load the agent roster from /api/agents" : null)}
                    onConfigChange={changeConfig}
                    onSubmitOrder={submit}
                    onSubmitFile={submitFile}
                    onResetState={resetState}
                    onSwitchMode={switchMode}
                    onClear={clearLocal}
                />
                <RoundTracker order={orderChain} config={config} />
            </div>
            <div className={`col-resize-handle ${left.dragging ? "dragging" : ""}`} onMouseDown={left.onMouseDown} />
            <div className="col col-center">
                <div className="panel fill graph-panel">
                    {/* keyed so a new order starts from a clean graph, and faded in
                        so that reset does not read as the page reloading */}
                    <div className="graph-swap" key={`${trackedOrderId ?? "none"}-${submitSeq}`}>
                        <GraphCanvas
                            roster={roster}
                            events={auctionEvents}
                            mode={status?.mode ?? null}
                            onNodeClick={(node, anchor) => setSelected({ node, anchor, orderId: trackedOrderId })}
                            onBackgroundClick={() => setSelected(null)}
                        />
                    </div>
                    <div className="legend">
                        <span className="legend-item"><span className="legend-dot" style={{ background: "#38bdf8" }} /> RFQ broadcast</span>
                        <span className="legend-item"><span className="legend-dot" style={{ background: "#8b5cf6" }} /> Bid</span>
                        <span className="legend-item"><span className="legend-dot" style={{ background: "#ef4444" }} /> No bid</span>
                        <span className="legend-item"><span className="legend-dot" style={{ background: "#f59e0b" }} /> Leading, not reserved</span>
                        <span className="legend-item"><span className="legend-dot" style={{ background: "#22c55e" }} /> Winner, reserved</span>
                        <span className="legend-item"><span className="legend-dot" style={{ background: "#334155" }} /> Idle, no reply yet</span>
                    </div>
                    {popup && (
                        <AgentDetailPopup
                            node={popup.node}
                            orderChain={orderChain}
                            config={config}
                            resourceSeq={resourceSeq}
                            restocks={popup.node.agent ? (restocks[popup.node.agent.agent_id] ?? []) : []}
                            anchor={popup.anchor}
                            onClose={() => setSelected(null)}
                        />
                    )}
                </div>
                <OrderDetailBar key={trackedOrderId ?? "none"} order={orderChain} />
            </div>
            {!log.collapsed && (
                <div className={`col-resize-handle ${right.dragging ? "dragging" : ""}`} onMouseDown={right.onMouseDown} />
            )}
            <div
                className={`col col-right ${log.collapsed ? "collapsed" : ""}`}
                style={{ width: rightWidth, flex: `0 0 ${rightWidth}px` }}
            >
                {/* events keep arriving while this is shut, the hook buffers them */}
                <LiveEventLog events={events} watching={config} collapsed={log.collapsed} />
            </div>
            <button
                className={`panel-handle ${log.collapsed ? "closed" : "open"}`}
                onClick={log.toggle}
                title={log.collapsed ? "show the activity log" : "hide the activity log"}
                aria-label={log.collapsed ? "Show the activity log" : "Hide the activity log"}
                aria-expanded={!log.collapsed}
            >
                {log.collapsed ? "‹" : "›"}
            </button>
        </div>
    );
}
