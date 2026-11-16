import { useMemo, useState } from "react";
import type { ConfigDescriptor, ConnectionState, LiveStatus } from "../types";

interface Props {
    connection: ConnectionState;
    status: LiveStatus | null;
    // driven by the mode_switching / mode_switched broadcasts, not by the poll
    switching: boolean;
    config: string;
    submitError: string | null;
    onConfigChange: (config: string) => void;
    onSubmitOrder: (config: string, orderId: string) => void;
    onSubmitFile: (config: string, path: string) => void;
    onResetState: () => void;
    onSwitchMode: (mode: string) => void;
    onClear: () => void;
}

const CONNECTION_LABEL: Record<ConnectionState, string> = {
    connecting: "Connecting…",
    connected: "Live",
    disconnected: "Disconnected, retrying…",
    error: "Connection error, retrying…",
};

const RULE_MODEL_KEY = "rule";

const MODES = [
    { value: "sealed", label: "Sealed" },
    { value: "reverse_auction", label: "Reverse auction" },
];

interface ModelOption {
    key: string;
    label: string;
    configs: ConfigDescriptor[];
    disabled: boolean;
    reason: string | null;
}

export default function LiveControlPanel({
    connection,
    status,
    switching,
    config,
    submitError,
    onConfigChange,
    onSubmitOrder,
    onSubmitFile,
    onResetState,
    onSwitchMode,
    onClear,
}: Props) {
    const configs = useMemo(() => status?.configs ?? [], [status]);

    // the menu is the backend's own model list, so every architecture stays
    // visible with the reason it cannot be picked. deriving it from the
    // running configurations instead silently hides four of the six
    const modelOptions = useMemo<ModelOption[]>(() => {
        const ruleConfigs = configs.filter((c) => c.model == null);
        const options: ModelOption[] = [{
            key: RULE_MODEL_KEY,
            label: "rule-based, no model",
            configs: ruleConfigs,
            disabled: ruleConfigs.length === 0,
            reason: ruleConfigs.length === 0 ? "not started in this session" : null,
        }];
        for (const entry of status?.models ?? []) {
            const running = configs.filter((c) => c.model === entry.model);
            options.push({
                key: entry.model,
                label: entry.label,
                configs: running,
                disabled: running.length === 0,
                // active says whether the ggufs are on this machine, which is
                // not the same question as whether a stack was started for it
                reason: running.length === 0 ? (entry.reason ?? "not started in this session") : null,
            });
        }
        return options;
    }, [configs, status]);

    const [orderId, setOrderId] = useState("");
    const [filePath, setFilePath] = useState("");

    const current = configs.find((c) => c.config === config);
    const modelKey = current ? (current.model ?? RULE_MODEL_KEY) : RULE_MODEL_KEY;
    const variantOptions = modelOptions.find((m) => m.key === modelKey)?.configs ?? [];

    const disconnected = connection !== "connected";
    const running = status?.queue.running ?? null;
    const queued = status?.queue.queued ?? 0;
    const busy = running != null;
    const deadStacks = Object.entries(status?.stacks ?? {}).filter(([, health]) => health !== "running");
    // the backend refuses a switch while anything is queued or running, so the
    // toggle says so instead of letting the click bounce
    const modeLocked = disconnected || switching || busy || queued > 0;
    const lockReason = switching
        ? "the stacks are restarting"
        : busy || queued > 0
          ? "the queue is busy, the backend refuses a switch until it is idle"
          : "not connected";

    function pickModel(key: string) {
        const first = modelOptions.find((m) => m.key === key)?.configs[0];
        if (first) onConfigChange(first.config);
    }

    function submitOrder(e: React.FormEvent) {
        e.preventDefault();
        const trimmed = orderId.trim();
        if (!trimmed) return;
        onSubmitOrder(config, trimmed);
    }

    function submitFile(e: React.FormEvent) {
        e.preventDefault();
        const trimmed = filePath.trim();
        if (!trimmed) return;
        onSubmitFile(config, trimmed);
    }

    return (
        <div className="panel live-control-panel">
            <div className="live-status-row">
                <span className={`conn-dot conn-${connection}`} />
                <span className="conn-label">{CONNECTION_LABEL[connection]}</span>
                {status && (
                    <span className="dim small">
                        · {status.clients} client{status.clients === 1 ? "" : "s"} · {status.events_seen} events
                    </span>
                )}
            </div>

            <div className="control-group">
                <div className="control-group-title">Configuration</div>
                <div className="live-field-row">
                    <label>
                        <span>Model</span>
                        <select value={modelKey} onChange={(e) => pickModel(e.target.value)}>
                            {modelOptions.map((m) => (
                                <option key={m.key} value={m.key} disabled={m.disabled} title={m.reason ?? undefined}>
                                    {m.label}
                                    {m.disabled ? ` (${m.reason})` : ""}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label className="field-narrow">
                        <span>Variant</span>
                        <select
                            value={config}
                            onChange={(e) => onConfigChange(e.target.value)}
                            disabled={variantOptions.length <= 1}
                        >
                            {variantOptions.length === 0 ? (
                                <option value={config}>—</option>
                            ) : (
                                variantOptions.map((c) => {
                                    // a stack that died is refused by the backend, so say so here
                                    const why = status?.unavailable[c.config]
                                        ?? (status?.stacks[c.config] === "dead" ? "stack not running" : null);
                                    return (
                                        <option key={c.config} value={c.config} disabled={why != null}>
                                            {c.variant ?? "default"}
                                            {why ? ` (${why})` : ""}
                                        </option>
                                    );
                                })
                            )}
                        </select>
                    </label>
                </div>
                <div className="mode-toggle-row">
                    <span className="mode-toggle-label">Bidding mode</span>
                    <div className={`mode-toggle ${switching ? "switching" : ""}`} role="group" aria-label="Bidding mode">
                        {MODES.map((m) => {
                            const active = status?.mode === m.value;
                            return (
                                <button
                                    key={m.value}
                                    type="button"
                                    className={`mode-segment ${active ? "active" : ""}`}
                                    aria-pressed={active}
                                    // clicking the mode already running must not restart the stacks
                                    disabled={active || modeLocked}
                                    title={modeLocked ? lockReason : `restart every stack in ${m.label} mode`}
                                    onClick={() => !active && onSwitchMode(m.value)}
                                >
                                    {m.label}
                                </button>
                            );
                        })}
                        {switching && <span className="mode-spinner" aria-hidden="true" />}
                    </div>
                </div>
                {switching && (
                    <div className="mode-switching-note">
                        Restarting both stacks in the other mode. The models reload, so this takes a few seconds.
                    </div>
                )}
            </div>

            <div className="control-group">
                <div className="control-group-title">Submit an order</div>
                <form className="live-field-row" onSubmit={submitOrder}>
                    <input type="text" placeholder="Order ID, e.g. ORD-34921" value={orderId} onChange={(e) => setOrderId(e.target.value)} />
                    <button type="submit" className="control-btn control-btn-primary" disabled={disconnected || !orderId.trim()}>
                        Submit
                    </button>
                </form>
                <form className="live-field-row" onSubmit={submitFile}>
                    <input type="text" placeholder="data/demo/restock_sequence.json" value={filePath} onChange={(e) => setFilePath(e.target.value)} />
                    <button type="submit" className="control-btn" disabled={disconnected || !filePath.trim()}>
                        File
                    </button>
                </form>
            </div>

            <div className="control-group">
                <div className="control-group-title">Session</div>
                <div className="live-field-row live-field-row-buttons">
                    <button className="control-btn" disabled={disconnected} onClick={onResetState} title="reseed agent stock and workload from the csv baseline">
                        Reset state
                    </button>
                    <button className="control-btn" onClick={onClear} title="clear this page only, the backend is untouched">
                        Clear view
                    </button>
                </div>
            </div>

            <div className="live-queue-row dim small">
                {status?.switching ? (
                    <span>switching bidding mode…</span>
                ) : busy ? (
                    <span>
                        running: {running!.order_id} on {running!.config}
                        {queued > 0 ? ` · ${queued} queued` : ""}
                    </span>
                ) : queued > 0 ? (
                    <span>{queued} queued</span>
                ) : (
                    <span>queue idle · {status?.queue.completed_jobs ?? 0} job(s) done</span>
                )}
            </div>

            {deadStacks.length > 0 && (
                <div className="live-submit-error">stack not running: {deadStacks.map(([name]) => name).join(", ")}</div>
            )}
            {submitError && <div className="live-submit-error">{submitError}</div>}
        </div>
    );
}
