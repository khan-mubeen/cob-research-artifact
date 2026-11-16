import { useCallback, useEffect, useRef, useState } from "react";
import type { AuctionEvent, ConnectionState, LiveEnvelope, LiveStatus, OrderChain, RestockNote } from "../types";
import { applyEnvelopeToChain, emptyChain, envelopeToAuctionEvents } from "../liveEvents";

// src/api's server, per src/api/settings.py:SERVER_HOST/SERVER_PORT defaults
export const API_BASE = "http://127.0.0.1:8000";
const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws";

const EVENT_LOG_LIMIT = 300;
const RECONNECT_DELAY_MS = 3000;
// the hello snapshot goes stale the moment anything happens, so ask again on
// a timer. one small frame every few seconds, answered from memory
const STATUS_POLL_MS = 4000;

export function useLiveSession(configPick: string) {
    const [connection, setConnection] = useState<ConnectionState>("connecting");
    const [status, setStatus] = useState<LiveStatus | null>(null);
    const [events, setEvents] = useState<LiveEnvelope[]>([]);
    const [auctionEvents, setAuctionEvents] = useState<AuctionEvent[]>([]);
    const [orderChain, setOrderChain] = useState<OrderChain | null>(null);
    const [trackedOrderId, setTrackedOrderId] = useState<string | null>(null);
    const [submitSeq, setSubmitSeq] = useState(0);
    const [submitError, setSubmitError] = useState<string | null>(null);
    // bumped by every resource event on the watched config, so an open popup
    // can re-read /api/state instead of waiting to be clicked again
    const [resourceSeq, setResourceSeq] = useState(0);
    const [restocks, setRestocks] = useState<Record<string, RestockNote[]>>({});
    // a mode switch is announced and confirmed on the stream. status.switching
    // only refreshes on the poll, and a switch can start and finish between two
    // polls, so the flag follows the broadcasts instead
    const [switching, setSwitching] = useState(false);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const trackedOrderIdRef = useRef<string | null>(null);
    const seenSeqRef = useRef<Set<number>>(new Set());
    const maxSeqRef = useRef(0);
    // the configuration the panel is pointed at, read inside the socket
    // callbacks without re-opening the socket every time it changes
    const configRef = useRef<string | null>(configPick);
    // the job this page submitted, so somebody else's order cannot take the
    // screen over while ours is still queued or running
    const pendingJobRef = useRef<string | null>(null);
    // the job the tracked order belongs to. a file submission runs several
    // orders back to back and they are one story, so the log is not wiped
    // between them
    const trackedJobRef = useRef<string | null>(null);

    // a configuration that is not running any more falls back to one that is,
    // rather than pointing the stream filter at nothing
    const running = status?.configs ?? [];
    const config = running.length > 0 && !running.some((c) => c.config === configPick)
        ? running[0].config
        : configPick;

    useEffect(() => {
        configRef.current = config;
    }, [config]);

    const selectedNow = useCallback(() => configRef.current, []);

    const handleEnvelope = useCallback((envelope: LiveEnvelope) => {
        if (envelope.type === "hello" || envelope.type === "status") {
            const fresh = (envelope.data as unknown as LiveStatus) ?? null;
            setStatus(fresh);
            // an api that restarted numbers its events from one again, and the
            // old seen-set would swallow the whole new stream as duplicates
            if (fresh && fresh.events_seen < maxSeqRef.current) {
                seenSeqRef.current.clear();
                maxSeqRef.current = 0;
            }
            return; // neither belongs in the activity log
        }
        if (envelope.type === "mode_switching") setSwitching(true);
        if (envelope.type === "mode_result" || envelope.type === "mode_switched") {
            const data = envelope.data as { status?: LiveStatus } | undefined;
            if (data?.status) setStatus(data.status);
            setSwitching(false);
        }
        if (envelope.type === "error") {
            // the message is top level on this frame, not inside data
            setSubmitError(envelope.message ?? "request failed");
            setSwitching(false);
        }
        if (envelope.type === "state_change" && (selectedNow() == null || envelope.config === selectedNow())) {
            noteResourceChange(envelope, setResourceSeq, setRestocks);
        }
        if (envelope.type === "accepted") {
            const data = envelope.data as { job_id?: string } | undefined;
            pendingJobRef.current = data?.job_id ?? null;
        }

        if (envelope.seq != null) {
            if (seenSeqRef.current.has(envelope.seq)) return;
            seenSeqRef.current.add(envelope.seq);
            if (envelope.seq > maxSeqRef.current) maxSeqRef.current = envelope.seq;
        }

        const orderId = envelope.order_id;
        const selected = configRef.current;
        // the stream carries every configuration and every client's orders.
        // only the selected configuration may take the screen, and while this
        // page has a job in flight only that job may
        const mine = orderId != null && (selected == null || envelope.config === selected);
        const jobOk = pendingJobRef.current == null || envelope.job_id == null
            || envelope.job_id === pendingJobRef.current;
        if (mine && jobOk && (envelope.type === "order_started" || trackedOrderIdRef.current == null)) {
            if (orderId !== trackedOrderIdRef.current) {
                trackedOrderIdRef.current = orderId;
                setTrackedOrderId(orderId);
                // from_start is only true once order_started is actually seen,
                // which is what tells the view whether its tables are complete
                setOrderChain(emptyChain(orderId, envelope.config ?? null, envelope.type === "order_started"));
                setAuctionEvents([]);
                const job = envelope.job_id ?? null;
                if (job == null || job !== trackedJobRef.current) setEvents([]);
                trackedJobRef.current = job;
            }
        }
        if (envelope.type === "order_finished" && envelope.job_id === pendingJobRef.current) {
            pendingJobRef.current = null;
        }

        setEvents((prev) => {
            const next = [...prev, envelope];
            return next.length > EVENT_LOG_LIMIT ? next.slice(next.length - EVENT_LOG_LIMIT) : next;
        });

        if (!orderId || orderId !== trackedOrderIdRef.current) return;
        if (selected != null && envelope.config != null && envelope.config !== selected) return;

        setOrderChain((prev) => applyEnvelopeToChain(prev ?? emptyChain(orderId, envelope.config ?? null), envelope));
        const mapped = envelopeToAuctionEvents(envelope);
        if (mapped.length > 0) setAuctionEvents((prev) => [...prev, ...mapped]);
    }, [selectedNow]);

    useEffect(() => {
        // one flag per effect run, not one shared ref: under StrictMode the
        // setup runs twice and a shared flag is already true again by the time
        // the first socket's close lands, so that close reconnects and leaves
        // two live sockets behind for the rest of the session
        let cancelled = false;

        function connect() {
            if (cancelled) return;
            setConnection((c) => (c === "connected" ? c : "connecting"));
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            // every callback checks that this socket is still the current one:
            // a superseded socket must not touch the connection state
            ws.onopen = () => !cancelled && wsRef.current === ws && setConnection("connected");
            ws.onmessage = (event) => {
                if (cancelled || wsRef.current !== ws) return;
                try {
                    handleEnvelope(JSON.parse(event.data));
                } catch {
                    // malformed frame, ignore rather than crash the session
                }
            };
            ws.onerror = () => !cancelled && wsRef.current === ws && setConnection("error");
            ws.onclose = () => {
                if (cancelled || wsRef.current !== ws) return;
                setConnection("disconnected");
                reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
            };
        }

        connect();
        return () => {
            cancelled = true;
            if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
            const ws = wsRef.current;
            wsRef.current = null;
            ws?.close();
        };
    }, [handleEnvelope]);

    const send = useCallback((payload: Record<string, unknown>) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(payload));
            return true;
        }
        return false;
    }, []);

    const requestStatus = useCallback(() => send({ action: "status" }), [send]);

    // clients, queue depth, stack health and the completed count all move
    // while the page is open; hello is only where they start
    useEffect(() => {
        if (connection !== "connected") return;
        requestStatus();
        const timer = setInterval(requestStatus, STATUS_POLL_MS);
        return () => clearInterval(timer);
    }, [connection, requestStatus]);

    const clearLocal = useCallback(() => {
        trackedOrderIdRef.current = null;
        pendingJobRef.current = null;
        trackedJobRef.current = null;
        setRestocks({});
        setTrackedOrderId(null);
        setOrderChain(null);
        setAuctionEvents([]);
        setEvents([]);
    }, []);

    const submit = useCallback(
        (config: string, orderId: string) => {
            setSubmitError(null);
            const ok = send({ action: "submit", config, order_id: orderId });
            if (ok) {
                clearLocal();
                setSubmitSeq((n) => n + 1);
            } else {
                setSubmitError("not connected to the live server");
            }
        },
        [send, clearLocal],
    );

    const submitFile = useCallback(
        (config: string, path: string) => {
            setSubmitError(null);
            const ok = send({ action: "submit", config, path });
            if (ok) {
                clearLocal();
                setSubmitSeq((n) => n + 1);
            } else {
                setSubmitError("not connected to the live server");
            }
        },
        [send, clearLocal],
    );

    const resetState = useCallback(() => {
        setSubmitError(null);
        if (!send({ action: "reset" })) setSubmitError("not connected to the live server");
    }, [send]);

    const switchMode = useCallback(
        (mode: string) => {
            setSubmitError(null);
            // set here as well as on the broadcast, so the control reacts to the
            // click rather than to the round trip
            if (send({ action: "mode", mode })) setSwitching(true);
            else setSubmitError("not connected to the live server");
        },
        [send],
    );

    return {
        config,
        connection,
        status,
        switching,
        resourceSeq,
        restocks,
        events,
        auctionEvents,
        orderChain,
        trackedOrderId,
        submitSeq,
        submitError,
        submit,
        submitFile,
        resetState,
        switchMode,
        requestStatus,
        clearLocal,
    };
}

/** keep what the resource feed said about one agent
 *
 * the snapshot behind /api/state says a restock is pending, never when it
 * lands, so the eta has to be kept from the event that ordered it */
function noteResourceChange(
    envelope: LiveEnvelope,
    bump: (fn: (n: number) => number) => void,
    note: (fn: (prev: Record<string, RestockNote[]>) => Record<string, RestockNote[]>) => void,
) {
    const data = envelope.data ?? {};
    const agentId = typeof data.agent_id === "string" ? data.agent_id : null;
    if (!agentId) return;
    bump((n) => n + 1);

    const event = data.event;
    if (event !== "restock_triggered" && event !== "restock_landed") return;
    const entry: RestockNote = {
        agent_id: agentId,
        material: typeof data.material === "string" ? data.material : "stock",
        status: event === "restock_triggered" ? "triggered" : "landed",
        amount_kg: typeof data.amount_kg === "number" ? data.amount_kg : null,
        days: typeof data.restock_time_days === "number" ? data.restock_time_days : null,
        available_after: typeof data.available_kg_after === "number" ? data.available_kg_after : null,
        at: Date.now(),
    };
    // the lead time is in simulated days, so an order and its arrival can be
    // milliseconds apart in real time. keeping both means a presenter sees the
    // whole story instead of whichever one happened to land last
    note((prev) => ({ ...prev, [agentId]: [...(prev[agentId] ?? []), entry].slice(-3) }));
}
