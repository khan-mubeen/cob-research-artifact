import { useMemo, useState } from "react";
import Bars from "./Bars";
import type { Gate, ReverseAuctionEntry } from "../../types";

interface Props {
    entries: ReverseAuctionEntry[];
}

// v4 only ever produces two variants per architecture -- base and tuned
const VARIANT_COLORS: Record<string, string> = {
    base: "#64748b",
    tuned: "#22c55e",
};

function variantOf(tag: string): string {
    return tag.endsWith("-base") ? "base" : "tuned";
}

function archOf(tag: string): string {
    return tag.replace(/^ra-e1-/, "").replace(/-base$/, "");
}

/** PASS, FAIL and NOT_EVALUATED are three different things, and a measurement
 * that was never taken is not a failed one */
function gateTone(status: Gate["status"]): string {
    if (status === "PASS") return "matched";
    if (status === "FAIL") return "mismatched";
    return "no-winner";
}

interface Row {
    arch: string;
    tuned: ReverseAuctionEntry | null;
    base: ReverseAuctionEntry | null;
    passed: number;
    evaluated: number;
}

export default function ReverseAuctionDashboard({ entries }: Props) {
    const [showAll, setShowAll] = useState(false);

    // the headline: one row per architecture, not one per run. the base
    // checkpoint belongs beside its fine-tune, not as a competing bar
    const rows = useMemo<Row[]>(() => {
        const byArch = new Map<string, Row>();
        for (const e of entries) {
            const arch = archOf(e.tag);
            const row = byArch.get(arch) ?? { arch, tuned: null, base: null, passed: 0, evaluated: 0 };
            if (variantOf(e.tag) === "base") row.base = e;
            else {
                row.tuned = e;
                const gates = Object.values(e.gates);
                row.evaluated = gates.filter((g) => g.status !== "NOT_EVALUATED").length;
                row.passed = gates.filter((g) => g.status === "PASS").length;
            }
            byArch.set(arch, row);
        }
        return [...byArch.values()].sort(
            (a, b) => (b.tuned?.summary.overall.accuracy_pct ?? 0) - (a.tuned?.summary.overall.accuracy_pct ?? 0),
        );
    }, [entries]);

    const bars = useMemo(
        () =>
            [...entries]
                .sort((a, b) => (b.summary.overall.accuracy_pct ?? 0) - (a.summary.overall.accuracy_pct ?? 0))
                .map((e) => ({
                    label: e.tag,
                    value: e.summary.overall.accuracy_pct ?? 0,
                    color: VARIANT_COLORS[variantOf(e.tag)],
                })),
        [entries],
    );

    const gateKeys = useMemo(() => {
        const keys = new Set<string>();
        for (const e of entries) Object.keys(e.gates).forEach((k) => keys.add(k));
        return Array.from(keys).sort();
    }, [entries]);

    return (
        <div className="dashboard-section">
            <div className="panel">
                <h3>Reverse auction, epoch 1 on v4</h3>
                <p className="dim small">
                    Six architectures, each fine-tuned for the auction and each measured against its own untrained
                    checkpoint. Gates counts only the ones that were actually evaluated for that run.
                </p>
                <div className="panel-content scrollable">
                    <table className="headline-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th className="border-left">Fine-tuned</th>
                                <th>Untrained</th>
                                <th>Gain</th>
                                <th>Gates</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((r) => {
                                const tuned = r.tuned?.summary.overall.accuracy_pct ?? null;
                                const base = r.base?.summary.overall.accuracy_pct ?? null;
                                const allPassed = r.evaluated > 0 && r.passed === r.evaluated;
                                return (
                                    <tr key={r.arch}>
                                        <td>{r.arch}</td>
                                        <td className="border-left cell-strong">{tuned != null ? `${tuned.toFixed(1)}%` : "—"}</td>
                                        <td className="dim">{base != null ? `${base.toFixed(1)}%` : "—"}</td>
                                        <td>{tuned != null && base != null ? `+${(tuned - base).toFixed(1)}` : "—"}</td>
                                        <td>
                                            <span className={`badge ${allPassed ? "matched" : "mismatched"}`}>
                                                {r.passed}/{r.evaluated} passed
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
                <button className="control-btn details-toggle" onClick={() => setShowAll((v) => !v)}>
                    {showAll ? "Hide the per-run detail" : "Show every run and every gate"}
                </button>
            </div>

            {showAll && (
                <>
                    <div className="panel">
                        <h3>Overall accuracy, every run</h3>
                        <p className="dim small">grey = base checkpoint, green = tuned (no suffix)</p>
                        <Bars items={bars} />
                    </div>

                    <div className="panel">
                        <h3>Gates</h3>
                        <div className="panel-content scrollable">
                            <table className="gates-table">
                                <thead>
                                    <tr>
                                        <th>Run</th>
                                        {gateKeys.map((k) => (
                                            <th key={k} title={k}>
                                                {k.replace(/_/g, " ")}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {entries.map((e) => (
                                        <tr key={e.tag}>
                                            <td>{e.tag}</td>
                                            {gateKeys.map((k) => {
                                                const gate = e.gates[k];
                                                if (!gate) return <td key={k}>—</td>;
                                                return (
                                                    <td key={k} title={gate.evidence}>
                                                        <span className={`badge ${gateTone(gate.status)}`}>
                                                            {gate.status === "NOT_EVALUATED" ? "not run" : gate.status}
                                                        </span>
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
