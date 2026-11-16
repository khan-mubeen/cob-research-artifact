import { useMemo, useState } from "react";
import Bars from "./Bars";
import type { Epoch2Entry, Stage } from "../../types";
import { STAGE_ORDER } from "../../types";

interface Props {
    entries: Epoch2Entry[];
}

const STAGE_LABELS: Record<Stage, string> = {
    material: "Material",
    tailoring: "Tailoring",
    logistics: "Logistics",
};

export default function Epoch2Dashboard({ entries }: Props) {
    const suiteNames = useMemo(() => (entries[0] ? Object.keys(entries[0].suites) : []), [entries]);
    const [suite, setSuite] = useState(suiteNames[0] ?? "");
    // the default view is the six-model headline; the nine suites are a live
    // follow-up question, not an opening slide
    const [showSuites, setShowSuites] = useState(false);

    // straight from reports/epoch-2/production/comparison/*.json
    // (arms.epoch_2.main_holdout_decisions) -- the exact numbers people are
    // shown directly, not recomputed here, so this can't disagree with them
    const headlineByStage = useMemo(() => {
        const result = {} as Record<Stage, { label: string; value: number }[]>;
        for (const stage of STAGE_ORDER) {
            result[stage] = entries
                .filter((e) => e.main_holdout_decisions != null)
                .map((e) => ({ label: e.tag, value: e.main_holdout_decisions![stage] }))
                .sort((a, b) => b.value - a.value);
        }
        return result;
    }, [entries]);

    const activeSuite = suite || suiteNames[0] || "";
    const substageNames = useMemo(() => {
        const first = entries.find((e) => e.suites[activeSuite]);
        return first ? Object.keys(first.suites[activeSuite]) : [];
    }, [entries, activeSuite]);

    return (
        <div className="dashboard-section">
            <div className="panel">
                <h3>Epoch-2 production: decision agreement on the main holdout</h3>
                <p className="dim small">
                    Six architectures on the sealed-bid holdout, per stage. Lifted straight from compare_epochs.py's
                    epoch_2 arm, so it cannot disagree with that report. The stress suites below are a deliberately
                    harder distribution and are not averaged in.
                </p>
                {STAGE_ORDER.map((stage) => (
                    <div key={stage} style={{ marginTop: 12 }}>
                        <p className="dim small" style={{ marginBottom: 4 }}>
                            {STAGE_LABELS[stage]}
                        </p>
                        <Bars items={headlineByStage[stage]} />
                    </div>
                ))}
            </div>

            <button className="control-btn details-toggle" onClick={() => setShowSuites((v) => !v)}>
                {showSuites ? "Hide the stress suites" : "Show the stress suites"}
            </button>

            {showSuites && (
            <div className="panel">
                <div className="picker-row">
                    <h3>Stress-suite detail</h3>
                    <select value={activeSuite} onChange={(e) => setSuite(e.target.value)}>
                        {suiteNames.map((s) => (
                            <option key={s} value={s}>
                                {s}
                            </option>
                        ))}
                    </select>
                </div>
                <p className="dim small">
                    A deliberately harder sample than the main holdout above, so these numbers are expected to run lower.
                </p>
                <div className="panel-content scrollable">
                    <table className="gates-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                {substageNames.map((s) => (
                                    <th key={s}>{s}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {entries.map((e) => (
                                <tr key={e.tag}>
                                    <td>{e.tag}</td>
                                    {substageNames.map((s) => {
                                        const cell = e.suites[activeSuite]?.[s];
                                        if (!cell) return <td key={s}>—</td>;
                                        return (
                                            <td key={s} title={`n=${cell.n}`}>
                                                {cell.decision_agreement_pct != null ? `${cell.decision_agreement_pct.toFixed(1)}%` : "—"}
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            )}
        </div>
    );
}
