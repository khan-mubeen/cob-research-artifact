"""summary of the latency diagnostic (b3); rerunnable on partial results

reads <dir>/<mode>/<model>/records.jsonl for every cell that has measured rows and writes summary.csv,
summary.md and the draft paper table (paper_table.md, paper_table.tex) into this folder.

  PYTHONDONTWRITEBYTECODE=1 python3 summarize.py [--dir runs]
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("sealed", "reverse_auction", "supervisor")
MODELS = ("gemma", "granite", "ministral", "phi", "qwen", "smollm3")
CONDS = ("base", "attached", "merged")
NAME = {"gemma": "Gemma", "granite": "Granite", "ministral": "Ministral", "phi": "Phi", "qwen": "Qwen",
        "smollm3": "SmolLM3"}
TITLE = {"sealed": "Sealed bid", "reverse_auction": "Reverse auction", "supervisor": "Supervisor"}


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def write_tables(stem, head, table, notes):
    """the same numbers as markdown and as a plain booktabs tabular"""
    md = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
    md += ["| " + " | ".join(r) + " |" for r in table]
    (HERE / f"{stem}.md").write_text("\n".join(md + [""] + notes) + "\n")
    tex = ["\\begin{tabular}{ll" + "r" * (len(head) - 2) + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in table]
    tex += ["\\bottomrule"] + [f"\\multicolumn{{{len(head)}}}{{l}}{{\\footnotesize {n}}} \\\\" for n in notes]
    (HERE / f"{stem}.tex").write_text("\n".join(tex + ["\\end{tabular}"]) + "\n")


def decision(text):
    """the decision field of the json answer (first { to last }), or None"""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)).get("decision") if m else None
    except ValueError:
        return None


def extra_lines(meas_by):
    """two notes from the saved records only: reverse-auction smollm3 first-token times, and, in sealed bid, whether
    merged outputs that differ from attached keep the same decision field (no labels, no scoring)"""
    out = []
    meas = meas_by.get(("reverse_auction", "smollm3"))
    if meas:
        t = {c: sorted(r["ttft_ms"] for r in meas[c].values()) for c in ("base", "merged")}
        med = {c: statistics.median(v) for c, v in t.items()}
        slow = [x for x in t["merged"] if x > 10 * med["merged"]]
        rest = [x for x in t["merged"] if x <= 10 * med["merged"]]
        out.append(f"- Reverse auction SmolLM3: TTFT median {med['base']:.2f} ms (base) and {med['merged']:.2f} ms (merged); "
                   f"the merged mean {mean(t['merged']):.2f} ms is raised by {len(slow)} slow "
                   f"request{'s' if len(slow) != 1 else ''} "
                   f"({', '.join(f'{x:.0f}' for x in slow)} ms); without it the merged mean is {mean(rest):.2f} ms "
                   f"(base mean {mean(t['base']):.2f} ms).")
    diff = same = 0
    for (mode, m), meas in meas_by.items():
        if mode != "sealed":
            continue
        for i, r in meas["merged"].items():
            if i in meas["attached"] and r["text"] != meas["attached"][i]["text"]:
                diff += 1
                d = decision(r["text"])
                same += d is not None and d == decision(meas["attached"][i]["text"])
    if diff:
        out.append(f"- Sealed bid, saved texts only: where merged and attached text differ ({diff} rows over all "
                   f"sealed cells), the decision field is identical in {same} ({100 * same / diff:.1f} %).")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs")
    a = ap.parse_args()
    root = HERE / a.dir
    rows, hook, status, meas_by = [], [], {}, {}
    for mode in MODES:
        for m in MODELS:
            p = root / mode / m / "records.jsonl"
            if not p.exists():
                status[(mode, m)] = "pending"
                continue
            recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            status[(mode, m)] = "complete" if (root / mode / m / "cell_done.json").exists() else "partial"
            meas = defaultdict(dict)
            for r in recs:
                if r["kind"] == "measure":
                    meas[r["condition"]][r["row"]] = r
                elif r["kind"] == "hook_check":
                    hook.append(r)
            common = set.intersection(*(set(meas[c]) for c in CONDS)) if all(meas[c] for c in CONDS) else set()
            if not common:
                continue
            out = {"mode": mode, "model": m, "status": status[(mode, m)], "rows": len(common)}
            for c in CONDS:
                rs = [meas[c][i] for i in common]
                warm = [r["total_ms"] for r in recs if r["kind"].startswith("warmup") and r["condition"] == c]
                out.update({f"{c}_in_tokens": round(mean(r["input_tokens"] for r in rs), 2),
                            f"{c}_out_tokens": round(mean(r["output_tokens"] for r in rs), 2),
                            f"{c}_ttft_ms": round(mean(r["ttft_ms"] for r in rs), 2),
                            f"{c}_tpot_ms": round(mean(r["tpot_ms"] for r in rs), 3),
                            f"{c}_total_ms": round(mean(r["total_ms"] for r in rs), 2),
                            f"{c}_warmup_mean_ms": round(mean(warm), 1) if warm else None,
                            f"{c}_cap_hits": sum(r["hit_cap"] for r in rs)})
            for c in ("attached", "merged"):
                out[f"{c}_to_base_tpot"] = round(out[f"{c}_tpot_ms"] / out["base_tpot_ms"], 3)
                out[f"{c}_to_base_total"] = round(out[f"{c}_total_ms"] / out["base_total_ms"], 3)
            out["merged_vs_attached_exact_match_pct"] = round(
                100 * sum(meas["merged"][i]["text"] == meas["attached"][i]["text"] for i in common) / len(common), 2)
            rows.append(out)
            meas_by[(mode, m)] = meas
    if rows:
        with open(HERE / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    n_rows = sorted({r["rows"] for r in rows})
    label = (f"{'/'.join(map(str, n_rows))} validation rows, one request at a time, not the official "
             "test profile")
    md = ["# Latency diagnostic summary\n",
          f"Cells: {sum(v == 'complete' for v in status.values())} complete, "
          f"{sum(v == 'partial' for v in status.values())} partial, {sum(v == 'pending' for v in status.values())} "
          f"pending (from `{a.dir}/`). {label}. Means over the rows measured in all three conditions; ttft is the time "
          "to the first generated token, tpot the time per output token after the first.\n",
          "| mode | model | rows | condition | in tokens | out tokens | ttft (ms) | tpot (ms) | total (ms) | warm-up mean (ms) |",
          "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        for c in CONDS:
            md.append(f"| {r['mode']} | {r['model']} | {r['rows']} | {c} | {r[f'{c}_in_tokens']} | {r[f'{c}_out_tokens']} | "
                      f"{r[f'{c}_ttft_ms']} | {r[f'{c}_tpot_ms']} | {r[f'{c}_total_ms']} | {r[f'{c}_warmup_mean_ms']} |")
    md += ["", "| mode | model | attached / base tpot | merged / base tpot | attached / base total | "
           "merged / base total | merged = attached text (%) |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['mode']} | {r['model']} | {r['attached_to_base_tpot']} | {r['merged_to_base_tpot']:.3f} | "
                  f"{r['attached_to_base_total']} | {r['merged_to_base_total']:.3f} | {r['merged_vs_attached_exact_match_pct']} |")
    if hook:
        by = defaultdict(dict)
        for h in hook:
            by[h["row"]][h["hook"]] = h["total_ms"]
        pairs = [(v[True], v[False]) for v in by.values() if True in v and False in v]
        if pairs:
            on, off = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
            fired = all(h["hook_calls"] and h["hook_calls"] > 0 for h in hook if h["hook"])
            md += ["", f"First-token hook check ({len(pairs)} rows, attached condition): mean total {on:.1f} ms with the "
                   f"hook, {off:.1f} ms without ({100 * (on - off) / off:+.2f} %); hook fired on every request: {fired}."]
    if rows:
        at = [r["attached_to_base_tpot"] for r in rows]
        mg = [r["merged_to_base_tpot"] for r in rows]
        em = [r["merged_vs_attached_exact_match_pct"] for r in rows]
        md += ["", "## Conclusions (data shown above only)", "",
               f"- Time per output token, attached adapter / base: {min(at):.2f} to {max(at):.2f} across {len(rows)} cells.",
               f"- Time per output token, merged adapter / base: {min(mg):.3f} to {max(mg):.3f}.",
               "- Merged and attached generate the same text on " + "; ".join(
                   f"{min(v):.1f} to {max(v):.1f} % of rows ({TITLE[mode].lower()})" for mode in MODES
                   if (v := [r["merged_vs_attached_exact_match_pct"] for r in rows if r["mode"] == mode])) + ".",
               "- This compares conditions on the same validation rows in this implementation (Unsloth, bf16, one gcd); "
               "it does not show deployment latency or behaviour of other runtimes.",
               "- Absolute times are not comparable with the official test profile, which ran eight processes per node."]
        md += extra_lines(meas_by)
    (HERE / "summary.md").write_text("\n".join(md) + "\n")
    # paper table: first-token time and time per output token, base and attached; merged only as a note
    head = ["Mode", "Model", "TTFT base", "TTFT attached", "TPOT base", "TPOT attached", "TPOT ratio (attached/base)"]
    table = [[TITLE[r["mode"]], NAME[r["model"]], f"{r['base_ttft_ms']:.1f}", f"{r['attached_ttft_ms']:.1f}",
              f"{r['base_tpot_ms']:.1f}", f"{r['attached_tpot_ms']:.1f}", f"{r['attached_to_base_tpot']:.2f}"] for r in rows]
    mg = [r["merged_to_base_tpot"] for r in rows]
    notes = [f"{label[0].upper()}{label[1:]}; times in ms.",
             f"Merged/base TPOT ranges from {min(mg):.3f} to {max(mg):.3f} in all {len(rows)} cells."]
    write_tables("paper_table", head, table, notes)
    print(f"{len(rows)} cell(s) summarised; status {dict((f'{k[0]}:{k[1]}', v) for k, v in status.items())}")


if __name__ == "__main__":
    main()
