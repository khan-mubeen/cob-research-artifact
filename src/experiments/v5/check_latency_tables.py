"""consistency check of the published latency tables, standard library only

reads tables/latency_breakdown_*.csv, the four latency_breakdown_table* tables (md and tex) and
tables/tokens_summary.md, recomputes the relations between them, prints one line per check and exits
non-zero on any mismatch.

  python src/experiments/v5/check_latency_tables.py [--tables reports/v5/evaluation/tables]
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "config.py").exists())
MODES = ("sealed", "reverse_auction", "supervisor")
MODELS = ("gemma", "granite", "ministral", "phi", "qwen", "smollm3")
TITLE = {"sealed": "Sealed bid", "reverse_auction": "Reverse auction", "supervisor": "Supervisor"}
NAME = {"gemma": "Gemma", "granite": "Granite", "ministral": "Ministral", "phi": "Phi", "qwen": "Qwen",
        "smollm3": "SmolLM3"}
STEMS = ("table1_tokens", "table2_times", "table3_warm_up", "table4_decomposition")


def md_rows(path):
    """body rows of a markdown table, keyed by (mode title, model name)"""
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in path.read_text().splitlines()
            if line.startswith("| ") and not line.startswith("| Mode")]
    return {(r[0], r[1]): r[2:] for r in rows}


def tex_rows(path):
    rows = [[c.strip() for c in line.rstrip(" \\").split(" & ")] for line in path.read_text().splitlines()
            if " & " in line and not line.startswith("Mode")]
    return {(r[0], r[1]): r[2:] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(REPO / "reports" / "v5" / "evaluation" / "tables"))
    t = Path(ap.parse_args().tables)
    arms = {(r["mode"], r["model"], r["arm"]): r for r in csv.DictReader(open(t / "latency_breakdown_arms.csv"))}
    cells = {(r["mode"], r["model"]): r for r in csv.DictReader(open(t / "latency_breakdown_cells.csv"))}
    md = {s: md_rows(t / f"latency_breakdown_{s}.md") for s in STEMS}
    tex = {s: tex_rows(t / f"latency_breakdown_{s}.tex") for s in STEMS}
    keys = [(mode, m) for mode in MODES for m in MODELS]
    rows = {(mode, m): (TITLE[mode], NAME[m]) for mode, m in keys}
    f = lambda v: float(v.replace(",", ""))  # noqa: E731
    checks = []

    def check(name, bad):
        checks.append(not bad)
        print(f"{'ok      ' if not bad else 'MISMATCH'} {name}" + (f": {bad[:3]}" if bad else ""))

    def agrees(shown, raw, d):
        """shown (d decimals, rounded from the unrounded value) is consistent with the csv value: equal at d
        decimals, or either neighbour when the csv value lies exactly half way (a tie at the csv's own precision)"""
        k = len(raw.split(".")[1]) if "." in raw else 0
        if shown.count(".") != (d > 0) or (d and len(shown.split(".")[1]) != d):
            return False
        if k <= d:
            return shown == f"{float(raw):.{d}f}"
        return abs(float(shown) - float(raw)) <= 0.5 * 10 ** -d + 0.5 * 10 ** -k + 1e-9

    check("tex equals md in tables 1 to 4", [s for s in STEMS if md[s] != tex[s]])
    check("every table has the 18 cells", [s for s in STEMS if set(md[s]) != set(rows.values())])
    # tables against the csv files at the shown precision
    spec = {"table1_tokens": [("a", "in_tokens", "base", 1), ("a", "out_tokens", "base", 1), ("a", "out_tokens", "selected", 1)],
            "table2_times": [("c", k, None, 1) for k in ("ttft_base", "ttft_ft", "tpot_base", "tpot_ft", "total_base",
                                                          "total_ft")] + [("c", "tpot_ratio", None, 2)],
            "table3_warm_up": [("a", "warmup_mean", "base", 1), ("a", "warmup_mean", "selected", 1),
                               ("a", "warmup_ratio", "base", 2), ("a", "warmup_ratio", "selected", 2),
                               ("a", "warmup_rows", "base", 0), ("a", "warmup_rows", "selected", 0)],
            "table4_decomposition": [("c", k, None, 1) for k in ("difference", "ttft_part", "length_effect",
                                                                  "per_token_effect")]}
    for s, fields in spec.items():
        bad = []
        for k in keys:
            raw = [arms[(*k, arm)][x] if src == "a" else cells[k][x] for src, x, arm, d in fields]
            shown = md[s][rows[k]]
            if len(shown) != len(fields) or not all(agrees(v, r, d) for v, r, (_, _, _, d) in zip(shown, raw, fields)):
                bad.append((k, shown, raw))
        check(f"{s} equals the csv files at the shown precision", bad)
    t2, t4 = md["table2_times"], md["table4_decomposition"]
    check("TPOT ratio = fine-tuned / base (csv, 3 decimals)",
          [k for k in keys if abs(float(cells[k]["tpot_ft"]) / float(cells[k]["tpot_base"]) - float(cells[k]["tpot_ratio"])) > 0.0011])
    check("the three terms sum to the latency difference (table 4, within 0.25 ms)",
          [k for k in keys if abs(sum(f(x) for x in t4[rows[k]][1:]) - f(t4[rows[k]][0])) > 0.25])
    check("the latency difference equals the table 2 latency gap (within 0.15 ms)",
          [k for k in keys if abs((f(t2[rows[k]][5]) - f(t2[rows[k]][4])) - f(t4[rows[k]][0])) > 0.15])
    check("the TTFT part equals the table 2 TTFT gap (within 0.15 ms)",
          [k for k in keys if abs((f(t2[rows[k]][1]) - f(t2[rows[k]][0])) - f(t4[rows[k]][1])) > 0.15])
    check("warm-up ratio = warm-up mean / steady mean (csv, 3 decimals)",
          [k for k in arms if abs(float(arms[k]["warmup_mean"]) / float(arms[k]["total_mean"]) - float(arms[k]["warmup_ratio"])) > 0.0011])
    slower_md = sum(f(t2[rows[k]][5]) > f(t2[rows[k]][4]) for k in keys)
    slower_csv = sum(float(cells[k]["total_ft"]) > float(cells[k]["total_base"]) for k in keys)
    check(f"cells with the fine-tuned model slower than base: {slower_md} of 18 (table 2) and {slower_csv} (csv)",
          [] if slower_md == slower_csv else ["counts differ"])
    largest = lambda r: max(range(3), key=lambda i: abs(f(r[1 + i])))  # noqa: E731
    per_token_md = sum(largest(t4[rows[k]]) == 2 for k in keys)
    per_token_csv = sum(max(("ttft_part", "length_effect", "per_token_effect"), key=lambda x: abs(float(cells[k][x])))
                        == "per_token_effect" for k in keys)
    check(f"cells where the per token effect is the largest term: {per_token_md} of 18 (table 4) and "
          f"{per_token_csv} (csv)", [] if per_token_md == per_token_csv else ["counts differ"])
    ts = {(r[0], r[1].lower()): r[2:] for r in [[c.strip() for c in line.strip().strip("|").split("|")]
          for line in (t / "tokens_summary.md").read_text().splitlines() if line.startswith("| ") and not line.startswith("| Mode")]}
    check("table 1 equals tokens_summary (input, output base, output fine-tuned)",
          [k for k in keys if [f(x) for x in md["table1_tokens"][rows[k]]] != [f(x) for x in ts[(rows[k][0], k[1])][:3]]])
    print(f"{sum(checks)} of {len(checks)} checks pass")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
