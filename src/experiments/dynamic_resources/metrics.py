"""one metrics row per run, in the format used for cases 1 to 4

reads any number of run files and prints a comparison table. used to report each
model as its pair of runs lands, and to build the final six-model table

usage:
  python -m src.experiments.dynamic_resources.metrics <run.jsonl> [<run.jsonl> ...]
  python -m src.experiments.dynamic_resources.metrics --all
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[3]
DIRS = [REPO / "reports/bidding/sealed/dynamic_resources",
        REPO / "reports/bidding/reverse_auction/dynamic_resources"]


def label(path):
    """mode and agent from the path and filename, no metadata lookup needed"""
    mode = "sealed" if "sealed" in path.parent.name or "/sealed/" in path.as_posix() else "auction"
    stem = path.stem
    if stem.startswith("rule_"):
        who = "rule" + (" 150" if stem.endswith("_first150") else "")
    else:
        parts = stem.split("_")
        who = parts[1] if len(parts) > 2 else "slm"
    return f"{who} {mode}"


def row(path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    st = Counter(r["outcome"]["status"] for r in rows)
    ev = Counter(e["event"] for r in rows for e in r["dynamic_events"])
    tr = Counter(e["trigger"] for r in rows for e in r["dynamic_events"]
                 if e["event"] == "restock_triggered")
    done = [r["outcome"] for r in rows if r["outcome"]["status"] == "completed"]
    pairs = len({(e["agent_id"], e["material"]) for r in rows for e in r["dynamic_events"]
                 if e["event"] == "restock_triggered"})
    return {
        "label": label(path), "n": len(rows),
        "completed": st["completed"], "infeasible": st["infeasible"],
        "other": sum(v for k, v in st.items() if k not in ("completed", "infeasible")),
        "cost_mean": mean(d["total_cost"] for d in done) if done else 0,
        "cost_med": median(d["total_cost"] for d in done) if done else 0,
        "days_mean": mean(d["total_days"] for d in done) if done else 0,
        "trig": sum(tr.values()), "cmt": tr["commitment"], "thr": tr["threshold"],
        "landed": ev["restock_landed"], "blocked": ev["restock_skipped_pending"],
        "rej": ev["reservation_rejected"], "rel": ev["reservation_released"],
        "pairs": pairs,
        "mins": sum(r["elapsed_seconds"] for r in rows) / 60,
    }


def main(argv):
    if argv and argv[0] == "--all":
        paths = sorted(p for d in DIRS if d.is_dir() for p in d.glob("*run*.jsonl"))
    else:
        paths = [Path(a) for a in argv]
    if not paths:
        print("no run files")
        return 1
    rows = [row(p) for p in paths]
    hdr = (f"{'run':<18}{'n':>5}{'done':>6}{'infeas':>7}{'other':>6}{'cost mean':>11}"
           f"{'cost med':>10}{'days':>7}{'trig':>6}{'cmt':>5}{'thr':>5}{'land':>6}"
           f"{'blk':>5}{'rej':>5}{'rel':>5}{'pair':>6}{'min':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<18}{r['n']:>5}{r['completed']:>6}{r['infeasible']:>7}{r['other']:>6}"
              f"{r['cost_mean']:>11.2f}{r['cost_med']:>10.2f}{r['days_mean']:>7.2f}"
              f"{r['trig']:>6}{r['cmt']:>5}{r['thr']:>5}{r['landed']:>6}{r['blocked']:>5}"
              f"{r['rej']:>5}{r['rel']:>5}{r['pairs']:>6}{r['mins']:>6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
