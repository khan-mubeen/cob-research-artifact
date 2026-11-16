"""derive the relocation mapping from the cleanup plan's Appendices A and B.

the plan is authoritative. this parser re-derives its tables as structured data
so no destination is ever retyped by hand. output feeds relocate.py.

stage numbers are assigned from the DESTINATION path, not by hand-listing files,
so the staged execution order is a property of the plan's own hierarchy:

  2 models/base-checkpoints/**            base snapshots + base GGUFs
  3 models/epoch-1/production/**          six v2 adapters + Qwen r1
  4 models/epoch-1/ablation/**            Qwen r2/r3, Granite v1 (renamed)
  5 models/epoch-1/deployment/{qwen-r1-gguf-fix,granite-v1}/**
  6 models/epoch-1/deployment/quarantine/**   broken Qwen conversion
  7 models/feasibility/smoke-artifacts/**     delete-hold isolation
  8 reports/**                            Appendix B
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*$")


def stage_for(dest: str) -> int | None:
    d = dest.replace("\\", "/")
    if d.startswith("models/base-checkpoints/"):
        return 2
    if d.startswith("models/epoch-1/production/"):
        return 3
    if d.startswith("models/epoch-1/ablation/"):
        return 4
    if d.startswith("models/epoch-1/deployment/quarantine/"):
        return 6
    if d.startswith("models/epoch-1/deployment/"):
        return 5
    if d.startswith("models/feasibility/smoke-artifacts/"):
        return 7
    if d.startswith("reports/"):
        return 8
    if d.startswith("prompt-reports/"):
        return 9  # conversational meta-report, moves to the ignored area
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = Path(args.plan).read_text(encoding="utf-8").splitlines()
    # locate appendix bounds so we never pick up tables from sections 3/4
    bounds = {}
    for i, ln in enumerate(lines):
        if ln.startswith("## Appendix A"):
            bounds["A"] = i
        elif ln.startswith("## Appendix B"):
            bounds["B"] = i
        elif ln.startswith("## Appendix C"):
            bounds["C"] = i

    entries, skipped, unmapped = [], [], []
    for label, lo, hi in (("A", bounds["A"], bounds["B"]), ("B", bounds["B"], bounds["C"])):
        for ln in lines[lo:hi]:
            m = ROW.match(ln)
            if not m:
                continue
            old, new, disp = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if old.lower() == "current file":
                continue
            if disp.upper().startswith("UNCHANGED") or old == new:
                skipped.append({"path": old, "reason": disp})
                continue
            stage = stage_for(new)
            if stage is None:
                unmapped.append({"old_path": old, "new_path": new, "disposition": disp})
                continue
            entries.append({"stage": stage, "old_path": old, "new_path": new,
                            "appendix": label, "disposition": "move",
                            "plan_disposition": disp})

    entries.sort(key=lambda e: (e["stage"], e["new_path"]))
    Path(args.out).write_text(json.dumps(entries, indent=1), encoding="utf-8")

    print(f"parsed rows -> {len(entries)} moves | {len(skipped)} unchanged | "
          f"{len(unmapped)} unmapped")
    by_app = {}
    for e in entries:
        by_app[e["appendix"]] = by_app.get(e["appendix"], 0) + 1
    print(f"  Appendix A (models): {by_app.get('A', 0)}   Appendix B (reports): {by_app.get('B', 0)}")
    print("  by stage:", {s: sum(1 for e in entries if e["stage"] == s)
                          for s in sorted({e["stage"] for e in entries})})
    for s in skipped:
        print(f"  UNCHANGED: {s['path']}")
    for u in unmapped:
        print(f"  UNMAPPED : {u['old_path']} -> {u['new_path']} [{u['disposition']}]")

    # existence pre-check against the live filesystem
    missing = [e["old_path"] for e in entries if not (REPO / e["old_path"]).exists()]
    print(f"\nsources missing on disk: {len(missing)}")
    for p in missing[:20]:
        print(f"  MISSING: {p}")


if __name__ == "__main__":
    main()
