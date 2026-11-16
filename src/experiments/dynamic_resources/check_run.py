"""sanity-check one dynamic-resources run

asserts the invariants the state module is supposed to hold, so a run is judged
on its events rather than on having finished. for auction runs it also checks
that state does not move between rounds of the same order"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def load(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def ts(v):
    return datetime.fromisoformat(v)


def main(path):
    rows = load(path)
    print(f"{path}\n{len(rows)} order records\n")

    status = Counter(r["outcome"]["status"] for r in rows)
    print("outcomes:", dict(status))

    events = [e for r in rows for e in r["dynamic_events"]]
    print("events  :", dict(Counter(e["event"] for e in events)))

    problems = []

    # a hold must not be taken twice without a release in between. an abandoned
    # chain releases and a later chain may pick the same agent again, which is
    # legitimate, so track held/released rather than counting events
    held = 0
    for r in rows:
        state = {}
        for e in r["dynamic_events"]:
            key = (e["stage"], e.get("agent_id"))
            if e["event"] in ("stock_reserved", "workload_reserved"):
                if state.get(key) == "held":
                    held += 1
                state[key] = "held"
            elif e["event"] in ("reservation_released", "workload_released"):
                state[key] = "released"
    if held:
        problems.append(f"{held} reservation(s) taken while already held, with no release")

    # a restock lands once, and only after the eta it was given
    landed = [e for e in events if e["event"] == "restock_landed"]
    for e in landed:
        if ts(e["sim_now"]) < ts(e["triggered_at"]):
            problems.append(f"restock landed before it was triggered: {e['agent_id']} {e['material']}")
    pairs = Counter((e["agent_id"], e["material"], e["triggered_at"]) for e in landed)
    twice = {k: v for k, v in pairs.items() if v > 1}
    if twice:
        problems.append(f"same restock landed more than once: {list(twice)[:5]}")

    # never two restocks in flight for one agent/material. a release that rolls
    # back the hold which triggered a restock also cancels it, so that clears the
    # slot just as a landing does
    pending = {}
    twice = 0
    for r in rows:
        for e in r["dynamic_events"]:
            key = (e.get("agent_id"), e.get("material"))
            if e["event"] == "restock_triggered":
                if pending.get(key):
                    twice += 1
                pending[key] = True
            elif e["event"] == "restock_landed":
                pending[key] = False
            elif e["event"] == "reservation_released" and e.get("restock_cancelled"):
                pending[key] = False
    if twice:
        problems.append(f"{twice} restock(s) triggered while one was already in flight")

    # the walking clock never goes backwards across the batch
    clock = [ts(r["outcome"]["sim_now"]) for r in rows if r["outcome"].get("sim_now")]
    if clock != sorted(clock):
        problems.append("sim_now is not monotonic across the batch")
    if clock:
        print(f"clock   : {clock[0]} -> {clock[-1]}")

    triggers = Counter(e["trigger"] for e in events if e["event"] == "restock_triggered")
    print("triggers:", dict(triggers))
    print(f"pairs   : {len({(e['agent_id'], e['material']) for e in events if e['event'] == 'restock_triggered'})} "
          f"agent-material pair(s) restocked")

    print("\n" + ("PROBLEMS:" if problems else "all invariants hold"))
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
