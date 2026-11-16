"""
how long replies actually take, read off a session log

  python -m src.api.timing                        the newest session log
  python -m src.api.timing .run/logs/session_*.jsonl

this is where the window numbers in settings.py come from. the relay stamps
every event with its own arrival time, so the offset from a stage opening to the
last bid inside it is measured rather than assumed. run the session with the
window wide open (DEMO_WINDOW_<CONFIG>) or a slow reply is dropped before it can
be measured, which is exactly the mistake the numbers are meant to prevent
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.api import settings

# a window has to clear the slowest reply with room to spare. inference time
# moves with what else the card is doing, so the headroom is a ratio not a
# constant, and it is rounded up to the next half second
HEADROOM = 1.35


def load(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def newest_log():
    logs = sorted(settings.LOG_DIR.glob("session_*.jsonl"))
    if not logs:
        raise SystemExit(f"no session logs in {settings.LOG_DIR}")
    return logs[-1]


def at(event):
    return datetime.fromisoformat(event["at"])


def stage_windows(events):
    """(config, stage, round) -> how long each reply took to arrive

    measured from the rfq that asked for it, not from the stage opening: an
    auction stage opens once and then publishes one rfq per round, so anchoring
    on the opening would charge round 3 for the two windows before it"""
    asked = {}
    offsets = defaultdict(list)
    counts = defaultdict(list)
    for event in events:
        config = event.get("config")
        if event["type"] == "rfq":
            rnd = event["data"].get("round_number")
            asked[(config, event["data"].get("stage") or event.get("stage"),
                   event["order_id"], rnd)] = at(event)
        elif event["type"] == "bid":
            rnd = event.get("round_number")
            start = asked.get((config, event["stage"], event["order_id"], rnd))
            if start is None:
                continue
            offsets[(config, event["stage"], rnd)].append((at(event) - start).total_seconds())
        elif event["type"] == "stage_closed":
            counts[(config, event["data"]["stage"])].append(event["data"]["bid_count"])
    return offsets, counts


def report(events):
    offsets, counts = stage_windows(events)
    if not offsets:
        print("no bids in this log")
        return

    print(f"{'config':<15}{'stage':<11}{'round':>6}{'bids':>6}{'slowest':>9}{'median':>8}")
    worst = defaultdict(float)
    for (config, stage, rnd), values in sorted(offsets.items(), key=lambda kv: str(kv[0])):
        values.sort()
        slowest = values[-1]
        median = values[len(values) // 2]
        worst[config] = max(worst[config], slowest)
        print(f"{config:<15}{stage:<11}{rnd if rnd else '-':>6}{len(values):>6}"
              f"{slowest:>9.2f}{median:>8.2f}")

    print(f"\n{'config':<8}{'slowest reply':>15}{'x' + str(HEADROOM):>8}{'window':>9}"
          f"{'  configured':>14}")
    for config, slowest in sorted(worst.items()):
        suggested = max(0.5, round(slowest * HEADROOM * 2 + 0.499) / 2)
        configured = settings.WINDOWS.get(config, "?")
        print(f"{config:<15}{slowest:>15.2f}{slowest * HEADROOM:>8.2f}{suggested:>9.1f}"
              f"{configured:>14}")

    per_stage = {stage: max(c) for (config, stage), values in counts.items()
                 for c in [values]}
    print("\nmost bids collected in one stage: "
          + ", ".join(f"{stage} {n}" for stage, n in sorted(per_stage.items())))


def main():
    paths = sys.argv[1:] or [newest_log()]
    events = [event for path in paths for event in load(path)]
    print(f"{len(events)} events from {len(paths)} log(s)\n")
    report(events)


if __name__ == "__main__":
    main()
