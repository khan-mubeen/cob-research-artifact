"""flatten the evaluation reports into the small, replay-ready JSON assets the
static frontend in ui/ reads at build/dev time. no live model calls happen
here or in the frontend.

two bidding scenarios feed the Auction Replay graph, from two differently
shaped report families:

  reverse_auction   reports/reverse_auction/epoch-1/v4/evaluation/raw/
                     rows carry round_number/current_best_bid -- a real
                     multi-round negotiation, up to 3 rounds per stage.
                     category filter: these rows mix a real multi-round
                     narrative (category == "auction_round") with synthetic
                     floor-boundary probes that reuse the same order/agent/
                     round id with an artificially injected current_best_bid
                     to test edge behavior. mixing them into one replay would
                     show contradictory best-bid values inside a single
                     round, so the replay export keeps auction_round rows
                     only. the floor-boundary rows are exactly what the
                     reverse-auction `gates` already summarize, so they still
                     surface in the dashboard.

  sealed             reports/epoch-2/production/evaluation/raw/, the
                     "main_holdout" suite specifically. rows carry no
                     round_number at all -- sealed bidding is a single
                     simultaneous submission, not a negotiation, so every
                     row for one order+stage becomes that stage's one and
                     only round. main_holdout is used (not the other 8
                     suites in this report) because it's the large,
                     representative sample -- the analogue of
                     reverse_auction's "auction_round" category -- while the
                     other suites are deliberately small adversarial/edge
                     probes meant for the dashboard's gates, not a replay
                     narrative.

both scenarios share one roster (data/synthetic/v4) and both are keyed by
the same 6 architecture names in the Auction Replay model picker; only the
Evaluation Dashboard still shows every base/tuned variant.
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVERSE_AUCTION_RAW = PROJECT_ROOT / "reports" / "reverse_auction" / "epoch-1" / "v4" / "evaluation" / "raw"
EPOCH2_RAW = PROJECT_ROOT / "reports" / "epoch-2" / "production" / "evaluation" / "raw"
EPOCH2_COMPARISON = PROJECT_ROOT / "reports" / "epoch-2" / "production" / "comparison"
OUT_DIR = PROJECT_ROOT / "ui" / "frontend" / "public" / "data"

# same version + filenames supervisor.py:_load_stage_rosters reads for the
# live NATS roster -- see src/supervisor/supervisor.py:STAGE_AGENT_FILES
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "v4"
STAGE_AGENT_FILES = {
    "material": "material_agents.csv",
    "tailoring": "manufacturer_agents.csv",
    "logistics": "logistics_agents.csv",
}

# the 6 architectures, used to key the Auction Replay model picker across
# both scenarios even though the two report families tag them differently
# (ra-e1-{arch} vs {arch}-e2)
ARCHES = ["gemma", "granite", "ministral", "phi", "qwen", "smollm3"]

AUCTION_ROUND = "auction_round"


def _latest_per_tag(raw_dir: Path) -> dict[str, Path]:
    """one file per tag, keeping the newest timestamp if a tag has several"""
    by_tag: dict[str, Path] = {}
    for path in sorted(raw_dir.glob("eval_*.json")):
        tag = path.stem.split("_", 1)[1].rsplit("_", 2)[0]
        by_tag[tag] = path  # sorted by name -> later timestamp wins
    return by_tag


def _parse_raw_payload(row: dict) -> dict:
    try:
        return json.loads(row.get("raw", "{}"))
    except json.JSONDecodeError:
        return {}


def _bid_from_row(row: dict) -> dict:
    payload = _parse_raw_payload(row)
    return {
        "agent_id": row["id"].split("/", 1)[1],
        "label_decision": row.get("label_decision"),
        "model_decision": row.get("model_decision"),
        "decision_matched": row.get("decision_matched"),
        "offer_amount": payload.get("offer_amount"),
        "estimated_completion_days": payload.get("estimated_completion_days"),
        "defect_probability": payload.get("defect_probability"),
        "offer_pct_dev": row.get("offer_pct_dev"),
        "days_matched": row.get("days_matched"),
        "withdrawn_in_earlier_round": row.get("withdrawn_in_earlier_round", False),
    }


def _winner_from_standing(standing_offer_by_agent: dict[str, float]) -> tuple[str | None, float | None]:
    if not standing_offer_by_agent:
        return None, None
    winner = min(standing_offer_by_agent, key=lambda a: standing_offer_by_agent[a])
    return winner, standing_offer_by_agent[winner]


def build_stage_auction(rows: list[dict]) -> dict:
    """one stage's reverse-auction (material, tailoring, or logistics) for
    one order -- a chain-of-bid order runs all three in sequence, material
    first, each stage's winner feeding the next
    (src/supervisor/supervisor.py:218-256)"""
    rows = sorted(rows, key=lambda r: r["round_number"])
    decrement = rows[0].get("platform_minimum_decrement")

    rounds_by_number: dict[int, list[dict]] = {}
    for row in rows:
        rounds_by_number.setdefault(row["round_number"], []).append(row)

    rounds = []
    for round_number in sorted(rounds_by_number):
        round_rows = rounds_by_number[round_number]
        rounds.append({
            "round_number": round_number,
            "announced_best_bid": round_rows[0].get("current_best_bid"),
            "bids": [_bid_from_row(row) for row in round_rows],
        })

    # the winner is the lowest offer still standing across every round, not
    # just whichever round happens to be sampled last: the live auction
    # carries each agent's latest bid forward as "standing" even in rounds
    # they don't reply again (src/supervisor/auction.py:156,
    # `standing.update(received)`), and our eval rows are a sample -- the
    # final sampled round for one order/stage may only contain a different
    # agent's later NO_BID, which must not erase an earlier real bid
    standing_offer_by_agent: dict[str, float] = {}
    for round_data in rounds:
        for bid in round_data["bids"]:
            if bid["model_decision"] == "BID" and bid["offer_amount"] is not None:
                standing_offer_by_agent[bid["agent_id"]] = bid["offer_amount"]
    winner_agent_id, winner_offer = _winner_from_standing(standing_offer_by_agent)

    return {
        "platform_minimum_decrement": decrement,
        "rounds": rounds,
        "winner_agent_id": winner_agent_id,
        "closing_offer": winner_offer,
    }


def build_stage_auction_sealed(rows: list[dict]) -> dict:
    """sealed bidding has no rounds -- it's one simultaneous submission, so
    every row recorded for an order+stage becomes that stage's single round.
    no current_best_bid/platform_minimum_decrement exist in this mode"""
    bids = [_bid_from_row(row) for row in rows]
    rounds = [{"round_number": 1, "announced_best_bid": None, "bids": bids}]

    standing_offer_by_agent = {
        b["agent_id"]: b["offer_amount"] for b in bids
        if b["model_decision"] == "BID" and b["offer_amount"] is not None
    }
    winner_agent_id, winner_offer = _winner_from_standing(standing_offer_by_agent)

    return {
        "platform_minimum_decrement": None,
        "rounds": rounds,
        "winner_agent_id": winner_agent_id,
        "closing_offer": winner_offer,
    }


def _group_by_order_and_stage(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    by_order: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        order_id = row["id"].split("/", 1)[0]
        by_order.setdefault(order_id, {}).setdefault(row["stage"], []).append(row)
    return by_order


def build_replay_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    auction_rows = [r for r in data["rows"] if r.get("category") == AUCTION_ROUND]
    by_order = _group_by_order_and_stage(auction_rows)
    priority_by_order = {row["id"].split("/", 1)[0]: row.get("priority") for row in auction_rows}

    orders = [
        {
            "order_id": order_id,
            "priority": priority_by_order.get(order_id),
            "stages": {stage: build_stage_auction(rows) for stage, rows in stage_rows.items()},
        }
        for order_id, stage_rows in sorted(by_order.items())
    ]

    summary = data.get("summary") or {}
    return {
        "tag": summary.get("tag"),
        "model": summary.get("model"),
        "generated_at": summary.get("generated_at"),
        "orders": orders,
    }


def build_replay_file_sealed(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = (data.get("main_holdout") or {}).get("rows") or []
    by_order = _group_by_order_and_stage(rows)

    orders = [
        {
            "order_id": order_id,
            "priority": None,  # not recorded for sealed rows
            "stages": {stage: build_stage_auction_sealed(rows) for stage, rows in stage_rows.items()},
        }
        for order_id, stage_rows in sorted(by_order.items())
    ]

    return {
        "tag": data.get("tag"),
        "model": data.get("model"),
        "generated_at": data.get("generated_at"),
        "orders": orders,
    }


def build_reverse_auction_dashboard(by_tag: dict[str, Path]) -> list[dict]:
    """summary/generated_at live under data['summary'] in these files (unlike
    the epoch-2 files, where they're top-level), and gates carries an extra
    {"gates": {...}} wrapper -- both unwrapped here so the frontend doesn't
    have to know about the raw report's inconsistent shape"""
    entries = []
    for tag, path in sorted(by_tag.items()):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary") or {}
        gates = (data.get("gates") or {}).get("gates") or {}
        entries.append({
            "tag": summary.get("tag"),
            "model": summary.get("model"),
            "generated_at": summary.get("generated_at"),
            "kind": "reverse_auction",
            "summary": summary,
            "gates": gates,
        })
    return entries


def build_epoch2_dashboard(by_tag: dict[str, Path]) -> list[dict]:
    """the headline `main_holdout_decisions` numbers come straight out of
    reports/epoch-2/production/comparison/{arch}_epoch1_vs_epoch2.json
    (produced by src/experiments/compare_epochs.py) instead of being
    recomputed here -- that comparison file is what people are shown
    directly, so the dashboard must reproduce its exact numbers rather than
    risk a different (lower) figure from averaging across every stress
    suite, most of which are deliberately adversarial edge probes with a
    much harder distribution than main_holdout"""
    entries = []
    for tag, path in sorted(by_tag.items()):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        suites = {
            key: value.get("summary")
            for key, value in data.items()
            if isinstance(value, dict) and "rows" in value
        }

        arch = tag[:-3] if tag.endswith("-e2") else tag
        comparison_path = EPOCH2_COMPARISON / f"{arch}_epoch1_vs_epoch2.json"
        main_holdout_decisions = None
        if comparison_path.exists():
            with open(comparison_path, encoding="utf-8") as f:
                comparison = json.load(f)
            main_holdout_decisions = (comparison.get("arms") or {}).get("epoch_2", {}).get("main_holdout_decisions")

        entries.append({
            "tag": data.get("tag"),
            "model": data.get("model"),
            "generated_at": data.get("generated_at"),
            "kind": "epoch2_production",
            "suites": suites,
            "main_holdout_decisions": main_holdout_decisions,
        })
    return entries


def build_roster() -> dict[str, list[dict]]:
    roster: dict[str, list[dict]] = {}
    for stage, filename in STAGE_AGENT_FILES.items():
        path = SYNTHETIC_DATA_DIR / filename
        agents = []
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                agents.append({
                    "agent_id": row["agent_id"],
                    "location": row.get("location"),
                    "specialty": row.get("specialty"),
                    "reputation_score": float(row["reputation_score"]) if row.get("reputation_score") else None,
                })
        roster[stage] = sorted(agents, key=lambda a: a["agent_id"])
    return roster


def main():
    replay_dir = OUT_DIR / "replay"
    (replay_dir / "reverse_auction").mkdir(parents=True, exist_ok=True)
    (replay_dir / "sealed").mkdir(parents=True, exist_ok=True)

    roster = build_roster()
    with open(OUT_DIR / "roster.json", "w", encoding="utf-8") as f:
        json.dump(roster, f)
    print(f"wrote roster.json ({', '.join(f'{s}={len(a)}' for s, a in roster.items())})")

    ra_by_tag = _latest_per_tag(REVERSE_AUCTION_RAW)
    epoch2_by_tag = _latest_per_tag(EPOCH2_RAW)

    # Auction Replay picker: one file per architecture per scenario, always
    # the tuned (non "-base") reverse-auction variant -- "-base" stays
    # available in the dashboard, where the base/tuned comparison matters,
    # but isn't a meaningful thing to *watch play out* in the replay
    for arch in ARCHES:
        ra_tag = f"ra-e1-{arch}"
        if ra_tag in ra_by_tag:
            replay = build_replay_file(ra_by_tag[ra_tag])
            with open(replay_dir / "reverse_auction" / f"{arch}.json", "w", encoding="utf-8") as f:
                json.dump(replay, f)
            print(f"wrote replay/reverse_auction/{arch}.json ({len(replay['orders'])} orders)")

        e2_tag = f"{arch}-e2"
        if e2_tag in epoch2_by_tag:
            replay = build_replay_file_sealed(epoch2_by_tag[e2_tag])
            with open(replay_dir / "sealed" / f"{arch}.json", "w", encoding="utf-8") as f:
                json.dump(replay, f)
            print(f"wrote replay/sealed/{arch}.json ({len(replay['orders'])} orders)")

    index = {"models": ARCHES, "scenarios": ["sealed", "reverse_auction"]}
    with open(replay_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    dashboard = {
        "reverse_auction": build_reverse_auction_dashboard(ra_by_tag),
        "epoch2_production": build_epoch2_dashboard(epoch2_by_tag),
    }
    with open(OUT_DIR / "dashboard.json", "w", encoding="utf-8") as f:
        json.dump(dashboard, f)
    print(f"wrote dashboard.json ({len(ra_by_tag)} reverse_auction + {len(epoch2_by_tag)} epoch2 tags)")


if __name__ == "__main__":
    main()
