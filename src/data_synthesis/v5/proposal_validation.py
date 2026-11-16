"""shared machinery for validating a logistics calibration proposal in isolation

a proposal writes its own calibration csv, logistics profile, orders and metadata into its
own directory, runs the full v5 diagnostic on them, and is compared with production.
nothing here writes data/synthetic/v5 or the production agent_calibration.csv, and
production_hashes lets a caller prove that before and after a run
"""

from src.data_synthesis.v5 import agent_calibration, diagnostics, logistics, orders
from src.data_synthesis.v5.settings import AGENT_CALIBRATION_PATH, V5_DIR, rel, sha256_file, write_csv

import pandas as pd

CALIBRATION_COLUMNS = ("agent_id", "calibration_corner", "calibration_reading_id", "fixed_charge_eur",
                       "alpha_eur_per_kg", "beta_eur_per_tonne_km", "fuel_surcharge_series_id",
                       "min_shipment_weight_kg", "max_shipment_weight_kg", "max_weight_inclusive",
                       "oversized_above_kg", "oversized_fee_eur")
PRODUCTION_FILES = (AGENT_CALIBRATION_PATH, agent_calibration.REASONING_PATH)


def production_hashes():
    files = sorted(p for p in V5_DIR.iterdir() if p.is_file()) + list(PRODUCTION_FILES)
    return {rel(p): sha256_file(p) for p in files}


def regenerate(out_dir, calibration_rows):
    """calibration csv, logistics profile, orders and metadata for a proposal, in out_dir only"""
    if out_dir.resolve() == V5_DIR.resolve():
        raise ValueError("a proposal must never be written into data/synthetic/v5")
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_path, profile_path = out_dir / "agent_calibration.csv", out_dir / "logistics_agents.csv"
    write_csv(pd.DataFrame([{c: row[c] for c in CALIBRATION_COLUMNS} for row in calibration_rows]), calibration_path)
    logistics.build(calibration_path, profile_path)
    orders.build(out_dir, profile_path, calibration_path)
    return profile_path, out_dir / orders.OUTPUT_PATH.name


def diagnose(label, report_dir, logistics_profile_path=None, orders_path=None):
    """the full v5 diagnostic, always on the production materials, manufacturers and products"""
    data = diagnostics.Dataset(V5_DIR, logistics_profile_path, orders_path)
    title = f"BID COMPETITION DIAGNOSTICS - v5 logistics calibration: {label}"
    return diagnostics.write_reports(data, diagnostics.run(data), report_dir / label, title)


def summarize(report, agent_ids):
    wins = {a: int(report["wins_lowest_cost"]["logistics"].get(a, 0)) for a in agent_ids}
    shares = {a: float(report["win_share_lowest_cost"]["logistics"].get(a, 0.0)) for a in agent_ids}
    top = max(shares.items(), key=lambda kv: kv[1])
    return {"feasible_orders": report["feasible_orders"],
            "feasibility_pct": round(report["feasible_orders"] / report["orders"] * 100, 1),
            "high_quality_feasibility_pct": report["feasibility_by_scenario_pct"]["high_quality"],
            "orders_with_no_eligible_logistics_bidder": report["logistics_zero_bidders"],
            "zero_win_logistics_agents": sorted(a for a, n in wins.items() if n == 0),
            "largest_logistics_share": {"agent": top[0], "share_pct": top[1]},
            "weight_domain_violations": report["weight_domain_violations"],
            "logistics_win_share_lowest_cost_pct": shares}


def compare(baseline, variant, agent_ids):
    """the logistics numbers a proposal is judged on, one report against another"""
    before, after = summarize(baseline, agent_ids), summarize(variant, agent_ids)
    return {
        "baseline": before, "variant": after,
        "share_change_points": {a: round(after["logistics_win_share_lowest_cost_pct"][a]
                                         - before["logistics_win_share_lowest_cost_pct"][a], 1) for a in agent_ids},
        "reduces_largest_share": after["largest_logistics_share"]["share_pct"]
                                 < before["largest_logistics_share"]["share_pct"],
        "reduces_zero_win_agents": len(after["zero_win_logistics_agents"]) < len(before["zero_win_logistics_agents"]),
        "change_in_orders_with_no_logistics_bidder": (after["orders_with_no_eligible_logistics_bidder"]
                                                      - before["orders_with_no_eligible_logistics_bidder"]),
    }


def render_side_by_side(reports, agent_ids):
    """win shares and headline numbers, one column per report"""
    names = list(reports)
    summaries = {n: summarize(reports[n], agent_ids) for n in names}
    width = max(14, max(len(n) for n in names) + 2)
    lines = ["lowest-cost logistics win share, %", f"  {'agent':34s}" + "".join(f"{n:>{width}s}" for n in names)]
    for agent in agent_ids:
        lines.append(f"  {agent:34s}" + "".join(
            f"{summaries[n]['logistics_win_share_lowest_cost_pct'][agent]:>{width}.1f}" for n in names))
    lines.append("")
    for label, key in (("feasible orders", "feasible_orders"), ("feasibility %", "feasibility_pct"),
                       ("high_quality feasibility %", "high_quality_feasibility_pct"),
                       ("orders with no logistics bidder", "orders_with_no_eligible_logistics_bidder"),
                       ("weight domain violations", "weight_domain_violations")):
        lines.append(f"  {label:34s}" + "".join(f"{summaries[n][key]:>{width}}" for n in names))
    lines.append(f"  {'zero-win logistics agents':34s}" + "".join(
        f"{','.join(summaries[n]['zero_win_logistics_agents']) or '-':>{width}s}" for n in names))
    lines.append(f"  {'largest logistics share':34s}" + "".join(
        f"{summaries[n]['largest_logistics_share']['agent'] + ' ' + str(summaries[n]['largest_logistics_share']['share_pct']):>{width}s}"
        for n in names))
    return lines


def render_band_table(name, report, agent_ids):
    bands = [label for _, _, label in diagnostics.WEIGHT_BANDS]
    table = report["logistics_wins_by_agent_and_weight"]
    lines = [f"lowest-cost logistics wins by agent and weight band: {name}",
             f"  {'agent':6s}{'domain kg':18s}" + "".join(f"{b:>17s}" for b in bands)]
    for agent in agent_ids:
        domain = report["logistics_weight_domains_kg"][agent]
        described = diagnostics.describe_domain((domain["min"], domain["max"], domain["max_inclusive"]))
        lines.append(f"  {agent:6s}{described:18s}" + "".join(f"{table.get(agent, {}).get(b, 0):17d}" for b in bands))
    return lines
