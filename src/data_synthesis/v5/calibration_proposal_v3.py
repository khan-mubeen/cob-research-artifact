"""proposal v3: the freight corner assigned by a blind bijective search, validated in isolation

  PYTHONPATH=. python -m src.data_synthesis.v5.calibration_proposal_v3

a proposal-and-validate pass, not a final regeneration. it writes
  data/processed/v5/logistics/agent_calibration_proposal_v3.json  the assignment and its evidence
  data/processed/v5/calibration_proposal_v3/                      calibration, logistics profile,
                                                                  orders and metadata built from it
  reports/v5/calibration_proposal_v3/                             diagnostics, the full assignment
                                                                  ranking and the comparison
and never writes data/synthetic/v5/ or data/processed/v5/logistics/agent_calibration.csv

part 1  the seven sourced freight candidates, reused exactly from the prior proposal output
        and verified against the pinned records; a difference stops the run
part 2  a calibration demand sample: the v5 order generator run in memory with seed
        CALIBRATION_SEED, reading only order shapes, never a price and never a production order
part 3  demand per (city, candidate) and the 1% meaningful threshold
part 4  every bijective assignment of the seven candidates to the seven freight agents, scored
        by freight_assignment.solve on the calibration demand and the v4 service regions alone
part 5  the parcel corner copied unchanged from production, the isolated regeneration, the full
        diagnostic on the production 50,000-order set, and the comparison with production and
        with the specialty-text proposal on file

no specialty text is read: the freight and parcel agents come from the production calibration's
corner column, and service regions are loaded with the specialty column excluded
"""

import csv
import json
import sys
from collections import Counter

import pandas as pd

from src.data_synthesis.v5 import freight_assignment, freight_candidates, orders, proposal_validation
from src.data_synthesis.v5 import products as products_module
from src.data_synthesis.v5.settings import (
    AGENT_CALIBRATION_PATH, BETA_EUR_PER_TONNE_KM, PROCESSED_V5, REPORTS_V5, SEED, SHARED_FUEL_SERIES_ID, V4_DIR,
    V5_DIR, rel, sha256_file, write_json,
)

# arbitrary: its only role is to differ from the production seed 42, it carries no meaning
CALIBRATION_SEED = 7

FREIGHT_AGENTS = ("L01", "L02", "L04", "L05", "L08", "L09", "L11")
PARCEL_AGENTS = ("L03", "L06", "L07", "L10", "L12")

PROPOSAL_PATH = PROCESSED_V5 / "logistics" / "agent_calibration_proposal_v3.json"
DATA_DIR = PROCESSED_V5 / "calibration_proposal_v3"
REPORT_DIR = REPORTS_V5 / "calibration_proposal_v3"
PRODUCTION_ORDERS_PATH = V5_DIR / orders.OUTPUT_PATH.name
SPECIALTY_PROPOSAL_DIR = PROCESSED_V5 / "calibration_proposal" / "proposal"
SPECIALTY_PROPOSAL_REPORT = REPORTS_V5 / "calibration_proposal" / "proposal" / "bid_competition_v5.json"
TOP_RANKS = 25

STATUS = ("proposal awaiting review before it is made final and regenerated for real. production "
          "data/synthetic/v5/ and data/processed/v5/logistics/agent_calibration.csv are untouched by it")

THRESHOLD_BASIS = (
    "a disclosed modeling threshold, not a mathematical derivation: a (city, candidate) cell is meaningful when "
    "its calibration-sample demand is at least 1% of that city's calibration-sample orders. 1% sits below every "
    "deliberately designed stress-scenario share in methodology section 8.2, the smallest being tight_low_budget "
    "at 3%, so it cannot be met by absorbing an intentional stress category. cells below it are rare residuals, "
    "accepted as stress conditions, consistent with section 2's deliberately difficult conditions")

OVERLAP_CONVENTION = (
    "an order counts toward every candidate domain that contains its weight when demand per (city, candidate) is "
    "tabulated and the 1% threshold is applied: meaningfulness belongs to a domain, and picking one domain for an "
    "order before any agent holds one would be an arbitrary attribution. the objective counts orders, not cells: "
    "an order is in scope once if any of its domains is meaningful in its city, and covered once if any eligible "
    "provider can carry it, so overlapping domains never count the same order twice and a domain boundary such "
    "as exactly 1000 kg is judged by the order's exact weight, not by whole-cell containment. the cell-summed "
    "alternative, which would count an order once per meaningful domain containing it, is scored for every "
    "assignment as a sensitivity")


class ShapeOnlyEstimator:
    """stands in for the budget estimator: the calibration sample needs order shapes only, so
    nothing is priced and every budget is a placeholder that no calculation reads"""

    def estimate(self, *order_shape):
        return {"market_price": 1.0}


def fleet():
    """freight service regions and the production parcel rows, with no specialty text loaded"""
    production = pd.read_csv(AGENT_CALIBRATION_PATH)
    corners = dict(zip(production["agent_id"], production["calibration_corner"]))
    expected = {**{a: "freight" for a in FREIGHT_AGENTS}, **{a: "parcel" for a in PARCEL_AGENTS}}
    if corners != expected:
        raise ValueError(f"production corners {corners} differ from the fixed freight and parcel sets")
    regions = pd.read_csv(V4_DIR / "logistics_agents.csv", usecols=["agent_id", "service_regions"])
    service_regions = {row.agent_id: tuple(row.service_regions.split("|"))
                       for row in regions.itertuples() if row.agent_id in FREIGHT_AGENTS}
    cities = sorted({city for value in regions["service_regions"] for city in value.split("|")})
    parcel_rows = production[production["agent_id"].isin(PARCEL_AGENTS)].to_dict("records")
    return service_regions, parcel_rows, cities


def product_pairs():
    return pd.read_csv(products_module.OUTPUT_PATH).to_dict("records")


def calibration_sample(cities, pairs, seed=CALIBRATION_SEED):
    """the v5 order generator with a seed other than production's, generated in memory only"""
    if seed == SEED:
        raise ValueError("the calibration sample must not use the production seed")
    return orders.OrderGenerator(pairs, ShapeOnlyEstimator(), cities, seed=seed).generate()


def piece_counts(order_rows, pairs, candidate_domains):
    """orders per (city, weight piece); the weight is quantity x finished weight, as everywhere in v5"""
    weight = {(p["product_id"], p["material_type"]): p["finished_weight_kg_per_unit"] for p in pairs}
    counts = Counter()
    for order in order_rows:
        kg = order["quantity"] * weight[(order["product_id"], order["material_type"])]
        counts[(order["deliver_to"], freight_assignment.piece_of(kg, candidate_domains))] += 1
    return dict(counts)


def assign_from_calibration():
    """parts 1 to 4. it reads the prior proposal output, the pinned records, the products table,
    the production calibration corners and the v4 service regions, and no order file"""
    candidates = freight_candidates.load_candidates()
    verification = freight_candidates.verify_candidates(candidates)
    domains = freight_candidates.domains(candidates)
    service_regions, parcel_rows, cities = fleet()
    pairs = product_pairs()
    sample = calibration_sample(cities, pairs)
    demand = piece_counts(sample, pairs, domains)
    ranking = freight_assignment.solve(demand, domains, service_regions)
    return {"candidates": candidates, "verification": verification, "domains": domains,
            "service_regions": service_regions, "parcel_rows": parcel_rows, "cities": cities, "pairs": pairs,
            "sample_orders": len(sample), "demand": demand, "ranking": ranking}


# ── reporting the calculation ───────────────────────────────────────────────

def demand_table(demand, domains, cities):
    totals = freight_assignment.city_orders(demand)
    by_cell = freight_assignment.domain_demand(demand, domains)
    meaningful = freight_assignment.meaningful_cells(demand, domains)
    table = {}
    for city in cities:
        table[city] = {"orders": totals[city], "by_candidate": {
            str(c): {"orders": by_cell.get((city, c), 0),
                     "share_pct": round(by_cell.get((city, c), 0) / totals[city] * 100, 2),
                     "meaningful": (city, c) in meaningful}
            for c in sorted(domains)}}
    residuals = [{"city": city, "candidate": c, "orders": by_cell.get((city, c), 0),
                  "share_pct": round(by_cell.get((city, c), 0) / totals[city] * 100, 2)}
                 for city in cities for c in sorted(domains) if (city, c) not in meaningful]
    pieces = {city: {freight_assignment.piece_label(piece): count
                     for (c, piece), count in sorted(demand.items()) if c == city} for city in cities}
    return table, residuals, pieces, meaningful


def ranking_summary(ranking, domains):
    agents = sorted(ranking[0]["assignment"])
    best = ranking[0]
    tied_once = [r for r in ranking if r["covered_once"] == best["covered_once"]]
    tied_both = [r for r in tied_once if r["covered_twice"] == best["covered_twice"]]
    pairing = {}
    for agent in agents:
        pairing[agent] = {}
        for c in sorted(domains):
            with_pair = [r for r in ranking if r["assignment"][agent] == c]
            top = with_pair[0]
            pairing[agent][str(c)] = {"best_covered_once": top["covered_once"], "best_covered_twice": top["covered_twice"]}
    sensitivity = sorted(ranking, key=lambda r: (-r["cell_weighted_once"], -r["cell_weighted_twice"], r["sequence"]))
    return {
        "assignments_evaluated": len(ranking),
        "all_bijective": all(sorted(r["sequence"]) == sorted(domains) for r in ranking)
                         and len({r["sequence"] for r in ranking}) == len(ranking),
        "winner": {"candidates_by_agent": best["assignment"], "covered_once": best["covered_once"],
                   "covered_twice": best["covered_twice"]},
        "assignments_tied_on_objective_1": len(tied_once),
        "assignments_tied_on_objectives_1_and_2": len(tied_both),
        "tie_break_applied": len(tied_both) > 1,
        "distinct_objective_1_values": len({r["covered_once"] for r in ranking}),
        "worst": {"candidates_by_agent": ranking[-1]["assignment"], "covered_once": ranking[-1]["covered_once"],
                  "covered_twice": ranking[-1]["covered_twice"]},
        "top": [{"rank": i + 1, "candidates_by_agent": r["assignment"], "covered_once": r["covered_once"],
                 "covered_twice": r["covered_twice"]} for i, r in enumerate(ranking[:TOP_RANKS])],
        "best_objective_with_each_agent_candidate_pairing": pairing,
        "sensitivity_cell_summed_objective": {
            "winner": sensitivity[0]["assignment"], "same_winner_as_primary": sensitivity[0]["sequence"] == best["sequence"],
            "primary_winner_rank_under_sensitivity": next(i + 1 for i, r in enumerate(sensitivity)
                                                          if r["sequence"] == best["sequence"])},
    }


def write_ranking_csv(ranking, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    agents = sorted(ranking[0]["assignment"])
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["rank", *agents, "covered_once", "covered_twice", "cell_weighted_once", "cell_weighted_twice"])
        for i, r in enumerate(ranking, 1):
            writer.writerow([i, *[r["assignment"][a] for a in agents], r["covered_once"], r["covered_twice"],
                             r["cell_weighted_once"], r["cell_weighted_twice"]])


def calibration_rows(assignment, candidates, parcel_rows):
    rows = []
    for agent_id, candidate in assignment.items():
        c = candidates[candidate]
        rows.append({"agent_id": agent_id, "calibration_corner": "freight",
                     "calibration_reading_id": freight_candidates.CANDIDATES[candidate][0],
                     "fixed_charge_eur": c["fixed_charge_eur"], "alpha_eur_per_kg": c["alpha_eur_per_kg"],
                     "beta_eur_per_tonne_km": BETA_EUR_PER_TONNE_KM, "fuel_surcharge_series_id": SHARED_FUEL_SERIES_ID,
                     "min_shipment_weight_kg": c["min_shipment_weight_kg"],
                     "max_shipment_weight_kg": c["max_shipment_weight_kg"],
                     "max_weight_inclusive": c["max_weight_inclusive"], "oversized_above_kg": None,
                     "oversized_fee_eur": None})
    for row in parcel_rows:
        # production parcel rows carry no minimum; 0 kg and an inclusive cap keep their eligibility identical
        rows.append({**{k: row[k] for k in proposal_validation.CALIBRATION_COLUMNS
                        if k not in ("min_shipment_weight_kg", "max_weight_inclusive")},
                     "min_shipment_weight_kg": 0.0, "max_weight_inclusive": True})
    return sorted(rows, key=lambda r: r["agent_id"])


def build():
    result = assign_from_calibration()
    ranking, domains, candidates = result["ranking"], result["domains"], result["candidates"]
    winner = ranking[0]["assignment"]
    table, residuals, pieces, meaningful = demand_table(result["demand"], domains, result["cities"])
    rows = calibration_rows(winner, candidates, result["parcel_rows"])
    write_ranking_csv(ranking, REPORT_DIR / "assignment_ranking.csv")
    document = {
        "status": STATUS,
        "method": ("freight candidates assigned by a global bijective search over all 7! assignments, scored on an "
                   "independent calibration demand sample and the agents' v4 service regions only; the parcel "
                   "corner is copied unchanged from production"),
        "part_1_candidates": {
            "candidates": [{"candidate": n, "label": c["label"], "reading_id": freight_candidates.CANDIDATES[n][0],
                            "fixed_charge_eur": c["fixed_charge_eur"], "alpha_eur_per_kg": c["alpha_eur_per_kg"],
                            "min_shipment_weight_kg": c["min_shipment_weight_kg"],
                            "max_shipment_weight_kg": c["max_shipment_weight_kg"],
                            "max_weight_inclusive": c["max_weight_inclusive"], "source": c["source"],
                            "arithmetic": c["arithmetic"], "domain_basis": c["domain_basis"]}
                           for n, c in candidates.items()],
            "verification_against_pinned_sources": result["verification"],
            "candidate_1_lower_bound": freight_candidates.candidate_1_lower_bound(),
        },
        "part_2_calibration_sample": {
            "seed": CALIBRATION_SEED,
            "seed_note": ("arbitrary. its only role is to differ from the production seed 42 (streams 42 and 43); "
                          f"it uses streams {CALIBRATION_SEED} and {CALIBRATION_SEED + 1}. it is not a tuned parameter"),
            "generator": "src/data_synthesis/v5/orders.py OrderGenerator, the production order design with another seed",
            "orders": result["sample_orders"],
            "persisted": ("no: generated in memory for the demand count and discarded. only its seed and the counts "
                          "below are recorded"),
            "fields_read": ["product_id", "material_type", "quantity", "deliver_to"],
            "never_read": [rel(PRODUCTION_ORDERS_PATH), "any price, budget, F, alpha or damage probability"],
            "used_for": "part 3 demand and part 4 assignment only; never for diagnostics, reporting of results or evaluation",
        },
        "part_3_demand": {
            "threshold": THRESHOLD_BASIS,
            "overlap_convention": OVERLAP_CONVENTION,
            "demand_by_city_and_candidate": table,
            "orders_by_city_and_weight_piece": pieces,
            "meaningful_cells": len(meaningful),
            "rare_residuals": residuals,
        },
        "part_4_assignment": {
            "inputs": ["calibration demand per (city, weight piece)", "the seven candidates' weight domains",
                       "the seven freight agents' v4 service_regions"],
            "not_read": ["price", "F", "alpha", "beta", "damage_probability", "specialty", "diagnostic output"],
            "objective": ["1. orders in a meaningful cell with at least one eligible provider, maximized",
                          "2. among ties, such orders with at least two eligible providers, maximized",
                          "3. among ties, the lexicographically smallest candidate sequence with agents sorted by id"],
            **ranking_summary(ranking, domains),
            "full_ranking_csv": rel(REPORT_DIR / "assignment_ranking.csv"),
        },
        "agents": rows,
        "production_files_sha256_at_proposal_time": proposal_validation.production_hashes(),
    }
    write_json(PROPOSAL_PATH, document)
    return document, result, rows, meaningful


# ── part 5: evaluation on the production order set ──────────────────────────

def production_coverage(result, meaningful, assignment):
    """structural coverage of each meaningful cell by the winning assignment, on production orders"""
    frame = pd.read_csv(PRODUCTION_ORDERS_PATH, usecols=["product_id", "material_type", "quantity", "deliver_to"])
    demand = piece_counts(frame.to_dict("records"), result["pairs"], result["domains"])
    coverage = freight_assignment.cell_coverage(assignment, demand, result["domains"], result["service_regions"],
                                                meaningful)
    cells = [{"city": city, "candidate": c, **counts} for (city, c), counts in sorted(coverage.items())]
    uncovered = [cell for cell in cells if cell["with_one_or_more"] < cell["orders"]]
    return {"basis": ("freight agents only, by weight domain and service region, before damage, reputation and "
                      "deadline filters; those filters are in the diagnostic's no-bidder count"),
            "cells": cells, "meaningful_cells": len(cells),
            "cells_fully_covered_by_one_or_more": len(cells) - len(uncovered),
            "cells_not_fully_covered": uncovered}


def highest_demand_cell(result, meaningful):
    by_cell = freight_assignment.domain_demand(result["demand"], result["domains"])
    city, candidate = max(meaningful, key=lambda cell: (by_cell[cell], cell))
    cell = (city, candidate)
    ranking = result["ranking"]

    def two_or_more(row):
        return freight_assignment.cell_coverage(row["assignment"], result["demand"], result["domains"],
                                                result["service_regions"], {cell})[cell]["with_two_or_more"]

    best = max(two_or_more(row) for row in ranking)
    return {"city": city, "candidate": candidate, "calibration_orders": by_cell[cell],
            "winner_orders_with_two_or_more_providers": two_or_more(ranking[0]),
            "best_any_bijection_orders_with_two_or_more_providers": best}


def main():
    before = proposal_validation.production_hashes()
    document, result, rows, meaningful = build()
    profile_path, orders_path = proposal_validation.regenerate(DATA_DIR, rows)
    agent_ids = sorted(FREIGHT_AGENTS + PARCEL_AGENTS)
    reports = {
        "production": proposal_validation.diagnose("production", REPORT_DIR),
        "specialty_proposal_on_file": proposal_validation.diagnose(
            "specialty_proposal_on_file", REPORT_DIR, SPECIALTY_PROPOSAL_DIR / "logistics_agents.csv",
            SPECIALTY_PROPOSAL_DIR / "orders.csv"),
        "v3": proposal_validation.diagnose("v3", REPORT_DIR, profile_path, orders_path),
    }
    on_file = json.loads(SPECIALTY_PROPOSAL_REPORT.read_text(encoding="utf-8"))
    rediagnosed = reports["specialty_proposal_on_file"]
    winner = result["ranking"][0]["assignment"]
    evaluation = {
        "status": STATUS,
        "evaluation_set": rel(PRODUCTION_ORDERS_PATH),
        "calibration_seed": CALIBRATION_SEED,
        "winning_assignment": {agent: {"candidate": c, "label": result["candidates"][c]["label"]}
                               for agent, c in winner.items()},
        "comparisons": {
            "v3_against_production": proposal_validation.compare(reports["production"], reports["v3"], agent_ids),
            "v3_against_specialty_proposal": proposal_validation.compare(rediagnosed, reports["v3"], agent_ids),
        },
        "specialty_proposal_rediagnosis_matches_file": all(
            rediagnosed[k] == on_file[k] for k in ("feasible_orders", "logistics_zero_bidders", "win_share_lowest_cost",
                                                   "wins_lowest_cost", "weight_domain_violations")),
        "tied_price_groups": {name: report["logistics_tied_price_groups"] for name, report in reports.items()},
        "production_set_coverage_of_meaningful_cells": production_coverage(result, meaningful, winner),
        "highest_demand_meaningful_cell": highest_demand_cell(result, meaningful),
        "rare_residuals_calibration_sample": document["part_3_demand"]["rare_residuals"],
    }
    after = proposal_validation.production_hashes()
    evaluation["production_files_unchanged"] = before == after
    write_json(REPORT_DIR / "comparison.json", evaluation)
    lines = [STATUS, "", f"winning assignment: {winner}", ""]
    lines += proposal_validation.render_side_by_side(reports, agent_ids)
    for name, report in reports.items():
        lines += ["", *proposal_validation.render_band_table(name, report, agent_ids)]
    (REPORT_DIR / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    print(f"\nproduction files unchanged: {before == after}")
    print(f"wrote {rel(PROPOSAL_PATH)}, {rel(DATA_DIR)}/ and {rel(REPORT_DIR)}/")
    return 0 if before == after else 1


if __name__ == "__main__":
    sys.exit(main())
