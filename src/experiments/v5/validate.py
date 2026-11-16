"""phase 5: validation of every file under data/finetune/v5

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.validate

  schema        every row of every jsonl file: required fields and types, the label's json shape,
                the label agreeing with rule_decision and with the row's own identity, the split
                field agreeing with the file name, the role of every supplementary row
  golden        a seeded sample of every main file, and every floor-boundary and edge row, has its
                label re-derived from the stored v5 csv files by code written here from the
                methodology, which imports nothing from src/common, the v5 pricing module or the
                generators, and must agree exactly: decision, offer, completion days, the floor a
                prompt shows, the chain a supervisor row selects, and the dated values a prompt prints
  balance       BID/NO_BID or strategy counts per file, and the same ratios from the committed v4
                files, read only, for comparison
  sanity        every offer finite and positive, and none implausible against its order's budget
  manifest      every file's sha256 and row count recomputed and compared with manifest.json
  leakage       every order id collected from every main file of every stage and mode; no order id
                in two splits, and every row in the split the split file assigns its order

writes reports/v5/phase5_validation.json and exits non-zero on any failure
"""

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.data_synthesis.v5.logistics import RouteCost
from src.data_synthesis.v5.settings import SHARED_FUEL_SERIES_ID, V5_DIR

FINETUNE_V5 = PROJECT_ROOT / "data" / "finetune" / "v5"
V4_FINETUNE = PROJECT_ROOT / "data" / "finetune" / "v4"
REPORT_PATH = PROJECT_ROOT / "reports" / "v5" / "phase5_validation.json"
SPLITS = ("train", "validation", "test")
STAGES = ("material", "tailoring", "logistics")
SAMPLE_PER_MAIN_FILE = 300
SAMPLE_SEED = 20260914
IMPLAUSIBLE_BUDGET_MULTIPLE = 10

# restated from the methodology, not imported, so a pricing bug cannot validate itself
PRIORITY = {"urgent": 0.30, "normal": 0.00, "flexible": -0.15}
DECIMALS = {"reference": 4, "fixed": 4, "alpha": 6, "beta": 5, "km": 3, "fuel": 2, "fx": 4, "weight": 4}
LABEL_KEYS = ("order_id", "stage", "agent_id", "decision", "offer_amount", "estimated_completion_days",
              "defect_probability")


def missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def ceil_cents(value):
    return math.ceil(round(value * 100, 6)) / 100


def printed(user, label):
    match = re.search(rf"^{re.escape(label)}: (.+)$", user, re.M)
    return None if match is None else match.group(1)


class Stored:
    """the v5 csv files and pinned series, read directly"""

    def __init__(self):
        self.agents = {
            "material": {a["agent_id"]: a for a in pd.read_csv(V5_DIR / "material_agents.csv").to_dict("records")},
            "tailoring": {a["agent_id"]: a for a in pd.read_csv(V5_DIR / "manufacturer_agents.csv").to_dict("records")},
            "logistics": {a["agent_id"]: a for a in pd.read_csv(V5_DIR / "logistics_agents.csv").to_dict("records")},
        }
        self.orders = {o["order_id"]: o for o in pd.read_csv(V5_DIR / "orders.csv").to_dict("records")}
        self.products = {(r["product_id"], r["material_type"]): (r["material_kg_per_unit"], r["finished_weight_kg_per_unit"])
                         for r in pd.read_csv(V5_DIR / "products.csv").to_dict("records")}
        self.reference = {(r.material, r.date): r.usd_per_kg
                          for r in pd.read_csv(V5_DIR / "material_reference_prices.csv").itertuples()}
        self.route = RouteCost()
        split = pd.read_csv(FINETUNE_V5 / "split" / "order_split.csv")
        self.split = dict(zip(split["order_id"], split["split"]))

    @staticmethod
    def days(order):
        created = datetime.strptime(order["order_created"], "%Y-%m-%dT%H:%M:%S")
        return max((datetime.strptime(order["deadline"], "%Y-%m-%d") - created).days, 1)

    def market(self, pricing_date):
        day = datetime.strptime(pricing_date, "%Y-%m-%d").date()
        return (round(self.route.series[SHARED_FUEL_SERIES_ID].as_of(day).value, DECIMALS["fuel"]),
                round(self.route.fx.rate("EUR", "USD", day).rate, DECIMALS["fx"]))


# ── the rules, re-derived ───────────────────────────────────────────────────

def material_rule(agent, material, required_kg, days, max_defect, min_rep, priority, reference):
    if material not in str(agent["material_types"]).split("|"):
        return {"decision": "NO_BID", "reason": "material_not_supplied"}
    if agent["defect_probability"] > max_defect:
        return {"decision": "NO_BID", "reason": "defect_probability_too_high"}
    if agent["reputation_score"] < min_rep:
        return {"decision": "NO_BID", "reason": "reputation_too_low"}
    available, restock = agent[f"{material}_available_kg"], agent[f"{material}_restock_amount_kg"]
    if available >= required_kg:
        completion = agent["default_completion_days"]
    elif not missing(restock) and available + restock >= required_kg:
        completion = agent["default_completion_days"] + agent[f"{material}_restock_time_days"]
    else:
        return {"decision": "NO_BID", "reason": "insufficient_stock_and_restock"}
    if completion > days:
        return {"decision": "NO_BID", "reason": "completion_exceeds_deadline"}
    internal = required_kg * agent[f"{material}_price_ratio_to_reference"] * round(reference, DECIMALS["reference"]) \
        * (1 + agent["defect_probability"])
    minimum = internal * (1 + agent["min_profit_margin"])
    initial = internal * (1 + agent["initial_markup"]) * (1 + PRIORITY[priority])
    return {"decision": "BID", "offer": round(max(minimum, initial), 2), "days": completion, "minimum": minimum}


def manufacturer_rule(agent, product, quantity, days, max_defect, min_rep, priority):
    if product not in str(agent["product_types"]).split("|"):
        return {"decision": "NO_BID", "reason": "product_not_made"}
    if agent["defect_probability"] > max_defect:
        return {"decision": "NO_BID", "reason": "defect_probability_too_high"}
    if agent["reputation_score"] < min_rep:
        return {"decision": "NO_BID", "reason": "reputation_too_low"}
    capacity = agent["daily_capacity_units"]
    if capacity <= 0:
        return {"decision": "NO_BID", "reason": "no_daily_capacity"}
    completion = agent["current_workload_units"] / capacity + quantity / capacity
    if completion > days:
        return {"decision": "NO_BID", "reason": "completion_exceeds_deadline"}
    ratio = (agent["current_workload_units"] + quantity) / (capacity * days)
    if ratio > 1.0:
        return {"decision": "NO_BID", "reason": "workload_ratio_above_one"}
    workload = 0.0 if ratio < 0.5 else 0.05 if ratio < 0.75 else 0.10 if ratio < 0.9 else 0.15
    internal = quantity * agent[f"unit_cost_usd_{product}"] * (1 + agent["defect_probability"])
    minimum = internal * (1 + agent["min_profit_margin"])
    initial = internal * (1 + agent["initial_markup"]) * (1 + workload) * (1 + PRIORITY[priority])
    return {"decision": "BID", "offer": round(max(minimum, initial), 2), "days": round(completion, 2), "minimum": minimum}


def inside_domain(agent, weight):
    low, cap = agent.get("min_shipment_weight_kg"), agent.get("max_shipment_weight_kg")
    inclusive = agent.get("max_weight_inclusive")
    inclusive = True if missing(inclusive) else bool(inclusive)
    if not missing(low) and weight < low:
        return False
    if missing(cap):
        return True
    return weight <= cap if inclusive else weight < cap


def logistics_rule(agent, city, weight, days, max_defect, min_rep, priority, km, fuel, fx):
    if city not in str(agent["service_regions"]).split("|"):
        return {"decision": "NO_BID", "reason": "region_not_serviced"}
    if agent["reputation_score"] < min_rep:
        return {"decision": "NO_BID", "reason": "reputation_too_low"}
    if agent["damage_probability"] > max_defect:
        return {"decision": "NO_BID", "reason": "damage_probability_too_high"}
    if not inside_domain(agent, weight):
        return {"decision": "NO_BID", "reason": "shipment_weight_outside_domain"}
    route_days = agent["default_completion_same_city_days"] if agent["location"] == city \
        else agent["default_completion_intercity_days"]
    if route_days > days:
        return {"decision": "NO_BID", "reason": "completion_exceeds_deadline"}
    freight = round(agent["fixed_charge_eur"], DECIMALS["fixed"]) + round(agent["alpha_eur_per_kg"], DECIMALS["alpha"]) * weight \
        + round(agent["beta_eur_per_tonne_km"], DECIMALS["beta"]) * (weight / 1000.0) * km
    fee = agent["oversized_fee_eur"]
    oversized = fee if not missing(fee) and weight > agent["oversized_above_kg"] else 0.0
    price = (freight * (1.0 + fuel / 100.0) + oversized) * (1.0 + agent["damage_probability"]) \
        * (1.0 + PRIORITY[priority]) * fx
    return {"decision": "BID", "offer": round(price, 2), "days": route_days, "minimum": price}


def stage_inputs(stored, order):
    """the dated values of one stored order, derived here without the rfq builder"""
    kg_unit, finished = stored.products[(order["product_id"], order["material_type"])]
    days = stored.days(order)
    manufacturers = stored.agents["tailoring"]
    bids = []
    for agent_id, agent in manufacturers.items():
        r = manufacturer_rule(agent, order["product_id"], order["quantity"], days, order["max_defect_rate"],
                              order["min_reputation_score"], order["priority"])
        if r["decision"] == "BID":
            bids.append((r["offer"], agent_id))
    origin = manufacturers[min(bids)[1]]["location"] if bids else None
    fuel, fx = stored.market(order["pricing_date"])
    weight = round(order["quantity"] * finished, DECIMALS["weight"])
    km = round(stored.route.distance_km[(origin, order["deliver_to"])], DECIMALS["km"]) if origin else None
    return {"days": days, "required_kg": round(order["quantity"] * kg_unit, DECIMALS["weight"]), "weight": weight,
            "reference": round(stored.reference[(order["material_type"], order["pricing_date"])], DECIMALS["reference"]),
            "origin": origin, "km": km, "fuel": fuel, "fx": fx}


def rule_for(stored, stage, agent, order, inputs):
    common = (order["max_defect_rate"], order["min_reputation_score"], order["priority"])
    if stage == "material":
        return material_rule(agent, order["material_type"], inputs["required_kg"], inputs["days"], *common,
                             inputs["reference"])
    if stage == "tailoring":
        return manufacturer_rule(agent, order["product_id"], order["quantity"], inputs["days"], *common)
    return logistics_rule(agent, order["deliver_to"], inputs["weight"], inputs["days"], *common, inputs["km"],
                          inputs["fuel"], inputs["fx"])


def label_matches(rule, label):
    if rule["decision"] != label["decision"]:
        return False
    if rule["decision"] == "NO_BID":
        return label["offer_amount"] is None
    return label["offer_amount"] == rule["offer"] and abs(label["estimated_completion_days"] - round(rule["days"], 2)) < 1e-9


def prompt_matches_inputs(stage, user, inputs):
    checks = {"days until deadline": inputs["days"]}
    if stage == "material":
        checks.update({"required quantity (kg)": inputs["required_kg"],
                       "reference price on the pricing date (USD per kg)": inputs["reference"]})
    if stage == "logistics":
        checks.update({"shipment weight (kg)": inputs["weight"], "route distance from pickup (km)": inputs["km"],
                       "fuel surcharge on the pricing date (%)": inputs["fuel"],
                       "EUR to USD on the pricing date": inputs["fx"]})
    bad = [k for k, v in checks.items() if float(printed(user, k)) != v]
    if stage == "logistics" and printed(user, "pickup from") != inputs["origin"]:
        bad.append("pickup from")
    return bad


def auction_label(rule, row):
    """what decide_round_bid must have said, from the sealed rule and the round state on the row"""
    if row["round_number"] <= 1 or rule["decision"] != "BID":
        return rule
    if row["current_best_bid"] is None:
        return {"decision": "NO_BID"}
    offer = round(float(row["current_best_bid"]) - float(row["platform_minimum_decrement"]), 2)
    if offer < rule["minimum"]:
        return {"decision": "NO_BID"}
    return {**rule, "offer": offer}


CANDIDATE = re.compile(
    r"candidate (\d+): material\(cost_usd=([^,]+), days=([^,]+), defect_p=([^)]+)\), "
    r"manufacturing\(cost_usd=([^,]+), days=([^,]+), defect_p=([^)]+)\), "
    r"logistics\(cost_usd=([^,]+), days=([^,]+), damage_p=([^)]+)\), "
    r"total_cost_usd=([^,]+), total_completion_days=(\S+)")


def chain_rule(candidates, priority, budget, deadline, defect_cap):
    def totals(c):
        cost = c["costs"][0] + c["costs"][1] + c["costs"][2]
        days = c["days"][0] + c["days"][1] + c["days"][2]
        survives = 1.0
        for p in c["risks"]:
            survives *= (1.0 - p)
        return cost, days, 1.0 - survives

    feasible = [c for c in candidates if totals(c)[0] <= budget and totals(c)[1] <= deadline]
    if not feasible:
        return []
    third = 1 / 3

    def key(c):
        # cost and days compared as printed, at two decimals
        cost, days, risk = totals(c)
        rounded = (round(cost, 2), round(days, 2))
        if priority == "urgent":
            return (rounded[1], rounded[0])
        if priority == "flexible":
            return (rounded[0], rounded[1])
        score = (third * (cost / budget if budget > 0 else 1.0)
                 + third * (days / deadline if deadline > 0 else 1.0)
                 + third * (risk / defect_cap if defect_cap > 0 else 1.0))
        return (score, rounded[0], rounded[1])

    best = min(key(c) for c in feasible)
    # every index that ties for the optimum: the agent-id tie-break that settles a real tie is not
    # printed in the prompt, so from the prompt alone any of these is the right answer
    return sorted(c["index"] for c in feasible if key(c) == best)


# ── the arms ────────────────────────────────────────────────────────────────

def main_files():
    files = []
    for split in SPLITS:
        files += [(f"{stage}_{split}.jsonl", "sealed", stage, split) for stage in STAGES]
        files += [(f"reverse_auction/{stage}_{split}.jsonl", "reverse_auction", stage, split) for stage in STAGES]
        files.append((f"supervisor/selection_{split}.jsonl", "supervisor", "supervisor", split))
    return files


def supplementary_files():
    files = [(f"edge/{stage}_edge.jsonl", "edge", stage) for stage in STAGES]
    files += [(f"reverse_auction/floor_boundary/{stage}_{case}_floor.jsonl", "floor", stage)
              for stage in STAGES for case in ("below", "at", "above")]
    return files


def schema_problems(row, mode, stage, split):
    problems = []
    need = {"order_id": str, "stage": str, "dataset_version": str, "system": str, "user": str, "completion": str}
    for key, kind in need.items():
        if not isinstance(row.get(key), kind) or (kind is str and not row.get(key)):
            problems.append(f"{key} missing or not {kind.__name__}")
    if row.get("dataset_version") != "v5":
        problems.append("dataset_version is not v5")
    if split is not None and row.get("split") != split:
        problems.append("split field disagrees with the file")
    if row.get("stage") != stage:
        problems.append("stage disagrees with the file")
    try:
        label = json.loads(row["completion"])
    except (KeyError, TypeError, ValueError):
        return problems + ["completion is not json"]
    if mode == "supervisor":
        if set(label) != {"selected_candidate", "reason"} or not isinstance(label["selected_candidate"], int) \
                or not 0 <= label["selected_candidate"] < row.get("n_candidates", 0):
            problems.append("selection label malformed")
        return problems
    if tuple(label) != LABEL_KEYS:
        problems.append("label keys differ from the output schema")
    if label.get("decision") != row.get("rule_decision"):
        problems.append("label decision disagrees with rule_decision")
    if (label.get("order_id"), label.get("agent_id"), label.get("stage")) != (row.get("order_id"), row.get("agent_id"), stage):
        problems.append("label identity disagrees with the row")
    if label.get("decision") == "BID":
        offer, days = label.get("offer_amount"), label.get("estimated_completion_days")
        if not isinstance(offer, (int, float)) or not math.isfinite(offer) or offer <= 0:
            problems.append("BID offer not finite and positive")
        if not isinstance(days, (int, float)) or not math.isfinite(days) or days < 0:
            problems.append("BID completion days invalid")
    elif label.get("decision") == "NO_BID":
        if label.get("offer_amount") is not None or label.get("estimated_completion_days") is not None:
            problems.append("NO_BID carries an offer")
    else:
        problems.append("decision is neither BID nor NO_BID")
    if mode == "reverse_auction" and (row.get("round_number") not in (1, 2, 3)
                                      or not isinstance(row.get("platform_minimum_decrement"), (int, float))):
        problems.append("auction round fields malformed")
    if mode in ("edge", "floor") and row.get("role") != ("robustness_eval" if mode == "edge" else "diagnostic"):
        problems.append("supplementary role missing")
    if "EUR)" in row["user"].replace("(EUR per shipment)", "").replace("(EUR per kg)", "").replace(
            "(EUR per tonne-km)", "").replace("oversized fee (EUR)", "") or "USD" not in row["user"]:
        problems.append("prompt currency wording is not v5")
    return problems


def golden_check(stored, row, mode, stage, cache):
    """re-derive one row's label from stored data; returns a list of problems"""
    user, label = row["user"], json.loads(row["completion"])
    if mode == "supervisor":
        order = stored.orders[row["order_id"]]
        budget, deadline = float(order["budget_usd"]), stored.days(order)
        if float(printed(user, "budget (USD)")) != budget:
            return ["printed budget differs from budget_usd"]
        candidates = []
        for m in CANDIDATE.finditer(user):
            g = [float(x) for x in m.groups()[1:]]
            candidates.append({"index": int(m.group(1)), "costs": (g[0], g[3], g[6]), "days": (g[1], g[4], g[7]),
                               "risks": (g[2], g[5], g[8])})
        if len(candidates) != row["n_candidates"]:
            return ["candidate lines do not parse"]
        chosen = chain_rule(candidates, order["priority"], budget, deadline, float(order["max_defect_rate"]))
        return [] if label["selected_candidate"] in chosen else [
            f"chain rule picks {chosen}, label {label['selected_candidate']}"]

    if mode == "edge":
        return edge_check(stored, row, stage, label)

    order = stored.orders[row["order_id"]]
    if row["order_id"] not in cache:
        cache[row["order_id"]] = stage_inputs(stored, order)
    inputs = cache[row["order_id"]]
    problems = prompt_matches_inputs(stage, user, inputs)
    agent = stored.agents[stage][row["agent_id"]]
    rule = rule_for(stored, stage, agent, order, inputs)
    if mode in ("reverse_auction", "floor"):
        floor_shown = printed(user, "your minimum_bid (USD)")
        expected_floor = ceil_cents(rule["minimum"]) if rule["decision"] == "BID" else None
        if (floor_shown == "null" and expected_floor is not None) or (
                floor_shown != "null" and float(floor_shown) != expected_floor):
            problems.append(f"prompt floor {floor_shown} is not {expected_floor}")
        rule = auction_label(rule, row)
        if mode == "floor":
            expected = {"below": "NO_BID", "at": "BID", "above": "BID"}[row["case"].split("_")[0]]
            if label["decision"] != expected or row["displayed_minimum_bid"] != expected_floor:
                problems.append("floor case label or displayed floor wrong")
    if not label_matches(rule, label):
        problems.append(f"label {label['decision']} {label['offer_amount']} is not {rule}")
    return problems


def edge_check(stored, row, stage, label):
    user = row["user"]
    agent = stored.agents[stage][row["agent_id"]]
    days = int(printed(user, "days until deadline"))
    max_defect = float(printed(user, "max allowed damage rate" if stage == "logistics" else "max allowed defect rate"))
    min_rep = float(printed(user, "min required reputation"))
    priority = printed(user, "priority")
    if stage == "material":
        material, date = printed(user, "material needed"), printed(user, "pricing date")
        reference = round(stored.reference[(material, date)], DECIMALS["reference"])
        if float(printed(user, "reference price on the pricing date (USD per kg)")) != reference:
            return ["printed reference differs from the stored series"]
        rule = material_rule(agent, material, float(printed(user, "required quantity (kg)")), days, max_defect, min_rep,
                             priority, reference)
    elif stage == "tailoring":
        rule = manufacturer_rule(agent, printed(user, "product"), int(printed(user, "quantity")), days, max_defect,
                                 min_rep, priority)
    else:
        fuel, fx = stored.market(printed(user, "pricing date"))
        km = round(stored.route.distance_km[(printed(user, "pickup from"), printed(user, "deliver to"))], DECIMALS["km"])
        if float(printed(user, "route distance from pickup (km)")) != km or float(printed(user, "fuel surcharge on the pricing date (%)")) != fuel:
            return ["printed logistics market values differ from stored data"]
        rule = logistics_rule(agent, printed(user, "deliver to"), float(printed(user, "shipment weight (kg)")), days,
                              max_defect, min_rep, priority, km, fuel, fx)
    return [] if label_matches(rule, label) else [f"label {label} is not {rule}"]


def v4_ratios():
    """BID shares of the committed v4 sets, read only, for the scale comparison"""
    out = {}
    for stage in STAGES:
        for split in ("train", "holdout"):
            counts = Counter()
            with open(V4_FINETUNE / f"{stage}_{split}.jsonl", encoding="utf-8") as f:
                for line in f:
                    counts[json.loads(line)["rule_decision"]] += 1
            out[f"sealed:{stage}_{split}"] = {"rows": sum(counts.values()), "BID": counts["BID"],
                                              "bid_pct": round(counts["BID"] / sum(counts.values()) * 100, 3)}
    meta = json.loads((V4_FINETUNE / "reverse_auction" / "v4" / "metadata.json").read_text(encoding="utf-8"))
    for key, v in meta["class_balance"].items():
        out[f"reverse_auction:{key}"] = {"rows": v["rows"], "BID": v["BID"], "bid_pct": round(v["BID"] / v["rows"] * 100, 3)}
    sup = json.loads((V4_FINETUNE / "supervisor" / "metadata.json").read_text(encoding="utf-8"))
    total = sum(sup["strategy_mix"].values())
    out["supervisor:strategy_mix"] = {k: round(v / total * 100, 3) for k, v in sup["strategy_mix"].items()}
    return out


def count_rows(path):
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip()) - (1 if path.suffix == ".csv" else 0)


def verify_manifest():
    stored = json.loads((FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    listed = {f["path"]: f for f in stored["files"]}
    on_disk = {str(p.relative_to(FINETUNE_V5)).replace("\\", "/") for p in FINETUNE_V5.rglob("*")
               if p.is_file() and p.name != "manifest.json"}
    problems += [f"{p}: listed, not on disk" for p in sorted(set(listed) - on_disk)]
    problems += [f"{p}: on disk, not listed" for p in sorted(on_disk - set(listed))]
    for relative in sorted(set(listed) & on_disk):
        path = FINETUNE_V5 / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != listed[relative]["sha256"]:
            problems.append(f"{relative}: sha256 differs")
        if listed[relative]["rows"] is not None and count_rows(path) != listed[relative]["rows"]:
            problems.append(f"{relative}: row count differs")
    return {"files_checked": len(listed), "problems": problems}


def main():
    stored = Stored()
    rng = random.Random(SAMPLE_SEED)
    result = {"schema": {}, "golden": {}, "balance": {}, "sanity": {}, "leakage": {}}
    ids_by_split = {s: set() for s in SPLITS}
    wrong_split = Counter()
    cache = {}
    offers_over_budget = []

    for relative, mode, stage, split in main_files():
        path = FINETUNE_V5 / relative
        rows = n = 0
        schema_bad, labels, offers, nan = Counter(), Counter(), [], 0
        sample = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                n += 1
                for p in schema_problems(row, mode, stage, split):
                    schema_bad[p] += 1
                ids_by_split[split].add(row["order_id"])
                if stored.split.get(row["order_id"]) != split:
                    wrong_split[relative] += 1
                label = json.loads(row["completion"])
                if mode == "supervisor":
                    labels[row["strategy"]] += 1
                    offers.append(row["label_total_cost"])
                    if row["label_total_cost"] > float(stored.orders[row["order_id"]]["budget_usd"]):
                        offers_over_budget.append((relative, row["order_id"], "chain total above budget"))
                else:
                    labels[label["decision"]] += 1
                    if label["decision"] == "BID":
                        offer = label["offer_amount"]
                        nan += not (isinstance(offer, (int, float)) and math.isfinite(offer))
                        offers.append(offer)
                        budget = float(stored.orders[row["order_id"]]["budget_usd"])
                        if offer > IMPLAUSIBLE_BUDGET_MULTIPLE * budget:
                            offers_over_budget.append((relative, row["order_id"], offer, budget))
                # reservoir sample, seeded, so every row has the same chance
                if len(sample) < SAMPLE_PER_MAIN_FILE:
                    sample.append(row)
                elif (j := rng.randrange(n)) < SAMPLE_PER_MAIN_FILE:
                    sample[j] = row
        result["schema"][relative] = {"rows": n, "problems": dict(schema_bad)}
        series = pd.Series(offers, dtype=float)
        result["balance"][relative] = {"rows": n, **dict(labels),
                                       **({"bid_pct": round(labels["BID"] / n * 100, 3)} if mode != "supervisor" else {})}
        result["sanity"][relative] = {"values": len(offers), "non_finite": nan, "non_positive": int((series <= 0).sum()),
                                      "min": float(series.min()) if len(series) else None,
                                      "median": float(series.median()) if len(series) else None,
                                      "p99": float(series.quantile(0.99)) if len(series) else None,
                                      "max": float(series.max()) if len(series) else None}
        failures = []
        for row in sample:
            failures += [(row["order_id"], row.get("agent_id"), p) for p in golden_check(stored, row, mode, stage, cache)]
        result["golden"][relative] = {"rows_re_derived": len(sample), "disagreements": len(failures),
                                      "examples": failures[:5]}
        print(f"{relative:45s} rows {n:7d} schema problems {sum(schema_bad.values())} "
              f"golden {len(sample)} re-derived, {len(failures)} disagree")

    for relative, mode, stage in supplementary_files():
        path = FINETUNE_V5 / relative
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        schema_bad = Counter(p for row in rows for p in schema_problems(row, mode, stage, "test" if mode == "floor" else None))
        failures = [(r["order_id"], r["agent_id"], p) for r in rows for p in golden_check(stored, r, mode, stage, cache)]
        if mode == "floor":
            misplaced = sum(1 for r in rows if stored.split.get(r["order_id"]) != "test")
            if misplaced:
                wrong_split[relative] += misplaced
        result["schema"][relative] = {"rows": len(rows), "problems": dict(schema_bad)}
        result["golden"][relative] = {"rows_re_derived": len(rows), "disagreements": len(failures), "examples": failures[:5]}
        result["balance"][relative] = {"rows": len(rows), **dict(Counter(r["rule_decision"] for r in rows))}
        print(f"{relative:45s} rows {len(rows):7d} schema problems {sum(schema_bad.values())} "
              f"golden all re-derived, {len(failures)} disagree")

    overlaps = {f"{a}_{b}": len(ids_by_split[a] & ids_by_split[b])
                for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    result["leakage"] = {"order_ids_by_split": {s: len(v) for s, v in ids_by_split.items()},
                         "overlaps": overlaps, "rows_in_a_split_other_than_their_orders": dict(wrong_split),
                         # summed, not truth-tested: a Counter holding a zero count is still non-empty
                         "passed": not any(overlaps.values()) and sum(wrong_split.values()) == 0}
    result["implausible_offers"] = {"rule": f"a stage offer above {IMPLAUSIBLE_BUDGET_MULTIPLE}x its order's budget_usd, "
                                            "or a supervisor chain total above budget",
                                    "count": len(offers_over_budget), "examples": offers_over_budget[:10]}
    result["manifest"] = verify_manifest()
    result["v4_reference_ratios"] = v4_ratios()
    checks = {
        "schema": all(not v["problems"] for v in result["schema"].values()),
        "golden_style_re_derivation": all(v["disagreements"] == 0 for v in result["golden"].values()),
        "no_non_finite_or_non_positive_offers": all(v["non_finite"] == 0 and v["non_positive"] == 0
                                                    for v in result["sanity"].values()),
        "no_implausible_offers": not offers_over_budget,
        "manifest_hashes_match_disk": not result["manifest"]["problems"],
        "no_order_id_in_two_splits": result["leakage"]["passed"],
    }
    result["checks"], result["passed"] = checks, all(checks.values())
    result["golden_rows_re_derived_total"] = sum(v["rows_re_derived"] for v in result["golden"].values())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    print(f"leakage {result['leakage']['order_ids_by_split']} overlaps {overlaps}")
    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
