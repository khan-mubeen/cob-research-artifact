"""v5 training sets for finetune_lora.py: the named datasets, their required windows, and a
stratified subsample that can only ever draw from the train split

finetune_lora.py dispatches here for any dataset named v5_*, so its v4 datasets, and v4's
stratified_subsample, run exactly as before. nothing here imports unsloth, so the leakage and
stratification guarantees are testable without a gpu

  datasets     v5_sealed, v5_reverse_auction, v5_supervisor: the train files of each mode under
               data/finetune/v5. validation and test files are never named here
  windows      the sequence window each mode needs, from the measured token lengths
               (reports/v5/phase6_token_lengths.json); a shorter window would truncate the label
  subsample    the split's own stratification, scenario_type x priority, never a new scheme.
               stratum quotas follow the train split's order population and are apportioned by
               largest remainder, exactly as the split apportioned validation and test, so a
               subsample of n rows holds exactly n rows. quotas fill in the order the records
               arrive, finetune_lora's seeded shuffle, and a shortfall is topped up from the rest
               of that order and reported, never silently

the leakage guarantee is checked, not assumed: every record's order id must be in the split file
and assigned to train, before and after subsampling, or TrainSplitLeak is raised
"""

import csv
import random
from collections import Counter
from pathlib import Path

from config import PROJECT_ROOT

FINETUNE_V5 = PROJECT_ROOT / "data" / "finetune" / "v5"
SPLIT_PATH = FINETUNE_V5 / "split" / "order_split.csv"
SEED = 42
STAGES = ("material", "tailoring", "logistics")

V5_TRAIN_SETS = {
    "v5_sealed": [f"{stage}_train.jsonl" for stage in STAGES],
    "v5_reverse_auction": [f"reverse_auction/{stage}_train.jsonl" for stage in STAGES],
    "v5_supervisor": ["supervisor/selection_train.jsonl"],
}
V5_WINDOWS = {"v5_sealed": 1024, "v5_reverse_auction": 1280, "v5_supervisor": 2048}


class TrainSplitLeak(ValueError):
    """a v5 training record whose order is not in the train split"""


def is_v5_dataset(name):
    return name in V5_TRAIN_SETS


def train_set_paths(name, root=FINETUNE_V5):
    return [Path(root) / relative for relative in V5_TRAIN_SETS[name]]


def check_window(name, max_seq_length):
    required = V5_WINDOWS[name]
    if max_seq_length < required:
        raise ValueError(f"{name} needs a {required}-token window, its longest rendered rows exceed "
                         f"{max_seq_length}; a shorter window truncates the output schema and label")


def read_split(path=SPLIT_PATH):
    """order id -> (split, (scenario_type, priority))"""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["order_id"]: (row["split"], (row["scenario_type"], row["priority"])) for row in csv.DictReader(f)}


def assert_train_only(records, split):
    wrong = Counter(split[r["order_id"]][0] if r["order_id"] in split else "not_in_split_file" for r in records
                    if r["order_id"] not in split or split[r["order_id"]][0] != "train")
    if wrong:
        raise TrainSplitLeak(f"{sum(wrong.values())} training record(s) come from outside the train split: {dict(wrong)}")


def apportion(sizes, total):
    """integer shares of total proportional to sizes, largest remainder first, the key breaking a tie"""
    population = sum(sizes.values())
    exact = {k: n * total / population for k, n in sizes.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def stratified_subsample(records, n, split_path=SPLIT_PATH, seed=SEED):
    """n records from the train split, stratified on scenario_type x priority

    returns (records, report). records arrive in finetune_lora's seeded order"""
    split = read_split(split_path)
    assert_train_only(records, split)
    if n > len(records):
        raise ValueError(f"cannot subsample {n} records from {len(records)}")
    population = Counter(stratum for split_name, stratum in split.values() if split_name == "train")
    quota = apportion(population, n)
    remaining = dict(quota)
    taken, leftover = [], []
    for record in records:
        stratum = split[record["order_id"]][1]
        if remaining.get(stratum, 0) > 0:
            taken.append(record)
            remaining[stratum] -= 1
        else:
            leftover.append(record)
    shortfall = {"|".join(k): v for k, v in remaining.items() if v > 0}
    taken.extend(leftover[: n - len(taken)])
    random.Random(seed).shuffle(taken)
    assert_train_only(taken, split)
    achieved = Counter(split[r["order_id"]][1] for r in taken)
    report = {
        "rows": len(taken),
        "orders": len({r["order_id"] for r in taken}),
        "strata": {"|".join(k): {"train_orders": population[k], "quota": quota[k], "achieved": achieved.get(k, 0),
                                 "achieved_share_pct": round(achieved.get(k, 0) / len(taken) * 100, 3),
                                 "population_share_pct": round(population[k] / sum(population.values()) * 100, 3)}
                   for k in sorted(population)},
        "shortfall_topped_up": shortfall,
        "all_orders_in_train_split": True,
    }
    return taken, report
