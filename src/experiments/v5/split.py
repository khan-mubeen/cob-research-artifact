"""the one global split of the 50,000 v5 orders into train, validation and test, by order id

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.split

every v5 sealed, reverse-auction, supervisor example is generated from these three fixed
order-id sets, so an order id can only ever appear in one split

stratified jointly on scenario_type and priority. each stratum's validation and test sizes are
its exact 10% shares apportioned by largest remainder, so the global totals are exactly
40,000 / 5,000 / 5,000 and no stratum is off its share by a whole order more than rounding
forces; the rest of the stratum is train. inside a stratum the order ids are sorted and then
shuffled by one random.Random(SPLIT_SEED) stream, visiting strata in sorted order, so the split
is exactly reproducible. no third stratification dimension is used; the dimensions not
stratified on are measured afterwards with a chi-square test of independence, and the result is
recorded rather than acted on silently

writes data/finetune/v5/split/order_split.csv and data/finetune/v5/split/metadata.json
"""

import math
import random
import sys
from collections import Counter

import pandas as pd

from src.data_synthesis.v5.settings import V5_DIR
from src.experiments.v5.common import FINETUNE_V5, SPLIT_PATH, SPLITS, rel, require_v5, sha256_file, write_json

SPLIT_SEED = 42
FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
STRATIFY_ON = ("scenario_type", "priority")
CHECKED_NOT_STRATIFIED = ("material_type", "product_id", "deliver_to", "month")
# a stratum whose smallest split would hold fewer orders than this is reported as rare
RARE_SPLIT_ORDERS = 30
# a dimension not stratified on counts as imbalanced only below this chi-square p-value
IMBALANCE_P_VALUE = 0.001


def apportion(sizes, total):
    """integer shares of total across strata proportional to their sizes, largest remainder first,
    the stratum key breaking an exact tie so the result is deterministic"""
    population = sum(sizes.values())
    exact = {k: n * total / population for k, n in sizes.items()}
    out = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def assign(orders):
    n = len(orders)
    targets = {"validation": math.floor(n * FRACTIONS["validation"]), "test": math.floor(n * FRACTIONS["test"])}
    targets["train"] = n - targets["validation"] - targets["test"]
    by_stratum = {}
    for row in orders.itertuples():
        by_stratum.setdefault(tuple(getattr(row, c) for c in STRATIFY_ON), []).append(row.order_id)
    sizes = {k: len(v) for k, v in by_stratum.items()}
    validation = apportion(sizes, targets["validation"])
    test = apportion(sizes, targets["test"])
    rng = random.Random(SPLIT_SEED)
    split_of = {}
    for key in sorted(by_stratum):
        ids = sorted(by_stratum[key])
        rng.shuffle(ids)
        v, t = validation[key], test[key]
        for i, order_id in enumerate(ids):
            split_of[order_id] = "validation" if i < v else "test" if i < v + t else "train"
    return split_of, targets, sizes


def _upper_regularized_gamma(a, x):
    """Q(a, x), by series below a + 1 and by continued fraction above, numerical recipes 6.2"""
    if x <= 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(10_000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(log_prefix)
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(log_prefix) * h


def chi_square(frame, column):
    """pearson chi-square test of independence between a column and the split, computed here
    because scipy is not installed in the project environment"""
    table = pd.crosstab(frame[column], frame["split"]).values.astype(float)
    rows, cols, total = table.sum(axis=1), table.sum(axis=0), table.sum()
    expected = rows[:, None] * cols[None, :] / total
    statistic = float(((table - expected) ** 2 / expected).sum())
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    return {"chi_square": round(statistic, 4), "degrees_of_freedom": int(dof),
            "p_value": _upper_regularized_gamma(dof / 2, statistic / 2)}


def shares(frame, column):
    out = {}
    for split in SPLITS:
        part = frame[frame["split"] == split]
        counts = part[column].value_counts()
        out[split] = {str(k): {"orders": int(counts.get(k, 0)), "share_pct": round(counts.get(k, 0) / len(part) * 100, 3)}
                      for k in sorted(frame[column].unique())}
    population = frame[column].value_counts()
    deviation = max(abs(out[s][str(k)]["share_pct"] - population[k] / len(frame) * 100)
                    for s in SPLITS for k in population.index)
    return out, round(float(deviation), 3)


def main():
    context = require_v5()
    orders = pd.read_csv(V5_DIR / "orders.csv",
                         usecols=["order_id", "order_created", "scenario_type", "priority", "material_type",
                                  "product_id", "deliver_to"])
    if orders["order_id"].duplicated().any():
        raise SystemExit("duplicate order ids in the v5 orders")
    split_of, targets, sizes = assign(orders)
    orders["split"] = orders["order_id"].map(split_of)
    orders["month"] = orders["order_created"].str[:7]

    counts = Counter(orders["split"])
    if dict(counts) != targets or orders["split"].isna().any():
        raise SystemExit(f"split totals {dict(counts)} differ from the targets {targets}")

    strata = []
    for (scenario, priority), group in orders.groupby(list(STRATIFY_ON)):
        per = Counter(group["split"])
        row = {"scenario_type": scenario, "priority": priority, "orders": int(len(group)),
               **{s: int(per.get(s, 0)) for s in SPLITS}}
        row["share_pct"] = {s: round(row[s] / row["orders"] * 100, 3) for s in SPLITS}
        row["in_every_split"] = all(row[s] > 0 for s in SPLITS)
        row["rare"] = min(row[s] for s in SPLITS) < RARE_SPLIT_ORDERS
        strata.append(row)

    stratified = {}
    for column in STRATIFY_ON:
        table, deviation = shares(orders, column)
        stratified[column] = {"by_split": table, "max_share_deviation_points": deviation}
    checked = {}
    for column in CHECKED_NOT_STRATIFIED:
        table, deviation = shares(orders, column)
        test = chi_square(orders, column)
        checked[column] = {"by_split": table, "max_share_deviation_points": deviation, "independence_test": test,
                           "imbalanced": test["p_value"] is not None and test["p_value"] < IMBALANCE_P_VALUE}

    out = orders[["order_id", "split", "scenario_type", "priority"]].sort_values("order_id")
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPLIT_PATH, index=False, lineterminator="\n")
    ids = {s: set(out.loc[out["split"] == s, "order_id"]) for s in SPLITS}
    metadata = {
        "dataset_version": context.version,
        "source_orders": rel(V5_DIR / "orders.csv"),
        "source_orders_sha256": sha256_file(V5_DIR / "orders.csv"),
        "source_order_count": int(len(orders)),
        "scale_decision": ("deliberate v5 scaling decision: the full 50,000-order v5 pool is split and generated, "
                           "not v4's ~3,000-order experimental sample"),
        "split_seed": SPLIT_SEED,
        "fractions": FRACTIONS,
        "targets": targets,
        "counts": {s: int(counts[s]) for s in SPLITS},
        "method": ("stratified jointly on scenario_type x priority; per-stratum validation and test sizes apportioned "
                   "by largest remainder of the exact 10% shares, the rest train; ids sorted then shuffled with one "
                   "random.Random(split_seed) stream over strata in sorted order"),
        "stratify_on": list(STRATIFY_ON),
        "third_dimension_added": False,
        "strata": strata,
        "stratified_distributions": stratified,
        "not_stratified_checked": checked,
        "imbalance_rule": (f"a dimension not stratified on is imbalanced when its chi-square test of independence "
                           f"from the split gives p < {IMBALANCE_P_VALUE}"),
        "disjoint": {"train_validation": len(ids["train"] & ids["validation"]),
                     "train_test": len(ids["train"] & ids["test"]),
                     "validation_test": len(ids["validation"] & ids["test"])},
        "order_split_csv": rel(SPLIT_PATH),
        "order_split_csv_sha256": sha256_file(SPLIT_PATH),
    }
    write_json(FINETUNE_V5 / "split" / "metadata.json", metadata)
    print(f"split {dict(counts)} over {len(sizes)} strata")
    for row in strata:
        print(f"  {row['scenario_type']:18s} {row['priority']:9s} {row['orders']:6d}  "
              f"train {row['train']:6d} validation {row['validation']:5d} test {row['test']:5d}"
              f"{'  RARE' if row['rare'] else ''}")
    for column, v in checked.items():
        print(f"  not stratified {column:14s} max share deviation {v['max_share_deviation_points']} points, "
              f"p {v['independence_test']['p_value']}, imbalanced {v['imbalanced']}")
    print(f"wrote {rel(SPLIT_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
