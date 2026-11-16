"""Load the raw evaluation data used by the epoch comparison charts."""

import glob
import json
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

MODELS = [
    ("qwen", "Qwen"),
    ("granite", "Granite"),
    ("phi", "Phi"),
    ("gemma", "Gemma"),
    ("ministral", "Ministral"),
    ("smollm3", "SmolLM3"),
]
STAGES = ("material", "tailoring", "logistics")
CLASSES = ("above_infeasible", "below_feasible", "exact_feasible")


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_report_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path

x
def _offer_deviation_stats(eval_path):
    """Compute offer deviation statistics from scorable main-holdout rows."""
    rows = _load_json(eval_path)["main_holdout"]["rows"]
    stage_values = {
        stage: [
            row["offer_pct_dev"]
            for row in rows
            if row["stage"] == stage and row.get("offer_pct_dev") is not None
        ]
        for stage in STAGES
    }
    pooled = [value for stage in STAGES for value in stage_values[stage]]
    if not pooled:
        raise ValueError(f"no scorable offer-deviation rows in {eval_path}")

    return {
        "by_stage": {
            stage: {
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "std": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0 if values else None
                ),
            }
            for stage, values in stage_values.items()
        },
        "pooled": {
            "n": len(pooled),
            "mean": statistics.mean(pooled),
            "std": statistics.stdev(pooled) if len(pooled) > 1 else 0.0,
        },
    }


def _stage_model_data():
    output = {}
    comparison_dir = REPO / "reports" / "epoch-2" / "production" / "comparison"

    for key, label in MODELS:
        comparison = _load_json(comparison_dir / f"{key}_epoch1_vs_epoch2.json")
        record = {"label": label}

        for arm in ("base", "epoch_1", "epoch_2"):
            summary = comparison["arms"][arm]
            if summary is None:
                continue

            raw_path = comparison["inputs"].get(arm)
            if not raw_path:
                raise ValueError(f"{key} {arm} has no raw evaluation path")
            raw_stats = _offer_deviation_stats(_resolve_report_path(raw_path))

            arm_data = {
                "decisions": {
                    stage: summary["main_holdout_decisions"][stage]
                    for stage in STAGES
                },
                "offer_dev": {
                    stage: raw_stats["by_stage"][stage]["mean"]
                    for stage in STAGES
                },
                "offer_dev_stats": raw_stats,
                "raw_eval_path": str(Path(raw_path).as_posix()),
                "json_valid": summary["json"]["valid_pct"],
                "malformed": summary["json"]["malformed"],
                "rows_scored": summary["json"]["rows_scored"],
                "baseline_six": summary.get("baseline_six"),
                "baseline_six_guarded": summary.get("baseline_six_guarded"),
                "edge_clean10": summary.get("edge_holdout_clean10"),
            }

            robustness = summary.get("robustness_holdout_v2_per_class")
            if robustness:
                arm_data["robustness"] = {
                    name: int(robustness[name].split("/")[0])
                    for name in CLASSES
                }
                arm_data["robustness_total"] = sum(
                    arm_data["robustness"].values()
                )

            guarded = summary.get("robustness_holdout_v2_per_class_guarded")
            if guarded:
                arm_data["robustness_guarded"] = {
                    name: int(guarded[name].split("/")[0])
                    for name in CLASSES
                }

            guard_effect = summary.get("guard_b", {}).get("robustness_holdout_v2")
            if guard_effect:
                arm_data["guard_corrected"] = guard_effect["false_bids_corrected"]
                arm_data["guard_residual"] = guard_effect["false_nobids_remaining"]
                arm_data["guard_new_errors"] = guard_effect["new_errors_introduced"]

            record[arm] = arm_data

        record["criteria"] = {
            "epoch_1": comparison["criteria"]["epoch_1"],
            "epoch_2": comparison["criteria"]["epoch_2"],
        }
        output[key] = record

    return output


def _single_glob(pattern):
    matches = sorted(glob.glob(str(REPO / pattern)))
    if len(matches) != 1:
        raise ValueError(f"expected one report for {pattern}, found {len(matches)}")
    return Path(matches[0])


def _supervisor_data():
    output = {}
    paths = (
        ("base", "reports/supervisor/smoke/eval_smoke-base_*.json"),
        (
            "epoch_1",
            "reports/supervisor/epoch-1/evaluation/raw/eval_sup-e1-full_*.json",
        ),
        (
            "epoch_2",
            "reports/supervisor/epoch-2/evaluation/raw/eval_sup-e2-full_*.json",
        ),
    )

    for tag, pattern in paths:
        report_path = _single_glob(pattern)
        summary = _load_json(report_path)["summary"]
        output[tag] = {
            "n": summary["n"],
            "accuracy": summary["selection_accuracy_pct"],
            "mean_regret": summary["mean_regret_pct"],
            "median_regret": summary["median_regret_pct"],
            "zero_regret": summary["zero_regret_pct"],
            "p90_regret": summary["p90_regret_pct"],
            "json_valid": summary["json_valid_pct"],
            "by_strategy": {
                key: {
                    "n": value["n"],
                    "accuracy": value["accuracy_pct"],
                    "regret": value["mean_regret_pct"],
                }
                for key, value in summary["by_strategy"].items()
            },
            "raw_eval_path": str(report_path.relative_to(REPO).as_posix()),
        }

    return output


def load_chart_data():
    """Return chart inputs computed from the evaluation reports on disk."""
    return {
        "stage": _stage_model_data(),
        "supervisor": _supervisor_data(),
    }


def main():
    data = load_chart_data()
    print("STAGE MODELS")
    for key, _ in MODELS:
        record = data["stage"][key]
        print(f"  {record['label']}")
        for arm in ("epoch_1", "epoch_2"):
            stats = record[arm]["offer_dev_stats"]["pooled"]
            counts = record[arm]["offer_dev_stats"]["by_stage"]
            count_text = ", ".join(
                f"{stage}={counts[stage]['n']}" for stage in STAGES
            )
            print(
                f"    {arm}: {stats['mean']:.4f} +/- {stats['std']:.4f}% "
                f"(n={stats['n']}; {count_text})"
            )

    print("\nSUPERVISOR")
    for tag in ("base", "epoch_1", "epoch_2"):
        summary = data["supervisor"][tag]
        print(
            f"  {tag:8s} n={summary['n']:5d} "
            f"accuracy={summary['accuracy']:6.2f}% "
            f"regret={summary['mean_regret']:7.3f}%"
        )


if __name__ == "__main__":
    main()
