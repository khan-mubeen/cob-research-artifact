"""resolve report locations after the epoch/result-family reorganization.

before the reorganization every evaluation JSON lived in one mixed directory
(`reports/slm_comparison/<version>/`). the plan splits them by result family:

  production   reports/epoch-{n}/production/evaluation/raw/     eval_*-v2*
  ablation     reports/epoch-{n}/ablation/evaluation/raw/       eval_finetuned*, r1/r2/r3, granite-ft/v1
  reference    reports/reference/base-evaluations/raw/          eval_base*, eval_*-base*
  deployment   reports/epoch-{n}/deployment/qwen-r1-gguf-fix/   eval_r1-merged-hf_*

consumers must therefore search several roots instead of globbing one. use
`find_evals(tag)` for reads and `report_dir(...)` for writes, so no script
hardcodes a post-move path.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPOCH = 1


def epoch_root(epoch: int = DEFAULT_EPOCH) -> Path:
    return PROJECT_ROOT / "reports" / f"epoch-{epoch}"


def report_dir(category: str = "production", subdir: str = "",
               epoch: int = DEFAULT_EPOCH) -> Path:
    """category: production | ablation | deployment | reference"""
    if category == "reference":
        base = PROJECT_ROOT / "reports" / "reference"
    else:
        base = epoch_root(epoch) / category
    return base / subdir if subdir else base


def eval_roots(epoch: int = DEFAULT_EPOCH) -> list[Path]:
    """every directory that can hold a raw eval JSON, in search order"""
    return [
        report_dir("production", "evaluation/raw", epoch),
        report_dir("ablation", "evaluation/raw", epoch),
        report_dir("reference", "base-evaluations/raw"),
        report_dir("deployment", "qwen-r1-gguf-fix", epoch),
    ]


def find_evals(pattern: str, epoch: int = DEFAULT_EPOCH) -> list[Path]:
    """all eval files matching a glob pattern across every raw root, sorted by
    name so the newest timestamped run is last (the previous glob semantics)"""
    hits: list[Path] = []
    for root in eval_roots(epoch):
        if root.is_dir():
            hits.extend(root.glob(pattern))
    return sorted(hits, key=lambda p: p.name)


def latest_eval(pattern: str, epoch: int = DEFAULT_EPOCH) -> Path | None:
    hits = find_evals(pattern, epoch)
    return hits[-1] if hits else None
