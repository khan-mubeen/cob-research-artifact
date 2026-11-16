"""phase 4 determinism: every v5 generator run again into a scratch directory, byte-compared with
the manifest

  PYTHONPATH=. COB_DATASET_VERSION=v5 python -m src.experiments.v5.rerun_check

each generator's main() runs unmodified with its OUT_DIR pointed at data/finetune/.v5_rerun/, so
the committed corpus is never rewritten. the edge and floor-boundary generators read the committed
sealed and reverse-auction test files, which the rerun must reproduce anyway. the split is
recomputed in memory and compared with order_split.csv row for row. a rerun that differs in any
byte from data/finetune/v5/manifest.json fails

writes reports/v5/phase4_rerun_check.json
"""

import io
import json
import shutil
import sys
from contextlib import redirect_stdout

import pandas as pd

from config import PROJECT_ROOT
from src.experiments.v5 import (
    generate_edge_examples, generate_finetune_data, generate_floor_boundary, generate_reverse_auction_data,
    generate_supervisor_data,
)
from src.experiments.v5.common import FINETUNE_V5, SPLIT_PATH, require_v5, sha256_file
from src.experiments.v5.split import assign

SCRATCH = PROJECT_ROOT / "data" / "finetune" / ".v5_rerun"
REPORT_PATH = PROJECT_ROOT / "reports" / "v5" / "phase4_rerun_check.json"
MODULES = (
    ("", generate_finetune_data),
    ("reverse_auction", generate_reverse_auction_data),
    ("supervisor", generate_supervisor_data),
    ("edge", generate_edge_examples),
    ("reverse_auction/floor_boundary", generate_floor_boundary),
)


def main():
    require_v5()
    manifest = {f["path"]: f["sha256"] for f in json.loads((FINETUNE_V5 / "manifest.json").read_text(encoding="utf-8"))["files"]}
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    compared, differing = 0, []
    try:
        for relative, module in MODULES:
            out = SCRATCH / relative if relative else SCRATCH
            module.OUT_DIR = out
            with redirect_stdout(io.StringIO()):
                module.main()
            for path in sorted(p for p in out.iterdir() if p.is_file()):
                key = f"{relative}/{path.name}" if relative else path.name
                compared += 1
                if manifest.get(key) != sha256_file(path):
                    differing.append(key)
            print(f"reran {relative or 'sealed'}: {sum(1 for p in out.iterdir() if p.is_file())} files")
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    orders = pd.read_csv(FINETUNE_V5.parent.parent / "synthetic" / "v5" / "orders.csv",
                         usecols=["order_id", "scenario_type", "priority"])
    split_of, _, _ = assign(orders)
    stored = pd.read_csv(SPLIT_PATH)
    split_identical = dict(zip(stored["order_id"], stored["split"])) == split_of

    result = {"files_compared": compared, "files_differing": differing, "split_recomputed_identical": split_identical,
              "passed": not differing and split_identical and compared == sum(1 for p in manifest if not p.startswith("split/"))}
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
