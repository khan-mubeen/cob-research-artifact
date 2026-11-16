"""phase 6: small seeded samples of real v5 training rows for the timing smoke run

  PYTHONPATH=. python -m src.experiments.v5.smoke_sample

finetune_lora.py trains on --train-jsonl for anything outside its two v4 dataset names, so the
smoke run reads one sample file per mode. each file holds a seeded, uniformly drawn sample of the
real v5 train rows of that mode, stages mixed in proportion to their row counts, so the measured
step time reflects the real v5 length mix. the samples live in data/finetune/.v5_smoke/, beside
data/finetune/v5 and outside its manifest, and are timing inputs only, never training data
"""

import json
import random
import sys

from config import PROJECT_ROOT

FINETUNE_V5 = PROJECT_ROOT / "data" / "finetune" / "v5"
OUT_DIR = PROJECT_ROOT / "data" / "finetune" / ".v5_smoke"
ROWS = 480
SEED = 42
FILES = {
    "sealed": [f"{stage}_train.jsonl" for stage in ("material", "tailoring", "logistics")],
    "reverse_auction": [f"reverse_auction/{stage}_train.jsonl" for stage in ("material", "tailoring", "logistics")],
    "supervisor": ["supervisor/selection_train.jsonl"],
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode, names in FILES.items():
        rng = random.Random(SEED)
        sample, seen = [], 0
        for name in names:
            with open(FINETUNE_V5 / name, encoding="utf-8") as f:
                for line in f:
                    seen += 1
                    if len(sample) < ROWS:
                        sample.append(line)
                    elif (j := rng.randrange(seen)) < ROWS:
                        sample[j] = line
        path = OUT_DIR / f"{mode}_smoke.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(sample)
        stages = {}
        for line in sample:
            stage = json.loads(line)["stage"]
            stages[stage] = stages.get(stage, 0) + 1
        print(f"{mode}: {len(sample)} of {seen} train rows -> {path.relative_to(PROJECT_ROOT)} {stages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
