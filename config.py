from pathlib import Path
import json
import os
from datetime import datetime

# ── single change point ───────────────────────────────────────────────────────
DATA_VERSION = "v4"

# the reverse-auction fine-tuning set is versioned separately: it is rebuilt
# whenever the auction rules change, while the synthetic order data does not move.
# REVERSE_AUCTION_DATASET points generation, training and evaluation at a sibling
# folder for a controlled ablation, without moving the version lineage itself
REVERSE_AUCTION_VERSION = os.getenv("REVERSE_AUCTION_DATASET", "v4")

VERSION_NOTES = {
    "v1": "initial baseline data from first meeting",
    "v2": "improved data with real-world grounded pricing and varied agent profiles",
    "v3": "enhanced data and make it competitive and enhance generated orders data",
    "v4": "scaled order data to 50,000 orders, agent profiles re-checked against bid_check at new scale",
}

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent
RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / DATA_VERSION


def get_version_info(num_rows: dict) -> dict:
    return {
        "version":      DATA_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "notes":        VERSION_NOTES.get(DATA_VERSION, ""),
        "row_counts":   num_rows,
    }


def save_version_info(num_rows: dict) -> None:
    info = get_version_info(num_rows)
    path = SYNTHETIC_DATA_DIR / "version_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"saved version_info.json for {DATA_VERSION}")
