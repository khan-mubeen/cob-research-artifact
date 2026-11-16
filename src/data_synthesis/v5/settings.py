"""fixed inputs shared by the v5 synthesis modules

the order window below is a provisional decision pending confirmation, not a
permanent methodology choice. it runs 180 days from 2025-10-15 so the monthly
price series that move v5 material prices carry real observations throughout,
and it ends at the tightest of them, the denim PPI, whose last observation is
2026-04-01. v4's own window, from 2026-07-11, is untouched and still describes v4

v4's day convention is kept: an order is created on the base date plus 0 to 179
days, so the last possible order_created date is 2026-04-12 and 2026-04-13 is the
exclusive end of the window
"""

import hashlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.dataset.acquire_v5_sources import write_json  # noqa: F401  re-exported for the v5 modules

WINDOW_STATUS = "provisional decision pending confirmation, not a permanent methodology choice"
WINDOW_BASIS = (
    "180 days ending at the last observation of the tightest monthly price series, the denim "
    "PPI WPU03420104 (2026-04-01), so cotton, denim and leather prices move on real "
    "observations across the whole window instead of carrying one stale value forward")
WINDOW_START = date(2025, 10, 15)
CREATION_WINDOW_DAYS = 180
WINDOW_END_EXCLUSIVE = WINDOW_START + timedelta(days=CREATION_WINDOW_DAYS)
WINDOW_DAYS = tuple(WINDOW_START + timedelta(days=i) for i in range(CREATION_WINDOW_DAYS))

SEED = 42
TOTAL_ORDERS = 50_000

# the one fixed rate the manufacturer unit costs are relabelled at
FX_REFERENCE_DATE = WINDOW_START
FX_REFERENCE_BASIS = (
    "the first day of the v5 order window: the earliest date any v5 order exists, and an "
    "ECB publication day, so the rate is read exactly rather than carried forward")

ALL_MATERIALS = ("cotton", "polyester", "linen", "denim", "leather")
ALL_PRODUCTS = ("shirt", "hoodie", "jacket", "jeans", "skirt")

# every material is priced. polyester's price is fixed rather than dated, because its
# temporal series is missing; methodology section 5.1 keeps a sourced price fixed in
# exactly that case instead of excluding the material
FIXED_PRICE_MATERIALS = {
    "polyester": ("fixed at the Tabb Textile TWPWD finished-fabric anchor for every date: "
                  "data/raw/v5/material/polyester/sunsirs_dty.json still declares "
                  "observations_status=missing, so no temporal series moves it"),
}
DATED_MATERIALS = tuple(m for m in ALL_MATERIALS if m not in FIXED_PRICE_MATERIALS)

# copied verbatim from PRIORITY_MARKUP in src/data_synthesis/synthesize_orders.py; the
# budget estimator and the v5 logistics price both apply it
PRIORITY_MARKUP = {"urgent": 0.30, "normal": 0.00, "flexible": -0.15}

BETA_EUR_PER_TONNE_KM = 0.07163
BETA_BASIS = (
    "used as directly specified by the project supervisor, identical for every agent because "
    "no documented reason to vary it exists. its derivation remains a candidate, not confirmed, "
    "exactly as data/processed/v5/logistics/beta_candidate_derivation.json records")

# every logistics agent follows this one fuel-surcharge series, resolved by each order's
# own date. mapping agents to different carriers' series made the carriers' incomparable
# percentage bases decide which agents won, so the series is shared, not per agent
SHARED_FUEL_SERIES_ID = "matkahuolto_lavarahti"
SHARED_FUEL_BASIS = (
    "one common temporal fuel-cost signal for every agent. Matkahuolto Lavarahti has the best "
    "coverage of the v5 window among the usable pinned series: 22 dated observations inside the "
    "window and no day carried past its last observation. absolute surcharge levels published by "
    "different carriers sit on different bases and are not comparable, so agents do not differ "
    "by fuel series in v5")

V4_DIR = PROJECT_ROOT / "data" / "synthetic" / "v4"
V5_DIR = PROJECT_ROOT / "data" / "synthetic" / "v5"
RAW_V5 = PROJECT_ROOT / "data" / "raw" / "v5"
PROCESSED_V5 = PROJECT_ROOT / "data" / "processed" / "v5"
REPORTS_V5 = PROJECT_ROOT / "reports" / "v5"

AGENT_CALIBRATION_PATH = PROCESSED_V5 / "logistics" / "agent_calibration.csv"
DISTANCE_MATRIX_PATH = PROCESSED_V5 / "logistics" / "manufacturer_to_city_distance_matrix.csv"


def rel(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(frame, path):
    """csv with unix newlines, so a rebuild on any machine gives the same bytes"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def read_csv(path):
    return pd.read_csv(path)
