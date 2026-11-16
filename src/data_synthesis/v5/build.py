"""build data/synthetic/v5 end to end

  PYTHONPATH=. python -m src.data_synthesis.v5.build

order matters: the agent calibration feeds the logistics profile, and every profile
feeds the order budgets. every file is then validated against its v5 schema. nothing
here touches data/synthetic/v4, the v4 generators, or any live bidding path
"""

import sys

import pandas as pd

from src.dataset.schemas import schema_for
from src.data_synthesis.v5 import (
    agent_calibration, logistics, manufacturers, materials, orders, products,
)
from src.data_synthesis.v5.settings import V5_DIR, rel

SCHEMA_FILES = {"material_agents": materials.PROFILE_PATH,
                "manufacturer_agents": V5_DIR / "manufacturer_agents.csv",
                "products": products.OUTPUT_PATH,
                "orders": orders.OUTPUT_PATH}


def validate():
    """(file, problems) for every v5 file; logistics uses its own v5 schema"""
    report = {}
    for kind, path in SCHEMA_FILES.items():
        report[rel(path)] = schema_for(kind, "v5").validate_csv(path, limit=5)
    rows = pd.read_csv(logistics.PROFILE_PATH).to_dict("records")
    report[rel(logistics.PROFILE_PATH)] = logistics.LOGISTICS_SCHEMA_V5.validate_rows(rows, limit=5)
    return report


def main():
    print("part 1  agent calibration")
    _, _, fuel_check = agent_calibration.build()
    print(f"        shared fuel series {fuel_check['shared_series']}")
    print("part 2  manufacturer usd conversion")
    _, rate = manufacturers.build()
    print(f"        EUR/USD {rate.rate} on {rate.rate_date}")
    print("part 3  materials")
    material_build = materials.build()
    for material, fixed in material_build["fixed_prices"].items():
        print(f"        {material} fixed at {fixed['usd_per_kg']:.4f} USD/kg, no temporal series")
    print("part 3  products")
    kept = products.build()
    print(f"        {len(kept)} pairs")
    print("part 3  logistics profile")
    logistics.build()
    print("part 3  orders and budgets")
    frame, _ = orders.build()
    print(f"        {len(frame)} orders")
    print("schema validation")
    report = validate()
    failures = {path: problems for path, problems in report.items() if problems}
    for path in sorted(report):
        print(f"  {'FAIL' if path in failures else 'ok  '} {path}")
    if failures:
        print(failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
