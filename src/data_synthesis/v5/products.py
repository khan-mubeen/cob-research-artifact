"""the v5 product-material table: v4's table, unchanged

  PYTHONPATH=. python -m src.data_synthesis.v5.products

every v4 pair has a v5 price, so all twelve are kept with v4's physical values. the
table is written into data/synthetic/v5 so the dataset directory is self-contained
"""

import sys

import pandas as pd

from src.data_synthesis.v5.settings import V4_DIR, V5_DIR, rel, write_csv

OUTPUT_PATH = V5_DIR / "products.csv"


def build():
    v4 = pd.read_csv(V4_DIR / "products.csv")
    write_csv(v4, OUTPUT_PATH)
    return v4


def main():
    frame = build()
    print(f"{len(frame)} product-material pairs, as in v4")
    print(f"wrote {rel(OUTPUT_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
