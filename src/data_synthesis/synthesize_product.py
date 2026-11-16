from pathlib import Path

import pandas as pd

from config import SYNTHETIC_DATA_DIR

OUTPUT_PATH = SYNTHETIC_DATA_DIR / "products.csv"

# finished weight includes thread, fasteners, lining, and other manufacturing inputs

PRODUCTS = [
    # shirts
    {
        "product_id":                  "shirt",
        "material_type":               "cotton",
        "material_kg_per_unit":        0.10,
        "finished_weight_kg_per_unit": 0.15,
    },
    {
        "product_id":                  "shirt",
        "material_type":               "linen",
        "material_kg_per_unit":        0.08,
        "finished_weight_kg_per_unit": 0.12,
    },
    {
        "product_id":                  "shirt",
        "material_type":               "polyester",
        "material_kg_per_unit":        0.09,
        "finished_weight_kg_per_unit": 0.13,
    },

    # hoodies
    {
        "product_id":                  "hoodie",
        "material_type":               "cotton",
        "material_kg_per_unit":        0.70,
        "finished_weight_kg_per_unit": 0.85,
    },
    {
        "product_id":                  "hoodie",
        "material_type":               "polyester",
        "material_kg_per_unit":        0.65,
        "finished_weight_kg_per_unit": 0.80,
    },

    # jackets
    {
        "product_id":                  "jacket",
        "material_type":               "leather",
        "material_kg_per_unit":        1.20,
        "finished_weight_kg_per_unit": 1.50,
    },
    {
        "product_id":                  "jacket",
        "material_type":               "polyester",
        "material_kg_per_unit":        0.80,
        "finished_weight_kg_per_unit": 1.00,
    },
    {
        "product_id":                  "jacket",
        "material_type":               "linen",
        "material_kg_per_unit":        0.75,
        "finished_weight_kg_per_unit": 0.95,
    },

    # jeans
    {
        "product_id":                  "jeans",
        "material_type":               "denim",
        "material_kg_per_unit":        0.85,
        "finished_weight_kg_per_unit": 1.00,
    },

    # skirts
    {
        "product_id":                  "skirt",
        "material_type":               "cotton",
        "material_kg_per_unit":        0.30,
        "finished_weight_kg_per_unit": 0.38,
    },
    {
        "product_id":                  "skirt",
        "material_type":               "linen",
        "material_kg_per_unit":        0.28,
        "finished_weight_kg_per_unit": 0.35,
    },
    {
        "product_id":                  "skirt",
        "material_type":               "polyester",
        "material_kg_per_unit":        0.25,
        "finished_weight_kg_per_unit": 0.32,
    },
]


def synthesize(output_path):
    df = pd.DataFrame(PRODUCTS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"saved {len(df)} product entries to {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    synthesize(OUTPUT_PATH)
