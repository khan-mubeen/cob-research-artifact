"""unit tests for the record schema contracts.
runnable directly (python ...) or under pytest.

the contract is only worth having if it does two opposite things at once: pass
the committed v4 corpus exactly as it is stored, untyped monetary column names
included, and refuse a v5-shaped record that has lost its unit, currency,
version or pricing date instead of letting it through with nulls
"""

from src.dataset.context import for_version
from src.dataset.schemas import (
    ALL_MATERIALS, ALL_PRODUCTS, Field, RecordSchema, SchemaError, schema_for,
    schemas_for_version, validate_dataset,
)


V4 = for_version("v4")


def raises(fn, fragment):
    try:
        fn()
    except SchemaError as e:
        assert fragment in str(e), f"expected {fragment!r} in the error, got {e}"
        return
    raise AssertionError(f"expected a SchemaError mentioning {fragment!r}")


def v5_order(**overrides):
    record = {
        "order_id": "ORD-V5-1",
        "order_created": "2026-07-17T11:00:00",
        "product_id": "shirt",
        "material_type": "cotton",
        "quantity": 250,
        "budget_usd": 1234.56,
        "deadline": "2026-08-01",
        "deliver_to": "Tampere",
        "max_defect_rate": 0.05,
        "min_reputation_score": 0.5,
        "priority": "normal",
        "dataset_version": "v5",
        "currency": "USD",
        "pricing_date": "2026-07-17",
        "unit": "per_order",
    }
    record.update(overrides)
    return record


# ── the committed v4 corpus satisfies its own contract ──────────────────────

def test_every_v4_csv_passes_its_schema():
    report = validate_dataset(V4)
    assert report == {}, f"v4 no longer satisfies its schema: {report}"


def test_the_v4_schemas_name_the_columns_the_v4_files_actually_have():
    import pandas as pd

    for kind, schema in schemas_for_version("v4").items():
        header = set(pd.read_csv(V4.synthetic_dir / f"{kind}.csv", nrows=0).columns)
        missing = [f.name for f in schema.fields if f.required and f.name not in header]
        assert not missing, f"{kind}.csv has no column for {missing}"


def test_v4_monetary_fields_declare_eur_even_where_the_column_name_does_not():
    """budget, base_delivery_cost, cost_per_kg and unit_cost_* say nothing about
    currency in the csv. the schema is where that is finally written down"""
    order = schema_for("orders", "v4")
    budget = next(f for f in order.fields if f.name == "budget")
    assert budget.currency == "EUR" and budget.unit == "per_order"

    logistics = schema_for("logistics_agents", "v4")
    assert {f.name for f in logistics.monetary_fields} == {"base_delivery_cost", "cost_per_kg"}
    assert all(f.currency == "EUR" for f in logistics.monetary_fields)

    manufacturer = schema_for("manufacturer_agents", "v4")
    assert {f.name for f in manufacturer.monetary_fields} == {
        f"unit_cost_{p}" for p in ALL_PRODUCTS}

    material = schema_for("material_agents", "v4")
    assert {f.name for f in material.monetary_fields} == {
        f"{m}_base_cost_eur_per_kg" for m in ALL_MATERIALS}


def test_v4_has_no_pricing_date_currency_or_unit_field():
    """v4 prices are fixed. a v4 schema that required a pricing date would be
    describing a dataset that does not exist"""
    for schema in schemas_for_version("v4").values():
        assert schema.requires_pricing_date is False
        names = set(schema.field_names)
        assert not (names & {"pricing_date", "currency", "dataset_version", "unit"})


def test_products_carry_no_currency_at_all():
    products = schema_for("products", "v5")
    assert products.monetary_fields == ()


# ── a v5 record without its envelope is not a v5 record ─────────────────────

def test_a_complete_v5_order_passes():
    assert schema_for("orders", "v5").errors(v5_order()) == []


def test_a_v5_order_missing_unit_currency_version_or_date_fails():
    schema = schema_for("orders", "v5")
    for missing in ("unit", "currency", "dataset_version", "pricing_date"):
        record = v5_order()
        record.pop(missing)
        problems = schema.errors(record)
        assert any(p.startswith(f"{missing}:") for p in problems), (
            f"dropping {missing} produced {problems}")


def test_a_v5_order_with_those_fields_present_but_null_fails():
    """the exact failure the contract exists to prevent: passing with nulls"""
    schema = schema_for("orders", "v5")
    for blanked, empty in (("unit", ""), ("currency", None),
                           ("pricing_date", float("nan")), ("dataset_version", "  ")):
        problems = schema.errors(v5_order(**{blanked: empty}))
        assert any(p.startswith(f"{blanked}:") for p in problems), (
            f"blanking {blanked} produced {problems}")


def test_a_v5_record_declaring_the_wrong_version_or_currency_fails():
    schema = schema_for("orders", "v5")
    problems = schema.errors(v5_order(dataset_version="v4"))
    assert any("this is the v5 schema" in p for p in problems), problems
    problems = schema.errors(v5_order(currency="EUR"))
    assert any("v5 data is USD" in p for p in problems), problems


def test_a_v4_order_row_does_not_pass_the_v5_schema():
    import pandas as pd

    row = pd.read_csv(V4.synthetic_dir / "orders.csv", nrows=1).to_dict("records")[0]
    problems = schema_for("orders", "v5").errors(row)
    for expected in ("budget_usd: missing", "dataset_version: missing",
                     "currency: missing", "pricing_date: missing", "unit: missing"):
        assert expected in problems, f"{expected!r} not in {problems}"


def test_every_v5_schema_that_prices_something_requires_the_envelope():
    for kind, schema in schemas_for_version("v5").items():
        if kind == "products":
            continue
        names = set(schema.field_names)
        assert {"unit", "currency", "dataset_version", "pricing_date"} <= names, kind


# ── ordinary field-level validation ─────────────────────────────────────────

def test_types_ranges_and_enums_are_enforced():
    schema = schema_for("orders", "v5")
    assert any("expected a number" in p for p in schema.errors(v5_order(budget_usd="free")))
    assert any("expected a whole number" in p for p in schema.errors(v5_order(quantity=2.5)))
    assert any("below the minimum" in p for p in schema.errors(v5_order(budget_usd=-1.0)))
    assert any("above the maximum" in p
               for p in schema.errors(v5_order(max_defect_rate=1.5)))
    assert any("is not one of" in p for p in schema.errors(v5_order(priority="whenever")))
    assert any("not an iso date" in p for p in schema.errors(v5_order(deadline="soon")))


def test_a_nullable_column_may_be_empty_but_a_required_one_may_not():
    material = schema_for("material_agents", "v4")
    row = {f.name: 0.5 for f in material.fields}
    row.update({"agent_id": "M01", "stage": "material", "location": "Helsinki",
                "material_types": "cotton"})
    assert material.errors(row) == []
    # a supplier that does not stock leather leaves the whole block empty
    row["leather_available_kg"] = float("nan")
    row["leather_base_cost_eur_per_kg"] = float("nan")
    assert material.errors(row) == []
    row["defect_probability"] = float("nan")
    assert any("not nullable" in p for p in material.errors(row))


def test_validate_raises_with_every_problem_listed():
    schema = schema_for("orders", "v5")
    try:
        schema.validate(v5_order(priority="whenever", quantity=0))
    except SchemaError as e:
        assert "priority" in str(e) and "quantity" in str(e)
        return
    raise AssertionError("an invalid record validated")


def test_an_unknown_record_kind_or_version_raises():
    raises(lambda: schema_for("carriers", "v4"), "unknown record kind")
    raises(lambda: schema_for("orders", "v9"), "unknown dataset version")


def test_a_malformed_schema_definition_raises_at_definition_time():
    raises(lambda: Field("x", "money"), "unknown field kind")
    raises(lambda: Field("x", "enum"), "needs choices")
    raises(lambda: Field("x", "number", currency="XBT"), "unknown currency")
    raises(lambda: RecordSchema(name="dupe", version="v4", currency="EUR",
                                fields=(Field("a", "str"), Field("a", "str"))),
           "duplicate field names")


# ── the contract is machine readable ────────────────────────────────────────

def test_every_schema_serializes_to_json():
    import json

    for version in ("v4", "v5"):
        for kind, schema in schemas_for_version(version).items():
            blob = json.loads(json.dumps(schema.as_dict()))
            assert blob["dataset_version"] == version
            assert blob["fields"], kind
            names = [f["name"] for f in blob["fields"]]
            assert len(names) == len(set(names))


def test_the_serialized_contract_carries_unit_and_currency():
    blob = schema_for("material_agents", "v5").as_dict()
    cotton = next(f for f in blob["fields"] if f["name"] == "cotton_base_cost_usd_per_kg")
    assert cotton["unit"] == "per_kg" and cotton["currency"] == "USD"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} dataset schema unit tests passed")


if __name__ == "__main__":
    main()
