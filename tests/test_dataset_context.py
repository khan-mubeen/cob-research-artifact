"""unit tests for the explicit dataset version/currency context.
runnable directly (python ...) or under pytest.

two things matter here. resolving nothing still gives exactly today's v4
behavior, so defining the context cannot have changed anything. and every way of
building a context that mixes a version with the wrong currency, the wrong
pricing-date rule or another version's column names raises rather than resolving
to something plausible
"""

import os
from dataclasses import replace

import config
from src.dataset.context import (
    DatasetContext, DatasetContextError, ENV_VAR, KNOWN_VERSIONS, REGISTRY,
    default_context, for_version, resolve,
)


def raises(fn, fragment):
    try:
        fn()
    except DatasetContextError as e:
        assert fragment in str(e), f"expected {fragment!r} in the error, got {e}"
        return
    raise AssertionError(f"expected a DatasetContextError mentioning {fragment!r}")


# ── the default still means v4 ───────────────────────────────────────────────

def test_resolving_nothing_gives_todays_v4_behavior():
    was = os.environ.pop(ENV_VAR, None)
    try:
        ctx = resolve()
    finally:
        if was is not None:
            os.environ[ENV_VAR] = was
    assert ctx.version == config.DATA_VERSION == "v4"
    assert ctx.currency == "EUR"
    assert ctx.requires_pricing_date is False
    assert ctx.synthetic_dir == config.SYNTHETIC_DATA_DIR


def test_the_default_context_resolves_the_v4_column_names_that_exist_today():
    ctx = default_context()
    assert ctx.material_cost_column("cotton") == "cotton_base_cost_eur_per_kg"
    assert ctx.manufacturer_unit_cost_column("shirt") == "unit_cost_shirt"
    assert ctx.logistics_base_cost_field == "base_delivery_cost"
    assert ctx.logistics_cost_per_kg_field == "cost_per_kg"
    assert ctx.order_budget_field == "budget"


def test_the_v4_columns_the_context_names_are_the_columns_the_formulas_read():
    """the names in the registry are the ones bidding_formulas looks up by hand"""
    import inspect

    from src.common import bidding_formulas

    source = inspect.getsource(bidding_formulas)
    ctx = default_context()
    assert 'f"{material_type}_base_cost_eur_per_kg"' in source
    assert ctx.material_cost_field == "{material}_base_cost_eur_per_kg"
    assert 'f"unit_cost_{product_id}"' in source
    assert ctx.manufacturer_unit_cost_field == "unit_cost_{product}"
    assert '"base_delivery_cost"' in source
    assert '"cost_per_kg"' in source


def test_the_environment_can_select_a_version_explicitly():
    was = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = "v5"
    try:
        ctx = resolve()
        assert ctx.version == "v5" and ctx.currency == "USD"
        # an explicit argument still beats the environment
        assert resolve("v4").version == "v4"
    finally:
        if was is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = was


def test_every_registered_version_builds():
    for version in KNOWN_VERSIONS:
        ctx = for_version(version)
        assert ctx.version == version
        assert ctx.currency == REGISTRY[version]["currency"]


# ── v5 is declared, and is not v4 with different words ──────────────────────

def test_v5_is_usd_dated_and_uses_its_own_column_names():
    ctx = for_version("v5")
    assert ctx.currency == "USD"
    assert ctx.requires_pricing_date is True
    assert ctx.material_cost_column("cotton") == "cotton_base_cost_usd_per_kg"
    assert ctx.manufacturer_unit_cost_column("shirt") == "unit_cost_usd_shirt"
    assert ctx.order_budget_field == "budget_usd"


def test_no_v5_column_name_collides_with_a_v4_one():
    """a v5 reader pointed at v4 data must find nothing, not a plausible number"""
    v4, v5 = for_version("v4"), for_version("v5")
    v4_names = {v4.material_cost_column(m) for m in ("cotton", "linen")} | {
        v4.manufacturer_unit_cost_column("shirt"), v4.logistics_base_cost_field,
        v4.logistics_cost_per_kg_field, v4.order_budget_field}
    v5_names = {v5.material_cost_column(m) for m in ("cotton", "linen")} | {
        v5.manufacturer_unit_cost_column("shirt"), v5.logistics_base_cost_field,
        v5.logistics_cost_per_kg_field, v5.order_budget_field}
    assert not (v4_names & v5_names), f"shared column names: {v4_names & v5_names}"


# ── nothing falls back ──────────────────────────────────────────────────────

def test_an_unknown_version_raises():
    raises(lambda: resolve("v9"), "unknown dataset version")
    raises(lambda: for_version(""), "unknown dataset version")


def test_an_unknown_version_in_the_environment_raises():
    was = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = "latest"
    try:
        raises(resolve, "unknown dataset version")
    finally:
        if was is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = was


def test_a_version_with_the_wrong_currency_raises():
    raises(lambda: replace(for_version("v4"), currency="USD"),
           "dataset v4 is denominated in EUR")
    raises(lambda: replace(for_version("v5"), currency="EUR"),
           "dataset v5 is denominated in USD")


def test_an_unknown_currency_raises():
    raises(lambda: replace(for_version("v4"), currency="GBP"), "unknown currency")


def test_a_version_with_the_wrong_pricing_date_rule_raises():
    raises(lambda: replace(for_version("v4"), requires_pricing_date=True),
           "requires_pricing_date")
    raises(lambda: replace(for_version("v5"), requires_pricing_date=False),
           "requires_pricing_date")


def test_borrowing_another_versions_column_names_raises():
    raises(lambda: replace(for_version("v5"), order_budget_field="budget"),
           "stores order_budget in")
    raises(lambda: replace(for_version("v5"),
                           material_cost_field="{material}_base_cost_eur_per_kg"),
           "stores material_cost in")
    raises(lambda: replace(for_version("v4"), logistics_cost_per_kg_field="cost_per_kg_usd"),
           "stores logistics_cost_per_kg in")


def test_a_hand_built_context_is_checked_the_same_way():
    raises(lambda: DatasetContext(
        version="v5", currency="USD", requires_pricing_date=True,
        material_cost_field="{material}_base_cost_usd_per_kg",
        manufacturer_unit_cost_field="unit_cost_{product}",   # the v4 name
        logistics_base_cost_field="base_delivery_cost_usd",
        logistics_cost_per_kg_field="cost_per_kg_usd",
        order_budget_field="budget_usd"), "stores manufacturer_unit_cost in")


def test_a_context_cannot_be_edited_in_place():
    ctx = for_version("v4")
    try:
        ctx.currency = "USD"
    except Exception as e:
        assert type(e).__name__ in ("FrozenInstanceError", "AttributeError")
        return
    raise AssertionError("a DatasetContext was mutated")


# ── the explicit checks a consumer will make ────────────────────────────────

def test_require_currency_and_version_raise_on_a_mismatch():
    ctx = for_version("v4")
    ctx.require_currency("EUR")
    ctx.require_version("v4")
    raises(lambda: ctx.require_currency("USD"), "currency mismatch")
    raises(lambda: ctx.require_version("v5"), "version mismatch")


def test_a_missing_dataset_directory_raises_rather_than_returning_a_path():
    """data/synthetic/v5 exists since the v5 synthesis task, so the missing case is
    checked against an empty root instead of relying on v5 being absent"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        raises(for_version("v5", root=tmp).require_synthetic_data, "has no directory at")
    assert for_version("v4").require_synthetic_data().is_dir()


def test_the_context_serializes_for_metadata_and_messages():
    assert for_version("v4").as_dict() == {
        "dataset_version": "v4", "currency": "EUR", "requires_pricing_date": False}
    assert for_version("v5").as_dict() == {
        "dataset_version": "v5", "currency": "USD", "requires_pricing_date": True}


# ── the import rule that keeps this out of a cycle ──────────────────────────

def test_the_lookup_and_fx_layers_do_not_import_the_context_or_config():
    """part 5 and part 6 take a resolved context as an argument, never fetch one

    this is the rule that stops config, context and the lookup layer forming a
    cycle, and it is easier to keep if something checks it"""
    import inspect

    from src.dataset import fx, history

    for module in (history, fx):
        source = inspect.getsource(module)
        for banned in ("import config", "from config", "dataset.context",
                       "dataset import context"):
            assert banned not in source, (
                f"{module.__name__} imports {banned!r}, which it must not")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} dataset context unit tests passed")


if __name__ == "__main__":
    main()
