"""the explicit dataset context downstream code will be handed instead of
resolving a global default at import time

config.DATA_VERSION is one module-level constant that nearly every pipeline
reads when it is first imported. that is fine while there is one dataset. it
stops being fine the moment v4 (EUR, fixed prices) and v5 (USD, date-matched
prices) have to be generated, priced, trained and evaluated side by side,
because "which dataset is this" would be answered by whichever process happened
to import first

a DatasetContext answers it per call instead. nothing consumes it yet: this
module defines it, proves it resolves to today's v4 behavior by default, and
proves that a mixed or unknown context raises rather than falling back. wiring
it into the agents, supervisor and synthesis is a later, separate step

import direction, deliberately one way:
    context -> config          (for the project root and the current default)
    context -> nothing else in src
the historical lookup and fx layers never import this module. they are handed
whatever a resolved context already decided, as plain arguments, so config,
context and the lookup layer cannot form a cycle
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import config


class DatasetContextError(ValueError):
    """an unknown, incomplete or internally inconsistent dataset context

    always raised, never swallowed into a default: a context that cannot be
    resolved is a configuration bug, and guessing hides which dataset produced
    a number"""


# iso 4217 codes this project actually denominates data in. a code outside this
# set is a typo until someone adds a dataset that uses it
KNOWN_CURRENCIES = ("EUR", "USD", "ZAR")

STAGES = ("material", "tailoring", "logistics")


# ── the registry ─────────────────────────────────────────────────────────────
# one row per dataset that exists or is being built. adding a version means
# adding a row here, not passing a free-form string around
#
# columns are the names as actually stored. three of v4's five monetary columns
# carry no currency token at all: unit_cost_shirt, base_delivery_cost, cost_per_kg
# and budget are EUR by prompt wording and convention only. the registry is what
# records that they are EUR, so nothing has to rename a v4 column to describe it.
# v5 names every monetary column after its currency, following the one v4 column
# that already does, so a v5 reader pointed at v4 finds nothing rather than a
# plausible wrong number

V4_COLUMNS = {
    "material_cost": "{material}_base_cost_eur_per_kg",
    "manufacturer_unit_cost": "unit_cost_{product}",
    "logistics_base_cost": "base_delivery_cost",
    "logistics_cost_per_kg": "cost_per_kg",
    "order_budget": "budget",
}

V5_COLUMNS = {
    "material_cost": "{material}_base_cost_usd_per_kg",
    "manufacturer_unit_cost": "unit_cost_usd_{product}",
    "logistics_base_cost": "base_delivery_cost_usd",
    "logistics_cost_per_kg": "cost_per_kg_usd",
    "order_budget": "budget_usd",
}

REGISTRY = {
    "v1": {"currency": "EUR", "requires_pricing_date": False, "columns": V4_COLUMNS},
    "v2": {"currency": "EUR", "requires_pricing_date": False, "columns": V4_COLUMNS},
    "v3": {"currency": "EUR", "requires_pricing_date": False, "columns": V4_COLUMNS},
    "v4": {"currency": "EUR", "requires_pricing_date": False, "columns": V4_COLUMNS},
    "v5": {"currency": "USD", "requires_pricing_date": True, "columns": V5_COLUMNS},
}

KNOWN_VERSIONS = tuple(sorted(REGISTRY))


@dataclass(frozen=True)
class DatasetContext:
    """everything a consumer needs to read one dataset without guessing

    frozen so a context cannot be edited halfway down a call stack. the column
    templates live here rather than in the consumers because that is the seam
    where a v4 number could be read as a v5 one: bidding_formulas reads
    {material}_base_cost_eur_per_kg by name today, and a v5 reader that finds
    that column would price v5 offers off frozen EUR data without complaining

    build one through resolve() or for_version(). constructing it directly is
    allowed and fully checked: every field must agree with REGISTRY"""

    version: str
    currency: str
    # the immutable date a v5 price is resolved as of. v4 prices are fixed, so
    # v4 has no pricing date and must not invent one
    requires_pricing_date: bool
    material_cost_field: str            # formatted with material=
    manufacturer_unit_cost_field: str   # formatted with product=
    logistics_base_cost_field: str
    logistics_cost_per_kg_field: str
    order_budget_field: str
    notes: str = ""
    _root: Path = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.version not in REGISTRY:
            raise DatasetContextError(
                f"unknown dataset version {self.version!r}, "
                f"known versions are {list(KNOWN_VERSIONS)}")
        if self.currency not in KNOWN_CURRENCIES:
            raise DatasetContextError(
                f"unknown currency {self.currency!r}, "
                f"known currencies are {list(KNOWN_CURRENCIES)}")

        registered = REGISTRY[self.version]
        if self.currency != registered["currency"]:
            raise DatasetContextError(
                f"mixed context: dataset {self.version} is denominated in "
                f"{registered['currency']}, not {self.currency}")
        if self.requires_pricing_date != registered["requires_pricing_date"]:
            raise DatasetContextError(
                f"mixed context: dataset {self.version} requires_pricing_date is "
                f"{registered['requires_pricing_date']}, not {self.requires_pricing_date}")

        for key, attr in (("material_cost", "material_cost_field"),
                          ("manufacturer_unit_cost", "manufacturer_unit_cost_field"),
                          ("logistics_base_cost", "logistics_base_cost_field"),
                          ("logistics_cost_per_kg", "logistics_cost_per_kg_field"),
                          ("order_budget", "order_budget_field")):
            expected = registered["columns"][key]
            actual = getattr(self, attr)
            if actual != expected:
                raise DatasetContextError(
                    f"mixed context: dataset {self.version} stores {key} in "
                    f"{expected!r}, not {actual!r}")

    # ── resolved locations ──────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root or config.PROJECT_ROOT

    @property
    def synthetic_dir(self) -> Path:
        return self.root / "data" / "synthetic" / self.version

    @property
    def raw_source_dir(self) -> Path:
        """where this dataset's pinned source snapshots live

        v4 reads a handful of loose files straight out of data/raw, v5 keeps a
        per-source tree with a manifest under data/raw/v5"""
        return self.root / "data" / "raw" / self.version

    # ── column name resolution ──────────────────────────────────────────────

    def material_cost_column(self, material: str) -> str:
        return self.material_cost_field.format(material=material)

    def manufacturer_unit_cost_column(self, product: str) -> str:
        return self.manufacturer_unit_cost_field.format(product=product)

    # ── checks a caller can make explicit ───────────────────────────────────

    def require_currency(self, currency: str) -> None:
        """raise unless an amount's currency is this dataset's currency

        the v4 hazard this closes is adding two numbers whose column names never
        said what they were denominated in"""
        if currency != self.currency:
            raise DatasetContextError(
                f"currency mismatch: {self.version} data is {self.currency}, "
                f"got {currency!r}")

    def require_version(self, version: str) -> None:
        if version != self.version:
            raise DatasetContextError(
                f"version mismatch: this context is {self.version}, got {version!r}")

    def require_synthetic_data(self) -> Path:
        """the dataset directory, or a clear error naming what is missing"""
        if not self.synthetic_dir.is_dir():
            raise DatasetContextError(
                f"dataset {self.version} has no directory at {self.synthetic_dir}")
        return self.synthetic_dir

    def as_dict(self) -> dict:
        """the shape that belongs in generated metadata and message envelopes"""
        return {
            "dataset_version": self.version,
            "currency": self.currency,
            "requires_pricing_date": self.requires_pricing_date,
        }


def for_version(version, root=None) -> DatasetContext:
    """the registered context for one dataset version"""
    if version not in REGISTRY:
        raise DatasetContextError(
            f"unknown dataset version {version!r}, "
            f"known versions are {list(KNOWN_VERSIONS)}")
    registered = REGISTRY[version]
    columns = registered["columns"]
    return DatasetContext(
        version=version,
        currency=registered["currency"],
        requires_pricing_date=registered["requires_pricing_date"],
        material_cost_field=columns["material_cost"],
        manufacturer_unit_cost_field=columns["manufacturer_unit_cost"],
        logistics_base_cost_field=columns["logistics_base_cost"],
        logistics_cost_per_kg_field=columns["logistics_cost_per_kg"],
        order_budget_field=columns["order_budget"],
        notes=config.VERSION_NOTES.get(version, ""),
        _root=Path(root) if root is not None else None,
    )


ENV_VAR = "COB_DATASET_VERSION"


def resolve(version=None, root=None) -> DatasetContext:
    """the dataset context for this call

    precedence: the explicit argument, then COB_DATASET_VERSION, then
    config.DATA_VERSION. with none of them set this returns v4, which is exactly
    what every consumer resolves today

    an unknown version raises. there is no "latest" and no fallback"""
    if version is None:
        version = os.getenv(ENV_VAR, "").strip() or config.DATA_VERSION
    return for_version(version, root)


def default_context() -> DatasetContext:
    """what config.DATA_VERSION resolves to, ignoring the environment

    this is the pin the golden harness uses: it must stay v4 and EUR"""
    return for_version(config.DATA_VERSION, None)
