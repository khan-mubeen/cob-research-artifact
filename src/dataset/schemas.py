"""machine-readable schemas for the profile, product and order records

the csv headers under data/synthetic are the only schema this project has ever
had. they are a de facto one: nothing declares a column's type, nothing declares
its unit, and three of v4's five monetary columns do not even say what currency
they are in. a v5 record that lost its currency or its pricing date would read
as perfectly valid v4-shaped data

these schemas are declarative and serialize to plain json through as_dict(), so
the contract can be written into generated metadata rather than lived in
somebody's head. they validate, they do not coerce: a record either satisfies the
contract or produces a list of errors naming every field that failed

the v4 schemas describe the corpus exactly as committed, including its untyped
monetary column names, so validating the live v4 files is a real check rather
than a rename. the v5 schemas add the four fields the methodology requires
everywhere: unit, currency, dataset version and pricing date

no dependency beyond the standard library and pandas, which the project already
has. this deliberately does not pull in jsonschema for four record shapes
"""

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime

from src.dataset.context import KNOWN_CURRENCIES, REGISTRY

ALL_MATERIALS = ("cotton", "polyester", "linen", "denim", "leather")
ALL_PRODUCTS = ("shirt", "hoodie", "jacket", "jeans", "skirt")
PRIORITIES = ("urgent", "normal", "flexible")


class SchemaError(ValueError):
    """a record, file or schema definition that does not satisfy its contract"""


def _is_missing(value):
    # pandas writes an unset numeric cell as nan, json writes it as null, and a
    # csv read as text writes it as the empty string. all three are absent
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return isinstance(value, str) and value.strip() == ""


@dataclass(frozen=True)
class Field:
    """one column, with the two things a csv header never says

    unit is the physical or monetary unit the number is in: kg, days, a
    probability, an iso currency code. currency is set only on monetary fields
    and is what makes a mixed-currency total detectable rather than plausible"""

    name: str
    kind: str                       # str | number | int | date | enum
    required: bool = True
    nullable: bool = False          # a column that exists but may be empty
    unit: str = None
    currency: str = None
    choices: tuple = None
    minimum: float = None
    maximum: float = None
    note: str = ""

    def __post_init__(self):
        if self.kind not in ("str", "number", "int", "date", "enum"):
            raise SchemaError(f"{self.name}: unknown field kind {self.kind!r}")
        if self.kind == "enum" and not self.choices:
            raise SchemaError(f"{self.name}: an enum field needs choices")
        if self.currency is not None and self.currency not in KNOWN_CURRENCIES:
            raise SchemaError(f"{self.name}: unknown currency {self.currency!r}")

    def errors(self, record) -> list:
        """every way this field fails in one record, as plain sentences"""
        if self.name not in record:
            return [] if not self.required else [f"{self.name}: missing"]

        value = record[self.name]
        if _is_missing(value):
            if self.nullable:
                return []
            return [f"{self.name}: empty, and this field is not nullable"]

        problems = []
        if self.kind == "str":
            if not isinstance(value, str):
                problems.append(f"{self.name}: expected text, got {type(value).__name__}")
        elif self.kind == "enum":
            if value not in self.choices:
                problems.append(f"{self.name}: {value!r} is not one of {list(self.choices)}")
        elif self.kind == "date":
            problems.extend(self._date_errors(value))
        else:
            problems.extend(self._number_errors(value))
        return problems

    def _date_errors(self, value):
        if isinstance(value, datetime):
            return []
        if not isinstance(value, str):
            return [f"{self.name}: expected an iso date string, got {type(value).__name__}"]
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                datetime.strptime(text, fmt)
                return []
            except ValueError:
                continue
        return [f"{self.name}: {value!r} is not an iso date or datetime"]

    def _number_errors(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{self.name}: expected a number, got {value!r}"]
        if self.kind == "int" and float(value) != int(value):
            return [f"{self.name}: expected a whole number, got {value!r}"]
        problems = []
        if self.minimum is not None and value < self.minimum:
            problems.append(f"{self.name}: {value} is below the minimum {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            problems.append(f"{self.name}: {value} is above the maximum {self.maximum}")
        return problems

    def as_dict(self):
        out = {"name": self.name, "kind": self.kind, "required": self.required,
               "nullable": self.nullable}
        for key in ("unit", "currency", "minimum", "maximum", "note"):
            value = getattr(self, key)
            if value not in (None, ""):
                out[key] = value
        if self.choices:
            out["choices"] = list(self.choices)
        return out


@dataclass(frozen=True)
class RecordSchema:
    """one record shape: the fields, and what dataset they belong to"""

    name: str
    version: str
    currency: str
    fields: tuple
    requires_pricing_date: bool = False
    allow_extra_columns: bool = True
    note: str = ""
    _by_name: dict = dataclass_field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_by_name", {f.name: f for f in self.fields})
        if len(self._by_name) != len(self.fields):
            raise SchemaError(f"{self.name}: duplicate field names")

    @property
    def field_names(self):
        return tuple(f.name for f in self.fields)

    @property
    def monetary_fields(self):
        return tuple(f for f in self.fields if f.currency is not None)

    def errors(self, record) -> list:
        """every contract violation in one record

        the version and currency fields are checked against this schema's own
        dataset, so a v4 row cannot satisfy a v5 schema by carrying the wrong
        version string any more than by omitting it"""
        problems = []
        for spec in self.fields:
            problems.extend(spec.errors(record))

        if not self.allow_extra_columns:
            extra = sorted(set(record) - set(self.field_names))
            if extra:
                problems.append(f"unexpected columns: {extra}")

        declared_version = record.get("dataset_version")
        if declared_version is not None and not _is_missing(declared_version):
            if declared_version != self.version:
                problems.append(
                    f"dataset_version: record says {declared_version!r}, "
                    f"this is the {self.version} schema")

        declared_currency = record.get("currency")
        if declared_currency is not None and not _is_missing(declared_currency):
            if declared_currency != self.currency:
                problems.append(
                    f"currency: record says {declared_currency!r}, "
                    f"{self.version} data is {self.currency}")
        return problems

    def validate(self, record) -> None:
        problems = self.errors(record)
        if problems:
            raise SchemaError(f"{self.name} record is invalid:\n  " + "\n  ".join(problems))

    def validate_rows(self, rows, limit=None) -> list:
        """(row index, problems) for every invalid row, newest contract first

        returns rather than raises: a whole-file check wants the full list, not
        the first failure"""
        failures = []
        for i, row in enumerate(rows):
            problems = self.errors(row)
            if problems:
                failures.append((i, problems))
                if limit is not None and len(failures) >= limit:
                    break
        return failures

    def validate_csv(self, path, limit=None) -> list:
        import pandas as pd
        rows = pd.read_csv(path).to_dict("records")
        return self.validate_rows(rows, limit)

    def as_dict(self):
        return {
            "schema": self.name,
            "dataset_version": self.version,
            "currency": self.currency,
            "requires_pricing_date": self.requires_pricing_date,
            "allow_extra_columns": self.allow_extra_columns,
            "note": self.note,
            "fields": [f.as_dict() for f in self.fields],
        }


# ── shared field groups ──────────────────────────────────────────────────────
# built per version so the monetary column names come from one place, the
# context registry, rather than being retyped here

def _columns(version):
    return REGISTRY[version]["columns"]


def _currency(version):
    return REGISTRY[version]["currency"]


def _v5_envelope(version):
    """the four fields v5 requires on every priced record

    v4 has none of them, which is the whole reason they are required in v5: a
    v5 row missing any one of them is not a v5 row"""
    if not REGISTRY[version]["requires_pricing_date"]:
        return ()
    return (
        Field("dataset_version", "enum", choices=(version,),
              note="which dataset this record belongs to"),
        Field("currency", "enum", choices=(_currency(version),), unit="iso4217",
              note="the currency every monetary field on this record is in"),
        Field("pricing_date", "date",
              note="the immutable as-of date the sourced prices were resolved on"),
        Field("unit", "str",
              note="the physical unit the priced quantity is measured in"),
    )


def _risk_fields():
    return (
        Field("min_profit_margin", "number", unit="ratio", minimum=0.0),
        Field("initial_markup", "number", unit="ratio", minimum=0.0),
        Field("reputation_score", "number", unit="score", minimum=0.0, maximum=1.0),
    )


def material_schema(version) -> RecordSchema:
    cols = _columns(version)
    currency = _currency(version)
    fields = [
        Field("agent_id", "str"),
        Field("stage", "enum", choices=("material",)),
        Field("location", "str"),
        Field("material_types", "str", note="pipe separated"),
        Field("default_completion_days", "number", unit="days", minimum=0),
        Field("defect_probability", "number", unit="probability", minimum=0.0, maximum=1.0),
        *_risk_fields(),
    ]
    for material in ALL_MATERIALS:
        # a supplier that does not stock a material leaves its whole block empty,
        # which is why these are nullable rather than optional
        fields.extend([
            Field(f"{material}_available_kg", "number", unit="kg",
                  nullable=True, minimum=0.0),
            Field(cols["material_cost"].format(material=material), "number",
                  unit="per_kg", currency=currency, nullable=True, minimum=0.0),
            Field(f"{material}_restock_amount_kg", "number", unit="kg",
                  nullable=True, minimum=0.0),
            Field(f"{material}_restock_time_days", "number", unit="days",
                  nullable=True, minimum=0.0),
        ])
    fields.extend(_v5_envelope(version))
    return RecordSchema(
        name="material_agent", version=version, currency=currency,
        requires_pricing_date=REGISTRY[version]["requires_pricing_date"],
        fields=tuple(fields),
        note="one material provider profile, the shape the material agent reads")


def manufacturer_schema(version) -> RecordSchema:
    cols = _columns(version)
    currency = _currency(version)
    fields = [
        Field("agent_id", "str"),
        Field("stage", "enum", choices=("tailoring",)),
        Field("location", "str"),
        Field("product_types", "str", note="pipe separated"),
        Field("daily_capacity_units", "int", unit="units_per_day", minimum=0),
        Field("current_workload_units", "int", unit="units", minimum=0),
        Field("defect_probability", "number", unit="probability", minimum=0.0, maximum=1.0),
        *_risk_fields(),
    ]
    for product in ALL_PRODUCTS:
        fields.append(
            Field(cols["manufacturer_unit_cost"].format(product=product), "number",
                  unit="per_unit", currency=currency, nullable=True, minimum=0.0))
    fields.extend(_v5_envelope(version))
    return RecordSchema(
        name="manufacturer_agent", version=version, currency=currency,
        requires_pricing_date=REGISTRY[version]["requires_pricing_date"],
        fields=tuple(fields),
        note="one manufacturer profile, the shape the tailoring agent reads")


def logistics_schema(version) -> RecordSchema:
    cols = _columns(version)
    currency = _currency(version)
    fields = [
        Field("agent_id", "str"),
        Field("stage", "enum", choices=("logistics",)),
        Field("location", "str"),
        Field("service_regions", "str", note="pipe separated"),
        Field(cols["logistics_base_cost"], "number", unit="per_shipment",
              currency=currency, minimum=0.0),
        Field(cols["logistics_cost_per_kg"], "number", unit="per_kg",
              currency=currency, minimum=0.0),
        Field("default_completion_same_city_days", "number", unit="days", minimum=0),
        Field("default_completion_intercity_days", "number", unit="days", minimum=0),
        Field("damage_probability", "number", unit="probability", minimum=0.0, maximum=1.0),
        *_risk_fields(),
    ]
    fields.extend(_v5_envelope(version))
    return RecordSchema(
        name="logistics_agent", version=version, currency=currency,
        requires_pricing_date=REGISTRY[version]["requires_pricing_date"],
        fields=tuple(fields),
        note="one carrier profile. route-aware v5 fields are deliberately absent, "
             "they wait on the logistics method decision")


def product_schema(version) -> RecordSchema:
    return RecordSchema(
        name="product", version=version, currency=_currency(version),
        requires_pricing_date=False,
        fields=(
            Field("product_id", "enum", choices=ALL_PRODUCTS),
            Field("material_type", "enum", choices=ALL_MATERIALS),
            Field("material_kg_per_unit", "number", unit="kg", minimum=0.0),
            Field("finished_weight_kg_per_unit", "number", unit="kg", minimum=0.0),
        ),
        note="physical conversions only, no monetary field and no currency")


def order_schema(version) -> RecordSchema:
    cols = _columns(version)
    currency = _currency(version)
    fields = [
        Field("order_id", "str"),
        Field("order_created", "date",
              note="v5 resolves market prices as of this date, never the run date"),
        Field("product_id", "enum", choices=ALL_PRODUCTS),
        Field("material_type", "enum", choices=ALL_MATERIALS),
        Field("quantity", "int", unit="units", minimum=1),
        Field(cols["order_budget"], "number", unit="per_order",
              currency=currency, minimum=0.0),
        Field("deadline", "date"),
        Field("deliver_to", "str"),
        Field("max_defect_rate", "number", unit="probability", minimum=0.0, maximum=1.0),
        Field("min_reputation_score", "number", unit="score", minimum=0.0, maximum=1.0),
        Field("priority", "enum", choices=PRIORITIES),
    ]
    fields.extend(_v5_envelope(version))
    return RecordSchema(
        name="order", version=version, currency=currency,
        requires_pricing_date=REGISTRY[version]["requires_pricing_date"],
        fields=tuple(fields),
        note="one order. scenario_type is generator bookkeeping and is allowed "
             "through as an extra column rather than contracted")


BUILDERS = {
    "material_agents": material_schema,
    "manufacturer_agents": manufacturer_schema,
    "logistics_agents": logistics_schema,
    "products": product_schema,
    "orders": order_schema,
}


def schema_for(record_kind, version) -> RecordSchema:
    build = BUILDERS.get(record_kind)
    if build is None:
        raise SchemaError(
            f"unknown record kind {record_kind!r}, known kinds are {sorted(BUILDERS)}")
    if version not in REGISTRY:
        raise SchemaError(f"unknown dataset version {version!r}")
    return build(version)


def schemas_for_version(version) -> dict:
    return {kind: schema_for(kind, version) for kind in BUILDERS}


def validate_dataset(context, limit_per_file=5) -> dict:
    """validate every csv of one dataset against its own schemas

    takes a resolved DatasetContext rather than a version string, so the caller
    has already decided which dataset this is. returns {file: [(row, problems)]},
    empty when the dataset satisfies its contract"""
    directory = context.require_synthetic_data()
    report = {}
    for kind, schema in schemas_for_version(context.version).items():
        path = directory / f"{kind}.csv"
        if not path.is_file():
            report[f"{kind}.csv"] = [(-1, [f"missing file {path}"])]
            continue
        failures = schema.validate_csv(path, limit=limit_per_file)
        if failures:
            report[f"{kind}.csv"] = failures
    return report
