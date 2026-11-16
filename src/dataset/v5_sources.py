"""the declared contents of data/raw/v5: what is downloaded, what is recorded by
hand, and what every stored value means

this module is data, not behavior. acquire_v5_sources.py reads it to build the
tree and manifest.json; the manifest tests read it to check that what is on disk
is what was declared. keeping the declaration in one file is what makes the
acquisition step re-runnable and the result diffable

two acquisition modes exist and only two:

  downloaded     the artifact was fetched from its url and stored byte for byte.
                 its sha-256 is source/artifact integrity: it proves the pinned
                 file has not changed since retrieval

  manual_record  a value was read off a catalog, pdf or web page by hand and
                 written into a local json record here. its sha-256 is record
                 integrity only. it proves this repository's record of the value
                 has not been edited since it was written. it is not, and never
                 becomes, proof that the original page still says the same thing

the manual values below were verified by hand during the methodology work and are
reproduced here exactly. they are deliberately not re-fetched: re-fetching risks
silent drift from what docs/data_synthesis_methodology.md already locks in, and
one of the sources actively blocks automated requests
"""

# the date the hand-verified values were committed into the methodology document.
# the original browsing dates were not recorded, so this is the date this
# repository can actually stand behind, and access_date_basis says so on every
# manual record rather than leaving a bare date to be misread
MANUAL_RECORD_ACCESS_DATE = "2026-09-11"
MANUAL_ACCESS_DATE_BASIS = (
    "the date these hand-verified values were committed into "
    "docs/data_synthesis_methodology.md (02a3fb5); the original browsing date was "
    "not recorded at verification time, so this is the earliest date this "
    "repository can evidence for the value"
)

HASH_PURPOSE = {
    "downloaded": "source_artifact_integrity",
    "manual_record": "record_integrity",
}

HASH_PURPOSE_MEANING = {
    "source_artifact_integrity":
        "sha-256 of the downloaded artifact itself. it proves the pinned file "
        "stored in this repository has not changed since it was retrieved from "
        "the source url on the recorded retrieval date. it says nothing about "
        "whether the source has since revised or withdrawn the series",
    "record_integrity":
        "record integrity only. this is the sha-256 of this repository's local json "
        "record file, not of anything "
        "external. it proves the recorded value has not been edited since it was "
        "written here. it is NOT evidence that the original web page, catalog or "
        "pdf still contains the same value, and it cannot become such evidence. "
        "re-checking the source is a manual step that produces a new record",
}


# ── downloaded sources ───────────────────────────────────────────────────────
# every url below was confirmed to return the real artifact before being written
# here, see the report. fred serves a series csv from the graph endpoint; the ecb
# historical reference rates are published as a single zip whose one member is the
# csv the fx layer reads

DOWNLOADS = [
    {
        "artifact": "material/cotton/PCOTTINDUSDM.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOTTINDUSDM",
        "series_page": "https://fred.stlouisfed.org/series/PCOTTINDUSDM",
        "source": "FRED / IMF, Global price of Cotton",
        "series_id": "PCOTTINDUSDM",
        "unit": "U.S. Cents per Pound",
        "frequency": "Monthly",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "unit_confirmed_from": "the Units line printed on the FRED series page",
        "role": "primary temporal reference for cotton",
        "licence_note": "IMF Primary Commodity Prices, reprinted by FRED with permission",
        "expected_header": "observation_date,PCOTTINDUSDM",
    },
    {
        "artifact": "material/denim/WPU03420104.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPU03420104",
        "series_page": "https://fred.stlouisfed.org/series/WPU03420104",
        "source": ("FRED / BLS, Producer Price Index by Commodity: Textile Products "
                   "and Apparel: Finished Cotton Broadwoven Fabrics"),
        "series_id": "WPU03420104",
        "unit": "Index Dec 1994=100",
        "frequency": "Monthly",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "unit_confirmed_from": "the Units line printed on the FRED series page",
        "role": ("monthly temporal proxy for denim; an index of finished cotton "
                 "broadwoven fabrics, not a denim market price"),
        "licence_note": "U.S. Bureau of Labor Statistics, Producer Price Index",
        "expected_header": "observation_date,WPU03420104",
    },
    {
        "artifact": "material/leather/PHIDEUSDM.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PHIDEUSDM",
        "series_page": "https://fred.stlouisfed.org/series/PHIDEUSDM",
        "source": "FRED / IMF, Global price of Hides",
        "series_id": "PHIDEUSDM",
        "unit": "U.S. Cents per Pound",
        "frequency": "Monthly",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "unit_confirmed_from": "the Units line printed on the FRED series page",
        "role": ("upstream temporal proxy for leather only; it is a hide price, "
                 "not a finished garment-leather price"),
        "licence_note": "IMF Primary Commodity Prices, reprinted by FRED with permission",
        "expected_header": "observation_date,PHIDEUSDM",
    },
    {
        "artifact": "fx/eurofxref-hist.zip",
        "url": "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip",
        "series_page": ("https://www.ecb.europa.eu/stats/policy_and_exchange_rates/"
                        "euro_reference_exchange_rates/html/index.en.html"),
        "dataset_page": "https://data.ecb.europa.eu/data/datasets/EXR",
        "source": "European Central Bank, euro foreign exchange reference rates",
        "series_id": "EXR",
        "unit": "units of currency per 1 EUR",
        "frequency": "every TARGET working day",
        "unit_confirmed_from": (
            "the ECB reference-rates page, which states the rates are quoted as "
            "units of currency per euro and are based on the daily concertation "
            "procedure at around 14:10 CET"),
        "role": "the only fx source; every non-USD price is converted through it",
        "licence_note": "ECB, reuse permitted with acknowledgement of the source",
        "extract_member": "eurofxref-hist.csv",
        "extract_to": "fx/ecb_eurofxref_hist.csv",
        "expected_header_startswith": "Date,",
    },
]


# ── manual records ───────────────────────────────────────────────────────────
# reproduced exactly as verified. the verification notes are the ones written
# during that verification and are not reworded here

MANUAL_RECORDS = [
    {
        "artifact": "material/cotton/tabb_jersey_knit.json",
        "record": {
            "source": "Tabb Textile Company, 2024 Product Catalog",
            "source_url": ("https://www.tabbtextileinc.com/wp-content/uploads/2024/11/"
                           "Tabb-Textile-Fabric-Catalog-Final-Version-1.pdf"),
            "page_or_section": "jersey knit section of the fabric catalog",
            "role": "finished-fabric price level for cotton",
            "observation_kind": "catalog list prices, one catalog edition, not a series",
            "catalog_edition": "2024",
            "currency": "USD",
            "unit": "USD per pound",
            "observations": [
                {"sku": "JKCWI", "description": "Imported 4.5oz Jersey Knit White",
                 "composition": "100% Cotton", "value": 6.65, "unit": "USD/lb"},
                {"sku": "JKCCI", "description": "Imported 4.5oz Jersey Knit Colors",
                 "composition": "100% Cotton", "value": 6.95, "unit": "USD/lb"},
                {"sku": "JKCWD", "description": "Domestic 4.5oz Jersey Knit White",
                 "composition": "100% Cotton", "value": 7.65, "unit": "USD/lb"},
                {"sku": "JKCCD", "description": "Domestic 4.5oz Jersey Knit Colors",
                 "composition": "100% Cotton", "value": 7.95, "unit": "USD/lb"},
            ],
            "verification_note": (
                "confirmed '100% Cotton' printed directly on each line; catalog's "
                "separate JKPCWI/JKPCCI/JKPCWD/JKPCCD 'Jersey Knit Poly/Cotton' SKUs "
                "checked and explicitly excluded as blends"),
        },
    },
    {
        "artifact": "material/polyester/tabb_polyester_twill.json",
        "record": {
            "source": "Tabb Textile Company, 2024 Product Catalog",
            "source_url": ("https://www.tabbtextileinc.com/wp-content/uploads/2024/11/"
                           "Tabb-Textile-Fabric-Catalog-Final-Version-1.pdf"),
            "page_or_section": "twill section of the fabric catalog",
            "role": "finished-fabric price level for polyester",
            "observation_kind": "catalog list price, one catalog edition, not a series",
            "catalog_edition": "2024",
            "currency": "USD",
            "unit": "USD per yard",
            "observations": [
                {"sku": "TWPWD",
                 "description": "Domestic 6.7oz Twill 100% Texturized Polyester White",
                 "composition": "100% Texturized Polyester",
                 "width_inches": 60, "value": 4.65, "unit": "USD/yard"},
            ],
            "verification_note": "printed as '100% Texturized Polyester', not a blend",
        },
    },
    {
        "artifact": "material/denim/tabb_denim.json",
        "record": {
            "source": "Tabb Textile Company, 2024 Product Catalog",
            "source_url": ("https://www.tabbtextileinc.com/wp-content/uploads/2024/11/"
                           "Tabb-Textile-Fabric-Catalog-Final-Version-1.pdf"),
            "page_or_section": "denim section of the fabric catalog",
            "role": "finished-fabric price level for denim",
            "observation_kind": "catalog list prices, one catalog edition, not a series",
            "catalog_edition": "2024",
            "currency": "USD",
            "unit": "USD per yard",
            "observations": [
                {"sku": "DEHI", "description": "Imported 14.5oz denim",
                 "composition": "100% Cotton", "width_inches": 60,
                 "value": 4.85, "unit": "USD/yard"},
                {"sku": "DELI", "description": "Imported 10oz denim",
                 "composition": "100% Cotton", "width_inches": 60,
                 "value": 3.85, "unit": "USD/yard"},
                {"sku": "DEHBLD", "description": "Domestic 14.5oz denim",
                 "composition": "100% Cotton", "width_inches": 68,
                 "value": 7.50, "unit": "USD/yard"},
                {"sku": "DEHBD", "description": "Domestic 13.3oz denim",
                 "composition": "100% Cotton", "width_inches": 68,
                 "value": 7.50, "unit": "USD/yard"},
            ],
            "verification_note": (
                "catalog list prices; catalog states bulk/repeat-purchase discounts "
                "are available but does not quantify them"),
        },
    },
    {
        "artifact": "material/polyester/sunsirs_dty.json",
        "record": {
            "source": "SunSirs, Polyester DTY",
            "source_url": "https://www.sunsirs.com/uk/prodetail-1007.html",
            "page_or_section": "the Polyester DTY product detail page",
            "role": "temporal reference for polyester",
            "observation_kind": "the source labels these daily spot prices",
            "currency": None,
            "unit": None,
            "observations": [],
            "observations_status": "missing",
            "observations_status_note": (
                "no hand-verified SunSirs date/value observations exist in this "
                "repository or in the methodology document, and this source must not "
                "be re-fetched or reconstructed. the provenance record is created "
                "here so the structure is complete and the gap is visible; the "
                "observations have to be supplied by hand before polyester can use a "
                "temporal series. treat this file as a declared blocker, not as data"),
            "verification_note": (
                "site's own page title states 'Daily prices'; the live data table "
                "could not be independently opened for verification because the site "
                "returns an automated security-check page to non-browser requests; "
                "the daily-frequency claim rests on the source's own label only, not "
                "on an independently confirmed table"),
        },
    },
    {
        "artifact": "material/linen/alliance_flax.json",
        "record": {
            "source": ("Alliance for European Flax-Linen and Hemp, "
                       "economic observatory"),
            "source_url": ("https://allianceflaxlinenhemp.eu/en/"
                           "flax-linen-hemp-economic-observatory"),
            "page_or_section": "quarterly narrative press releases of the observatory",
            "role": "upstream temporal proxy for linen",
            "observation_kind": (
                "irregular observations embedded in narrative press releases, "
                "not a continuous series"),
            "product": "European long flax fibre, all qualities and regions",
            "currency": "EUR",
            "unit": "EUR per kg",
            "carry_forward": True,
            "interpolate": False,
            "observations": [
                {"date": "2023-06-01", "value": 6.05, "unit": "EUR/kg",
                 "published_as": "June 2023"},
                {"date": "2023-11-01", "value": 8.23, "unit": "EUR/kg",
                 "published_as": "November 2023"},
                {"date": "2024-03-01", "value": 9.08, "unit": "EUR/kg",
                 "published_as": "March 2024", "note": "year-on-year +55%"},
                {"date": "2026-03-01", "value": 5.89, "unit": "EUR/kg",
                 "published_as": "March 2026"},
            ],
            "verification_note": (
                "figures are irregular observations embedded in the Alliance's "
                "quarterly narrative press releases, not a continuous downloadable "
                "series; carry the most recent observation forward until the next one "
                "when resolving a date that falls between two observations; do not "
                "interpolate"),
        },
    },
    {
        "artifact": "material/linen/imaterial_linen.json",
        "record": {
            "source": "iMaterial, January 2024 Fabric Pricelist",
            "source_url": ("https://www.imaterial.co.za/Portals/81/Price-Lists/"
                           "Fabric-pricelist-Jan-2024.pdf"),
            "page_or_section": "linen and cotton sections of the January 2024 pricelist",
            "role": "finished-fabric price level for linen",
            "observation_kind": "catalog list prices, one pricelist edition, not a series",
            "catalog_edition": "January 2024",
            "currency": "ZAR",
            "unit": "ZAR per metre",
            "conversion_formula": (
                "USD/kg = (price_per_metre_ZAR x ZAR_to_USD(date)) / "
                "(width_m x GSM / 1000)"),
            "observations": [
                {"sku": "IMA10", "composition": "100% Linen", "gsm": 310,
                 "width_cm": 148, "value": 265.00, "unit": "ZAR/m", "use": "linen"},
                {"sku": "IMA23", "composition": "100% Linen", "gsm": 300,
                 "width_cm": 145, "value": 320.00, "unit": "ZAR/m", "use": "linen"},
                {"sku": "IMA34", "composition": "100% Linen", "gsm": 160,
                 "width_cm": 130, "value": 170.00, "unit": "ZAR/m", "use": "linen"},
                {"sku": "IMA9", "composition": "100% Linen", "gsm": 245,
                 "width_cm": 148, "value": 303.45, "unit": "ZAR/m", "use": "linen"},
                {"sku": "IMA5", "description": "Poplin", "composition": "100% Cotton",
                 "gsm": 119, "width_cm": 147, "value": 90.45, "unit": "ZAR/m",
                 "use": "cotton comparator, documentation-only ratio"},
                {"sku": "IMA8", "description": "Twill", "composition": "100% Cotton",
                 "gsm": 220, "width_cm": 150, "value": 72.00, "unit": "ZAR/m",
                 "use": "cotton comparator, documentation-only ratio"},
                {"sku": "IMA20", "description": "Canvas", "composition": "100% Cotton",
                 "gsm": 230, "width_cm": 150, "value": 85.00, "unit": "ZAR/m",
                 "use": "cotton comparator, documentation-only ratio"},
                {"sku": "IMA24", "description": "2/1 Fine Twill",
                 "composition": "100% Cotton", "gsm": 145, "width_cm": 150,
                 "value": 90.00, "unit": "ZAR/m",
                 "use": "cotton comparator, documentation-only ratio"},
                {"sku": "IMA27", "description": "Calico", "composition": "100% Cotton",
                 "gsm": 125, "width_cm": 135, "value": 58.45, "unit": "ZAR/m",
                 "use": "cotton comparator, documentation-only ratio"},
            ],
            "comparator_note": (
                "the cotton SKUs are kept only for the documentation-only "
                "linen:cotton ratio and are not used to generate v5 cotton prices"),
            "verification_note": (
                "ZAR-denominated; convert using the FX layer built in Part 6, not a "
                "hardcoded rate; conversion formula is "
                "USD/kg = (price_per_metre_ZAR x ZAR_to_USD(date)) / "
                "(width_m x GSM / 1000)"),
        },
    },
    {
        "artifact": "material/leather/leather_unlimited.json",
        "record": {
            "source": "Leather Unlimited, Garment & Bag Leather Hides collection",
            "source_url": ("https://leatherunltd.com/collections/garment-bag-leather/"
                           "products/premium-soft-light-weight-garment-leather-hide-"
                           "20-square-feet-2-3-oz"),
            "page_or_section": "the product page for the premium garment leather hide",
            "role": "finished-material price level for leather",
            "observation_kind": "one listed product price, not a series",
            "currency": "USD",
            "unit": "USD per hide",
            "observations": [
                {"product": "Premium Soft Light Weight Garment Leather Hide",
                 "area_sqft": 20, "weight_oz_per_sqft_range": "2-3",
                 "hide_type": "cowhide", "value": 89.95, "unit": "USD/hide"},
            ],
            "verification_note": (
                "this is the premium, garment-labeled tier, deliberately chosen over "
                "the same supplier's cheaper 'value priced' 2-4oz upholstery bundles "
                "after discovering those bundles sit in the same product line the "
                "supplier elsewhere labels 'crust leather' (a semi-finished stage this "
                "project excludes); do not substitute the value-priced bundles here"),
        },
    },
    {
        "artifact": "material/leather/townsend_conversion_rule.json",
        "record": {
            "source": "Townsend Leather, technical bulletin on leather thickness and weight",
            "source_url": ("https://www.townsendleather.com/wp-content/uploads/2013/12/"
                           "Bulletin-6_Thickness.pdf"),
            "page_or_section": "Bulletin 6, leather thickness and weight",
            "role": "physical conversion rule for leather, not a price",
            "observation_kind": "a stated industry conversion rule",
            "rule": ("leather ounce weight is a literal areal mass (oz per square "
                     "foot), not a thickness proxy requiring a density assumption"),
            "conversion": {
                "total_oz": "oz_per_sqft x area_sqft",
                "total_kg": "total_oz x 0.0283495",
                "oz_to_kg": 0.0283495,
            },
            "observations": [],
            "verification_note": (
                "cross-checked against Hoplok Leather, Stonestreet Leather, and "
                "LeatherSkinShop, all independently confirm oz/sqft as a direct weight "
                "measure; this avoids introducing a separate, unsupported leather "
                "density assumption"),
        },
    },
]


# the fx cross-check values are recorded on the fx source entry, not as a separate
# artifact: they exist only to sanity-check the downloaded ECB series and must
# never stand in for it
FX_CROSS_CHECK = {
    "purpose": (
        "cross-confirmed against three independent secondary sources during "
        "methodology development; use these only to sanity-check the real "
        "downloaded ECB series, never as the fx layer's actual source of truth"),
    "values": [
        {"period": "2024 annual average", "base": "EUR", "quote": "USD", "rate": 1.0824},
        {"period": "2024 annual average", "base": "EUR", "quote": "ZAR", "rate": 19.8297},
        {"period": "2026-08 monthly average", "base": "EUR", "quote": "USD", "rate": 1.1593},
    ],
    "verification_note": (
        "cross-confirmed against three independent secondary sources during "
        "methodology development; use these only to sanity-check the real downloaded "
        "ECB series in Part 6, never as the FX layer's actual source of truth"),
}




def group_of(artifact_path):
    """the folder an artifact belongs to, which is also its source.json group"""
    return artifact_path.rsplit("/", 1)[0]


def declared_artifacts():
    """every artifact this module declares, downloaded and manual, in tree order"""
    entries = []
    for spec in DOWNLOADS:
        entries.append((spec["artifact"], "downloaded"))
        if "extract_to" in spec:
            entries.append((spec["extract_to"], "downloaded"))
    for spec in MANUAL_RECORDS:
        entries.append((spec["artifact"], "manual_record"))
    return entries
