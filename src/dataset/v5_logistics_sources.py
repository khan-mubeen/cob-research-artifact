"""the declared contents of data/raw/v5/logistics: every tariff, rate and
fuel-surcharge figure the v5 logistics calibration is allowed to use, with the
metadata that has to be known before any of them is compared with another

this module is data, not behavior. acquire_v5_logistics_sources.py reads it to
write the record files, the per-folder source.json files and the logistics
manifest. the conventions are the ones data/raw/v5/manifest.json already uses:
exactly two acquisition modes, a record-integrity hash for anything read by
hand, and a verification note on every manual record

every price value here was supplied pre-verified in the task that created this
module and was then re-read against its source on MANUAL_ACCESS_DATE, except
where access_date_basis says the re-read was blocked. the six verification notes
given with that task are reproduced verbatim. where a source does not state a
required metadata field the field says NOT_STATED rather than a guess, and where
a source could not be opened at all the field says UNREADABLE instead, because
"not stated" would be a claim about a document nobody could read

quoted source wording is kept as printed except that typographic dashes are
written as a plain hyphen. tables that are too large to retype safely are kept as
the text the pdf or page actually yields, and the builder parses them, so there is
no hand transcription step between the source and the record
"""

MANUAL_ACCESS_DATE = "2026-09-13"

NOT_STATED = "not stated in source"
UNREADABLE = (
    "not verified: the source returned HTTP 403 to every automated request made "
    "on the access date and has no Internet Archive capture, so this field could "
    "not be read from the document itself"
)

GROUP_FEEDS = {
    "rate_calibration": "F_i and alpha_i",
    "line_haul_rate": "beta_i",
    "fuel_surcharge": "fuel_surcharge_i(t)",
    "distance": "distance_km",
}


def field(value, wording=NOT_STATED, note=None):
    """one required metadata field: the value and the source's own words for it"""
    out = {"value": value, "source_wording": wording}
    if note:
        out["note"] = note
    return out


REVERIFIED_BASIS = (
    "values supplied pre-verified in the task prompt; the source was opened again "
    "on this date and every value and metadata field in this record was checked "
    "against it"
)


# ── rate_calibration: F_i and alpha_i ────────────────────────────────────────

POSTI_PARCEL = {
    "artifact": "rate_calibration/posti_parcel.json",
    "record": {
        "source": "Posti Parcel Services, digital contract prices",
        "source_url": ("https://www.posti.fi/mzj3zpe8qb7p/63bn2u74jGXSHu1IK0KTOk/"
                       "8ed1177829b3ef1d403e467de8588375/"
                       "posti-digital-contract-prices-202501.pdf"),
        "page_or_section": ("page 1, 'Prices level 1', shipments of goods and parcels "
                            "picked up from a Posti outlet; VAT wording page 5; "
                            "weight limits page 6"),
        "role": "lower-bound calibration evidence for F_i and alpha_i",
        "calibration_bound": "lower",
        "currency": "EUR",
        "unit": "EUR per shipment plus EUR per kg",
        "document": {
            "title_as_printed": "Parcel services - digital contract prices",
            "file_name": "posti-digital-contract-prices-202501.pdf",
            "footer_as_printed": "© Posti June 15, 2024 All rights reserved.",
            "pages": 12,
        },
        "access_date_basis": REVERIFIED_BASIS,
        "reverification_status": "reverified",
        "tariff_metadata": {
            "vat_status": field(
                "exclusive",
                "VAT valid at the time will be added to prices during invoicing.",
                "page 5, 'Value added tax (VAT)'"),
            "effective_date": field(
                "valid until further notice; no start date is printed",
                "Valid until further notice, less than 500 parcels per calendar year",
                "page 1. the file name carries 202501 and the page 12 footer reads "
                "June 15, 2024; neither is stated as an effective date"),
            "currency_as_published": field("EUR", "€/item, €/kg column headings"),
            "unit_basis": field(
                ("per shipment (EUR/item) plus per kg (EUR/kg) on the greater of actual "
                 "and volume weight; the per-kg charge applies only to parcels larger "
                 "than 26 x 32 x 40 cm; minimum charge 1 kg"),
                ("The prices are determined by the parcel’s actual weight or volume "
                 "weight (width x length x height x 250 kg/m3), whichever is greater. "
                 "The minimum charge of a parcel is for 1 kilogram."),
                "page 1"),
            "service_category": field("domestic parcel", "Parcel services"),
            "fuel_surcharge_treatment": field(
                NOT_STATED, NOT_STATED,
                "this document never mentions a fuel surcharge. Posti publishes a "
                "separate 'Parcel' fuel and energy surcharge, and its 16.11.2025 "
                "domestic contract price list says the fuel surcharge is charged "
                "separately; that suggests, but does not establish, the same "
                "treatment for this digital price list"),
            "pickup_treatment": field(
                "not included: parcels are handed in at a Posti outlet or parcel locker",
                "Parcels picked up from a Posti outlet or Parcel Locker",
                "page 1 section heading; separate pick-ups are priced on page 5"),
        },
        "weight_limits_as_printed": {
            "Postal Parcel, Home Parcel, Express Parcel and Return": "maximum weight 25 kg",
            "with Oversized additional service": (
                "postal parcel’s maximum weight 25 kg; Home Parcel’s, Express Parcel’s "
                "and Return’s maximum weight 35 kg"),
            "Oversized additional service price": "€ 3.50",
        },
        "observations": [
            {"service": "Postal Parcel", "price_level": 1,
             "size_condition": "larger than 26 x 32 x 40 cm",
             "eur_per_item": 5.90, "eur_per_kg": 0.30, "page": 1,
             "printed_as": "Postal Parcel (larger than 26 x 32 x 40 cm) 5.90 0.30"},
            {"service": "Express Parcel", "price_level": 1,
             "size_condition": "over 26 x 32 x 40 cm",
             "eur_per_item": 8.90, "eur_per_kg": 0.30, "page": 1,
             "printed_as": "Express Parcel (over 26 x 32 x 40 cm) 8.90 0.30",
             "additional_charge_as_printed": (
                 "A delivery zone fee will be added to Express Parcels in accordance "
                 "with the parcel’s destination postal code: € 0.60/item for zone 1 "
                 "and € 1.20/item for zone 2.")},
            {"service": "Home Parcel", "price_level": 1,
             "size_condition": "over 26 x 32 x 40 cm",
             "eur_per_item": 10.90, "eur_per_kg": 0.30, "page": 1,
             "printed_as": "Home Parcel (over 26 x 32 x 40 cm) 10.90 0.30"},
        ],
        "verification_note": (
            "lower-bound calibration evidence; these parcel services have "
            "substantially lower shipment-weight limits (25-35kg) than this "
            "simulation's typical 50-1000+ kg shipments, per "
            "docs/data_synthesis_methodology.md Section 7.1"),
    },
}

POSTNORD_PARCEL_ZONES = {
    "artifact": "rate_calibration/postnord_parcel_zones.json",
    "record": {
        "source": "PostNord Finland, domestic service price list",
        "source_url": ("https://www.postnord.fi/siteassets/pdf/hinnastot/"
                       "online_prices_2026-02-01_en.pdf"),
        "page_or_section": ("15.01-20 kg weight bracket of the zone A, B and C domestic "
                            "price table; page number not verified, the document could "
                            "not be opened"),
        "role": "lower-bound calibration evidence for F_i and alpha_i",
        "calibration_bound": "lower",
        "currency": "EUR",
        "unit": "EUR per parcel within a weight bracket and delivery zone",
        "access_date_basis": (
            "values supplied pre-verified in the task prompt. re-reading the document "
            "on this date failed: HTTP 403 from urllib with a browser user agent, from "
            "WebFetch and from PowerShell Invoke-WebRequest, and the Internet Archive "
            "holds no capture of this url. the values are recorded as supplied and "
            "every metadata field the document would have to state is marked "
            "unverified rather than guessed"),
        "reverification_status": "blocked: HTTP 403, no archive capture",
        "document_dating_evidence": (
            "2026-02-01 appears in the file name online_prices_2026-02-01_en.pdf, and "
            "a web search index lists the document title as 'PostNord Oy service price "
            "list 1.2.2026'. neither was read from the document itself"),
        "tariff_metadata": {
            "vat_status": field(UNREADABLE, UNREADABLE),
            "effective_date": field(UNREADABLE, UNREADABLE,
                                    "see document_dating_evidence"),
            "currency_as_published": field(
                "EUR, as supplied with the pre-verified values", UNREADABLE),
            "unit_basis": field(
                "per parcel within a weight bracket and delivery zone, as supplied",
                UNREADABLE),
            "service_category": field(
                "domestic parcel, as supplied; the service name could not be re-read",
                UNREADABLE),
            "fuel_surcharge_treatment": field(
                UNREADABLE, UNREADABLE,
                "PostNord Finland publishes a separate monthly fuel surcharge, see "
                "fuel_surcharge/postnord_fuel_surcharge.json; whether this price list "
                "states it is charged on top could not be read"),
        },
        "observations": [
            {"weight_bracket_kg": "15.01-20", "zone": "A", "eur_per_parcel": 12.00},
            {"weight_bracket_kg": "15.01-20", "zone": "B", "eur_per_parcel": 14.00},
            {"weight_bracket_kg": "15.01-20", "zone": "C", "eur_per_parcel": 19.00},
        ],
        "verification_note": (
            "lower-bound calibration evidence, same category caveat as Posti parcel; "
            "zone-and-weight-bracket structure, not a constant EUR/kg rate"),
    },
}

POSTI_EXPRESS_FREIGHT_COLUMNS = [
    {"key": "loose_package",
     "printed_fi": "Irtokolli", "printed_en": "Loose package"},
    {"key": "half_pallet_first_unit",
     "printed_fi": "Teholava/Rullakko 1.yksikkö",
     "printed_en": "Rolltainer/Half pallet 1. unit €"},
    {"key": "half_pallet_next_units",
     "printed_fi": "Teholava/Rullakko seuraavat yksiköt*€",
     "printed_en": "Rolltainer/Half pallet next units* €"},
    {"key": "pallet_first_unit",
     "printed_fi": "Lava 1. yksikkö €", "printed_en": "Pallet 1. unit €"},
    {"key": "pallet_next_units",
     "printed_fi": "Lava seuraavat yksiköt* €", "printed_en": "Pallet next units* €"},
    {"key": "furniture_pallet_first_unit",
     "printed_fi": "Huonekalulava 1. yksikkö €",
     "printed_en": "Furniture pallet 1. unit €"},
    {"key": "furniture_pallet_next_units",
     "printed_fi": "Huonekalulava seuraavat yksiköt* €",
     "printed_en": "Furniture pallet next units* €"},
]


def _zone_rows(rows):
    return [{"zone": zone, "eur": dict(zip([c["key"] for c in POSTI_EXPRESS_FREIGHT_COLUMNS],
                                           values))}
            for zone, values in rows]


POSTI_EXPRESS_FREIGHT = {
    "artifact": "rate_calibration/posti_express_freight.json",
    "record": {
        "source": ("Posti Express Freight, domestic goods transport services contract "
                   "price list 16.11.2025"),
        "source_url": ("https://assets.ctfassets.net/dvxpcmq06s7e/2KEbIynQDBmrCouchzrJ2Q/"
                       "453d09672354ec0a4a5d2eb7bf618766/"
                       "posti-paketti-ja-express-rahti-sopimushinnasto-kotimaan-"
                       "kuljetuspalvelut-20251116.pdf"),
        "source_url_given_in_task": ("https://www.posti.fi/en/for-businesses/"
                                     "parcels-and-logistics/freight"),
        "source_location_note": (
            "the url given in the task returned HTTP 404 on the access date. its "
            "successor page, https://www.posti.fi/en/for-businesses/parcels-and-"
            "logistics/all-services/express-freight (linked from page 12 of the Posti "
            "parcel price list), states no prices. the zone and transport-unit prices "
            "below were therefore read from Posti's current domestic contract price "
            "list. Posti's english contract price list of 30.9.2024, "
            "https://www.posti.fi/mzj3zpe8qb7p/7s9AsOwmnECySlx35uEAFO/"
            "2554fda2bfcab7eb8d7ac9a0ee51dcae/posti-parcel-services-and-express-"
            "freight-contract-prices-20240930.pdf, prints identical Express Freight "
            "figures on its page 3 and supplies the english wording and the "
            "transport-unit dimensions"),
        "page_or_section": ("page 4 'Rahtipalvelut' Express-rahti tables, page 5 "
                            "Express-rahti Samana päivänä 00 and Illaksi 21 tables and "
                            "delivery charge, page 6 separate pick-up; VAT and fuel "
                            "wording page 1; transport-unit limits from the 30.9.2024 "
                            "english list pages 9-10"),
        "role": "upper-bound calibration evidence for F_i and alpha_i",
        "calibration_bound": "upper",
        "currency": "EUR",
        "unit": "EUR per transport unit by delivery zone",
        "access_date_basis": (
            "no price values were supplied with the task for this source; the prices "
            "were read on this date from the price list named in source_url and "
            "checked against the 30.9.2024 english list"),
        "reverification_status": "read from the current price list; given url returned 404",
        "tariff_metadata": {
            "vat_status": field(
                "exclusive",
                ("Hinnat on esitetty ilman arvonlisäveroa, ellei toisin mainita. "
                 "Hintoihin lisätään laskutuksessa kulloinkin voimassa oleva "
                 "arvonlisävero."),
                ("page 1. english list wording: 'The tariffs and rates are quoted "
                 "excluding VAT, unless otherwise stated. VAT valid at the time will be "
                 "added to prices during invoicing.'")),
            "effective_date": field(
                "2025-11-16",
                "Kotimaan tavarankuljetuspalvelut - Sopimushinnasto 16.11.2025",
                "page 1 title"),
            "currency_as_published": field("EUR", "€ column headings"),
            "unit_basis": field(
                ("per transport unit (loose package, rolltainer or half pallet, pallet, "
                 "furniture pallet), first unit and each next unit to the same address, "
                 "by delivery zone 0-5; not weight-based within a unit's capacity"),
                ("Express-rahdin hinnoittelu perustuu kappaleisiin, kuljetusyksikön "
                 "tyyppiin, vyöhykkeisiin sekä käytettyihin lisäpalveluihin."),
                ("page 1. english list wording: 'The pricing of Express Freight is based "
                 "on the transport unit type, the size of the shipment lot, transport "
                 "distance, and the additional service used.'")),
            "service_category": field("express freight (transport-unit road freight)",
                                      "Rahtipalvelut, Express-rahti"),
            "fuel_surcharge_treatment": field(
                "charged separately",
                ("Kulloinkin voimassaoleva polttoainelisä veloitetaan erikseen. Voimassa "
                 "oleva polttoainelisä löytyy osoitteesta posti.fi/polttoainelisa."),
                ("page 1. english list wording: 'An applicable fuel surcharge will be "
                 "charged separately.'")),
            "pickup_treatment": field(
                "not included: prices run from a Posti outlet; a pick-up is a separate charge",
                ("Ellei toisin mainita, hinnat sisältävät kuljetuksen postista "
                 "vastaanottajalle tai luovutuspisteeseen. Lähetyksille voi tilata "
                 "erillismaksusta noudon"),
                ("page 1. separate pick-up of one transport unit is printed as 40,33 on "
                 "page 6 of this list and as 40.35 in both the 30.9.2024 english list "
                 "and the digital parcel price list; recorded as printed in each")),
            "delivery_zone_definition": field(
                NOT_STATED, "Katso vyöhykkeet täältä.",
                "the zone list is a link; zones 0-5 are not defined in the document"),
        },
        "transport_unit_limits_as_printed_en_20240930": {
            "Pallet": "Maximum width 120 cm, Maximum length 100 cm, Maximum gross weight 1000 kg",
            "Half pallet": "Maximum width 60 cm, Maximum length 80 cm, Maximum gross weight 500 kg",
            "Rolltainer": "Maximum width 63 cm, Maximum length 81 cm, Maximum gross weight 250 kg",
            "Furniture pallet": ("Maximum width 80 cm, Maximum length 200 cm, "
                                 "Maximum gross weight 1200 kg"),
            "Loose colli": "Maximum gross weight 35 kg",
        },
        "columns": POSTI_EXPRESS_FREIGHT_COLUMNS,
        "column_footnote_as_printed": {"fi": "*Samalle vastaanottajalle",
                                       "en": "*To the same address"},
        "observations": [
            {"service_printed_fi": "Express-rahti 14", "service_printed_en": "Express Freight",
             "page": 4, "rows": _zone_rows([
                 (0, [40.00, 62.32, 53.25, 89.59, 75.04, 129.91, 108.80]),
                 (1, [50.00, 77.91, 68.84, 120.77, 106.22, 175.12, 154.02]),
                 (2, [60.00, 97.38, 88.31, 155.83, 141.28, 225.97, 204.86]),
                 (3, [70.00, 116.86, 107.79, 194.76, 180.21, 282.40, 261.29]),
                 (4, [80.00, 136.34, 127.27, 233.70, 219.15, 338.86, 317.76]),
                 (5, [90.00, 155.80, 146.73, 272.65, 258.10, 395.36, 374.26]),
             ])},
            {"service_printed_fi": "Express-rahti Aamuksi 09",
             "service_printed_en": "Express Freight Morning 09", "page": 4,
             "rows": _zone_rows([
                 (0, [50.00, 87.26, 78.19, 125.39, 110.86, 181.81, 160.75]),
                 (1, [60.00, 109.09, 100.01, 169.02, 154.50, 245.10, 224.04]),
                 (2, [70.00, 136.34, 127.27, 218.16, 203.63, 316.32, 295.26]),
                 (3, [80.00, 163.60, 154.53, 272.63, 258.10, 395.32, 374.26]),
                 (4, [90.00, 190.89, 181.82, 327.18, 312.65, 474.39, 453.33]),
                 (5, [100.00, 218.14, 209.07, 381.68, 367.16, 553.44, 532.38]),
             ])},
            {"service_printed_fi": "Express-rahti Samana päivänä 00",
             "service_printed_en": "Express Freight same-day 00", "page": 5,
             "rows": _zone_rows([
                 (0, [40.00, 43.39, 36.86, 63.78, 54.46, 92.47, 78.95]),
                 (1, [50.00, 68.04, 61.52, 103.41, 94.09, 149.96, 136.44]),
             ])},
            {"service_printed_fi": "Express-rahti Illaksi 21",
             "service_printed_en": "Express Freight Evening 21", "page": 5,
             "rows": _zone_rows([
                 (0, [50.00, 74.79, 63.91, 107.50, 90.06, 214.99, 197.56]),
                 (1, [60.00, 93.47, 82.59, 144.89, 127.46, 289.82, 272.38]),
                 (2, [70.00, 116.86, 105.98, 186.96, 169.52, 373.95, 356.52]),
                 (3, [80.00, 140.25, 129.37, 233.70, 216.26, 467.39, 449.95]),
                 (4, [90.00, 163.60, 152.72, 280.44, 263.00, 560.87, 543.43]),
                 (5, [100.00, 186.98, 176.10, 327.16, 309.72, 654.35, 636.91]),
             ])},
        ],
        "delivery_charge_as_printed": {
            "service_printed_fi": "Express-rahdin jakelumaksu", "page": 5,
            "eur_by_zone": {"1": 30.17, "2": 60.33},
        },
        "separate_pickup_as_printed": {
            "page": 6,
            "eur_by_transport_unit_count": {"1": 40.33, "2": 30.40, "3": 24.63, "4": 20.13,
                                            "5": 17.13, "6": 15.06, "7": 13.46, "8": 12.25,
                                            "9": 11.55, "10 tai enemmän": 10.76},
        },
        "verification_note": (
            "upper-bound calibration evidence; prices by transport-unit type and "
            "delivery zone (0-5), not by a constant per-kg rate; record whatever "
            "specific zone/unit prices are stated on this page with their exact "
            "wording, do not force them into an F+alpha shape here, that translation "
            "happens in Part 3"),
    },
}

# the page 1 freight table exactly as the pdf text layer yields it: weight bracket
# (kg), pallet-metre equivalent (lvm), then one price per distance band
LAVARAHTI_FREIGHT_TABLE_AS_PRINTED = """
0 - 79 0,00 - 0,24 29,71 30,58 31,21 32,21 34,15 35,99 37,15 38,51 40,18 42,32 43,63 44,75 46,57 49,04 51,92 53,26 53,39 57,67 62,02 62,38 62,87 63,36 63,86 64,36 64,87 65,38 65,89 66,41 66,95 67,47
80 - 99 0,25 - 0,30 31,55 32,33 33,22 34,37 36,49 38,70 39,99 41,56 43,45 45,86 47,37 48,84 50,84 53,62 57,62 59,24 60,72 64,36 70,06 70,84 71,40 71,95 72,51 73,08 73,65 74,23 74,81 75,40 76,00 76,60
100 - 149 0,31 - 0,45 34,72 35,63 36,76 38,42 41,11 43,73 45,61 47,89 50,39 53,57 55,48 57,64 60,49 63,86 69,20 71,68 73,59 78,08 84,36 85,01 85,69 86,38 87,07 87,77 88,46 89,18 89,90 90,62 91,35 92,09
150 - 199 0,46 - 0,60 40,81 41,86 43,25 45,31 48,43 51,71 54,17 57,08 60,15 63,99 66,42 69,30 72,89 77,10 83,67 86,94 89,34 93,59 100,35 101,89 102,70 103,53 104,36 105,20 106,04 106,90 107,76 108,64 109,52 110,41
200 - 249 0,61 - 0,75 44,86 46,05 47,78 50,21 53,80 57,78 60,67 64,29 68,07 72,59 75,41 78,85 83,39 88,18 96,21 100,16 103,17 107,14 114,38 117,00 117,93 118,86 119,82 120,77 121,74 122,73 123,70 124,70 125,71 126,73
250 - 299 0,76 - 0,90 50,66 51,87 53,98 56,82 60,87 65,54 68,93 73,18 77,53 82,69 86,06 90,27 95,41 101,07 110,36 115,13 119,62 122,44 130,06 133,60 134,67 135,75 136,84 137,95 139,06 140,19 141,34 142,48 143,65 144,81
300 - 349 0,91 - 1,05 55,28 56,73 59,08 62,31 66,92 72,16 76,12 81,01 86,03 91,93 95,55 100,55 106,49 112,75 123,48 129,03 134,94 136,54 144,71 149,28 150,48 151,69 152,91 154,14 155,39 156,65 157,92 159,21 160,50 161,81
350 - 399 1,06 - 1,20 61,11 62,65 65,20 68,89 74,00 79,96 84,45 89,89 95,48 102,12 106,33 111,88 118,66 125,69 137,70 144,09 151,51 151,83 160,43 165,95 167,28 168,63 169,98 171,35 172,73 174,13 175,54 176,96 178,41 179,86
400 - 449 1,21 - 1,35 65,04 66,75 69,65 73,70 79,21 85,86 90,77 97,07 103,30 110,54 115,09 121,48 129,01 136,70 149,19 156,24 165,19 166,32 175,46 180,81 182,27 183,73 185,21 186,70 188,21 189,74 191,27 192,82 194,40 195,97
450 - 499 1,36 - 1,50 71,41 73,16 76,35 80,87 86,88 94,18 99,64 106,47 113,31 121,16 126,40 133,38 141,57 150,10 163,16 170,85 182,11 182,11 191,73 198,05 199,61 201,20 202,81 204,43 206,06 207,72 209,39 211,08 212,78 214,50
500 - 549 1,51 - 1,65 75,21 77,21 80,67 85,60 92,12 99,96 105,97 113,54 121,00 129,56 135,20 142,89 151,89 161,14 174,49 182,94 196,68 195,51 205,63 212,89 214,60 216,33 218,08 219,85 221,63 223,43 225,25 227,08 228,94 230,81
550 - 599 1,66 - 1,80 79,87 81,94 85,77 91,02 98,05 106,60 113,11 121,37 129,37 138,65 144,76 153,10 162,93 172,80 186,68 195,96 212,04 209,59 220,19 228,40 230,23 232,09 233,96 235,85 237,76 239,69 241,63 243,60 245,58 247,59
600 - 649 1,81 - 1,95 85,12 87,33 91,34 97,09 104,56 113,79 120,82 129,74 138,34 148,37 154,79 163,91 174,49 185,11 199,51 209,47 227,97 224,33 235,50 244,55 246,52 248,50 250,51 252,53 254,57 256,65 258,74 260,84 262,97 265,12
650 - 699 1,96 - 2,10 89,80 92,12 96,41 102,60 110,49 120,42 127,99 137,50 146,75 157,43 164,48 174,19 185,52 196,90 211,80 222,40 243,30 238,50 250,08 260,20 262,29 264,41 266,55 268,70 270,89 273,09 275,32 277,56 279,83 282,12
700 - 749 2,11 - 2,25 94,34 96,87 101,53 108,09 116,54 126,96 135,02 145,34 155,18 166,57 173,84 184,33 198,05 210,16 223,92 235,36 256,68 252,65 265,72 275,72 277,92 280,15 282,41 284,68 286,99 289,31 291,66 294,03 296,43 298,84
750 - 799 2,26 - 2,40 99,57 102,13 107,10 114,05 123,02 134,11 142,77 153,65 164,13 176,15 184,09 195,19 211,00 222,42 236,62 248,80 270,67 267,17 280,91 291,81 294,16 296,52 298,92 301,33 303,77 306,24 308,73 311,24 313,78 316,35
800 - 849 2,41 - 2,55 105,17 107,85 113,06 120,48 129,87 141,55 150,74 162,34 173,41 186,11 194,46 206,32 222,81 235,08 249,79 262,64 284,94 282,32 296,40 308,32 310,81 313,32 315,85 318,40 320,99 323,59 326,22 328,88 331,57 334,28
850 - 899 2,56 - 2,70 110,63 113,47 118,97 126,83 136,61 149,09 158,78 170,94 182,66 196,01 204,83 217,42 234,71 247,67 262,91 276,36 299,18 297,25 311,82 324,75 327,37 330,02 332,68 335,38 338,11 340,86 343,65 346,46 349,29 352,15
900 - 949 2,71 - 2,85 114,45 117,36 123,15 131,45 141,77 154,81 165,02 177,87 190,20 204,31 213,54 226,74 244,87 258,48 274,20 288,46 311,75 310,49 325,64 339,46 342,19 344,95 347,73 350,54 353,38 356,26 359,15 362,07 365,03 368,01
950 - 999 2,86 - 3,00 119,62 122,68 128,74 137,43 148,28 161,97 172,69 186,21 199,17 213,93 223,58 237,53 256,43 270,81 286,91 301,91 325,66 325,11 340,74 355,52 358,39 361,27 364,20 367,14 370,13 373,14 376,18 379,24 382,35 385,48
1000 - 1099 3,01 - 3,30 129,78 132,97 139,41 148,46 159,73 174,10 185,34 199,54 213,07 228,58 238,68 253,33 274,41 289,64 304,64 320,42 344,63 344,88 363,04 376,68 379,75 382,85 385,99 389,15 392,36 395,59 398,85 402,15 405,48 408,84
1100 - 1199 3,31 - 3,60 130,56 133,90 140,64 150,13 161,88 176,92 188,70 203,55 217,59 233,78 244,48 259,75 283,04 297,49 313,07 329,50 354,25 355,11 373,86 388,44 391,64 394,86 398,11 401,40 404,72 408,07 411,47 414,89 418,34 421,84
1200 - 1299 3,61 - 3,90 131,43 134,86 141,81 151,83 164,02 179,65 191,96 207,49 222,23 239,14 250,12 268,71 291,69 305,46 321,45 338,67 363,83 365,49 384,62 400,14 403,45 406,80 410,17 413,58 417,02 420,50 424,01 427,56 431,15 434,77
1300 - 1399 3,91 - 4,20 132,26 135,85 143,14 153,45 166,25 182,49 195,35 211,48 226,81 244,33 255,93 277,68 300,27 313,26 329,91 347,79 373,38 375,81 395,36 411,87 415,29 418,75 422,25 425,78 429,34 432,95 436,59 440,26 443,97 447,72
1400 - 1499 4,21 - 4,50 133,12 136,83 144,39 155,12 168,36 185,26 198,60 215,44 231,43 249,69 261,62 284,07 308,96 321,29 338,30 356,82 383,03 386,14 406,17 423,72 427,26 430,85 434,47 438,12 441,80 445,53 449,29 453,09 456,93 460,80
1500 - 1599 4,51 - 4,80 144,48 148,33 156,59 168,12 182,31 200,47 214,89 233,02 250,27 264,60 283,98 306,59 333,01 350,00 366,42 386,62 411,87 417,50 437,48 459,17 463,07 467,01 470,99 475,01 479,06 483,17 487,31 491,48 495,71 499,97
1600 - 1699 4,81 - 5,10 146,16 150,26 159,06 171,44 186,58 206,02 221,53 241,02 259,46 269,90 295,47 319,51 347,40 365,73 384,89 404,82 431,07 438,08 459,13 485,06 489,23 493,43 497,67 501,95 506,28 510,64 515,06 519,52 524,03 528,57
1700 - 1799 5,11 - 5,40 147,73 152,08 161,55 174,88 190,91 211,66 228,15 248,99 268,67 275,11 306,94 332,32 361,74 381,62 401,66 424,89 450,27 458,78 480,69 508,54 512,93 517,37 521,84 526,37 530,94 535,56 540,22 544,94 549,69 554,49
1800 - 1899 5,41 - 5,70 149,42 153,99 164,02 178,13 195,21 217,28 234,75 256,97 277,81 280,45 318,47 345,22 376,16 397,43 418,45 443,13 469,38 479,44 502,28 532,08 536,70 541,38 546,09 550,87 555,68 560,54 565,46 570,42 575,44 580,50
1900 - 1999 5,71 - 6,00 151,13 155,85 166,57 181,54 199,47 222,88 241,48 264,94 286,98 285,75 329,91 358,00 390,63 413,33 435,29 461,33 488,63 499,98 523,87 555,52 560,38 565,29 570,25 575,26 580,32 585,44 590,60 595,81 601,07 606,39
2000 - 2099 6,01 - 6,30 164,43 169,44 180,75 196,48 215,66 240,11 259,80 284,55 309,64 313,21 353,00 382,50 416,63 440,75 463,70 491,22 519,46 532,35 559,22 590,73 595,95 601,22 606,53 611,91 617,33 622,82 628,35 633,95 639,59 645,29
2100 - 2199 6,31 - 6,60 166,12 171,26 183,22 199,85 220,19 245,69 266,37 292,53 320,67 329,09 364,49 395,44 431,07 456,62 480,51 509,45 538,72 552,94 580,85 614,27 619,72 625,22 630,78 636,40 642,07 647,80 653,58 659,43 665,32 671,29
2200 - 2299 6,61 - 6,90 167,75 173,19 185,74 203,18 224,76 251,31 273,02 300,52 331,68 344,92 376,02 408,21 445,41 472,46 497,27 527,69 557,92 573,59 602,42 637,78 643,46 649,21 655,00 660,86 666,78 672,76 678,80 684,89 691,05 697,26
2300 - 2399 6,91 - 7,20 169,44 175,05 188,23 206,45 229,16 256,84 279,68 308,43 342,69 360,76 387,49 421,11 459,87 488,31 514,11 545,94 577,12 594,28 624,10 661,36 667,28 673,26 679,29 685,40 691,56 697,78 704,07 710,41 716,83 723,30
2400 - 2499 7,21 - 7,51 171,17 176,92 190,75 209,90 233,75 262,43 286,33 316,41 353,75 376,59 398,97 433,94 474,21 504,10 539,29 564,16 596,28 614,91 645,62 684,82 690,98 697,20 703,48 709,82 716,23 722,70 729,23 735,83 742,50 749,23
2500 - 2599 7,51 - 7,80 188,07 194,21 209,11 229,51 255,38 285,98 311,57 343,90 385,44 413,86 434,73 469,79 512,81 544,98 582,20 609,24 643,21 663,78 696,42 738,74 745,36 752,06 758,81 765,64 772,54 779,50 786,54 793,64 800,81 808,06
2600 - 2699 7,81 - 8,10 189,79 196,19 211,66 233,02 260,02 291,75 319,08 352,12 396,70 430,09 448,79 483,01 530,53 561,24 599,42 627,99 662,94 685,04 718,58 762,84 769,70 776,64 783,64 790,72 797,87 805,08 812,37 819,73 827,17 834,68
2700 - 2799 8,11 - 8,40 191,48 198,05 214,23 236,46 264,60 297,47 326,57 360,27 408,05 446,47 463,06 496,21 548,30 577,60 616,69 646,69 682,66 706,21 740,82 787,05 794,15 801,33 808,58 815,90 823,31 830,78 838,32 845,95 853,64 861,42
2800 - 2899 8,41 - 8,70 193,23 199,97 216,85 239,85 269,26 303,23 334,09 368,52 419,38 462,68 479,48 509,45 566,03 593,89 633,97 665,46 702,42 727,38 763,01 811,17 818,51 825,94 833,43 841,01 848,66 856,38 864,18 872,06 880,02 888,05
2900 - 2999 8,71 - 9,00 194,92 201,86 219,39 243,30 273,93 308,93 341,56 376,68 430,71 479,00 496,05 522,58 583,79 610,09 651,22 684,20 722,15 748,68 785,18 835,39 842,98 850,65 858,40 866,21 874,11 882,09 890,15 898,30 906,52 914,82
3000 - 3199 9,01 - 9,60 209,59 216,89 235,38 262,28 294,89 330,82 367,24 403,48 459,78 507,30 530,53 557,81 624,80 653,99 695,22 731,00 768,74 800,02 840,68 883,74 891,82 899,97 908,21 916,52 924,93 933,41 941,99 950,65 959,39 968,23
3200 - 3399 9,61 - 10,20 212,11 219,89 239,21 269,10 303,64 340,57 380,94 418,20 476,77 523,60 552,98 580,96 653,62 685,78 727,25 765,65 803,32 839,24 883,95 920,04 928,47 937,00 945,60 954,30 963,08 971,95 980,90 989,96 999,09 1008,32
3400 - 3599 10,21 - 10,80 214,75 222,79 243,05 275,90 312,51 350,34 394,59 432,91 494,75 539,96 575,39 604,13 682,40 717,51 759,21 800,32 837,81 878,53 927,28 963,52 972,40 981,36 990,41 999,56 1008,79 1018,12 1027,54 1037,05 1046,66 1056,36
3600 - 3799 10,81 - 11,40 217,33 225,70 246,96 282,84 321,35 360,14 408,92 447,68 512,61 556,17 597,82 629,89 711,32 749,29 791,29 835,05 872,36 917,78 967,27 1006,97 1016,28 1025,68 1035,18 1044,77 1054,45 1064,24 1074,12 1084,09 1094,18 1104,36
3800 - 3999 11,41 - 12,00 219,89 228,58 250,79 289,71 330,19 369,93 423,27 462,40 531,53 572,48 620,21 655,63 737,23 781,87 823,15 869,76 906,85 957,09 1007,22 1050,51 1060,26 1070,10 1080,03 1090,08 1100,21 1110,45 1120,80 1131,24 1141,80 1152,45
4000 - 4199 12,01 - 12,60 232,98 242,03 265,18 307,07 349,50 390,17 448,09 490,81 560,89 599,21 653,08 693,90 773,61 824,97 865,65 914,88 951,91 1012,09 1061,02 1108,13 1118,46 1128,88 1139,40 1150,03 1160,77 1171,62 1182,57 1193,64 1204,81 1216,10
4200 - 4399 12,61 - 13,20 235,55 244,98 269,01 313,94 358,34 399,93 463,13 508,84 579,70 615,52 675,55 721,72 799,57 857,43 897,59 949,55 986,50 1056,65 1104,27 1155,36 1166,14 1177,04 1188,05 1199,17 1210,40 1221,75 1233,20 1244,77 1256,45 1268,26
4400 - 4599 13,21 - 13,80 238,14 247,89 272,81 320,81 367,15 409,72 478,14 526,96 598,58 631,79 701,49 749,45 825,45 889,24 929,60 984,27 1021,02 1101,26 1147,66 1202,46 1213,72 1225,10 1236,59 1248,19 1259,91 1271,76 1283,71 1295,78 1307,98 1320,30
4600 - 4799 13,81 - 14,40 240,64 250,79 276,68 327,68 376,02 419,43 493,73 544,98 617,54 648,16 727,38 777,18 851,32 921,05 961,56 1018,96 1055,54 1145,69 1190,98 1249,70 1261,44 1273,29 1285,26 1297,35 1309,57 1321,90 1334,36 1346,94 1359,64 1372,48
4800 - 4999 14,41 - 15,00 243,25 253,70 280,52 334,55 384,87 429,16 509,49 562,95 636,41 664,37 753,41 804,88 877,30 952,80 993,57 1053,65 1090,05 1190,28 1234,30 1296,82 1309,02 1321,35 1333,81 1346,38 1359,08 1371,91 1384,87 1397,96 1411,17 1424,52
5000 - 5999 15,02 - 18,02 253,25 296,49 337,90 380,72 423,09 465,06 507,18 549,44 591,51 633,59 675,64 718,34 760,42 802,72 844,64 887,08 929,30 1203,32 1014,16 1310,46 1322,96 1335,61 1348,38 1361,28 1374,29 1387,46 1400,74 1414,16 1427,71 1441,41
6000 - 6999 18,02 - 21,02 278,19 328,50 375,11 424,46 472,97 520,46 568,88 616,50 664,86 712,39 760,73 809,32 857,70 906,14 953,60 1002,12 1050,52 1098,06 1147,48 1194,93 1206,29 1217,76 1229,36 1241,06 1252,89 1264,82 1276,88 1289,07 1301,37 1313,80
7000 - 7999 21,02 - 24,02 296,99 351,94 402,49 456,58 509,36 560,93 613,84 665,67 718,36 769,97 823,02 875,56 928,47 981,36 1032,97 1085,69 1138,77 1190,53 1244,13 1295,92 1308,29 1320,79 1333,41 1346,15 1359,02 1372,03 1385,16 1398,42 1411,82 1425,35
8000 - 8999 24,02 - 27,02 322,53 381,29 437,54 497,42 554,95 611,76 670,15 728,01 785,49 842,31 901,72 958,25 1016,74 1075,07 1131,94 1189,50 1248,78 1306,52 1363,67 1421,53 1435,16 1448,92 1462,81 1476,85 1491,04 1505,37 1519,83 1534,43 1549,19 1564,10
9000 - 9999 27,03 - 30,03 348,02 410,69 472,58 538,27 600,56 662,63 726,46 790,38 852,63 914,69 980,38 1040,87 1104,98 1168,74 1230,97 1293,24 1358,73 1422,46 1483,22 1547,09 1561,98 1577,02 1592,20 1607,53 1623,02 1638,66 1654,46 1670,42 1686,53 1702,80
10000 - 10999 30,03 - 33,03 359,10 424,24 488,64 556,46 621,23 685,25 751,42 817,64 882,12 946,65 1014,24 1077,40 1143,23 1209,59 1274,35 1338,54 1406,49 1472,54 1535,91 1601,85 1617,29 1632,87 1648,61 1664,51 1680,57 1696,79 1713,17 1729,71 1746,42 1763,30
11000 - 11999 33,03 - 36,03 381,27 451,21 520,64 592,75 662,46 730,44 801,24 872,06 940,98 1010,48 1081,82 1150,29 1219,59 1291,17 1360,93 1428,89 1501,72 1572,44 1641,13 1711,16 1727,69 1744,37 1761,23 1778,25 1795,44 1812,81 1830,36 1848,07 1865,96 1884,03
12000 - 12999 36,04 - 39,04 403,41 478,19 552,67 629,06 703,76 775,61 851,12 926,55 999,81 1074,28 1149,46 1223,18 1295,95 1372,75 1447,50 1519,32 1596,96 1672,35 1746,30 1820,41 1838,03 1855,82 1873,78 1891,94 1910,27 1928,79 1947,49 1966,37 1985,45 2004,71
13000 - 13999 39,04 - 42,04 425,57 505,23 584,68 665,29 745,03 820,75 900,95 981,01 1058,67 1138,11 1217,09 1296,07 1372,32 1454,30 1534,09 1609,70 1692,21 1772,26 1851,49 1929,71 1948,42 1967,31 1986,40 2005,68 2025,14 2044,81 2064,66 2084,73 2104,99 2125,45
14000 - 14999 42,04 - 45,04 447,77 532,23 616,73 701,61 786,28 865,92 950,76 1035,47 1117,52 1201,94 1284,67 1368,92 1448,69 1535,87 1620,65 1700,09 1787,46 1872,15 1956,66 2038,97 2058,78 2078,78 2098,98 2119,38 2139,99 2160,80 2181,82 2203,05 2224,49 2246,14
"""

# the distance band header of the same table, as the text layer yields its two rows
LAVARAHTI_KM_BAND_LOWER_AS_PRINTED = (
    "1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 "
    "1000 1050 1100 1150 1200 1250 1300 1350 1400 1450")
LAVARAHTI_KM_BAND_UPPER_AS_PRINTED = (
    "49 99 149 199 249 299 349 399 449 499 549 599 649 699 749 799 849 899 949 999 "
    "1049 1099 1149 1199 1249 1299 1349 1400 1449 1500")

MATKAHUOLTO_LAVARAHTI = {
    "artifact": "rate_calibration/matkahuolto_lavarahti.json",
    "record": {
        "source": ("Oy Matkahuolto Ab, Lavarahtihinnasto terminaalista vastaanottajalle "
                   "(pallet freight price list, terminal to recipient)"),
        "source_url": ("https://assets.ctfassets.net/lt6mvg8ztynj/yy0EMMk4lEJuvV0F81L1R/"
                       "ee3537135b96d5b5e303188ebe605327/Lavarahtihinnasto.pdf"),
        "source_url_given_in_task": "https://www.matkahuolto.fi/",
        "source_location_note": (
            "the task named the Matkahuolto home page, which prints no tariff. the "
            "Lavarahti service page, https://www.matkahuolto.fi/yrityksille/"
            "pakettipalvelut/toimitukset-yritykselta-yritykselle/isot-paketit-ja-"
            "lavarahti, describes the service and its limits but prints no prices. the "
            "tariff below was read from Matkahuolto's published Lavarahti price list "
            "pdf named in source_url"),
        "page_or_section": ("page 1 weight x distance freight table; page 2 'Lavarahdin "
                            "nouto' pickup table; page 3 section 1.1 pricing rules and "
                            "fuel surcharge wording; service limits from the Lavarahti "
                            "service page"),
        "role": "upper-bound calibration evidence for F_i and alpha_i",
        "calibration_bound": "upper",
        "currency": "EUR",
        "unit": "EUR per shipment by chargeable-weight bracket and road-distance band",
        "access_date_basis": (
            "no price values were supplied with the task for this source; the tariff "
            "was read on this date from the price list named in source_url"),
        "reverification_status": "read from the published price list; given url has no tariff",
        "tariff_metadata": {
            "vat_status": field("exclusive",
                                "Hinnasto voimassa 1.11.2024 alkaen (eur, alv 0%)",
                                "page 1 and page 2 headings"),
            "effective_date": field("2024-11-01",
                                    "Hinnasto voimassa 1.11.2024 alkaen (eur, alv 0%)"),
            "currency_as_published": field("EUR", "(eur, alv 0%)"),
            "unit_basis": field(
                ("per shipment by chargeable-weight bracket (kg, with a pallet-metre "
                 "equivalent column) and road-distance band (km), terminal to recipient; "
                 "chargeable weight is the greater of actual weight and cubic weight "
                 "1 m3 = 333 kg, or pallet-metre weight 1 lvm = 1850 kg when nothing can "
                 "be loaded on top"),
                ("Lavarahdin veloitus perustuu kilometreihin ja rahdituspainoon. ... "
                 "Kuutiopaino 1 m3 = 333 kg ... Lavametri (lvm) 1 lvm = 1850 kg "
                 "(pituus x leveys / 2,4m)"),
                "page 3, section 1.1"),
            "service_category": field("pallet freight", "Lavarahti"),
            "fuel_surcharge_treatment": field(
                "charged separately, reviewed weekly",
                ("Lavarahdin polttoainelisä tarkistetaan viikoittain ja astuu voimaan "
                 "maanantaisin. ... Kulloinkin voimassa oleva polttoainelisä on "
                 "nähtävillä Matkahuollon verkkosivuilla."),
                "page 3, section 1.1"),
            "pickup_treatment": field(
                ("not included in the freight table; pickup is always part of the "
                 "service and is priced separately on page 2"),
                ("Rahtihinta sisältää kuljetuksen terminaalista vastaanottajalle minne "
                 "tahansa Suomessa, noutohinta esitetty erikseen. Palveluun kuuluu aina "
                 "nouto"),
                "page 3, section 1.1"),
        },
        "service_limits_as_printed": [
            "Lavarahti on palvelumme yli 35 kg tai isokokoisille lähetyksille.",
            "Lähetys saa painaa enintään 3000 kg",
            "Lähetyksen yksittäisen paketin enimmäispaino on 1200 kg",
            "Lähetyksen kokonaistilavuus voi olla enintään 9 m3",
        ],
        "table_interpretation_note": (
            "the pdf text layer yields the distance header as two rows of band bounds "
            "and prefixes each price row with its weight bracket and pallet-metre "
            "bracket. the 1400 km bound is printed twice, as the upper bound of the "
            "1350 band and the lower bound of the 1400 band, and is kept as printed. "
            "the table extends to 14999 kg while the service page caps a shipment at "
            "3000 kg; both are recorded as printed"),
        "freight_table_as_printed": LAVARAHTI_FREIGHT_TABLE_AS_PRINTED.strip(),
        "km_band_lower_as_printed": LAVARAHTI_KM_BAND_LOWER_AS_PRINTED,
        "km_band_upper_as_printed": LAVARAHTI_KM_BAND_UPPER_AS_PRINTED,
        "pickup_table_as_printed": {
            "page": 2, "heading": "Lavarahdin nouto",
            "columns": ["Paino kg", "Hinta € (alv 0 %)"],
            "rows": [[79, 17.70], [99, 19.87], [149, 22.18], [199, 24.09], [249, 27.38],
                     [299, 27.87], [349, 30.21], [399, 32.18], [449, 33.99], [499, 36.66],
                     [549, 38.44], [599, 40.42], [649, 42.17], [699, 44.70], [1999, 46.46],
                     [14999, 51.13]],
        },
        "verification_note": (
            "upper-bound calibration evidence for heavier shipments outside ordinary "
            "parcel weight limits; record the actual tariff structure found, do not "
            "assume it matches Posti's zone count or unit types"),
    },
}


# ── line_haul_rate: beta_i ───────────────────────────────────────────────────

UPPLY_PDF = {
    "file_name_given_in_task": ("upplyirutiq22026europeanroadfreightratesbenchmark.pdf "
                                "(project root)"),
    "file_read": "upply-iru-ti-q2-2026-european-road-freight-rates-benchmark.pdf",
    "location_at_access": "the user's Downloads folder; no copy exists at the project root",
    "sha256": "281076fa8d6b37c8abbfc3ef485876f0103517294c2227d11dce9456a1a316f9",
    "bytes": 7335305,
    "pages": 38,
    "pdf_creation_date": "2026-08-11T15:10:14Z",
    "not_committed_because": (
        "page 38 states 'No part of this publication may be reproduced in any "
        "material form including photocopying or storing it by electronic means "
        "without the permission of the copyright owners, Transport Intelligence "
        "Limited / Upply / IRU.' the hash above identifies the exact copy read"),
}

UPPLY_CORRIDOR_RATES = {
    "artifact": "line_haul_rate/upply_corridor_rates.json",
    "record": {
        "source": ("IRU / Upply / Ti, The European Road Freight Rate Benchmark, "
                   "Q2 2026 - August 2026"),
        "source_url": "https://www.upply.com/en/ti-upply-iru-benchmarks-reports",
        "source_url_note": ("the publisher's report page as listed in "
                            "docs/data_synthesis_methodology.md section 4; the pdf's own "
                            "download url was not recorded when it was obtained"),
        "local_copy": UPPLY_PDF,
        "page_or_section": ("page 22 'City pairs - Contract, Headhaul Contract Rates, Data "
                            "on Rates'; page 25 'City pairs - Spot, Headhaul Spot Rates, "
                            "Data on Rates'; methodology page 36"),
        "role": "source values for the beta_i candidate derivation",
        "currency": "EUR",
        "unit": "EUR per km",
        "access_date_basis": (
            "read on this date from the local pdf copy identified in local_copy. the "
            "euro sign was confirmed on rendered images of pages 22 and 25, because "
            "the pdf text layer does not map that glyph"),
        "reverification_status": "reverified from the local copy",
        "tariff_metadata": {
            "vat_status": field(NOT_STATED, NOT_STATED,
                                "the 38 pages were searched for VAT wording; none applies to the rates"),
            "effective_date": field(
                "Q2 2026, quarterly average of weekly median rates",
                ("Ti then used the median rates provided by Upply on each corridor, "
                 "averaging weekly rates over each quarter."),
                "page 36, methodology"),
            "currency_as_published": field("EUR", "€1.45/km"),
            "unit_basis": field(
                ("EUR per km on an international headhaul lane, printed beside an index "
                 "with Q1-2017 = 100; not per tonne and not per tonne-km"),
                "166.8 index points, or €1.45/km",
                "page 22; the chart axis reads 'Index,Q1-2017=100'"),
            "service_category": field(
                "international road freight line-haul, truck load (LTL and FTL)",
                ("Upply provides Truck Load (LTL & FTL) weekly rates estimations based on "
                 "observed transactions for each major European trade lanes, associated "
                 "with a confidence index."),
                "page 36"),
            "fuel_surcharge_treatment": field(
                NOT_STATED, NOT_STATED,
                ("the report attributes rate increases to diesel costs being passed on "
                 "('Fuel has been the dominant driver in pushing rates upwards QoQ and "
                 "YoY, with operators unable to absorb the scale of diesel increases "
                 "without passing them through to customers', page 4) but never states "
                 "whether the per-km rates include or exclude a separate fuel surcharge")),
            "payload_or_load_factor": field(
                NOT_STATED, NOT_STATED,
                ("no payload, load factor or tonnage behind the per-km rates is stated in "
                 "the 38 pages; the document was searched for payload, tonne, capacity "
                 "and load factor")),
        },
        "observations": [
            {"page": 22, "rate_type": "contract", "corridor": "Warsaw to Duisburg",
             "index_points": 166.8, "eur_per_km": 1.45,
             "printed_as": ("The Warsaw to Duisburg route closed Q2 2026 at 166.8 index "
                            "points, or €1.45/km, up 6.1 points QoQ and 14.9 points "
                            "compared with the same quarter last year.")},
            {"page": 22, "rate_type": "contract", "corridor": "Duisburg to Lille",
             "index_points": 159.4, "eur_per_km": 2.58,
             "printed_as": ("Contract rates on the Duisburg to Lille lane reached 159.4 "
                            "index points (€2.58/km) in Q2 2026, up 13.1 points QoQ and "
                            "22.2 points YoY.")},
            {"page": 22, "rate_type": "contract", "corridor": "Duisburg to Prague",
             "index_points": 149.2, "eur_per_km": 1.73,
             "printed_as": ("Duisburg to Prague contract rates reached 149.2 index points "
                            "in Q2 2026, equivalent to €1.73/km. The index rose by 11.6 "
                            "points QoQ and by 19.7 points YoY.")},
            {"page": 22, "rate_type": "contract", "corridor": "Antwerp to Duisburg",
             "index_points": 181.8, "eur_per_km": 3.05,
             "printed_as": ("Antwerp to Duisburg closed Q2 2026 at 181.8 index points, or "
                            "€3.05/km, up 17.1 points QoQ and 25.1 points compared with "
                            "the same quarter last year.")},
            {"page": 25, "rate_type": "spot", "corridor": "Warsaw to Duisburg",
             "index_points": 170.4, "eur_per_km": 1.51,
             "printed_as": ("The Warsaw to Duisburg route closed Q2 2026 at 170.4 index "
                            "points, or €1.51/km, up 10.9 points QoQ and 17.3 points "
                            "compared with the same quarter last year.")},
            {"page": 25, "rate_type": "spot", "corridor": "Duisburg to Lille",
             "index_points": 168.2, "eur_per_km": 2.89,
             "printed_as": ("Spot rates on the Duisburg to Lille lane reached 168.2 index "
                            "points (€2.89/km) in Q2 2026, up 20.0 points QoQ and 23.2 "
                            "points YoY.")},
            {"page": 25, "rate_type": "spot", "corridor": "Duisburg to Prague",
             "index_points": 156.4, "eur_per_km": 2.00,
             "printed_as": ("Duisburg to Prague spot rates reached 156.4 index points in "
                            "Q2 2026, equivalent to €2.00/km. The index rose by 13.8 "
                            "points QoQ and by 19.5 points YoY.")},
            {"page": 25, "rate_type": "spot", "corridor": "Antwerp to Duisburg",
             "index_points": 167.6, "eur_per_km": 3.19,
             "printed_as": ("Antwerp to Duisburg spot rates stood at 167.6 index points "
                            "(€3.19/km), up 13.9 points QoQ and 17.9 points YoY.")},
        ],
        "verification_note": (
            "corridor rates and page references only, as printed on pages 22 and 25; no "
            "payload division, derivation arithmetic or comparison to 0.07163 is "
            "recorded here, that lives in "
            "data/processed/v5/logistics/beta_candidate_derivation.json"),
    },
}


# ── fuel_surcharge: evidence of mechanisms, never a per-agent value ──────────

POSTNORD_FUEL_SURCHARGE = {
    "artifact": "fuel_surcharge/postnord_fuel_surcharge.json",
    "record": {
        "source": "PostNord Finland, fuel and sulphur surcharge",
        "source_url": "https://www.postnord.fi/en/sending/fuel-and-sulphur-surcharge",
        "read_from": ("https://web.archive.org/web/20260218112022/"
                      "https://www.postnord.fi/en/sending/fuel-and-sulphur-surcharge/"),
        "archive_capture": {
            "timestamp": "2026-02-18T11:20:22Z",
            "sha256_of_captured_html": ("83cb8d991d8f81f6b1422df672fca501cd3f389bb0b36301"
                                        "61339f3aca972b89"),
        },
        "page_or_section": ("'Calculating fuel surcharge in Finland' and the parcels / "
                            "pallets and groupage table"),
        "role": "source evidence of a fuel-surcharge mechanism",
        "currency": None,
        "unit": "percent of freight costs excluding additional services",
        "access_date_basis": (
            "the live page returned HTTP 403 to urllib and to WebFetch on this date. the "
            "text was read from the most recent Internet Archive capture, 18 February "
            "2026; any percentage PostNord published after that capture is not in this "
            "record"),
        "reverification_status": "read from an archive capture; live page HTTP 403",
        "observations_status": "partial: the capture ends at the row starting 1.2.2026",
        "mechanism_as_printed": [
            "The fuel surcharge cost is updated monthly.",
            ("The fuel surcharge for both domestic and international transport is updated "
             "monthly. The surcharge is the percentage of freight costs (excluding "
             "additional services) and is tied to the previous month's diesel consumer "
             "price index maintained by Statistics Finland."),
            "The fuel surcharge can not be negative.",
        ],
        "tariff_metadata": {
            "vat_status": field(NOT_STATED),
            "effective_date": field("per row, the date in the 'Starting from' column",
                                    "Starting from"),
            "currency_as_published": field(
                "not applicable: published as a percentage of freight cost",
                "Fuel surcharge parcels, Fuel surcharge pallets and groupage"),
            "unit_basis": field(
                "percent of freight costs excluding additional services",
                ("The surcharge is the percentage of freight costs (excluding additional "
                 "services)")),
            "service_category": field("parcels; pallets and groupage",
                                      "Fuel surcharge parcels | Fuel surcharge pallets and groupage"),
            "fuel_surcharge_treatment": field(
                "not applicable: this record is the surcharge itself, charged on freight cost",
                "The surcharge is the percentage of freight costs"),
        },
        "observations": [
            {"starting_from": d, "parcels_percent": p, "pallets_and_groupage_percent": g}
            for d, p, g in (
                ("2026-02-01", 4.6, 6.7), ("2026-01-01", 4.2, 6.2),
                ("2025-12-01", 3.2, 4.7), ("2025-11-01", 2.6, 3.8),
                ("2025-10-01", 2.8, 4.2), ("2025-09-01", 2.6, 3.8),
                ("2025-08-01", 3.3, 4.9), ("2025-07-01", 2.8, 4.2),
                ("2025-06-01", 3.0, 4.4), ("2025-05-01", 3.9, 5.8),
                ("2025-04-01", 4.8, 7.1), ("2025-03-01", 4.8, 7.1),
                ("2025-02-01", 4.9, 7.2), ("2025-01-01", 4.6, 6.7))
        ],
        "sulphur_surcharge_as_printed": {
            "wording": ("The sulphur surcharge is calculated as a percentage of the freight "
                        "cost. ... Sulphur surcharge is valid until further notice."),
            "percent_by_destination": {
                "Parcels": {"SE, DK, NO": 1.00, "EE, LT, LV, PL": 0.58, "Other": 2.48},
                "Pallets": {"SE, DK, NO": 3.90, "EE, LT, LV, PL": 1.97, "Other": 3.94},
                "Groupage": {"SE, DK, NO": 4.20},
            },
            "note": "destination-country surcharges outside Finland, kept as context only",
        },
        "verification_note": (
            "this is SOURCE EVIDENCE of a real fuel-surcharge mechanism, tied to the "
            "previous month's diesel consumer price index from Statistics Finland. It is "
            "not automatically the surcharge value for every simulated logistics agent. "
            "Huong expects this component to differ between agents; record the mechanism "
            "and whatever historical percentages are actually published, do not produce "
            "or imply a single common value here."),
    },
}

POSTI_FUEL_SURCHARGE = {
    "artifact": "fuel_surcharge/posti_fuel_surcharge.json",
    "record": {
        "source": "Posti, fuel and energy surcharge",
        "source_url": "https://www.posti.fi/en/for-businesses/customer-support/fuel-surcharge",
        "page_or_section": ("the Parcel, Freight/Express freight/Transport units, Pick-up "
                            "and delivery service, Printed matter transport and Warehousing "
                            "Services lists"),
        "role": "source evidence of a fuel-surcharge mechanism",
        "currency": None,
        "unit": "percent; the base the percentage is applied to is not stated on this page",
        "access_date_basis": "read on this date from the live page named in source_url",
        "reverification_status": "read from the live page",
        "observations_status": ("partial history: the page publishes only the two most "
                                "recent values per service"),
        "tariff_metadata": {
            "vat_status": field(NOT_STATED),
            "effective_date": field("per row, the 'from' date", "from September 9, 2026"),
            "currency_as_published": field("not applicable: published as a percentage", "%"),
            "unit_basis": field(
                "percent", NOT_STATED,
                "the page does not state what the percentage is applied to or how it is set"),
            "service_category": field("per series, as printed below",
                                      "Parcel; Freight, Express freight and Transport units"),
            "fuel_surcharge_treatment": field(
                "not applicable: this record is the surcharge itself",
                NOT_STATED,
                ("Posti's 16.11.2025 contract price list states 'Kulloinkin voimassaoleva "
                 "polttoainelisä veloitetaan erikseen.' and points to posti.fi/polttoainelisa")),
        },
        "observations": [
            {"series": "Parcel", "from": "2026-09-09", "percent": 21.10,
             "printed_as": "21.10 % from September 9, 2026"},
            {"series": "Parcel", "from": "2026-09-02", "percent": 20.76,
             "printed_as": "20.76 % from September 2, 2026"},
            {"series": "Freight, Express freight and Transport units", "from": "2026-09-09",
             "percent": 33.93, "printed_as": "33.93 % from September 9, 2026"},
            {"series": "Freight, Express freight and Transport units", "from": "2026-09-02",
             "percent": 33.21, "printed_as": "33.21 % from September 2, 2026"},
            {"series": "Pick-up and delivery service", "from": "2026-09-01", "percent": 19.61,
             "printed_as": "19.61 % from September 1, 2026"},
            {"series": "Pick-up and delivery service", "from": "2026-08-01", "percent": 18.07,
             "printed_as": "18.07 % from August 1, 2026"},
            {"series": "Printed matter transport", "from": "2026-09-01", "percent": 34.59,
             "printed_as": "34.59 % from September 1, 2026"},
            {"series": "Printed matter transport", "from": "2026-08-01", "percent": 32.05,
             "printed_as": "32.05 % from August 1, 2026"},
            {"series": "Warehousing Services energy surcharge", "from": "2026-04-01",
             "percent": 1.21, "printed_as": "From April 1, 2026 the energy supplement is 1,21%",
             "note": "an energy supplement for warehousing, not a transport fuel surcharge"},
            {"series": "Warehousing Services energy surcharge", "from": "2026-07-01",
             "percent": 1.16, "printed_as": "From July 1, 2026 the energy supplement is 1,16%",
             "note": "an energy supplement for warehousing, not a transport fuel surcharge"},
        ],
        "verification_note": (
            "source evidence of Posti's published fuel and energy surcharge percentages, "
            "recorded as printed; the page states neither how the surcharge is set nor "
            "what it is applied to nor its VAT treatment, so it is not comparable "
            "percentage for percentage with PostNord's or Matkahuolto's; not a per-agent "
            "value"),
    },
}

PAKETTIPISTE_FUEL_SURCHARGE = {
    "artifact": "fuel_surcharge/pakettipiste_fuel_surcharge.json",
    "record": {
        "source": "Pakettipiste, fuel surcharge",
        "source_url": "https://www.pakettipiste.fi/en/fuel-surcharge/",
        "page_or_section": "the monthly 'Fuel surcharge starting' list",
        "role": "source evidence of a fuel-surcharge mechanism",
        "currency": None,
        "unit": "percent; the base the percentage is applied to is not stated on this page",
        "access_date_basis": "read on this date from the live page named in source_url",
        "reverification_status": "read from the live page",
        "mechanism_as_printed": [
            ("The Fuel surcharge is updated monthly based on the cost index of road "
             "transport of goods published by Statistics Finland."),
        ],
        "tariff_metadata": {
            "vat_status": field(NOT_STATED),
            "effective_date": field("per row, the month the surcharge starts",
                                    "Fuel surcharge starting September 1st, 2026"),
            "currency_as_published": field("not applicable: published as a percentage", "%"),
            "unit_basis": field("percent", NOT_STATED,
                                "the page does not state what the percentage is applied to"),
            "service_category": field("domestic parcel and logistics services",
                                      "Domestic parcel and logistics services"),
            "fuel_surcharge_treatment": field(
                "not applicable: this record is the surcharge itself", NOT_STATED),
        },
        "observations": [
            {"starting": d, "percent": p,
             "printed_as": f"Fuel surcharge starting {label} Domestic parcel and logistics services {printed} %"}
            for d, p, label, printed in (
                ("2026-09-01", 23.38, "September 1st, 2026", "23,38"),
                ("2026-08-01", 21.74, "August 1st, 2026", "21,74"),
                ("2026-07-01", 24.80, "July 1st, 2026", "24,80"),
                ("2026-06-01", 26.38, "June 1st, 2026", "26,38"),
                ("2026-05-01", 21.28, "May 1st, 2026", "21,28"),
                ("2026-04-01", 14.62, "April 1st, 2026", "14,62"),
                ("2026-03-01", 13.97, "March 1st, 2026", "13,97"),
                ("2026-02-01", 12.42, "February 1st, 2026", "12,42"),
                ("2026-01-01", 13.51, "January 1st, 2026", "13,51"),
                ("2025-12-01", 12.30, "December 1st, 2025", "12,30"),
                ("2025-11-01", 12.56, "November 1st, 2025", "12,56"),
                ("2025-10-01", 12.32, "October 1st, 2025", "12,32"),
                ("2025-09-01", 13.00, "September 1st, 2025", "13,00"),
                ("2025-08-01", 12.36, "August 1st, 2025", "12,36"),
                ("2025-07-01", 11.73, "July 1st, 2025", "11,73"),
                ("2025-06-01", 12.20, "June 1st, 2025", "12,20"),
                ("2025-05-01", 13.49, "May 1st, 2025", "13,49"),
                ("2025-04-01", 14.88, "April 1st, 2025", "14,88"),
                ("2025-03-01", 15.02, "March 1st, 2025", "15,02"),
                ("2025-02-01", 13.73, "February 1st, 2025", "13,73"),
                ("2025-01-01", 13.73, "January 1st, 2025", "13,73"))
        ],
        "verification_note": (
            "source evidence of Pakettipiste's published monthly fuel surcharge, recorded "
            "as printed; tied to Statistics Finland's road goods transport cost index, "
            "a different base from PostNord's diesel consumer price index, so the two "
            "percentages are not directly comparable; not a per-agent value"),
    },
}

# the three published series as the page text yields them, parsed by the builder.
# year headings such as '2025' sit inside the text and carry no value of their own
MATKAHUOLTO_FUEL_SERIES_AS_PRINTED = {
    "Paketti- ja kuljetuspalvelut kotimaassa ja ulkomaille": (
        "2026 3.9.2026 20,86 % 27.8.2026 20,65 % 20.8.2026 20,28 % 13.8.2026 20,31 % "
        "6.8.2026 20,05 % 30.7.2026 19,44 % 16.7.2026 18,21% 9.7.2026 18,08 % 2.7.2026 "
        "18,31 % 25.6.2026 18,80 % 18.6.2026 18,35 % 11.6.2026 18,52 % 4.6.2026 18,74 % "
        "28.5.2026 19,07 % 21.5.2026 19,69 % 14.5.2026 19,44 % 7.5.2026 19,36 % 30.4.2026 "
        "18,89 % 23.4.2026 19,03 % 16.4.2026 19,25 % 9.4.2026 18,76 % 2.4.2026 18,96 % "
        "26.3.2026 17,85 % 19.3.2026 16,44 % 1.3.2026 14,53 % 1.2.2026 13,72 % 1.1.2026 "
        "14,27 % 2025 1.12.2025 13,64 % 1.11.2025 13,77 % 1.10.2025 13,65% 1.9.2025 "
        "13,99 % 1.8.2025 13,04% 1.7.2025 12,70 % 1.6.2025 12,95 % 1.5.2025 13,59 % "
        "1.4.2025 14,25 % 1.3.2025 14,31 % 1.2.2025 13,99 % 1.1.2025 13,68 %"),
    "Lavarahti": (
        "2026 7.9.2026 32,08 % 31.8.2026 32,25 % 24.8.2026 31,97 % 3.8.2026 31,41 % "
        "27.7.2026 30,21% 20.7.2026 27,51% 13.7.2026 25,46 % 6.7.2026 24,67% 29.6.2026 "
        "24,32 % 22.6.2026 25,72 % 15.6.2026 26,67 % 8.6.2026 26,15 % 1.6.2026 27,91 % "
        "25.5.2026 28,35 % 18.5.2026 29,74 % 11.5.2026 29,63 % 4.5.2026 29,52 % 27.4.2026 "
        "27,50 % 22.4.2026 26,54 % 20.4.2026 28,96 % 13.4.2026 30,97 % 10.4.2026 33,14 % "
        "7.4.2026 31,74 % 27.3.2026 31,89 % 20.3.2026 29,49 % 16.3.2026 27,52 % 9.3.2026 "
        "23,48 % 2.3.2026 20,98 % 16.2.2026 20,05 % 9.2.2026 20,21 % 26.1.2026 19,67 % "
        "19.1.2026 19,17 % 12.1.2026 18,97 % 2025 22.12.2025 17,35 % 15.12.2025 17,88 % "
        "8.12.2025 18,13 % 1.12.2025 19,07 % 24.11.2025 18,91 % 10.11.2025 18,28 % "
        "3.11.2025 18,07 % 27.10.2025 16,98 % 20.10.2025 17,33 % 13.10.2025 17,31 % "
        "6.10.2025 17,99 % 29.9.2025 17,77 % 22.9.2025 17,75 % 15.9.2025 17,85 % 8.9.2025 "
        "17,65 % 1.9.2025 17,52 % 25.8.2025 16,97 % 18.8.2025 17,40 % 11.8.2025 17,60 % "
        "4.8.2025 17,67 % 21.7.2025 18,20% 30.6.2025 17,60 % 23.6.2025 17,09 % 16.6.2025 "
        "16,62 % 9.6.2025 16,47 % 2.6.2025 16,37 % 26.5.2025 16,67 % 19.5.2025 16,25 % "
        "12.5.2025 16,13 % 5.5.2025 15,69 % 28.4.2025 15,58% 21.4.2025 15,45 % 14.4.2025 "
        "16,36 % 7.4.2025 17,66 % 31.3.2025 17,36 % 24.3.2025 17,22 % 10.3.2025 18,16 % "
        "3.3.2025 18,63 % 24.2.2025 18,63 % 17.2.2025 18,53 % 10.2.2025 18,43 % 3.2.2025 "
        "18,55 % 27.1.2025 19,29 % 20.1.2025 18,99 % 13.1.2025 18,26% 7.1.2025 17,96 %"),
    "Rahti": (
        "3.9.2026 30,77 % 27.8.2026 30,52 % 20.8.2026 29,59 % 13.8.2026 29,66 % 6.8.2026 "
        "29,08 % 30.7.2026 28,01 % 16.7.2026 26,34 % 9.7.2026 26,02 % 2.7.2026 26,53 % "
        "25.6.2026 27,44 % 18.6.2026 27,23 % 11.6.2026 27,67 % 4.6.2026 28,26 % 28.5.2026 "
        "28,41 % 21.5.2026 28,87 % 14.5.2026 28,84 % 7.5.2026 28,36 % 30.4.2026 28,86 % "
        "23.4.2026 29,23 % 16.4.2026 29,48 % 9.4.2026 28,69 % 2.4.2026 29,49 % 26.3.2026 "
        "27,85 % 19.3.2026 22,78 % 1.3.2026 18,50 % 1.2.2026 16,19 % 1.1.2026 17,81 %"),
}

MATKAHUOLTO_FUEL_SURCHARGE = {
    "artifact": "fuel_surcharge/matkahuolto_fuel_surcharge.json",
    "record": {
        "source": "Matkahuolto, polttoainelisä (fuel surcharge)",
        "source_url": "https://www.matkahuolto.fi/yrityksille/polttoainelisa",
        "page_or_section": ("the 'Paketti- ja kuljetuspalvelut kotimaassa ja ulkomaille', "
                            "'Lavarahti' and 'Rahti' lists"),
        "role": "source evidence of a fuel-surcharge mechanism",
        "currency": None,
        "unit": "percent; the base the percentage is applied to is not stated on this page",
        "access_date_basis": "read on this date from the live page named in source_url",
        "reverification_status": "read from the live page",
        "mechanism_as_printed": [
            ("Polttoainelisä pohjautuu Tilastokeskuksen ylläpitämään kuorma-autoliikenteen "
             "kustannusindeksiin, jossa vertailulukuna on vuoden 2020 Keskiraskaiden ja "
             "raskaiden kuorma-autojen indeksin polttoaineiden kustannusindeksi, sekä "
             "erikseen määritettyyn päästövähennystoimenpiteiden osuuteen."),
            ("Lavarahdin polttoainelisä tarkistetaan viikoittain ja astuu voimaan "
             "maanantaisin. (Lavarahti price list page 3)"),
        ],
        "tariff_metadata": {
            "vat_status": field(NOT_STATED),
            "effective_date": field("per row, the date printed before each percentage",
                                    "3.9.2026 20,86 %"),
            "currency_as_published": field("not applicable: published as a percentage", "%"),
            "unit_basis": field("percent", NOT_STATED,
                                "the page does not state what the percentage is applied to"),
            "service_category": field(
                "three series: parcel and transport services, Lavarahti pallet freight, Rahti freight",
                "Paketti- ja kuljetuspalvelut kotimaassa ja ulkomaille; Lavarahti; Rahti"),
            "fuel_surcharge_treatment": field(
                "not applicable: this record is the surcharge itself",
                "Kulloinkin voimassa oleva polttoainelisä on nähtävillä Matkahuollon verkkosivuilla.",
                "Lavarahti price list page 3"),
        },
        "series_as_printed": MATKAHUOLTO_FUEL_SERIES_AS_PRINTED,
        "verification_note": (
            "source evidence of Matkahuolto's published fuel surcharge for three service "
            "groups, including Lavarahti, recorded as printed; tied to a Statistics "
            "Finland truck transport cost index with base year 2020 plus a separately "
            "defined emissions-reduction share, so it is not directly comparable to "
            "PostNord's diesel consumer price index percentage; not a per-agent value"),
    },
}


MANUAL_RECORDS = [
    POSTI_PARCEL,
    POSTNORD_PARCEL_ZONES,
    POSTI_EXPRESS_FREIGHT,
    MATKAHUOLTO_LAVARAHTI,
    UPPLY_CORRIDOR_RATES,
    POSTNORD_FUEL_SURCHARGE,
    POSTI_FUEL_SURCHARGE,
    PAKETTIPISTE_FUEL_SURCHARGE,
    MATKAHUOLTO_FUEL_SURCHARGE,
]

# written by src/dataset/route_distance.py, indexed by the builder when present
COORDINATES_ARTIFACT = "distance/location_coordinates.json"


def group_of(artifact_path):
    return artifact_path.split("/", 1)[0]
