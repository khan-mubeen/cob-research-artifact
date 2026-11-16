# Data Synthesis Methodology

**Project:** Chain-of-Bid, a supervisor-coordinated multi-agent bidding system for garment order fulfilment
**Scope of this document:** synthetic data generation, agent-profile design, calibration, validation methodology, provenance, reproducibility, and the language-model-assisted workflow used to produce and refine the data

---

## 1. Purpose and Scope

This document explains how the synthetic dataset behind Chain-of-Bid was designed, generated, calibrated, and validated. It covers the generated files and public sources, identifies the modelling assumptions, and explains the agent profiles, order generator, competitiveness checks, versioning, and reproducibility workflow.

Chain-of-Bid models a three-stage garment order fulfilment chain:

1. Material providers supply finished or otherwise ready-to-use fabric and leather to the manufacturing stage.
2. Manufacturers or tailoring agents convert the material into finished products.
3. Logistics providers deliver the completed products to customers.

Independent agents compete within each stage. If one agent were consistently cheaper, faster, and lower risk than every alternative, the bidding process would have little value. The profiles therefore contain deliberate trade-offs and are evaluated through static checks as well as order-level bid simulation.

The live system uses three sequential, stage-specific bidding windows:

1. The material bidding window opens and closes.
2. The manufacturing or tailoring bidding window opens and closes.
3. The logistics bidding window opens and closes.
4. The supervisor evaluates the complete fulfilment chain.

Late bids are not accepted after the relevant stage window closes.

This document focuses on the data used by that system. Detailed NATS message flow, live-agent behaviour, final-chain selection, reverse-auction logic, dynamic runtime state, and small-language-model integration are documented separately in `agent_logic_design.md`.

The document distinguishes among three kinds of claim: methods used to generate the current dataset, methods validated by the checks in Section 10, and system design specified in `agent_logic_design.md` but not reflected in the generated data or live implementation described here. Each passage identifies which of these states applies.

---

## 2. System and Dataset Overview

The system contains three agent populations:

- **15 material providers**, supplying one or more of five materials: cotton, polyester, linen, denim, leather
- **12 manufacturers or tailoring agents**, producing one or more of five products: shirt, hoodie, jacket, jeans, skirt
- **12 logistics providers**, serving one or more of seven Finnish cities: Espoo, Helsinki, Jyväskylä, Lahti, Oulu, Tampere, Turku

The product reference table defines the product–material combinations modelled by the current simulation. For example, the current schema allows a jacket to use leather, polyester, or linen. Combinations outside the current project schema are not generated.

Each synthetic order specifies: product, material, quantity, delivery city, customer budget, creation time, deadline, maximum acceptable defect or damage rate, minimum reputation requirement, priority, and generation scenario.

The generator combines ordinary orders with deliberate stress test scenarios. This allows the dataset to evaluate both everyday competition and difficult conditions such as limited budgets, large quantities, strict quality requirements, and short deadlines.

Five Python synthesis scripts generate the data:

1. `synthesize_product.py`
2. `synthesize_material_providers.py`
3. `synthesize_manufacturers.py`
4. `synthesize_logistics.py`
5. `synthesize_orders.py`

Python scripts write the final CSV values from public-source transformations, explicit formulas, calibrated parameters, or documented modelling assumptions. Language models assisted with drafting, review, and calibration reasoning, but did not directly produce the final CSV rows.

---

## 3. Generated Data Files

| File | Contents |
|---|---|
| `material_agents.csv` | Fifteen material-provider profiles, including material coverage, initial stock, deterministic restock parameters, cost per kilogram, completion time, defect probability, pricing markup, and reputation |
| `manufacturer_agents.csv` | Twelve manufacturer profiles, including product coverage, production capacity, initial workload, unit costs, defect probability, markup, and reputation |
| `logistics_agents.csv` | Twelve logistics-provider profiles, including service regions, delivery-cost structure, route-dependent delivery time, damage probability, markup, and reputation |
| `products.csv` | The reference table of product–material combinations modelled by the current system, including material and finished-product weights |
| `orders.csv` | Fifty thousand synthetic garment orders containing order requirements, constraints, priority, and scenario labels |
| `orders_metadata.json` | A provenance and audit record for `orders.csv`, containing the generation settings, distributions, budget factors, and hashes of the input CSV files |

The generated CSVs and metadata file are written to a versioned folder under `data/synthetic/`. The active dataset version is controlled through `DATA_VERSION = "v4"` in `config.py`. All synthesis, diagnostic, and visualization scripts use the same version setting.

---

## 4. Raw Data Sources and Modelling Inputs

The v5 material-price calibration combines finished-material reference prices with historical market series where credible time-series data are available. The finished-material reference establishes the appropriate price level, while the historical series capture how prices change over time. Where a historical series for the finished material is unavailable, a credible historical series for a relevant upstream raw material or broader product category may be used only as a temporal proxy. In such cases, the upstream series is used to represent historical price movements rather than being treated as the literal finished-material price. If no credible historical series is available, the project retains the sourced fixed finished-material price rather than creating artificial fluctuations.

USD is the common monetary unit for v5. Any non-USD price observation is converted to USD using the exchange rate corresponding to that observation date. Exchange rates are therefore date-dependent rather than represented by one fixed conversion value across the dataset. Fixed physical conversions, such as pounds to kilograms or yards to metres, remain constant.

The source type used for each material is documented explicitly, including whether the historical series represents the finished material itself or an upstream raw material/product-category proxy.

This calibration is planned for v5. The currently generated and validated v4 dataset remains unchanged until the v5 synthesis and validation pipeline is completed.

| Source | Link | Role in the project |
|---|---|---|
| FRED / IMF, Global price of Cotton (`PCOTTINDUSDM`) | https://fred.stlouisfed.org/series/PCOTTINDUSDM | Monthly historical cotton-price series in U.S. cents per pound; used as the primary temporal reference for cotton. Trading Economics was also considered, but FRED/IMF was selected because its series definition and units are clearly documented. Trading Economics displays cotton prices in US cents per pound labeled "USd," which can be misread as US dollars; FRED avoids this ambiguity. |
| SunSirs, Polyester DTY | https://www.sunsirs.com/uk/prodetail-1007.html | Historical polyester DTY price series; used as the temporal reference for polyester. The source-reported currency is converted to USD using the rate corresponding to each observation date. The source's own listing is titled "Daily prices." The live data table could not be independently opened for verification; the frequency claim rests on the source's own label only. |
| Tabb Textile Company, 2024 Product Catalog | https://www.tabbtextileinc.com/wp-content/uploads/2024/11/Tabb-Textile-Fabric-Catalog-Final-Version-1.pdf | Finished-fabric reference for polyester and denim. The cotton SKUs `JKCWI`, `JKCCI`, `JKCWD`, and `JKCCD` are listed as "Jersey Knit 100% Cotton"; the separate `JKPCWI`, `JKPCCI`, `JKPCWD`, and `JKPCCD` "Jersey Knit Poly/Cotton" SKUs are excluded. |
| iMaterial, January 2024 Fabric Pricelist | https://www.imaterial.co.za/Portals/81/Price-Lists/Fabric-pricelist-Jan-2024.pdf?ver=jwO2hgSpqBOo4OyOL2fDrA%3D%3D | Finished-fabric price-level reference for 100% linen and an independent cotton-price cross-check. |
| Alliance for European Flax-Linen and Hemp, Economic Observatory | https://allianceflaxlinenhemp.eu/en/flax-linen-hemp-economic-observatory | Historical European long-flax-fibre price observations used as an upstream temporal proxy for linen. The observations are published irregularly and are not treated as a continuous monthly series. |
| FRED / BLS, Finished Cotton Broadwoven Fabrics PPI (`WPU03420104`) | https://fred.stlouisfed.org/series/WPU03420104 | Monthly temporal proxy for denim. It is a broader finished-cotton-fabric producer-price index, not a denim-specific market price. |
| Leather Unlimited | https://leatherunltd.com/collections/garment-bag-leather/products/premium-soft-light-weight-garment-leather-hide-20-square-feet-2-3-oz | Finished garment-leather price reference used to establish the leather price level. |
| FRED / IMF, Global price of Hides (`PHIDEUSDM`) | https://fred.stlouisfed.org/series/PHIDEUSDM | Monthly upstream hide-price series used only as a temporal proxy for leather; it is not treated as a finished garment-leather price. |
| Townsend Leather, technical bulletin on leather thickness and weight | https://www.townsendleather.com/wp-content/uploads/2013/12/Bulletin-6_Thickness.pdf | Industry source for the leather ounce-per-square-foot weight conversion. |
| U.S. International Trade Commission, Harmonized Tariff Schedule, Chapter 41 | https://hts.usitc.gov/ | Product-classification context for further-prepared garment leather rather than raw or crust leather. |
| European Central Bank, reference exchange rates | https://data.ecb.europa.eu/data/datasets/EXR/ | Exchange-rate source used when a historical or catalog price is not already reported in USD. The applicable rate is matched to the date of the price observation rather than fixed for the whole dataset. |
| UCI Machine Learning Repository, Productivity Prediction of Garment Employees | https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees (DOI: 10.24432/C51S6D) | Manufacturer-data source used directly for team-level worker counts, SMV, actual productivity, and related fields from which capacity, quality-risk proxies, and reputation values are derived. |
| DataCo Smart Supply Chain for Big Data Analysis | https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis | Qualitative reference for order-quantity and delivery-pattern design. It is not directly ingested by the synthesis scripts and does not determine the final numerical distributions. |
| European Road Freight Rate Development Benchmark, Q2 2026, Upply / Ti / IRU | https://www.upply.com/en/ti-upply-iru-benchmarks-reports | European road-freight market reference used to contextualize the selected parametric logistics-pricing method. Contract rates are reported on p. 22, spot rates on p. 25, and the methodology on p. 36. These values are treated as benchmark evidence rather than as direct Finnish domestic tariffs. |
| Posti Express Freight | https://www.posti.fi/en/for-businesses/customer-support/service-prices-and-terms | Finnish domestic freight reference used to calibrate realistic ranges for the selected parametric logistics model. Its transport-unit, shipment-size, and delivery-zone structure provides freight-tier evidence for the fixed and weight-dependent cost components rather than being reproduced as a literal tariff formula. |
| Posti Parcel Services, digital contract prices | https://www.posti.fi/mzj3zpe8qb7p/63bn2u74jGXSHu1IK0KTOk/8ed1177829b3ef1d403e467de8588375/posti-digital-contract-prices-202501.pdf | Finnish parcel-price reference used as lower-bound calibration evidence for the fixed and weight-based components of the selected parametric logistics model. The published contract prices include fixed shipment charges and, for larger parcels, a EUR 0.30/kg weight component. Relevant reference values are Postal Parcel EUR 5.90/shipment + EUR 0.30/kg, Express Parcel EUR 8.90/shipment + EUR 0.30/kg, and Home Parcel EUR 10.90/shipment + EUR 0.30/kg. These parcel services have substantially lower shipment-weight limits than the project's heavier simulated shipments and are therefore treated as lower-bound calibration evidence rather than direct freight tariffs. |
| Matkahuolto Lavarahti | https://www.matkahuolto.fi/ | Finnish freight reference used to support calibration of the fixed and weight-dependent components of the selected parametric logistics model, particularly for heavier shipments that fall outside ordinary parcel-service weight limits. |
| PostNord Finland, domestic service price list | https://www.postnord.fi/siteassets/pdf/hinnastot/online_prices_2026-02-01_en.pdf | Finnish parcel-price reference used as additional lower-bound calibration evidence for the selected parametric logistics model. PostNord prices vary by shipment-weight bracket and delivery zone rather than using one constant EUR/kg rate. For the 15.01-20 kg bracket, the published prices are EUR 12.00 in Zone A, EUR 14.00 in Zone B, and EUR 19.00 in Zone C. These values are used as calibration evidence for realistic carrier-price ranges rather than interpreted as a direct constant weight coefficient or distance-based tariff. |
| PostNord Finland, fuel surcharge | https://www.postnord.fi/en/sending/fuel-and-sulphur-surcharge | Finland-specific reference for date-dependent fuel surcharges. PostNord states that its fuel surcharge is updated over time and linked to fuel-price development. |
| OpenStreetMap / OSRM | https://www.openstreetmap.org/ ; https://project-osrm.org/ | Proposed reproducible road-distance source for v5. Route distances can be generated once from a pinned map snapshot and routing configuration and stored locally rather than queried during bidding. |

The earlier v4 calibration used a raw cotton commodity anchor and cotton-relative multipliers for the remaining materials. V5 removes that generative ratio approach. Each material is handled independently using its own price source, finished-material reference, or documented temporal proxy.

Historical price data are used at the frequency or observation dates provided by the source. Monthly sources remain monthly and daily sources remain daily. For irregular historical observations, such as the linen proxy, the most recent published observation is carried forward until the next available observation. Missing intermediate price movements are not fabricated. A generated order uses the price applicable to its `order_created` date.

Cotton-relative ratios may still be reported descriptively if useful, but they are not used to generate v5 material prices.

Formal source references, access dates, catalog versions, exact products used, and conversion formulas should be retained in the paper bibliography or supplementary material.

---
## 5. Material-Provider Synthesis

### 5.1 Material-price calibration method

Material-provider costs represent finished or ready-to-use materials supplied to the garment manufacturer. V5 keeps the finished-material price level separate from the historical series used to model temporal movement.

All v5 monetary values are expressed in USD. Fixed physical-unit conversions remain constant, while currency conversion is applied using the exchange rate corresponding to the relevant source observation date.

For a historical source already reported in USD per pound:

```text
USD/kg = USD/lb × 2.20462
```

For a source reported in U.S. cents per pound:

```text
USD/kg = (US_cents_per_lb / 100) × 2.20462
```

For fabric priced per linear metre with a stated GSM and width:

```text
kg_per_metre = width_m × GSM / 1000
USD/kg = (price_per_metre_source_currency × source_currency_to_USD(date)) / kg_per_metre
```

For fabric priced per linear yard with a stated oz/yd² and width:

```text
weight_per_linear_yard_oz = oz_per_yd² × width_in / 36
weight_per_linear_yard_kg = weight_per_linear_yard_oz × 28.3495 / 1000
USD/kg = (price_per_yard_source_currency × source_currency_to_USD(date)) / weight_per_linear_yard_kg
```

For leather sold by surface area and ounce weight:

```text
total_oz = oz_per_ft² × area_ft²
total_kg = total_oz × 28.3495 / 1000
USD/kg = (bundle_price_source_currency × source_currency_to_USD(date)) / total_kg
```

Townsend Leather documents the same ounce-per-square-foot method in its technical bulletin: leather ounce weight multiplied by hide area gives total ounces, which can then be converted to kilograms.

When a historical series is used only as a temporal proxy rather than as the finished-material price itself, v5 scales a finished-material reference price by the relative movement of that series:

```text
dynamic_price_usd_per_kg(t)
    = anchor_price_usd_per_kg
    × reference_series_usd(t) / reference_series_usd(anchor_period)
```

This approach assumes that relative changes in the selected raw-material or broader-market proxy are transmitted proportionally to the finished-material price. Because finished-material prices also reflect processing and other production costs, the resulting temporal variation should be interpreted as a modelling approximation rather than a precise reconstruction of historical finished-fabric prices.

If the temporal series is reported in another currency, its observations are first normalized to USD using the exchange rate for each observation date before the relative movement is calculated.

If no credible historical series is available for a material, the sourced price remains fixed. The generator does not introduce synthetic temporal noise merely to make all materials dynamic.

### 5.2 Material-price sources and temporal treatment

The v5 materials are treated independently. Cotton-relative multipliers are not used for generation.

| Material | Price reference | Historical source | V5 treatment |
|---|---|---|---|
| Cotton | Verified 100% cotton finished-fabric reference | FRED / IMF `PCOTTINDUSDM` | Monthly dynamic price movement. The historical source is in U.S. cents/lb. Any finished-fabric anchor used for v5 must be explicitly verified as 100% cotton. |
| Polyester | Tabb Textile 100% texturized polyester twill | SunSirs Polyester DTY | Dynamic historical movement using the SunSirs series; non-USD observations are converted using date-matched FX. |
| Linen | iMaterial 100% linen products | Alliance for European Flax-Linen and Hemp long-flax-fibre price observations | Dynamic temporal movement using irregular historical long-flax-fibre observations as an upstream proxy. The latest available observation is carried forward until the next published observation. |
| Denim | Tabb Textile 100% cotton denim products | FRED / BLS `WPU03420104` | Monthly dynamic movement using the broader finished-cotton-fabric PPI as an explicitly documented proxy. |
| Leather | Leather Unlimited garment-leather product | FRED / IMF `PHIDEUSDM` | Monthly dynamic movement using the upstream hide series as an explicitly documented proxy while retaining the finished garment-leather reference for the price level. |

**Cotton.** The primary temporal source for v5 is FRED / IMF series `PCOTTINDUSDM`, which provides monthly global cotton prices in U.S. cents per pound. Because this is an upstream cotton commodity series rather than a finished-fabric price, it is used to provide the historical movement. The finished-fabric reference used to establish the material-provider price level must be explicitly verified as 100% cotton; blended cotton-polyester products are excluded.

**Polyester.** SunSirs Polyester DTY is used as the temporal reference. DTY (Drawn Textured Yarn) was selected over the alternative SunSirs FDY series because it provides a closer technical match to the "100% Texturized Polyester" finished-fabric reference used in this project, rather than averaging two different polyester product series. The existing Tabb Textile 100% texturized polyester twill remains the finished-fabric reference for the provider price level. SunSirs labels these as daily spot-price observations; this has not been independently verified beyond the source's own listing.

**Linen.** The iMaterial January 2024 catalog provides the finished-material price-level reference using 100% linen products with composition, GSM, width, and price per metre. Historical European long-flax-fibre price observations published by the Alliance for European Flax-Linen and Hemp are used as an upstream temporal proxy. Because these observations are published irregularly rather than as a continuous time series, the most recent available observation is carried forward until the next published observation. No intermediate price movements are fabricated.

**Denim.** Tabb Textile provides the finished-denim reference. FRED / BLS series `WPU03420104` is used only as a monthly temporal proxy because it covers finished cotton broadwoven fabrics rather than denim specifically. The methodology therefore does not describe this index as a denim market price.

**Leather.** Leather Unlimited provides the finished garment-leather price reference. FRED / IMF series `PHIDEUSDM` provides monthly hide-price movement and is used only as an upstream temporal proxy. It does not replace the finished-leather reference or imply that raw hides and garment leather have the same absolute price.

For documentation, material-to-cotton ratios may still be calculated after generation if a descriptive comparison is useful. They are not part of the v5 pricing formula.

### 5.3 Supplier coverage

The current fifteen material agents provide the following coverage:

| Material | Number of suppliers |
|---|---|
| Cotton | 5 |
| Polyester | 6 |
| Linen | 5 |
| Denim | 5 |
| Leather | 6 |

Every material is supported by multiple agents so that orders normally have more than one possible supplier.

Leather remains the most constrained material by design. Earlier versions contained fewer leather suppliers. Additional leather coverage was introduced during calibration because a disproportionate number of material-stage failures resulted from insufficient stock and supplier availability rather than meaningful price, quality, or timing competition.

The calibration was applied mainly through material-specific stock and restock parameters. Agent-wide variables were not used to solve material-specific problems unless their effects on every other material supplied by the same agent were reviewed.

### 5.4 Intended trade-offs

| Supplier type | Typical strength | Typical weakness |
|---|---|---|
| Budget supplier | Lower material cost | Higher defect probability or slower completion |
| Premium supplier | Lower quality risk | Higher price |
| Bulk supplier | Larger available stock | Not necessarily fastest or cheapest |
| Fast supplier | Short completion time | Higher markup or narrower coverage |

The goal is not to make every agent equally successful. The goal is to ensure that agents have different competitive niches and that no single profile is obviously superior on every relevant dimension.

### 5.5 Material-provider field reference

Each material provider occupies one row in `material_agents.csv`. Material-specific fields repeat for cotton, polyester, linen, denim, and leather. A material-specific field is set to `null` when the agent does not supply that material. This differs from `0`, which can represent a supported material with no immediately available stock.

| Field | Meaning and current use |
|---|---|
| `agent_id` | Unique identifier, M01–M15 |
| `stage` | Always `material` |
| `location` | Agent location; currently descriptive and reserved for future route-aware interactions |
| `specialty` | Human-readable description of the intended competitive niche |
| `material_types` | Pipe-separated list of supported materials; used for eligibility |
| `default_completion_days` | Baseline completion time when the current stock path is sufficient |
| `lead_time_variance` | Reserved for future stochastic timing; not used by the current deterministic formula |
| `min_profit_margin` | Minimum pricing floor |
| `initial_markup` | Agent-specific pricing adjustment |
| `defect_probability` | Quality-risk value used for eligibility and pricing |
| `reputation_score` | Initial neutral reputation value; fixed at 0.5 for material agents |
| `fx_usd_to_eur` | Legacy v4 field. It is not used by v5. V5 uses USD as the common currency and applies date-matched exchange rates only when a source observation is reported in another currency. |
| `{material}_available_kg` | Initial available stock at the start of the current simulation model |
| `{material}_base_cost_eur_per_kg` | Legacy v4 material-cost field. V5 replaces this with USD-denominated material-pricing inputs, with date-dependent prices used where a credible historical source or proxy is available. |
| `{material}_restock_amount_kg` | Additional amount available through the deterministic restock path |
| `{material}_restock_time_days` | Additional completion time required when the restock path is used |
The exact v5 schema change is implemented before v5 generation. The legacy EUR and fixed-FX fields should not be described as v5 operational inputs until that implementation is complete.

The current v4 dataset represents the initial state of each material agent only. Stock reservation, stock depletion across accepted orders, time-based restock arrival, and cancellation rollback are part of the runtime-state design described in `agent_logic_design.md`; they are not represented in the current offline dataset, which reflects only the initial state of each agent.

---

## 6. Manufacturer Synthesis

Manufacturer profiles are grounded in the UCI Garment Worker Productivity dataset. The source contains daily observations from twelve teams across sewing and finishing departments.

### 6.1 Capacity formula

```
daily_capacity_units = workers × (480 / SMV) × actual_productivity
```

Here, 480 is the number of minutes in an eight-hour shift, SMV is the standard minute value required per unit, and `actual_productivity` represents observed team productivity.

### 6.2 Main design decisions

- Sewing and finishing are represented as one manufacturer agent for each team.
- Effective capacity uses the slower department as the throughput bottleneck.
- Defect probability is modelled from the difference between actual and targeted productivity.
- Reputation is derived from actual productivity and constrained to the range from 0 to 1.
- Initial workload is assigned as a fraction of capacity across several workload tiers. Raw work-in-progress values were not used directly because they describe a broader production pipeline rather than a single day of immediately committed work.
- Agent-specific cost ratios create price differences between manufacturers producing the same product.
- Product specializations are synthetic because the source dataset does not contain garment product categories. Generation validates that every product has at least three potential manufacturers.

The exact transformation coefficients, clamping rules, team aggregation logic, and cost-ratio values are implemented in `synthesize_manufacturers.py`. These implementation details should be reproduced in a paper appendix or supplementary table if exact numerical replication is required.

### 6.3 Manufacturer field reference

| Field | Meaning and current use |
|---|---|
| `agent_id` | Unique identifier, T01–T12 |
| `source_team_id` | Original source-team identifier |
| `stage` | Always `tailoring` |
| `specialty` | Human-readable competitive niche |
| `scale` | Descriptive small, medium, or large category |
| `product_types` | Pipe-separated supported products; used for eligibility |
| `daily_capacity_units` | Daily production-capacity estimate |
| `current_workload_units` | Initial committed workload used for backlog and workload-markup calculations |
| `unit_cost_shirt` | Per-unit shirt cost, or `null` when unsupported |
| `unit_cost_hoodie` | Per-unit hoodie cost, or `null` when unsupported |
| `unit_cost_jacket` | Per-unit jacket cost, or `null` when unsupported |
| `unit_cost_jeans` | Per-unit jeans cost, or `null` when unsupported |
| `unit_cost_skirt` | Per-unit skirt cost, or `null` when unsupported |
| `min_profit_margin` | Minimum pricing floor |
| `initial_markup` | Agent-specific pricing adjustment |
| `defect_probability` | Productivity-derived quality-risk proxy |
| `reputation_score` | Productivity-derived reputation proxy |
| `location` | Manufacturer city. It is descriptive in the current v4 manufacturer bidding formula. Under the selected v5 logistics method, it becomes the shipment origin used together with `order.deliver_to` to determine the delivery route. |

`current_workload_units` in the current dataset is a single initial value per agent, not a value that changes as orders are processed. `agent_logic_design.md` describes a runtime design in which workload increases after an accepted order and capacity is released on completion or cancellation; this runtime behaviour is not part of the v4 data-generation pipeline described here.

For v5, manufacturer monetary values and offers are expressed in USD so that all stages and order budgets use the same currency.

---

## 7. Logistics Synthesis

The current v4 logistics profiles use synthetic fixed and weight-based delivery-cost parameters. For v5, the logistics-pricing method is being revised using Finnish freight-pricing references, the Q2 2026 Upply / Ti / IRU road-freight benchmark, reproducible road-distance data, and a shared date-dependent fuel surcharge.

Two candidate methods were reviewed. The freight-calibrated parametric model (Option A) was selected for v5 logistics pricing. Option B is retained below as a considered alternative for completeness.

### 7.1 Selected method: freight-calibrated parametric model

The first option follows the parametric structure proposed during methodology review:

```text
base_transport_cost_eur_i(t)
    = [F_i
        + alpha_i × weight_kg
        + beta_i × (weight_kg / 1000) × distance_km]
        × [1 + fuel_surcharge(t)]
```

The terms are:

- `F_i`: fixed shipment or dispatch component for logistics agent `i`
- `alpha_i`: weight-based handling coefficient in EUR/kg
- `beta_i`: line-haul transport-work coefficient in EUR/tonne-km
- `weight_kg`: finished shipment weight
- `distance_km`: road distance from the candidate manufacturer's `location` to the order's `deliver_to` city
- `fuel_surcharge(t)`: the shared fuel-surcharge percentage at the order date, the same series for every logistics agent (Section 7.3)

The resulting EUR value is converted to the common v5 currency using the date-matched EUR-to-USD exchange rate. The conversion is shown here for dimensional completeness; in the full bidding sequence it is applied last, as specified in Section 7.5:

```text
transport_cost_usd_i(t) = base_transport_cost_eur_i(t) × EUR_to_USD(t)
```

This formula is dimensionally consistent and separates fixed shipment cost, weight-dependent handling, transport work, and fuel-price effects. It should nevertheless be described as a **freight-calibrated parametric approximation**, not as the literal tariff formula used by Finnish carriers.

Beta. `beta_i` is set to 0.07163 EUR/tonne-km. The corresponding source reference and calibration evidence will be recorded with the remaining logistics-calibration sources before v5 generation.

**F_i and alpha_i**. The fixed shipment component `F_i` and weight-based coefficient `alpha_i` are calibrated from published Finnish carrier pricing rather than assigned arbitrary values. References from Posti, PostNord, Matkahuolto, and other applicable carrier services are used to establish realistic calibration ranges. Because published services use different pricing structures, including fixed shipment charges, per-kilogram charges, weight brackets, transport units, and delivery zones, their values are treated as calibration evidence rather than copied directly as one universal tariff. The synthetic logistics agents are assigned values within the documented calibration range while retaining their existing differences in service coverage, speed, damage risk, and reputation.

**Parcel weight limits**. A parcel-calibrated agent is only eligible for shipments its own parcel service can carry, so its `F_i` and `alpha_i` are never used above the weights that service documents. The limits come from the pinned Posti parcel price list. Postal Parcel carries up to 25 kg, and the Oversized additional service does not raise that limit. Home Parcel and Express Parcel carry up to 25 kg, or up to 35 kg with the Oversized additional service. Because the €3.50 Oversized fee is pinned in the same price list, home and express parcel agents remain eligible up to 35 kg and the fee is added to the price of any shipment above 25 kg rather than absorbed into `F_i` or `alpha_i`. The fee is an additional-service charge, not freight, so the fuel surcharge is not applied to it. In the current v5 profile, L03 and L06 use the Postal Parcel reading and stop at 25 kg, L07 uses the Express Parcel reading, and L10 and L12 use the Home Parcel reading, both reaching 35 kg with the fee. Freight-calibrated agents have no weight cap.

### 7.2 Considered alternative: Finnish freight tariff-matrix model

The second option follows Finnish freight tariffs more directly.

Instead of representing freight cost using continuous fixed, per-kilogram, and tonne-kilometre coefficients, the base transport price is selected from a freight tariff according to shipment characteristics such as:

```text
base_transport_cost_eur
    = freight_tariff(
        transport_unit,
        weight_or_capacity_bracket,
        delivery_zone
      )
```

Posti Express Freight and Matkahuolto Lavarahti provide suitable Finnish references because they cover palletised or heavier freight rather than ordinary parcel shipments.

Under this option, shipment weight determines the applicable transport-unit or weight category, while route distance or destination determines the delivery zone. The shared fuel surcharge described in Section 7.3 is then applied to the base freight component:

```text
transport_cost_eur(t) = base_transport_cost_eur × [1 + fuel_surcharge(t)]
```

The result is converted to USD using the exchange rate applicable to the order date, applied at the point specified in Section 7.5.

This option more closely resembles published Finnish freight-pricing structures and avoids introducing independently calibrated `alpha` and `beta` coefficients. Its disadvantage is that the pricing model becomes discrete and source-specific rather than a compact continuous equation.

This alternative was reviewed but was not selected for the v5 implementation. The parametric model in Section 7.1 is used instead.

### 7.3 Shared route and temporal treatment

The selected method uses the following shipment route:

```text
origin      = candidate_manufacturer.location
destination = order.deliver_to
```

Road distances are generated from a pinned routing dataset rather than retrieved live during bidding. The preferred reproducible approach is to use OpenStreetMap road data with a fixed OSRM version, routing profile, and map-data snapshot. The resulting manufacturer-city-to-destination-city distance matrix is stored with the dataset-generation inputs.

Fuel surcharge is treated separately from ordinary agent markup. Public Finnish carrier sources show that fuel surcharges vary over time, but each carrier publishes its percentage against its own base. Matkahuolto ties its surcharge to a Statistics Finland truck-transport cost index, while PostNord uses a diesel price index, so the absolute percentage levels of different carriers are not comparable.

All v5 logistics agents therefore use one shared series, the Matkahuolto Lavarahti fuel surcharge, resolved at each order's `order_created` date. It is a common temporal fuel-cost signal: it moves every agent's price over time in the same way, and agent differentiation does not come from fuel-series levels. Lavarahti was chosen because it covers the v5 order window most densely among the pinned series, with 22 distinct weekly percentages and no window day after its last observation. The fuel-surcharge mechanism stays explicit rather than being absorbed into `initial_markup`.

For v5, logistics pricing is first calculated in EUR because the external freight references are EUR-denominated. The resulting price is then converted to USD using the EUR-to-USD exchange rate applicable to the order date.

### 7.4 Agent differentiation

The external freight references provide calibration anchors rather than twelve independently observed carrier tariffs.

The synthetic logistics agents continue to differ through:

- service-region coverage
- same-city and intercity completion time
- damage probability
- reputation
- the calibration reading assigned from the agent's specialty, which sets `F` and `alpha` and, for parcel agents, the maximum shipment weight

Agent-specific differences in `F` and `alpha` are introduced only where their calibration rule is documented. `beta` and the fuel-surcharge series are the same for every agent. Unsupported random variation is not added merely to make agents different. Speed, coverage, damage risk, reputation, and parcel weight limits provide the remaining competitive differences. `min_profit_margin` and `initial_markup` stay on the profile, but they do not enter the v5 logistics price (Section 7.5). Every modelled destination city is covered by multiple logistics providers.

### 7.5 Pricing-operation order

Fuel surcharge is conceptually separated from ordinary agent bidding adjustments.

The intended sequence is:

```text
1. determine the externally calibrated base freight cost
2. apply the shared fuel surcharge to the freight component
3. add the Oversized fee where a home or express parcel shipment is above 25 kg
4. apply the damage-risk and priority adjustments
5. convert the resulting EUR offer to USD using the order-date exchange rate
```

In v5, "no markup, no floor" refers only to `initial_markup` and `min_profit_margin`. `F`, `alpha`, and `beta` come from published market prices that already include the carrier's commercial margin, so `initial_markup` is not applied on top of them. `min_profit_margin` is not used as a floor either, because a floor needs a cost basis that no pinned source provides. Damage-risk and priority adjustments remain applied, in the same positions they take in the v4 logistics bid:

```text
logistics_price_eur = (freight_with_fuel + oversized_fee)
                      × (1 + damage_probability)
                      × (1 + priority_markup)
```

Leaving out margin and markup is a provisional decision. A cost-basis methodology that could support them is future work.

### 7.6 Current v4 schema and v5 transition

The current v4 logistics CSV still contains the existing synthetic pricing fields:

| Field | Meaning and current use |
|---|---|
| `agent_id` | Unique identifier, L01–L12 |
| `stage` | Always `logistics` |
| `location` | Logistics-provider home city; currently used for same-city or intercity timing rather than as the shipment origin |
| `specialty` | Human-readable competitive niche |
| `service_regions` | Pipe-separated destination cities; used for eligibility |
| `base_delivery_cost` | Legacy v4 fixed shipment-cost parameter |
| `cost_per_kg` | Legacy v4 weight-based shipment-cost parameter |
| `default_completion_same_city_days` | Delivery duration for same-city orders |
| `default_completion_intercity_days` | Delivery duration for intercity orders |
| `delay_probability` | Reserved descriptive or future stochastic parameter |
| `damage_probability` | Eligibility and pricing risk input |
| `min_profit_margin` | Minimum pricing floor |
| `initial_markup` | Agent-specific pricing adjustment |
| `reputation_score` | Initial neutral reputation, fixed at 0.5 |

The v5 logistics schema will represent the fixed shipment component (`F_i`), weight-based coefficient (`alpha_i`), and tonne-kilometre coefficient (`beta_i`) required by the selected parametric model, together with a date-resolved fuel-surcharge lookup. The exact field names are finalized during v5 implementation. The existing `base_delivery_cost` and `cost_per_kg` fields remain unchanged as v4 fields.

### 7.7 Removed static capacity fields

Earlier versions contained `vehicle_capacity_kg`, `current_load_kg`, and `available_capacity_kg`. These fields were removed from the exported static profile because transport availability is a changing operational state rather than a permanent agent characteristic.

`agent_logic_design.md` describes a separate runtime state for logistics agents, covering current assigned load, active deliveries, reserved shipment capacity, available capacity, and expected capacity-release time. The static CSV described here represents only the stable provider characteristics that do not change between orders.

Logistics capacity is not depleted across successive orders in the current offline diagnostic. Reservation and release of capacity are part of the runtime design in `agent_logic_design.md` and are not represented in the current dataset.

---

## 8. Product Reference and Order Synthesis

### 8.1 Product-reference table

`products.csv` defines the product–material combinations modelled by the current system, and contains the material and shipment-weight assumptions needed by later stages.

| Field | Meaning |
|---|---|
| `product_id` | Product category |
| `material_type` | Allowed material within the current simulation schema |
| `material_kg_per_unit` | Material required per finished unit |
| `finished_weight_kg_per_unit` | Finished shipment weight per unit |

The table prevents orders from using combinations outside the current project schema.

### 8.2 Order scenarios

The current dataset contains 50,000 orders, distributed across one ordinary scenario and six stress-oriented scenarios.

| Scenario | Approximate share | Purpose |
|---|---|---|
| `random_realistic` | 71% | General mixed orders across products, quantities, cities, deadlines, and priorities |
| `tight_deadline` | 6% | Orders with shorter completion windows |
| `low_budget` | 6% | Orders with reduced willingness to pay |
| `large_quantity` | 5% | Orders that place greater pressure on material stock and manufacturing capacity |
| `high_quality` | 5% | Orders with stricter quality requirements |
| `rare_combo` | 4% | Lower-frequency product–material combinations within the current simulation distribution |
| `tight_low_budget` | 3% | Orders combining time pressure with limited budget |

The scenario names describe their role in the simulation. They should not be interpreted as measured real-world frequencies.

### 8.3 Creation dates and order density

Orders are distributed across a 180-day creation window. An earlier round-robin approach produced almost identical order counts for every day and appeared artificially uniform.

The current generator uses a seeded weekday-weighted sampler. Weekdays receive greater probability than weekends, and daily order counts vary naturally. A separate random stream prevents changes to date-generation logic from shifting the random values used for unrelated order fields.

### 8.4 Quantity distribution

Order quantities are generated from three weighted bands:

| Quantity group | Range | Share |
|---|---|---|
| Small | 100–300 units | 35% |
| Medium | 300–900 units | 45% |
| Large | 900–2,000 units | 20% |

These ranges are modelling assumptions informed by public supply-chain data and project requirements.

### 8.5 Budget calibration

Budgets are anchored to an estimate of expected chain-market price rather than raw input cost alone. For each stage, the estimate considers the typical effect of material or unit cost, defect or damage risk, initial markup, workload effects where relevant, and priority markup. For v5, all monetary values in this calculation are expressed in USD, and the material component uses the material price applicable to the order's `order_created` date. A simplified representation is:

```
expected stage price ≈ raw stage cost × (1 + expected risk factor) × (1 + expected markup factor)
```

The estimated stage values are combined into an expected chain price, and a scenario-dependent budget band is then applied.

| Budget condition | Multiplier band |
|---|---|
| Urgent ordinary order | 1.00–1.30 |
| Normal ordinary order | 0.95–1.25 |
| Flexible ordinary order | 0.90–1.15 |
| Low-budget scenario | 0.88–0.98 |
| Tight-low-budget scenario | 0.93–1.03 |

For ordinary scenarios, the priority-specific band is used. The low-budget and tight-low-budget scenarios use their own calibrated budget bands rather than multiplying an additional budget penalty onto the ordinary priority band.

The tight-low-budget band is slightly less restrictive than the low-budget band because combining a short deadline with an extremely low budget made the scenario almost universally infeasible during calibration. The purpose is to create a difficult but still competitive stress condition rather than a deliberately impossible order class.

### 8.6 Other order-design decisions

Defect tolerance is generated within ranges checked against the current agent population. Minimum reputation requirements are capped at 0.50. All material and logistics agents currently start at reputation 0.5, so a requirement above 0.5 would eliminate every agent in those two stages before price, time, or capacity were considered. Delivery cities are selected from the current logistics coverage catalogue, and product and material values are restricted to the product-reference table.

### 8.7 Order field reference

| Field | Meaning and current use |
|---|---|
| `order_id` | Unique order identifier |
| `order_created` | Timestamp at which the order enters the simulated system |
| `product_id` | Manufacturer product requirement |
| `quantity` | Drives material requirement, manufacturing time, and shipment weight |
| `material_type` | Material-provider requirement |
| `budget` | Maximum accepted total chain price; expressed in USD for v5 |
| `deadline` | Absolute completion deadline for the full chain |
| `deliver_to` | Delivery destination |
| `max_defect_rate` | Maximum accepted stage defect or damage probability |
| `min_reputation_score` | Minimum accepted stage reputation |
| `priority` | Urgent, normal, or flexible |
| `scenario_type` | Generation category used for analysis only |

`order_created` and `deadline` define the originally generated order lead time. In a live system, remaining time should be evaluated against the current processing time rather than repeatedly treating the full original interval as still available.

### 8.8 orders_metadata.json

`orders_metadata.json` is the provenance record for `orders.csv`, generated alongside the order CSV by `synthesize_orders.py`. It is not used by bidding agents during runtime.

The current metadata records: random seed, base date, order-creation window, number of generated orders, scenario distribution, priority distribution, budget-calculation stage factors, and SHA-256 hashes of the four input CSV files (`material_agents.csv`, `manufacturer_agents.csv`, `logistics_agents.csv`, `products.csv`).

The metadata file has four main uses:

- **Reproduction.** Its generation settings provide what is needed to recreate the same order-generation conditions.
- For **input identification**, the hashes tie an order dataset to the exact agent and product files used to generate it.
- **Change detection** follows from the input hashes. If one CSV changes, its SHA-256 hash also changes, showing a later reviewer whether two order datasets were generated from different agent profiles.
- The scenario, priority, and budget-factor records provide a **methodological explanation** of the assumptions behind the generated order distribution.

The file serves as the dataset's generation receipt or audit trail. It does not replace software-version control; a full publication package should also associate the dataset with a Git commit, generator-code version, configuration version, Python version, and dependency versions.

---

## 9. Synthesis and Calibration Process

The current dataset was revised in several rounds rather than produced in one pass. Each major revision responded to a problem observed in the validation output.

Each round followed the same general procedure: run `make bid_check` to produce fresh diagnostics, review the specific eligibility, win-share, and infeasibility numbers it reported, consult a language model for one targeted, specific change addressing the observed problem, apply that change to the relevant synthesis script, then rerun `make bid_check` to confirm the intended effect and check that no unrelated metric regressed. The four rounds in 9.1 through 9.4 below are instances of this same loop, not four separate methods.

### 9.1 Establishing competition

Some early profiles were simultaneously cheaper, faster, lower risk, and more broadly available than their direct competitors. Static checks were added to detect this kind of dominance. The profiles were then revised to pair lower cost with slower completion, lower defect probability with higher markup, larger stock with narrower coverage, and faster service with higher price. The aim was meaningful competitive choice, not equal win share.

### 9.2 Scaling from 500 to 50,000 orders

The first working validation used approximately 500 orders. Scaling to 50,000 exposed behaviours that were difficult to observe at the smaller size.

**Budget-anchor correction.** The original budget model relied too heavily on raw input cost. Agents price above raw cost because bids also contain factors such as defect or damage risk, initial markup, workload markup, and priority adjustment. The earlier budget anchor therefore underestimated realistic chain prices and made urgent and constrained orders systematically more difficult than intended. The budget anchor was revised around expected market-facing bid prices rather than raw costs alone.

**Leather-scarcity calibration.** Leather generated a disproportionate share of material-stage failures. The original leather population contained fewer suppliers and insufficient large-order stock. Additional coverage and material-specific stock adjustments were introduced. Agent-wide fields were not changed merely to solve leather-specific problems, because one agent may supply several materials, and an agent-wide change can unintentionally alter competition in unrelated material categories.

### 9.3 Manufacturer competition

At several points, some manufacturers won most of the orders for which they were eligible, while other frequently eligible agents rarely won. Cost ratios and markups were adjusted incrementally, and each change was followed by regeneration and full-scale diagnostic review. The purpose was to preserve trade-offs across lowest-cost, fastest, and balanced selection. Improving one criterion could reduce competitiveness under another, so the final profiles represent a compromise rather than an optimization toward one metric.

### 9.4 Order-density realism

The initial round-robin date assignment generated an unrealistically regular order book. The current seeded weekday-weighted distribution introduced natural daily variation while preserving reproducibility.

---

## 10. Validation Methodology

Validation combines static profile diagnostics with rule-based bid-competition diagnostics because the two checks examine different properties. The numerical results in Sections 10.3 and 10.4 refer to v4 and must be regenerated after v5 is created.

### 10.1 Static profile and dominance diagnostics

The static diagnostics compare agents within relevant competition groups, such as material agents supplying the same material and manufacturers producing the same product. Depending on the stage and diagnostic, the evaluated dimensions include cost, defect or damage probability, completion time, stock, restock behaviour, and service coverage. A profile is flagged when another profile is at least as strong on all evaluated dimensions and strictly stronger on at least one.

This is a coarse profile-level screen, not a mathematical proof that one agent can never win under any order-dependent bidding condition. The static check does not fully reproduce every effect of workload markup, priority, eligibility, changing order quantity, scenario constraints, or runtime state. The order-level bid simulation provides the stronger empirical competition check.

### 10.2 Rule-based bid-competition diagnostic

`make bid_check` converts the current deterministic bidding logic into an offline Python simulation. It runs without live agents, NATS, Docker, or an SLM, using the generated agent profiles, products, orders, and an offline implementation intended to mirror the project's deterministic eligibility and pricing logic.

The diagnostic checks whether the generated data produces multiple eligible bidders, meaningful price competition, both feasible and infeasible orders, different winners under different selection criteria, and a reasonable market-share distribution.

The diagnostic reports eligible-bidder counts, orders with no bidder at each stage, stage-level eligibility rates, reasons for material-stage rejection, candidate-chain feasibility, budget and deadline failures, top-offer price gaps, agent win shares, scenario- and material-level feasibility, and differences among lowest-cost, fastest, and balanced selections.

This diagnostic is a quality-assurance and competitiveness tool, not a replacement for live NATS integration testing.

### 10.3 Current internal validation snapshot

The latest internal run used 50,000 orders and reported:

| Metric | Current internal result |
|---|---|
| Orders with no feasible candidate chain | 4,629 / 50,000, or 9.3% |
| Orders with at least one feasible candidate chain | 45,371 / 50,000, or 90.7% |
| Orders with zero eligible material bidders | 1,564, or 3.1% |
| Orders with zero eligible manufacturer bidders | 0 |
| Orders with zero eligible logistics bidders | 0 |
| Orders with at least two eligible material bidders | 89.4% |
| Orders with at least two eligible manufacturer bidders | 99.9% |
| Orders with at least two eligible logistics bidders | 99.7% |
| Mean reported feasible candidate chains per order | 67.7 |
| Feasible orders where selection criteria differ | 98.4% |

The current infeasibility breakdown:

| Case | Definition | Current result |
|---|---|---|
| Case 1 | All three stages have zero eligible bidders | 0 |
| Case 2 | At least one, but not all, stages has zero eligible bidders | 1,564 |
| Case 3 | Every stage has at least one bidder, but no evaluated chain passes the final budget and deadline checks | 3,065 |

Case 3 breakdown:

| Reason | Orders |
|---|---|
| Over budget only | 2,570 |
| Over deadline only | 423 |
| Mixed or both | 72 |

Current selected scenario results:

| Scenario | Feasible orders |
|---|---|
| `random_realistic` | 95.2% |
| `high_quality` | 86.9% |
| `large_quantity` | 89.4% |
| `tight_deadline` | 83.9% |
| `rare_combo` | 93.8% |
| `low_budget` | 61.2% |
| `tight_low_budget` | 61.6% |

Current material-level feasibility:

| Material | Feasibility |
|---|---|
| Cotton | 92.5% |
| Polyester | 94.9% |
| Linen | 93.2% |
| Denim | 91.9% |
| Leather | 64.3% |

Leather remains the most constrained material.

### 10.4 Current competition indicators

The highest overall agent win shares reported by the current diagnostic:

| Criterion | Material maximum | Manufacturer maximum | Logistics maximum |
|---|---|---|---|
| Lowest total cost | 24.8% | 21.0% | 23.5% |
| Fastest completion | 47.8% | 36.8% | 13.2% |
| Balanced score | 25.2% | 28.5% | 14.5% |

The selection criteria produced different chains for 98.4% of orders with at least one feasible candidate chain, indicating that cost, speed, and balanced scoring usually identify different alternatives rather than repeatedly selecting one identical chain.

### 10.5 Interpretation and publication caution

As an internal validation baseline, the current results show multiple eligible bidders for most orders and distinct competitive niches across the agent population. Cost, speed, and balanced selection often disagree, while the order set still contains both feasible cases and deliberately difficult ones.

The candidate-chain construction in the diagnostic uses a shortlist of the cheapest and fastest bids per stage rather than the complete eligible-bidder set (Section 11), which can in principle omit a moderately priced, moderately fast bidder that forms the only feasible chain for a given order. The figures in Sections 10.3 and 10.4 therefore reflect this specific shortlist-based implementation and should be read as characterising the dataset's competitive structure under that implementation, rather than as figures independent of it.

Regardless of that implementation detail, the deterministic bidding logic was converted into an offline validation script and used repeatedly to evaluate data quality, feasibility, competition, and agent concentration.

For the v4 dataset, public-source grounding was followed by iterative dominance and bid-competition analysis. The two diagnostics described above provided the validation, including the scenario and material breakdowns in Section 10.3.

---

## 11. Verification and Review Process

Calibration followed observed output rather than visual preference alone. The examples below show how specific validation results changed the method.

**Budget-anchor diagnosis.** Large-scale validation showed that the original raw-cost budget anchor did not represent the prices produced by the bidding formulas. The budget model was therefore aligned with expected market-facing bid prices.

**Material scarcity.** Material-stage failure analysis showed that leather scarcity was stronger than intended. Supplier coverage and material-specific stock parameters were adjusted without changing unrelated agent-wide fields.

**Competitive-balance review.** Eligibility and win-share reports were reviewed together. A low overall market share was not automatically considered a problem; the review also considered how often the agent was eligible, whether it occupied a specialist niche, whether it won under another criterion, and whether the profile contributed useful trade-offs.

**Diagnostic shortlist review.** A code audit identified that the current diagnostic shortlist can exclude a bidder outside the cheapest group even when that bidder forms a feasible cost-time combination. Correcting this would involve evaluating the complete eligible bidder set rather than a shortlist, which the current eligible-bidder counts per stage (a maximum of six, eight, and nine respectively) make computationally practical.

**Runtime implementation scope.** This document's role is limited to the generated data, not the live system that consumes it. Dynamic state updates, reverse-auction execution, final-chain validation, and SLM-assisted bidding are runtime features that evolve independently of the synthetic dataset and are documented authoritatively in `agent_logic_design.md`, not here. An earlier version of this document included specific implementation-status claims about these features; those claims fell out of date as the runtime evolved faster than this document was updated, which is the reason this document no longer restates them. Anyone citing this project's current implementation status should consult `agent_logic_design.md` and the repository commit used for the experiment in question, not this document.

---

## 12. Visualization

The visualization workflow is designed to produce review charts for each generated file:

- **Material agents:** supplier coverage, stock, cost, defect probability, completion time, feature relationships
- **Manufacturers:** product coverage, capacity, workload, unit cost, quality, reputation
- **Logistics providers:** regional coverage, base cost, cost per kilogram, route timing, damage probability, reputation
- **Orders:** scenario distribution, priority distribution, product distribution, material distribution, quantity distribution, budget distribution, deadline distribution, delivery-city distribution, product–material combinations

Figures are written to `reports/reference/dataset/figures/{version}/`. Earlier stale references to the removed logistics-capacity columns in the visualization and diagnostic scripts have since been corrected, so `make diagnose` and `make figures` both run cleanly against the current schema.

---

## 13. Versioning Policy

| Version | Description |
|---|---|
| `data/synthetic/v1/` | Baseline dataset |
| `data/synthetic/v2/` | Grounded pricing and more varied agent profiles |
| `data/synthetic/v3/` | Improved competitive structure and order generation |
| `data/synthetic/v4/` | Scaled to 50,000 orders and recalibrated using full-scale diagnostics |
| `data/synthetic/v5/` (planned) | USD-denominated, date-aware material pricing using historical series or documented proxies where available, together with route-aware logistics pricing based on the freight-calibrated parametric model documented in Section 7.1. |

The active version is configured through `DATA_VERSION = "v4"`. Every synthesis, diagnostic, and visualization script imports the same setting. Older versions remain available rather than being overwritten.

---

## 14. Language-Model Assistance and Workflow

Language models were used for script drafting, code review, agent-profile design, diagnostic interpretation, calibration reasoning, documentation, and architecture review. Their suggestions were not treated as authoritative data sources.

Each material change proposed by a language model was checked through manual code inspection, formula checking, dataset regeneration, diagnostic comparison, review by another model, live testing, or a combination of these methods.

| Tool | Configuration | Main role |
|---|---|---|
| Claude Sonnet 4.6 | Max effort | Initial scripting, profile design, and early diagnostic development |
| Claude Sonnet 5 | Primarily high effort, occasionally max | Continued calibration, planning, documentation, and critical review |
| Claude Code with Fable 5 | Primarily high effort, occasionally max | Filesystem-based implementation, NATS integration, synthesis updates, and architecture work |
| Codex using GPT-5.6 Sol | Extra-high reasoning | Read-only repository audits, comment cleanup, and targeted code review |
| GPT-5.6 Sol | High intelligence | Independent reasoning, diagnostic review, planning, and cross-checking |

One model was often used to review another model's recommendation. For example, a proposed stock adjustment was checked against existing competitors before implementation, a claim about an existing retry mechanism was checked directly against the codebase, and code changes were evaluated through regeneration and diagnostics rather than accepted from explanation alone.

---

## 15. Representative Prompt Patterns

The prompts below summarize recurring request types, in the order the corresponding work appears in this document. They are representative patterns rather than complete raw chat transcripts.

**1. Agent competition**
> Review the current material, manufacturer, and logistics profiles using their actual eligibility and pricing rules. Identify profiles with no meaningful competitive niche and suggest the smallest realistic trade-off that improves competition without making one agent globally superior.

**2. Synthesis development**
> Given a public dataset and target CSV schema, derive agent fields using documented formulas. Clearly separate directly sourced values from calibrated assumptions and add validation for missing fields, invalid values, and insufficient coverage.

**3. Budget-anchor diagnosis**
> Compare a plain unweighted average of manufacturer unit costs against a lowest-cost-win-share-weighted estimate for the same products. Determine whether the plain average is being pulled upward by manufacturers that are eligible often but almost never win a bid, and quantify how much this dilution affects the budget anchor used for order generation.

**4. Material-specific calibration**
> Reduce excessive infeasibility for one material using material-specific stock, price, and restock fields. Avoid changing agent-wide attributes unless the effect on every other supported material is reviewed.

**5. Manufacturer balance**
> Investigate manufacturers with high eligibility but very low or very high win rates. Change one economic parameter at a time and validate the result through a full diagnostic rerun.

**6. Order-density realism**
> Replace uniform round-robin dates with a seeded weekday-weighted distribution while preventing the date generator from changing another field's random sequence.

**7. Static dominance validation**
> Compare agents only within relevant competition groups. Report profiles that are at least as strong on every evaluated static dimension and strictly stronger on at least one.

**8. Bid-competition validation**
> Simulate deterministic bidding over all generated orders without NATS, containers, live agents, or an SLM. Report eligibility, competition, candidate-chain feasibility, win shares, scenario behaviour, and infeasibility causes.

**9. Diagnostic shortlist audit**
> Review the candidate-chain construction inside the bid-competition diagnostic for cases where a bid outside the cheapest shortlist could still form the only feasible chain for an order. Determine whether the current shortlist approach can produce a false infeasible classification, and quantify the maximum number of eligible bidders per stage to assess whether complete enumeration is computationally practical instead.

**10. Live architecture verification**
> Compare the design document, RFQ schema, diagnostic model, and live implementation to determine whether later-stage bidding should depend on a provisional upstream winner or whether full-chain selection should occur after the sequential stage windows close.

**11. Visualization**
> Create a Python visualization script that reviews the generated material, manufacturer, logistics, product, and order files without requiring anyone to open the CSVs directly. Include coverage charts, cost and stock heatmaps, cost-versus-risk scatter plots, workload and capacity views, order distribution charts, and correlation heatmaps, and confirm that every chart references a field that still exists in the current schema after any column removal.

**12. Reproducibility verification**
> Regenerate the order dataset twice under identical seed, configuration, and input files, and compare the two output files by hash to confirm byte-level reproduction. Separately, regenerate the dataset under a shifted base date and confirm that scenario distribution and infeasibility-case classification remain unchanged even though the order timestamps differ.

GPT-5.6 Sol also provided independent review using similar validation-oriented prompts.

---

## 16. Reproducibility

The order generator is deterministic when all relevant inputs remain unchanged. An identical `orders.csv` requires the same input CSV files and contents, generator code, configuration, random seed, base date, Python version, and dependency environment.

`orders_metadata.json` records the seed, base date, creation window, order count, scenario distribution, priority distribution, budget-calculation factors, and input-file hashes. This provides a strong provenance record, but it does not by itself record every part of the software environment. A publication package should additionally include a Git commit, environment specification, configuration snapshot, generator-script version, and output-file hash.

There are two distinct reproducibility checks:

- **Byte-level reproduction.** With unchanged inputs, configuration, code, environment, base date, and seed, two generation runs produce the same output, verified through an output SHA-256 comparison.
- **Calendar-shift stability.** Changing the base date changes `order_created` and `deadline`, and therefore does not produce a byte-identical order file. A uniform calendar shift instead verifies that scenario and case-classification behaviour remains unchanged when relative lead times are preserved.

The agent-profile scripts do not use random sampling when producing the current material, manufacturer, and logistics files. With unchanged source inputs, scripts, and configuration, they are therefore deterministic.

Multi-seed robustness was checked by rerunning the full pipeline under 15 additional independently drawn seeds (46147, 37729, 23746, 65863, 28471, 29745, 14583, 76998, 90854, 95888, 53853, 56578, 80195, 13998, 3484), drawn using a fixed meta-seed rather than hand-selected, alongside the primary seed 42. Each of the 16 runs was verified to exit cleanly, produce exactly 50,000 orders with the expected schema, and yield an infeasibility rate within a valid range before any competitive metric was examined. Full bid_check diagnostic output for each of the 15 additional seeds is retained under `reports/reference/dataset/seed-check/seed_<seed>.txt`. The stability findings from this check are summarized in Section 17.

---

## 17. Limitations

- The dataset is a controlled simulation benchmark and should not be interpreted as a direct estimate of real Finnish garment-sector prices, capacity, quality, or failure rates.
- V5 material prices use historical movement only where a credible source or clearly documented proxy is available. Cotton uses a monthly global cotton series, polyester uses a historical DTY series, denim uses a broader finished-cotton-fabric PPI proxy, and leather uses an upstream hide-price proxy. Linen uses irregular European long-flax-fibre price observations as an upstream temporal proxy. Raw-material and proxy series are not presented as literal finished-fabric prices, and unsupported fluctuations are not fabricated.
- USD is the common v5 currency. Non-USD observations are converted using the exchange rate corresponding to the relevant source observation date; no single fixed exchange-rate value is used for the entire dataset. Physical unit conversions remain fixed.
- The selected v5 logistics-pricing method is a freight-calibrated parametric approximation rather than a literal Finnish carrier tariff. Its fixed and weight-based parameters require documented calibration from published carrier references, while the tonne-kilometre coefficient, route-distance matrix, and date-dependent fuel-surcharge treatment must be fully sourced and validated before v5 generation.
- Material and logistics reputation values begin at a neutral 0.5 rather than being derived from historical performance.
- Manufacturer reputation is derived from productivity and is not an independently observed reputation measure.
- Manufacturer defect probability is a modelled productivity-based proxy because the source dataset does not provide direct product-level defect rates.
- Manufacturer product specialization is synthetic because the source dataset does not contain product categories.
- Material stock values represent initial profile state in the current dataset; manufacturer workload values represent initial workload rather than continuously updated live state.
- Stock reservation, inventory depletion, restock arrival, manufacturing workload growth, job completion, logistics-capacity reservation, and capacity release are part of the runtime design described in `agent_logic_design.md` and are not included in the current offline data model.
- `lead_time_variance` and `delay_probability` are retained for a future stochastic model but are not used by the current deterministic bidding formula.
- The current 50,000-order dataset is generated under seed 42. A multi-seed robustness check across 15 additional seeds (see Section 16 for the seed list and artifact location) shows scenario-level feasibility preserving its ordering exactly (spreads of 0.5 to 4.6 percentage points, widest for the two budget-stress scenarios), material feasibility staying within roughly 1 point for cotton, polyester, linen, and denim and 63.7 to 66.0 percent for leather, mean feasible chains per order ranging from 67.3 to 68.0 against a baseline of 67.7, and win-share concentration under lowest-cost selection proving structural rather than seed-driven, the same leading agent tops each stage across all 16 seeds with leader share moving by at most 1 point. This supports treating the reported feasibility, competition, and win-share patterns as properties of the agent and product configuration rather than artifacts of the seed-42 draw specifically.
- The diagnostic figures reported in Section 10 reflect a candidate-chain construction that uses a shortlist rather than the complete eligible-bidder set, as described in Section 11.
- This document deliberately does not track runtime implementation status for dynamic state, reverse-auction execution, final-chain validation, or SLM-assisted bidding. An earlier version's specific claims about these features became stale as the runtime evolved independently of this document; `agent_logic_design.md` is the authoritative current source instead.
- The prompt patterns in this document are consolidated methodological examples rather than raw conversation transcripts.

## Anonymous review copy

The raw DataCo customer CSV is omitted. It was only the qualitative reference identified in the source table and is not ingested by the synthesis scripts. Its omission does not change generated distributions or saved results. Public source citations and third-party notices are retained.
