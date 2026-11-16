#### sealed: `material_train.jsonl`, ORD-00200 / M09

fields: `{"order_id": "ORD-00200", "agent_id": "M09", "stage": "material", "dataset_version": "v5", "split": "train", "rule_decision": "BID"}`

```text
[system]
You are a material provider agent in a multi-agent supply chain bidding system.
Decide whether to bid on the incoming material request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: M09
location: Helsinki
materials supplied: leather|linen
available stock for requested material (kg): 280.0
price ratio to reference for requested material: 1.12
restock amount (kg): 495.0
restock time (days): 4.0
default completion days: 1
defect probability: 0.02
reputation score: 0.5
min profit margin: 0.1
initial markup: 0.33

RFQ
order_id: ORD-00200
pricing date: 2026-03-06
material needed: linen
reference price on the pricing date (USD per kg): 32.3019
required quantity (kg): 115.36
max allowed defect rate: 0.046
min required reputation: 0.47
days until deadline: 13
priority: normal

TASK
1. Check if you supply this material. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Check stock:
   if available_kg greater than or equal to (>=) required_kg: completion = default_completion_days
   else if available_kg plus (+) restock_amount_kg greater than or equal to (>=) required_kg: completion = default_completion_days plus (+) restock_time_days
   else: return NO_BID
5. Check completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate, in USD:
   material_cost  = required_kg multiplied by (x) price_ratio_to_reference multiplied by (x) reference_price_usd_per_kg
   internal_cost  = material_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
7. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "ORD-00200",
  "stage": "material",
  "agent_id": "M09",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}

[completion]
{
  "order_id": "ORD-00200",
  "stage": "material",
  "agent_id": "M09",
  "decision": "BID",
  "offer_amount": 5661.78,
  "estimated_completion_days": 1,
  "defect_probability": 0.02
}
```

#### sealed: `tailoring_train.jsonl`, ORD-00200 / T4

fields: `{"order_id": "ORD-00200", "agent_id": "T4", "stage": "tailoring", "dataset_version": "v5", "split": "train", "rule_decision": "BID"}`

```text
[system]
You are a manufacturer agent in a multi-agent garment supply chain bidding system.
Decide whether to bid on the manufacturing request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: T4
location: Turku
products you make: jeans|shirt|skirt
daily capacity (units): 752
current workload (units): 113
unit cost for requested product (USD): 4.5442
defect probability: 0.02
reputation score: 0.77
min profit margin: 0.1
initial markup: 0.3

RFQ
order_id: ORD-00200
product: skirt
quantity: 412
max allowed defect rate: 0.046
min required reputation: 0.47
days until deadline: 13
priority: normal

TASK
1. Check if you make this product. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Calculate:
   backlog_days         = current_workload_units divided by (/) daily_capacity_units
   production_days      = quantity divided by (/) daily_capacity_units
   estimated_completion = backlog_days plus (+) production_days
5. Check estimated_completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate workload ratio and markup:
   workload_ratio = (current_workload_units plus (+) quantity) divided by (/) (daily_capacity_units multiplied by (x) days_until_deadline)
   if workload_ratio less than (<) 0.50:    workload_markup = 0.00
   else if ratio less than (<) 0.75:        workload_markup = 0.05
   else if ratio less than (<) 0.90:        workload_markup = 0.10
   else if ratio less than or equal to (<=) 1.00:       workload_markup = 0.15
   else:                        return NO_BID
7. Calculate, in USD:
   base_cost      = quantity multiplied by (x) unit_cost_for_product
   internal_cost  = base_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) workload_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
8. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "ORD-00200",
  "stage": "tailoring",
  "agent_id": "T4",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}

[completion]
{
  "order_id": "ORD-00200",
  "stage": "tailoring",
  "agent_id": "T4",
  "decision": "BID",
  "offer_amount": 2482.55,
  "estimated_completion_days": 0.7,
  "defect_probability": 0.02
}
```

#### sealed: `logistics_train.jsonl`, ORD-00200 / L01

fields: `{"order_id": "ORD-00200", "agent_id": "L01", "stage": "logistics", "dataset_version": "v5", "split": "train", "rule_decision": "BID"}`

```text
[system]
You are a logistics agent in a multi-agent supply chain bidding system.
Decide whether to bid on the delivery request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: L01
location: Helsinki
service regions: Espoo|Helsinki|Jyvaskyla|Lahti|Oulu|Tampere|Turku
fixed charge (EUR per shipment): 42.3967
alpha (EUR per kg): 0.12692
beta (EUR per tonne-km): 0.07163
shipment weight limit (kg): none
oversized fee (EUR): none
same-city delivery days: 1
intercity delivery days: 3.0
damage probability: 0.035
reputation score: 0.5

RFQ
order_id: ORD-00200
pricing date: 2026-03-06
pickup from: Lahti
deliver to: Lahti
route distance from pickup (km): 0.0
shipment weight (kg): 144.2
fuel surcharge on the pricing date (%): 20.98
EUR to USD on the pricing date: 1.1561
max allowed damage rate: 0.046
min required reputation: 0.47
days until deadline: 13
priority: normal

TASK
1. Check if deliver_to is in your service_regions. If not, return NO_BID.
2. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
3. Check damage_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
4. Check shipment_weight_kg is inside your shipment weight limit. If not, return NO_BID.
5. Calculate route days:
   if deliver_to == your location: route_days = default_completion_same_city_days
   else: route_days = default_completion_intercity_days
6. Check route_days less than or equal to (<=) days_until_deadline. If not, return NO_BID.
7. Calculate, in EUR:
   freight_eur    = fixed_charge plus (+) alpha multiplied by (x) shipment_weight_kg plus (+) beta multiplied by (x) shipment_weight_kg divided by (/) 1000 multiplied by (x) route_distance_km
   with_fuel_eur  = freight_eur multiplied by (x) (1 plus (+) fuel_surcharge_percent divided by (/) 100)
   oversized_eur  = the oversized fee if it applies to this shipment, else 0
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   price_eur      = (with_fuel_eur plus (+) oversized_eur) multiplied by (x) (1 plus (+) damage_probability) multiplied by (x) (1 plus (+) priority_markup)
8. Convert to USD, the currency of every offer:
   offer_amount   = price_eur multiplied by (x) eur_to_usd
   minimum_bid    = offer_amount, a market price with no markup on top and no margin below
9. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "ORD-00200",
  "stage": "logistics",
  "agent_id": "L01",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.035
}

[completion]
{
  "order_id": "ORD-00200",
  "stage": "logistics",
  "agent_id": "L01",
  "decision": "BID",
  "offer_amount": 87.87,
  "estimated_completion_days": 3,
  "defect_probability": 0.035
}
```

#### edge: `edge/material_edge.jsonl`, SYNEDGE5-M-RESTOCK-M01-cotton / M01

fields: `{"order_id": "SYNEDGE5-M-RESTOCK-M01-cotton", "agent_id": "M01", "stage": "material", "dataset_version": "v5", "role": "robustness_eval", "category": "restock_too_slow", "rule_decision": "NO_BID"}`

```text
[system]
You are a material provider agent in a multi-agent supply chain bidding system.
Decide whether to bid on the incoming material request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: M01
location: Helsinki
materials supplied: cotton|polyester
available stock for requested material (kg): 2200.0
price ratio to reference for requested material: 1.04
restock amount (kg): 3750.0
restock time (days): 3.0
default completion days: 2
defect probability: 0.04
reputation score: 0.5
min profit margin: 0.1
initial markup: 0.26

RFQ
order_id: SYNEDGE5-M-RESTOCK-M01-cotton
pricing date: 2025-10-15
material needed: cotton
reference price on the pricing date (USD per kg): 14.9949
required quantity (kg): 4075.0
max allowed defect rate: 0.5
min required reputation: 0.1
days until deadline: 4
priority: normal

TASK
1. Check if you supply this material. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Check stock:
   if available_kg greater than or equal to (>=) required_kg: completion = default_completion_days
   else if available_kg plus (+) restock_amount_kg greater than or equal to (>=) required_kg: completion = default_completion_days plus (+) restock_time_days
   else: return NO_BID
5. Check completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate, in USD:
   material_cost  = required_kg multiplied by (x) price_ratio_to_reference multiplied by (x) reference_price_usd_per_kg
   internal_cost  = material_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
7. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "SYNEDGE5-M-RESTOCK-M01-cotton",
  "stage": "material",
  "agent_id": "M01",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.04
}

[completion]
{
  "order_id": "SYNEDGE5-M-RESTOCK-M01-cotton",
  "stage": "material",
  "agent_id": "M01",
  "decision": "NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.04
}
```

#### edge: `edge/tailoring_edge.jsonl`, SYNEDGE-T-INF-T1 / T1

fields: `{"order_id": "SYNEDGE-T-INF-T1", "agent_id": "T1", "stage": "tailoring", "dataset_version": "v5", "role": "robustness_eval", "category": "infeasible", "rule_decision": "NO_BID"}`

```text
[system]
You are a manufacturer agent in a multi-agent garment supply chain bidding system.
Decide whether to bid on the manufacturing request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: T1
location: Oulu
products you make: hoodie|jacket|shirt
daily capacity (units): 883
current workload (units): 177
unit cost for requested product (USD): 8.5538
defect probability: 0.02
reputation score: 0.821
min profit margin: 0.1
initial markup: 0.28

RFQ
order_id: SYNEDGE-T-INF-T1
product: hoodie
quantity: 4284
max allowed defect rate: 0.5
min required reputation: 0.1
days until deadline: 3
priority: normal

TASK
1. Check if you make this product. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Calculate:
   backlog_days         = current_workload_units divided by (/) daily_capacity_units
   production_days      = quantity divided by (/) daily_capacity_units
   estimated_completion = backlog_days plus (+) production_days
5. Check estimated_completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate workload ratio and markup:
   workload_ratio = (current_workload_units plus (+) quantity) divided by (/) (daily_capacity_units multiplied by (x) days_until_deadline)
   if workload_ratio less than (<) 0.50:    workload_markup = 0.00
   else if ratio less than (<) 0.75:        workload_markup = 0.05
   else if ratio less than (<) 0.90:        workload_markup = 0.10
   else if ratio less than or equal to (<=) 1.00:       workload_markup = 0.15
   else:                        return NO_BID
7. Calculate, in USD:
   base_cost      = quantity multiplied by (x) unit_cost_for_product
   internal_cost  = base_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) workload_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
8. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "SYNEDGE-T-INF-T1",
  "stage": "tailoring",
  "agent_id": "T1",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}

[completion]
{
  "order_id": "SYNEDGE-T-INF-T1",
  "stage": "tailoring",
  "agent_id": "T1",
  "decision": "NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}
```

#### edge: `edge/logistics_edge.jsonl`, SYNEDGE-L-INF-L01 / L01

fields: `{"order_id": "SYNEDGE-L-INF-L01", "agent_id": "L01", "stage": "logistics", "dataset_version": "v5", "role": "robustness_eval", "category": "infeasible", "rule_decision": "NO_BID"}`

```text
[system]
You are a logistics agent in a multi-agent supply chain bidding system.
Decide whether to bid on the delivery request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: L01
location: Helsinki
service regions: Espoo|Helsinki|Jyvaskyla|Lahti|Oulu|Tampere|Turku
fixed charge (EUR per shipment): 42.3967
alpha (EUR per kg): 0.12692
beta (EUR per tonne-km): 0.07163
shipment weight limit (kg): none
oversized fee (EUR): none
same-city delivery days: 1
intercity delivery days: 3.0
damage probability: 0.035
reputation score: 0.5

RFQ
order_id: SYNEDGE-L-INF-L01
pricing date: 2026-01-19
pickup from: Espoo
deliver to: Espoo
route distance from pickup (km): 0.0
shipment weight (kg): 571.9
fuel surcharge on the pricing date (%): 19.17
EUR to USD on the pricing date: 1.1631
max allowed damage rate: 0.5
min required reputation: 0.1
days until deadline: 2
priority: normal

TASK
1. Check if deliver_to is in your service_regions. If not, return NO_BID.
2. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
3. Check damage_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
4. Check shipment_weight_kg is inside your shipment weight limit. If not, return NO_BID.
5. Calculate route days:
   if deliver_to == your location: route_days = default_completion_same_city_days
   else: route_days = default_completion_intercity_days
6. Check route_days less than or equal to (<=) days_until_deadline. If not, return NO_BID.
7. Calculate, in EUR:
   freight_eur    = fixed_charge plus (+) alpha multiplied by (x) shipment_weight_kg plus (+) beta multiplied by (x) shipment_weight_kg divided by (/) 1000 multiplied by (x) route_distance_km
   with_fuel_eur  = freight_eur multiplied by (x) (1 plus (+) fuel_surcharge_percent divided by (/) 100)
   oversized_eur  = the oversized fee if it applies to this shipment, else 0
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   price_eur      = (with_fuel_eur plus (+) oversized_eur) multiplied by (x) (1 plus (+) damage_probability) multiplied by (x) (1 plus (+) priority_markup)
8. Convert to USD, the currency of every offer:
   offer_amount   = price_eur multiplied by (x) eur_to_usd
   minimum_bid    = offer_amount, a market price with no markup on top and no margin below
9. Return BID with calculated values.

Return this JSON only:
{
  "order_id": "SYNEDGE-L-INF-L01",
  "stage": "logistics",
  "agent_id": "L01",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.035
}

[completion]
{
  "order_id": "SYNEDGE-L-INF-L01",
  "stage": "logistics",
  "agent_id": "L01",
  "decision": "NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.035
}
```

#### reverse auction: `reverse_auction/logistics_train.jsonl`, ORD-00200 / L01

fields: `{"order_id": "ORD-00200", "agent_id": "L01", "stage": "logistics", "category": "auction_round", "round_number": 2, "current_best_bid": 85.78, "platform_minimum_decrement": 5.0, "withdrawn_in_earlier_round": false, "rule_decision": "NO_BID", "dataset_version": "v5", "split": "train"}`

```text
[system]
You are a logistics agent in a multi-agent supply chain bidding system.
Decide whether to bid on the delivery request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: L01
location: Helsinki
service regions: Espoo|Helsinki|Jyvaskyla|Lahti|Oulu|Tampere|Turku
fixed charge (EUR per shipment): 42.3967
alpha (EUR per kg): 0.12692
beta (EUR per tonne-km): 0.07163
shipment weight limit (kg): none
oversized fee (EUR): none
same-city delivery days: 1
intercity delivery days: 3.0
damage probability: 0.035
reputation score: 0.5

RFQ
order_id: ORD-00200
pricing date: 2026-03-06
pickup from: Lahti
deliver to: Lahti
route distance from pickup (km): 0.0
shipment weight (kg): 144.2
fuel surcharge on the pricing date (%): 20.98
EUR to USD on the pricing date: 1.1561
max allowed damage rate: 0.046
min required reputation: 0.47
days until deadline: 13
priority: normal
current_best_bid (USD): 85.78
platform_minimum_decrement (USD): 5.0
round_number: 2
your minimum_bid (USD): 87.87

TASK
1. Check if deliver_to is in your service_regions. If not, return NO_BID.
2. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
3. Check damage_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
4. Check shipment_weight_kg is inside your shipment weight limit. If not, return NO_BID.
5. Calculate route days:
   if deliver_to == your location: route_days = default_completion_same_city_days
   else: route_days = default_completion_intercity_days
6. Check route_days less than or equal to (<=) days_until_deadline. If not, return NO_BID.
7. Calculate, in EUR:
   freight_eur    = fixed_charge plus (+) alpha multiplied by (x) shipment_weight_kg plus (+) beta multiplied by (x) shipment_weight_kg divided by (/) 1000 multiplied by (x) route_distance_km
   with_fuel_eur  = freight_eur multiplied by (x) (1 plus (+) fuel_surcharge_percent divided by (/) 100)
   oversized_eur  = the oversized fee if it applies to this shipment, else 0
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   price_eur      = (with_fuel_eur plus (+) oversized_eur) multiplied by (x) (1 plus (+) damage_probability) multiplied by (x) (1 plus (+) priority_markup)
8. Convert to USD, the currency of every offer:
   offer_amount   = price_eur multiplied by (x) eur_to_usd
   minimum_bid    = offer_amount, a market price with no markup on top and no margin below
9. Return BID with calculated values.
In rounds after round 1: if current_best_bid minus (-) platform_minimum_decrement greater than or equal to (>=) minimum_bid, submit that as offer. Otherwise stay silent.

Return this JSON only:
{
  "order_id": "ORD-00200",
  "stage": "logistics",
  "agent_id": "L01",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.035
}

[completion]
{
  "order_id": "ORD-00200",
  "stage": "logistics",
  "agent_id": "L01",
  "decision": "NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.035
}
```

#### reverse auction: `reverse_auction/material_train.jsonl`, ORD-00200 / M09

fields: `{"order_id": "ORD-00200", "agent_id": "M09", "stage": "material", "category": "auction_round", "round_number": 2, "current_best_bid": 4297.04, "platform_minimum_decrement": 5.0, "withdrawn_in_earlier_round": false, "rule_decision": "NO_BID", "dataset_version": "v5", "split": "train"}`

```text
[system]
You are a material provider agent in a multi-agent supply chain bidding system.
Decide whether to bid on the incoming material request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: M09
location: Helsinki
materials supplied: leather|linen
available stock for requested material (kg): 280.0
price ratio to reference for requested material: 1.12
restock amount (kg): 495.0
restock time (days): 4.0
default completion days: 1
defect probability: 0.02
reputation score: 0.5
min profit margin: 0.1
initial markup: 0.33

RFQ
order_id: ORD-00200
pricing date: 2026-03-06
material needed: linen
reference price on the pricing date (USD per kg): 32.3019
required quantity (kg): 115.36
max allowed defect rate: 0.046
min required reputation: 0.47
days until deadline: 13
priority: normal
current_best_bid (USD): 4297.04
platform_minimum_decrement (USD): 5.0
round_number: 2
your minimum_bid (USD): 4682.68

TASK
1. Check if you supply this material. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Check stock:
   if available_kg greater than or equal to (>=) required_kg: completion = default_completion_days
   else if available_kg plus (+) restock_amount_kg greater than or equal to (>=) required_kg: completion = default_completion_days plus (+) restock_time_days
   else: return NO_BID
5. Check completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate, in USD:
   material_cost  = required_kg multiplied by (x) price_ratio_to_reference multiplied by (x) reference_price_usd_per_kg
   internal_cost  = material_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
7. Return BID with calculated values.
In rounds after round 1: if current_best_bid minus (-) platform_minimum_decrement greater than or equal to (>=) minimum_bid, submit that as offer. Otherwise stay silent.

Return this JSON only:
{
  "order_id": "ORD-00200",
  "stage": "material",
  "agent_id": "M09",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}

[completion]
{
  "order_id": "ORD-00200",
  "stage": "material",
  "agent_id": "M09",
  "decision": "NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}
```

#### reverse auction: `reverse_auction/tailoring_train.jsonl`, ORD-00200 / T4

fields: `{"order_id": "ORD-00200", "agent_id": "T4", "stage": "tailoring", "category": "auction_round", "round_number": 2, "current_best_bid": 2443.8, "platform_minimum_decrement": 5.0, "withdrawn_in_earlier_round": false, "rule_decision": "BID", "dataset_version": "v5", "split": "train"}`

```text
[system]
You are a manufacturer agent in a multi-agent garment supply chain bidding system.
Decide whether to bid on the manufacturing request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: T4
location: Turku
products you make: jeans|shirt|skirt
daily capacity (units): 752
current workload (units): 113
unit cost for requested product (USD): 4.5442
defect probability: 0.02
reputation score: 0.77
min profit margin: 0.1
initial markup: 0.3

RFQ
order_id: ORD-00200
product: skirt
quantity: 412
max allowed defect rate: 0.046
min required reputation: 0.47
days until deadline: 13
priority: normal
current_best_bid (USD): 2443.8
platform_minimum_decrement (USD): 5.0
round_number: 2
your minimum_bid (USD): 2100.63

TASK
1. Check if you make this product. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Calculate:
   backlog_days         = current_workload_units divided by (/) daily_capacity_units
   production_days      = quantity divided by (/) daily_capacity_units
   estimated_completion = backlog_days plus (+) production_days
5. Check estimated_completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate workload ratio and markup:
   workload_ratio = (current_workload_units plus (+) quantity) divided by (/) (daily_capacity_units multiplied by (x) days_until_deadline)
   if workload_ratio less than (<) 0.50:    workload_markup = 0.00
   else if ratio less than (<) 0.75:        workload_markup = 0.05
   else if ratio less than (<) 0.90:        workload_markup = 0.10
   else if ratio less than or equal to (<=) 1.00:       workload_markup = 0.15
   else:                        return NO_BID
7. Calculate, in USD:
   base_cost      = quantity multiplied by (x) unit_cost_for_product
   internal_cost  = base_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) workload_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
8. Return BID with calculated values.
In rounds after round 1: if current_best_bid minus (-) platform_minimum_decrement greater than or equal to (>=) minimum_bid, submit that as offer. Otherwise stay silent.

Return this JSON only:
{
  "order_id": "ORD-00200",
  "stage": "tailoring",
  "agent_id": "T4",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}

[completion]
{
  "order_id": "ORD-00200",
  "stage": "tailoring",
  "agent_id": "T4",
  "decision": "BID",
  "offer_amount": 2438.8,
  "estimated_completion_days": 0.7,
  "defect_probability": 0.02
}
```

#### floor boundary: `reverse_auction/floor_boundary/material_below_floor.jsonl`, ORD-33554 / M11

fields: `{"order_id": "ORD-33554", "agent_id": "M11", "stage": "material", "category": "floor_boundary:below_floor", "round_number": 3, "current_best_bid": 2366.99, "platform_minimum_decrement": 5.0, "withdrawn_in_earlier_round": false, "rule_decision": "NO_BID", "dataset_version": "v5", "split": "test", "role": "diagnostic", "case": "below_floor", "displayed_minimum_bid": 2362.0, "mandated_next_offer": 2361.99}`

```text
[system]
You are a material provider agent in a multi-agent supply chain bidding system.
Decide whether to bid on the incoming material request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: M11
location: Espoo
materials supplied: linen|polyester
available stock for requested material (kg): 320.0
price ratio to reference for requested material: 1.08
restock amount (kg): 550.0
restock time (days): 4.0
default completion days: 1
defect probability: 0.03
reputation score: 0.5
min profit margin: 0.1
initial markup: 0.3

RFQ
order_id: ORD-33554
pricing date: 2026-02-16
material needed: linen
reference price on the pricing date (USD per kg): 45.612
required quantity (kg): 42.32
max allowed defect rate: 0.033
min required reputation: 0.48
days until deadline: 7
priority: urgent
current_best_bid (USD): 2366.99
platform_minimum_decrement (USD): 5.0
round_number: 3
your minimum_bid (USD): 2362.0

TASK
1. Check if you supply this material. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Check stock:
   if available_kg greater than or equal to (>=) required_kg: completion = default_completion_days
   else if available_kg plus (+) restock_amount_kg greater than or equal to (>=) required_kg: completion = default_completion_days plus (+) restock_time_days
   else: return NO_BID
5. Check completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate, in USD:
   material_cost  = required_kg multiplied by (x) price_ratio_to_reference multiplied by (x) reference_price_usd_per_kg
   internal_cost  = material_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
7. Return BID with calculated values.
In rounds after round 1: if current_best_bid minus (-) platform_minimum_decrement greater than or equal to (>=) minimum_bid, submit that as offer. Otherwise stay silent.

Return this JSON only:
{
  "order_id": "ORD-33554",
  "stage": "material",
  "agent_id": "M11",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.03
}

[completion]
{
  "order_id": "ORD-33554",
  "stage": "material",
  "agent_id": "M11",
  "decision": "NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.03
}
```

#### floor boundary: `reverse_auction/floor_boundary/tailoring_at_floor.jsonl`, ORD-33554 / T2

fields: `{"order_id": "ORD-33554", "agent_id": "T2", "stage": "tailoring", "category": "floor_boundary:at_floor", "round_number": 3, "current_best_bid": 2888.41, "platform_minimum_decrement": 5.0, "withdrawn_in_earlier_round": false, "rule_decision": "BID", "dataset_version": "v5", "split": "test", "role": "diagnostic", "case": "at_floor", "displayed_minimum_bid": 2883.41, "mandated_next_offer": 2883.41}`

```text
[system]
You are a manufacturer agent in a multi-agent garment supply chain bidding system.
Decide whether to bid on the manufacturing request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: T2
location: Helsinki
products you make: hoodie|shirt
daily capacity (units): 837
current workload (units): 460
unit cost for requested product (USD): 4.858
defect probability: 0.02
reputation score: 0.771
min profit margin: 0.1
initial markup: 0.29

RFQ
order_id: ORD-33554
product: shirt
quantity: 529
max allowed defect rate: 0.033
min required reputation: 0.48
days until deadline: 7
priority: urgent
current_best_bid (USD): 2888.41
platform_minimum_decrement (USD): 5.0
round_number: 3
your minimum_bid (USD): 2883.41

TASK
1. Check if you make this product. If not, return NO_BID.
2. Check defect_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
3. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
4. Calculate:
   backlog_days         = current_workload_units divided by (/) daily_capacity_units
   production_days      = quantity divided by (/) daily_capacity_units
   estimated_completion = backlog_days plus (+) production_days
5. Check estimated_completion less than or equal to (<=) days_until_deadline. If not, return NO_BID.
6. Calculate workload ratio and markup:
   workload_ratio = (current_workload_units plus (+) quantity) divided by (/) (daily_capacity_units multiplied by (x) days_until_deadline)
   if workload_ratio less than (<) 0.50:    workload_markup = 0.00
   else if ratio less than (<) 0.75:        workload_markup = 0.05
   else if ratio less than (<) 0.90:        workload_markup = 0.10
   else if ratio less than or equal to (<=) 1.00:       workload_markup = 0.15
   else:                        return NO_BID
7. Calculate, in USD:
   base_cost      = quantity multiplied by (x) unit_cost_for_product
   internal_cost  = base_cost multiplied by (x) (1 plus (+) defect_probability)
   minimum_bid    = internal_cost multiplied by (x) (1 plus (+) min_profit_margin)
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   initial_bid    = internal_cost multiplied by (x) (1 plus (+) initial_markup) multiplied by (x) (1 plus (+) workload_markup) multiplied by (x) (1 plus (+) priority_markup)
   offer_amount   = the larger of the two values max(minimum_bid, initial_bid)
8. Return BID with calculated values.
In rounds after round 1: if current_best_bid minus (-) platform_minimum_decrement greater than or equal to (>=) minimum_bid, submit that as offer. Otherwise stay silent.

Return this JSON only:
{
  "order_id": "ORD-33554",
  "stage": "tailoring",
  "agent_id": "T2",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.02
}

[completion]
{
  "order_id": "ORD-33554",
  "stage": "tailoring",
  "agent_id": "T2",
  "decision": "BID",
  "offer_amount": 2883.41,
  "estimated_completion_days": 1.18,
  "defect_probability": 0.02
}
```

#### floor boundary: `reverse_auction/floor_boundary/logistics_above_floor.jsonl`, ORD-33554 / L04

fields: `{"order_id": "ORD-33554", "agent_id": "L04", "stage": "logistics", "category": "floor_boundary:above_floor", "round_number": 3, "current_best_bid": 102.32, "platform_minimum_decrement": 5.0, "withdrawn_in_earlier_round": false, "rule_decision": "BID", "dataset_version": "v5", "split": "test", "role": "diagnostic", "case": "above_floor", "displayed_minimum_bid": 96.36, "mandated_next_offer": 97.32}`

```text
[system]
You are a logistics agent in a multi-agent supply chain bidding system.
Decide whether to bid on the delivery request.
Check all constraints carefully. Return valid JSON only, nothing outside the JSON.

[user]
AGENT PROFILE
agent_id: L04
location: Turku
service regions: Espoo|Helsinki|Lahti|Tampere|Turku
fixed charge (EUR per shipment): 40.9083
alpha (EUR per kg): 0.128447
beta (EUR per tonne-km): 0.07163
shipment weight limit (kg): none
oversized fee (EUR): none
same-city delivery days: 1
intercity delivery days: 2.0
damage probability: 0.032
reputation score: 0.5

RFQ
order_id: ORD-33554
pricing date: 2026-02-16
pickup from: Jyvaskyla
deliver to: Turku
route distance from pickup (km): 308.536
shipment weight (kg): 63.48
fuel surcharge on the pricing date (%): 20.05
EUR to USD on the pricing date: 1.1855
max allowed damage rate: 0.033
min required reputation: 0.48
days until deadline: 7
priority: urgent
current_best_bid (USD): 102.32
platform_minimum_decrement (USD): 5.0
round_number: 3
your minimum_bid (USD): 96.36

TASK
1. Check if deliver_to is in your service_regions. If not, return NO_BID.
2. Check reputation_score greater than or equal to (>=) min_reputation_score. If not, return NO_BID.
3. Check damage_probability less than or equal to (<=) max_defect_rate. If not, return NO_BID.
4. Check shipment_weight_kg is inside your shipment weight limit. If not, return NO_BID.
5. Calculate route days:
   if deliver_to == your location: route_days = default_completion_same_city_days
   else: route_days = default_completion_intercity_days
6. Check route_days less than or equal to (<=) days_until_deadline. If not, return NO_BID.
7. Calculate, in EUR:
   freight_eur    = fixed_charge plus (+) alpha multiplied by (x) shipment_weight_kg plus (+) beta multiplied by (x) shipment_weight_kg divided by (/) 1000 multiplied by (x) route_distance_km
   with_fuel_eur  = freight_eur multiplied by (x) (1 plus (+) fuel_surcharge_percent divided by (/) 100)
   oversized_eur  = the oversized fee if it applies to this shipment, else 0
   priority_markup: urgent=0.30, normal=0.00, flexible=-0.15
   price_eur      = (with_fuel_eur plus (+) oversized_eur) multiplied by (x) (1 plus (+) damage_probability) multiplied by (x) (1 plus (+) priority_markup)
8. Convert to USD, the currency of every offer:
   offer_amount   = price_eur multiplied by (x) eur_to_usd
   minimum_bid    = offer_amount, a market price with no markup on top and no margin below
9. Return BID with calculated values.
In rounds after round 1: if current_best_bid minus (-) platform_minimum_decrement greater than or equal to (>=) minimum_bid, submit that as offer. Otherwise stay silent.

Return this JSON only:
{
  "order_id": "ORD-33554",
  "stage": "logistics",
  "agent_id": "L04",
  "decision": "BID or NO_BID",
  "offer_amount": null,
  "estimated_completion_days": null,
  "defect_probability": 0.032
}

[completion]
{
  "order_id": "ORD-33554",
  "stage": "logistics",
  "agent_id": "L04",
  "decision": "BID",
  "offer_amount": 97.32,
  "estimated_completion_days": 1,
  "defect_probability": 0.032
}
```
