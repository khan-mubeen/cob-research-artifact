# v5 evaluation metrics

## Sealed-bid

price and days deviations are absolute: |model - rule-based| relative to the rule-based value, never signed


### validation

| model | arm | rows | macro agree % | overall agree % | JSON valid % | NO_BID recall % | NO_BID precision % | macro \|price dev\| % | overall \|price dev\| % | macro \|days dev\| (d) | macro \|days dev\| % | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma | base | 67,729 | 97.67 | 97.81 | 99.98 | 0.00 | - | 51.43 | 51.60 | 0.77 | 60.64 |  |
| gemma | epoch 1 | 67,729 | 99.97 | 99.97 | 100.00 | 99.52 | 99.05 | 0.65 | 0.64 | 0.00 | 0.03 |  |
| gemma | epoch 2 (selected) | 67,729 | 99.98 | 99.98 | 100.00 | 99.66 | 99.59 | 0.28 | 0.28 | 0.00 | 0.01 | selection not statistically distinguishable from epoch 3 (p = 0.69) |
| gemma | epoch 3 | 67,729 | 99.98 | 99.98 | 99.98 | 100.00 | 100.00 | 0.10 | 0.10 | 0.00 | 0.00 |  |
| granite | base | 67,729 | 77.52 | 77.73 | 99.99 | 97.62 | 8.70 | 107.71 | 102.21 | 3.18 | 373.94 |  |
| granite | epoch 1 | 67,729 | 99.94 | 99.94 | 100.00 | 99.80 | 97.67 | 0.33 | 0.33 | 0.00 | 0.03 |  |
| granite | epoch 2 (selected) | 67,729 | 100.00 | 100.00 | 100.00 | 99.80 | 100.00 | 0.15 | 0.15 | 0.00 | 0.00 | selection not statistically distinguishable from epoch 3 (p = 0.109) |
| granite | epoch 3 | 67,729 | 99.99 | 99.99 | 99.99 | 99.93 | 99.73 | 0.06 | 0.05 | 0.00 | 0.00 |  |
| ministral | base | 67,729 | 75.67 | 74.85 | 98.30 | 67.85 | 6.06 | 63.56 | 64.86 | 0.83 | 79.11 |  |
| ministral | epoch 1 | 67,729 | 99.89 | 99.90 | 100.00 | 96.66 | 98.47 | 0.61 | 0.60 | 0.00 | 0.03 |  |
| ministral | epoch 2 | 67,729 | 99.98 | 99.99 | 100.00 | 99.80 | 99.52 | 0.29 | 0.28 | 0.00 | 0.01 |  |
| ministral | epoch 3 (selected) | 67,729 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.14 | 0.14 | 0.00 | 0.00 |  |
| phi | base | 67,729 | 38.27 | 38.04 | 99.67 | 99.05 | 3.37 | 41.34 | 41.34 | 0.16 | 3.36 |  |
| phi | epoch 1 | 67,729 | 99.98 | 99.98 | 100.00 | 99.45 | 99.59 | 0.36 | 0.35 | 0.00 | 0.01 |  |
| phi | epoch 2 (selected) | 67,729 | 99.99 | 99.99 | 100.00 | 99.80 | 99.86 | 0.18 | 0.18 | 0.00 | 0.00 | selection not statistically distinguishable from epoch 3 (p = 0.0574) |
| phi | epoch 3 | 67,729 | 99.98 | 99.98 | 99.99 | 99.80 | 99.86 | 0.62 | 0.68 | 0.00 | 0.00 |  |
| qwen | base | 67,729 | 97.72 | 97.86 | 100.00 | 1.63 | 92.31 | 60.48 | 58.42 | 0.56 | 62.51 |  |
| qwen | epoch 1 | 67,729 | 99.95 | 99.95 | 100.00 | 98.98 | 98.71 | 0.24 | 0.24 | 0.00 | 0.01 |  |
| qwen | epoch 2 | 67,729 | 99.99 | 99.99 | 100.00 | 100.00 | 99.66 | 0.12 | 0.12 | 0.00 | 0.00 |  |
| qwen | epoch 3 (selected) | 67,729 | 100.00 | 99.99 | 100.00 | 100.00 | 99.80 | 0.05 | 0.05 | 0.00 | 0.00 | selection not statistically distinguishable from epoch 2 (p = 1) |
| smollm3 | base | 67,729 | 70.85 | 71.40 | 72.89 | 0.00 | - | 310.79 | 355.43 | 3.54 | 316.34 |  |
| smollm3 | epoch 1 (selected) | 67,729 | 99.99 | 99.99 | 100.00 | 99.86 | 99.52 | 0.34 | 0.33 | 0.00 | 0.01 |  |
| smollm3 | epoch 2 | 67,729 | 99.97 | 99.97 | 100.00 | 98.77 | 99.72 | 0.16 | 0.16 | 0.00 | 0.00 |  |
| smollm3 | epoch 3 | 67,729 | 99.99 | 99.99 | 99.99 | 100.00 | 99.86 | 0.07 | 0.07 | 0.00 | 0.00 |  |

### test

base and the selected epoch are the one-shot test reads; the other epochs are descriptive, read after selection and not used for selection

| model | arm | rows | macro agree % | overall agree % | JSON valid % | NO_BID recall % | NO_BID precision % | macro \|price dev\| % | overall \|price dev\| % | macro \|days dev\| (d) | macro \|days dev\| % | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma | base (one-shot) | 67,597 | 97.56 | 97.71 | 99.99 | 0.00 | - | 51.73 | 51.89 | 0.77 | 60.54 |  |
| gemma | epoch 1 (descriptive, not used for selection) | 67,597 | 99.97 | 99.97 | 100.00 | 99.68 | 98.97 | 0.65 | 0.64 | 0.00 | 0.03 |  |
| gemma | epoch 2 (selected, one-shot) | 67,597 | 99.98 | 99.98 | 100.00 | 99.81 | 99.55 | 0.29 | 0.29 | 0.00 | 0.01 | selection not statistically distinguishable from epoch 3 (p = 0.69) |
| gemma | epoch 3 (descriptive, not used for selection) | 67,597 | 99.98 | 99.97 | 99.98 | 100.00 | 99.94 | 0.10 | 0.10 | 0.00 | 0.00 |  |
| granite | base (one-shot) | 67,597 | 77.29 | 77.49 | 99.99 | 97.41 | 9.01 | 107.65 | 102.23 | 3.16 | 368.34 |  |
| granite | epoch 1 (descriptive, not used for selection) | 67,597 | 99.92 | 99.93 | 99.99 | 99.48 | 97.71 | 0.33 | 0.33 | 0.00 | 0.02 |  |
| granite | epoch 2 (selected, one-shot) | 67,597 | 99.99 | 99.99 | 99.99 | 99.87 | 100.00 | 0.14 | 0.14 | 0.00 | 0.00 | selection not statistically distinguishable from epoch 3 (p = 0.109) |
| granite | epoch 3 (descriptive, not used for selection) | 67,597 | 99.99 | 99.99 | 99.99 | 100.00 | 99.94 | 0.05 | 0.05 | 0.00 | 0.00 |  |
| ministral | base (one-shot) | 67,597 | 75.44 | 74.66 | 98.17 | 68.89 | 6.44 | 63.68 | 64.98 | 0.81 | 76.59 |  |
| ministral | epoch 1 (descriptive, not used for selection) | 67,597 | 99.90 | 99.91 | 100.00 | 98.06 | 97.99 | 0.65 | 0.64 | 0.00 | 0.03 |  |
| ministral | epoch 2 (descriptive, not used for selection) | 67,597 | 99.98 | 99.98 | 100.00 | 99.81 | 99.29 | 0.29 | 0.29 | 0.00 | 0.00 |  |
| ministral | epoch 3 (selected, one-shot) | 67,597 | 100.00 | 100.00 | 100.00 | 100.00 | 99.94 | 0.15 | 0.15 | 0.00 | 0.00 |  |
| phi | base (one-shot) | 67,597 | 38.13 | 37.91 | 99.68 | 98.44 | 3.51 | 41.07 | 41.07 | 0.15 | 3.16 |  |
| phi | epoch 1 (descriptive, not used for selection) | 67,597 | 99.98 | 99.98 | 100.00 | 99.48 | 99.68 | 0.35 | 0.35 | 0.00 | 0.02 |  |
| phi | epoch 2 (selected, one-shot) | 67,597 | 99.99 | 99.99 | 100.00 | 99.87 | 99.87 | 0.18 | 0.18 | 0.00 | 0.00 | selection not statistically distinguishable from epoch 3 (p = 0.0574) |
| phi | epoch 3 (descriptive, not used for selection) | 67,597 | 99.98 | 99.98 | 99.99 | 99.87 | 99.87 | 1.44 | 1.59 | 0.00 | 0.00 |  |
| qwen | base (one-shot) | 67,597 | 97.62 | 97.76 | 100.00 | 1.94 | 90.91 | 60.55 | 58.51 | 0.56 | 61.62 |  |
| qwen | epoch 1 (descriptive, not used for selection) | 67,597 | 99.94 | 99.94 | 100.00 | 98.90 | 98.52 | 0.24 | 0.24 | 0.00 | 0.01 |  |
| qwen | epoch 2 (descriptive, not used for selection) | 67,597 | 99.99 | 99.99 | 100.00 | 100.00 | 99.68 | 0.12 | 0.12 | 0.00 | 0.00 |  |
| qwen | epoch 3 (selected, one-shot) | 67,597 | 100.00 | 100.00 | 100.00 | 99.87 | 100.00 | 0.04 | 0.04 | 0.00 | 0.00 | selection not statistically distinguishable from epoch 2 (p = 1) |
| smollm3 | base (one-shot) | 67,597 | 71.64 | 72.13 | 73.69 | 0.00 | - | 311.58 | 360.22 | 3.63 | 322.47 |  |
| smollm3 | epoch 1 (selected, one-shot) | 67,597 | 99.99 | 99.99 | 100.00 | 99.87 | 99.74 | 0.34 | 0.33 | 0.00 | 0.01 |  |
| smollm3 | epoch 2 (descriptive, not used for selection) | 67,597 | 99.95 | 99.96 | 100.00 | 98.25 | 99.87 | 0.16 | 0.16 | 0.00 | 0.00 |  |
| smollm3 | epoch 3 (descriptive, not used for selection) | 67,597 | 99.98 | 99.98 | 99.98 | 99.94 | 99.87 | 0.06 | 0.06 | 0.00 | 0.00 |  |

## Reverse auction

price and days deviations are absolute: |model - rule-based| relative to the rule-based value, never signed


### validation

| model | arm | rows | macro agree % | overall agree % | JSON valid % | NO_BID recall % | NO_BID precision % | macro \|price dev\| % | overall \|price dev\| % | macro \|days dev\| (d) | macro \|days dev\| % | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma | base | 75,310 | 61.35 | 61.70 | 100.00 | 0.00 | 100.00 | 12.86 | 12.92 | 0.62 | 46.18 |  |
| gemma | epoch 1 | 75,310 | 99.57 | 99.56 | 100.00 | 98.86 | 99.99 | 0.06 | 0.06 | 0.00 | 0.05 |  |
| gemma | epoch 2 | 75,310 | 99.96 | 99.96 | 100.00 | 99.96 | 99.93 | 0.04 | 0.04 | 0.00 | 0.01 |  |
| gemma | epoch 3 (selected) | 75,310 | 99.99 | 99.99 | 100.00 | 99.99 | 99.99 | 0.02 | 0.02 | 0.00 | 0.00 |  |
| granite | base | 75,310 | 66.64 | 66.68 | 100.00 | 64.13 | 55.64 | 23.37 | 19.96 | 3.53 | 397.76 |  |
| granite | epoch 1 | 75,310 | 99.78 | 99.78 | 100.00 | 99.67 | 99.75 | 0.03 | 0.03 | 0.00 | 0.03 |  |
| granite | epoch 2 | 75,310 | 99.98 | 99.97 | 100.00 | 99.98 | 99.95 | 0.43 | 0.41 | 0.00 | 0.01 |  |
| granite | epoch 3 (selected) | 75,310 | 99.99 | 99.99 | 100.00 | 99.99 | 99.99 | 0.03 | 0.02 | 0.00 | 0.00 |  |
| ministral | base | 75,310 | 54.42 | 54.19 | 99.88 | 34.41 | 38.98 | 11.38 | 10.43 | 1.03 | 96.58 |  |
| ministral | epoch 1 | 75,310 | 99.95 | 99.95 | 100.00 | 99.95 | 99.93 | 0.14 | 0.15 | 0.00 | 0.05 |  |
| ministral | epoch 2 | 75,310 | 99.97 | 99.96 | 100.00 | 99.98 | 99.93 | 0.06 | 0.06 | 0.00 | 0.04 |  |
| ministral | epoch 3 (selected) | 75,310 | 100.00 | 99.99 | 100.00 | 100.00 | 99.99 | 0.02 | 0.02 | 0.00 | 0.00 |  |
| phi | base | 75,310 | 38.70 | 38.34 | 100.00 | 99.07 | 38.23 | 4.04 | 7.21 | 2.01 | 279.28 |  |
| phi | epoch 1 | 75,310 | 99.84 | 99.84 | 100.00 | 99.60 | 99.98 | 0.09 | 0.10 | 0.00 | 0.02 |  |
| phi | epoch 2 | 75,310 | 99.95 | 99.95 | 100.00 | 99.88 | 99.99 | 0.03 | 0.03 | 0.00 | 0.00 |  |
| phi | epoch 3 (selected) | 75,310 | 99.99 | 99.99 | 100.00 | 99.98 | 100.00 | 0.01 | 0.01 | 0.00 | 0.00 |  |
| qwen | base | 75,310 | 61.37 | 61.73 | 100.00 | 0.08 | 85.19 | 10.90 | 11.04 | 0.51 | 51.71 |  |
| qwen | epoch 1 | 75,310 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.02 | 0.02 | 0.00 | 0.01 |  |
| qwen | epoch 2 | 75,310 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.01 | 0.01 | 0.00 | 0.00 |  |
| qwen | epoch 3 (selected) | 75,310 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |  |
| smollm3 | base | 75,310 | 61.26 | 61.61 | 99.40 | 0.00 | - | 15.77 | 15.63 | 4.88 | 406.46 |  |
| smollm3 | epoch 1 | 75,310 | 99.88 | 99.88 | 100.00 | 99.80 | 99.88 | 0.14 | 0.15 | 0.00 | 0.04 |  |
| smollm3 | epoch 2 | 75,310 | 99.98 | 99.98 | 100.00 | 99.98 | 99.96 | 0.07 | 0.08 | 0.00 | 0.00 |  |
| smollm3 | epoch 3 (selected) | 75,310 | 99.99 | 99.99 | 100.00 | 100.00 | 99.99 | 0.01 | 0.02 | 0.00 | 0.00 |  |

### test

base and the selected epoch are the one-shot test reads; the other epochs are descriptive, read after selection and not used for selection

| model | arm | rows | macro agree % | overall agree % | JSON valid % | NO_BID recall % | NO_BID precision % | macro \|price dev\| % | overall \|price dev\| % | macro \|days dev\| (d) | macro \|days dev\| % | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma | base (one-shot) | 75,397 | 61.42 | 61.76 | 100.00 | 0.00 | - | 12.74 | 12.78 | 0.59 | 44.96 |  |
| gemma | epoch 1 (descriptive, not used for selection) | 75,397 | 99.63 | 99.61 | 100.00 | 98.99 | 99.99 | 0.06 | 0.07 | 0.00 | 0.05 |  |
| gemma | epoch 2 (descriptive, not used for selection) | 75,397 | 99.95 | 99.95 | 100.00 | 99.95 | 99.92 | 0.04 | 0.04 | 0.00 | 0.01 |  |
| gemma | epoch 3 (selected, one-shot) | 75,397 | 99.99 | 99.99 | 100.00 | 99.99 | 99.99 | 0.02 | 0.02 | 0.00 | 0.00 |  |
| granite | base (one-shot) | 75,397 | 66.23 | 66.26 | 100.00 | 63.94 | 55.07 | 23.50 | 19.97 | 3.51 | 396.72 |  |
| granite | epoch 1 (descriptive, not used for selection) | 75,397 | 99.78 | 99.78 | 100.00 | 99.64 | 99.78 | 0.02 | 0.02 | 0.00 | 0.03 |  |
| granite | epoch 2 (descriptive, not used for selection) | 75,397 | 99.97 | 99.97 | 100.00 | 99.98 | 99.94 | 0.53 | 0.50 | 0.03 | 1.74 |  |
| granite | epoch 3 (selected, one-shot) | 75,397 | 99.99 | 99.99 | 100.00 | 99.99 | 99.99 | 0.05 | 0.04 | 0.00 | 0.00 |  |
| ministral | base (one-shot) | 75,397 | 54.44 | 54.21 | 99.88 | 34.67 | 38.99 | 11.28 | 10.35 | 1.01 | 94.39 |  |
| ministral | epoch 1 (descriptive, not used for selection) | 75,397 | 99.96 | 99.95 | 100.00 | 99.94 | 99.93 | 0.12 | 0.13 | 0.00 | 0.03 |  |
| ministral | epoch 2 (descriptive, not used for selection) | 75,397 | 99.97 | 99.96 | 100.00 | 99.97 | 99.93 | 0.06 | 0.06 | 0.00 | 0.04 |  |
| ministral | epoch 3 (selected, one-shot) | 75,397 | 100.00 | 100.00 | 100.00 | 100.00 | 99.99 | 0.02 | 0.02 | 0.00 | 0.00 |  |
| phi | base (one-shot) | 75,397 | 38.68 | 38.34 | 100.00 | 99.15 | 38.20 | 10.49 | 8.17 | 0.44 | 42.88 |  |
| phi | epoch 1 (descriptive, not used for selection) | 75,397 | 99.82 | 99.82 | 100.00 | 99.55 | 99.98 | 0.08 | 0.08 | 0.00 | 0.02 |  |
| phi | epoch 2 (descriptive, not used for selection) | 75,397 | 99.94 | 99.94 | 100.00 | 99.85 | 99.99 | 0.03 | 0.04 | 0.00 | 0.00 |  |
| phi | epoch 3 (selected, one-shot) | 75,397 | 99.99 | 99.99 | 100.00 | 99.99 | 99.99 | 0.03 | 0.03 | 0.00 | 0.00 |  |
| qwen | base (one-shot) | 75,397 | 61.45 | 61.79 | 100.00 | 0.12 | 80.95 | 11.01 | 11.12 | 0.50 | 51.13 |  |
| qwen | epoch 1 (descriptive, not used for selection) | 75,397 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.02 | 0.02 | 0.00 | 0.01 |  |
| qwen | epoch 2 (descriptive, not used for selection) | 75,397 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.01 | 0.01 | 0.00 | 0.00 |  |
| qwen | epoch 3 (selected, one-shot) | 75,397 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |  |
| smollm3 | base (one-shot) | 75,397 | 61.32 | 61.66 | 99.38 | 0.00 | - | 16.34 | 16.18 | 4.88 | 405.11 |  |
| smollm3 | epoch 1 (descriptive, not used for selection) | 75,397 | 99.88 | 99.88 | 100.00 | 99.78 | 99.90 | 0.08 | 0.09 | 0.00 | 0.05 |  |
| smollm3 | epoch 2 (descriptive, not used for selection) | 75,397 | 99.98 | 99.98 | 100.00 | 99.98 | 99.96 | 0.09 | 0.09 | 0.00 | 0.00 |  |
| smollm3 | epoch 3 (selected, one-shot) | 75,397 | 99.99 | 99.99 | 100.00 | 100.00 | 99.98 | 0.02 | 0.02 | 0.00 | 0.00 |  |

## Supervisor


### validation

| model | arm | rows | sel. acc % | JSON valid % | mean regret % | \|cost dev\| % | cost dev % | \|days dev\| % | days dev % | infeasible % | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma | base | 4,455 | 19.24 | 100.00 | 16.61 | 4.80 | -2.47 | 19.94 | 8.62 | 0.00 |  |
| gemma | epoch 1 | 4,455 | 95.65 | 100.00 | 0.03 | 0.12 | 0.01 | 0.55 | -0.03 | 0.00 |  |
| gemma | epoch 2 | 4,455 | 96.43 | 100.00 | 0.03 | 0.09 | -0.00 | 0.45 | 0.02 | 0.00 |  |
| gemma | epoch 3 (selected) | 4,455 | 97.40 | 100.00 | 0.02 | 0.08 | -0.00 | 0.37 | 0.03 | 0.00 |  |
| granite | base | 4,455 | 11.87 | 100.00 | 27.78 | 6.98 | -5.97 | 30.37 | 24.25 | 0.00 |  |
| granite | epoch 1 | 4,455 | 93.85 | 100.00 | 0.07 | 0.19 | 0.08 | 0.89 | 0.25 | 0.00 |  |
| granite | epoch 2 | 4,455 | 96.72 | 100.00 | 0.03 | 0.09 | -0.03 | 0.38 | -0.17 | 0.00 |  |
| granite | epoch 3 (selected) | 4,455 | 98.00 | 100.00 | 0.01 | 0.06 | 0.01 | 0.29 | 0.02 | 0.00 |  |
| ministral | base | 4,455 | 17.69 | 100.00 | 18.62 | 5.84 | -2.93 | 22.18 | 5.46 | 0.00 |  |
| ministral | epoch 1 | 4,455 | 94.28 | 100.00 | 0.08 | 0.15 | -0.02 | 0.75 | 0.12 | 0.00 |  |
| ministral | epoch 2 | 4,455 | 96.88 | 100.00 | 0.02 | 0.08 | -0.01 | 0.41 | -0.01 | 0.00 |  |
| ministral | epoch 3 (selected) | 4,455 | 97.82 | 100.00 | 0.01 | 0.06 | 0.00 | 0.39 | 0.12 | 0.00 |  |
| phi | base | 4,455 | 15.74 | 100.00 | 18.97 | 5.66 | -3.19 | 22.98 | 9.33 | 0.00 |  |
| phi | epoch 1 | 4,455 | 94.21 | 100.00 | 0.06 | 0.14 | -0.01 | 0.64 | 0.03 | 0.00 |  |
| phi | epoch 2 | 4,455 | 96.12 | 100.00 | 0.03 | 0.08 | -0.01 | 0.47 | 0.04 | 0.00 |  |
| phi | epoch 3 (selected) | 4,455 | 97.51 | 100.00 | 0.02 | 0.07 | -0.00 | 0.35 | 0.05 | 0.00 |  |
| qwen | base | 4,455 | 25.30 | 99.98 | 14.53 | 5.12 | -2.86 | 16.72 | 6.03 | 0.00 |  |
| qwen | epoch 1 | 4,455 | 96.27 | 100.00 | 0.05 | 0.13 | 0.03 | 0.64 | 0.29 | 0.00 |  |
| qwen | epoch 2 | 4,455 | 98.32 | 100.00 | 0.02 | 0.06 | -0.02 | 0.28 | -0.06 | 0.00 |  |
| qwen | epoch 3 (selected) | 4,455 | 98.83 | 100.00 | 0.01 | 0.04 | 0.02 | 0.23 | 0.10 | 0.00 |  |
| smollm3 | base | 4,455 | 14.23 | 100.00 | 20.27 | 5.52 | -2.12 | 24.53 | 12.20 | 0.00 |  |
| smollm3 | epoch 1 | 4,455 | 94.30 | 100.00 | 0.05 | 0.14 | -0.01 | 0.79 | 0.16 | 0.00 |  |
| smollm3 | epoch 2 | 4,455 | 96.70 | 100.00 | 0.02 | 0.08 | -0.01 | 0.48 | 0.10 | 0.00 |  |
| smollm3 | epoch 3 (selected) | 4,455 | 97.76 | 100.00 | 0.01 | 0.06 | 0.01 | 0.38 | 0.10 | 0.00 |  |

### test

base and the selected epoch are the one-shot test reads; the other epochs are descriptive, read after selection and not used for selection

| model | arm | rows | sel. acc % | JSON valid % | mean regret % | \|cost dev\| % | cost dev % | \|days dev\| % | days dev % | infeasible % | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma | base (one-shot) | 4,435 | 19.46 | 100.00 | 17.03 | 4.94 | -2.39 | 20.13 | 8.88 | 0.00 |  |
| gemma | epoch 1 (descriptive, not used for selection) | 4,435 | 95.63 | 100.00 | 0.04 | 0.12 | 0.02 | 0.63 | 0.16 | 0.00 |  |
| gemma | epoch 2 (descriptive, not used for selection) | 4,435 | 97.02 | 100.00 | 0.03 | 0.08 | -0.02 | 0.40 | 0.06 | 0.00 |  |
| gemma | epoch 3 (selected, one-shot) | 4,435 | 97.52 | 100.00 | 0.01 | 0.07 | 0.01 | 0.38 | 0.12 | 0.00 |  |
| granite | base (one-shot) | 4,435 | 11.79 | 100.00 | 27.98 | 6.80 | -5.79 | 30.29 | 23.81 | 0.00 |  |
| granite | epoch 1 (descriptive, not used for selection) | 4,435 | 94.39 | 100.00 | 0.05 | 0.18 | 0.06 | 0.78 | 0.21 | 0.00 |  |
| granite | epoch 2 (descriptive, not used for selection) | 4,435 | 96.75 | 100.00 | 0.02 | 0.09 | -0.01 | 0.41 | -0.11 | 0.00 |  |
| granite | epoch 3 (selected, one-shot) | 4,435 | 97.95 | 100.00 | 0.01 | 0.05 | 0.02 | 0.30 | 0.09 | 0.00 |  |
| ministral | base (one-shot) | 4,435 | 18.06 | 100.00 | 18.71 | 5.88 | -3.06 | 21.53 | 5.35 | 0.00 |  |
| ministral | epoch 1 (descriptive, not used for selection) | 4,435 | 94.16 | 100.00 | 0.08 | 0.16 | -0.02 | 0.74 | 0.14 | 0.00 |  |
| ministral | epoch 2 (descriptive, not used for selection) | 4,435 | 96.66 | 100.00 | 0.02 | 0.09 | 0.01 | 0.47 | 0.15 | 0.00 |  |
| ministral | epoch 3 (selected, one-shot) | 4,435 | 97.72 | 100.00 | 0.01 | 0.06 | 0.01 | 0.35 | 0.13 | 0.00 |  |
| phi | base (one-shot) | 4,435 | 16.89 | 100.00 | 18.88 | 5.46 | -3.01 | 21.89 | 8.72 | 0.00 |  |
| phi | epoch 1 (descriptive, not used for selection) | 4,435 | 94.54 | 100.00 | 0.05 | 0.13 | -0.01 | 0.64 | 0.13 | 0.00 |  |
| phi | epoch 2 (descriptive, not used for selection) | 4,435 | 95.90 | 100.00 | 0.03 | 0.09 | 0.02 | 0.45 | 0.06 | 0.00 |  |
| phi | epoch 3 (selected, one-shot) | 4,435 | 97.11 | 100.00 | 0.02 | 0.08 | 0.01 | 0.38 | 0.05 | 0.00 |  |
| qwen | base (one-shot) | 4,435 | 25.77 | 99.91 | 14.85 | 4.92 | -2.80 | 16.57 | 5.96 | 0.00 |  |
| qwen | epoch 1 (descriptive, not used for selection) | 4,435 | 96.55 | 100.00 | 0.03 | 0.11 | 0.02 | 0.63 | 0.39 | 0.00 |  |
| qwen | epoch 2 (descriptive, not used for selection) | 4,435 | 98.22 | 100.00 | 0.02 | 0.05 | -0.01 | 0.25 | -0.11 | 0.00 |  |
| qwen | epoch 3 (selected, one-shot) | 4,435 | 98.67 | 100.00 | 0.01 | 0.03 | 0.00 | 0.21 | 0.09 | 0.00 |  |
| smollm3 | base (one-shot) | 4,435 | 15.13 | 100.00 | 20.42 | 5.35 | -2.06 | 24.72 | 12.31 | 0.00 |  |
| smollm3 | epoch 1 (descriptive, not used for selection) | 4,435 | 94.39 | 100.00 | 0.05 | 0.13 | -0.01 | 0.65 | 0.12 | 0.00 |  |
| smollm3 | epoch 2 (descriptive, not used for selection) | 4,435 | 96.41 | 100.00 | 0.02 | 0.08 | 0.00 | 0.43 | 0.14 | 0.00 |  |
| smollm3 | epoch 3 (selected, one-shot) | 4,435 | 97.54 | 100.00 | 0.02 | 0.06 | 0.01 | 0.36 | 0.07 | 0.00 |  |

## Selection margin: chosen epoch vs runner-up (validation, exact McNemar)

the pre-registered rule is applied as written; this only reports whether its choice is statistically distinguishable on the key that decided it: the first of the ordered keys on which chosen and runner-up differ, tested on that key's per-row field (rows pooled across stages, while the key is a stage macro: a close proxy)

| mode | model | chosen | runner-up | decided by | gap on that key | chosen-only | runner-up-only | p (exact McNemar) | note |
|---|---|---|---|---|---|---|---|---|---|
| sealed | gemma | epoch 2 | epoch 3 | macro decision agreement % | 0.0033 | 14 | 11 | 0.69 (decision_matched) | selection not statistically distinguishable |
| sealed | granite | epoch 2 | epoch 3 | macro decision agreement % | 0.0100 | 8 | 2 | 0.109 (decision_matched) | selection not statistically distinguishable |
| sealed | ministral | epoch 3 | epoch 2 | macro decision agreement % | 0.0167 | 10 | 0 | 0.00195 (decision_matched) | distinguishable at 0.05 |
| sealed | phi | epoch 2 | epoch 3 | macro decision agreement % | 0.0100 | 11 | 3 | 0.0574 (decision_matched) | selection not statistically distinguishable |
| sealed | qwen | epoch 3 | epoch 2 | macro decision agreement % | 0.0033 | 3 | 2 | 1 (decision_matched) | selection not statistically distinguishable |
| sealed | smollm3 | epoch 1 | epoch 3 | macro json validity % | 0.0100 | 8 | 0 | 0.00781 (json_valid) | distinguishable at 0.05 |
| reverse_auction | gemma | epoch 3 | epoch 2 | macro decision accuracy % | 0.0367 | 31 | 2 | 1.31e-07 (decision_matched) | distinguishable at 0.05 |
| reverse_auction | granite | epoch 3 | epoch 2 | macro decision accuracy % | 0.0133 | 15 | 4 | 0.0192 (decision_matched) | distinguishable at 0.05 |
| reverse_auction | ministral | epoch 3 | epoch 2 | macro decision accuracy % | 0.0300 | 23 | 0 | 2.38e-07 (decision_matched) | distinguishable at 0.05 |
| reverse_auction | phi | epoch 3 | epoch 2 | macro decision accuracy % | 0.0433 | 33 | 1 | 4.07e-09 (decision_matched) | distinguishable at 0.05 |
| reverse_auction | qwen | epoch 3 | epoch 2 | offer exact-match % | 4.5500 | 4693 | 2577 | 8.8e-138 (offer_exact) | distinguishable at 0.05 |
| reverse_auction | smollm3 | epoch 3 | epoch 2 | macro decision accuracy % | 0.0167 | 15 | 2 | 0.00235 (decision_matched) | distinguishable at 0.05 |
| supervisor | gemma | epoch 3 | epoch 2 | selection accuracy % | 0.9700 | 60 | 17 | 8.91e-07 (correct) | distinguishable at 0.05 |
| supervisor | granite | epoch 3 | epoch 2 | selection accuracy % | 1.2800 | 94 | 37 | 6.72e-07 (correct) | distinguishable at 0.05 |
| supervisor | ministral | epoch 3 | epoch 2 | selection accuracy % | 0.9400 | 66 | 24 | 1.09e-05 (correct) | distinguishable at 0.05 |
| supervisor | phi | epoch 3 | epoch 2 | selection accuracy % | 1.3900 | 97 | 35 | 6.36e-08 (correct) | distinguishable at 0.05 |
| supervisor | qwen | epoch 3 | epoch 2 | selection accuracy % | 0.5100 | 43 | 20 | 0.00515 (correct) | distinguishable at 0.05 |
| supervisor | smollm3 | epoch 3 | epoch 2 | selection accuracy % | 1.0600 | 81 | 34 | 1.38e-05 (correct) | distinguishable at 0.05 |
