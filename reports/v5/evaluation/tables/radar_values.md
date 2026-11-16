# Radar values

Six fine-tuned models per mode, each at its epoch selected on validation. Scores run from 0 to 10 and outward is better on every axis: accuracy scores 10 x (value - floor) / (100 - floor); price or cost deviation, completion-time deviation, latency and output tokens score 10 x (1 - value / reference). Each split's floor and references are the worst value among all twelve arms of the mode (six models, base and selected), rounded down (floor) or up (references) to a multiple of half a power of ten. The ranking uses the mean of the three validation quality scores, ties broken by the accuracy score, then alphabetically; the three top-ranked models are plotted with their test scores. Latency and output tokens were measured on the test split only.


## Sealed-bid


### validation

decision agreement floor 35 %, price deviation reference 350 %, completion-time deviation reference 4 days

| model | epoch | decision agreement (%) | price deviation (%) | completion-time deviation (days) | decision agreement score | price accuracy score | time accuracy score | mean score | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 3 | 99.9967 | 0.0536 | 0.0000 | 9.999 | 9.998 | 10.000 | 9.999 | 1 |
| Ministral | 3 | 100.0000 | 0.1398 | 0.0000 | 10.000 | 9.996 | 10.000 | 9.999 | 2 |
| Granite | 2 | 99.9967 | 0.1490 | 0.0001 | 9.999 | 9.996 | 10.000 | 9.998 | 3 |
| Phi | 2 | 99.9933 | 0.1812 | 0.0000 | 9.999 | 9.995 | 10.000 | 9.998 | 4 |
| Gemma | 2 | 99.9833 | 0.2831 | 0.0001 | 9.997 | 9.992 | 10.000 | 9.996 | 5 |
| SmolLM3 | 1 | 99.9867 | 0.3366 | 0.0001 | 9.998 | 9.990 | 10.000 | 9.996 | 6 |

### test

decision agreement floor 35 %, price deviation reference 350 %, completion-time deviation reference 4 days, latency reference 8.5 s, output tokens reference 150 tokens

| model | epoch | decision agreement (%) | price deviation (%) | completion-time deviation (days) | latency (s) | output tokens (tokens) | decision agreement score | price accuracy score | time accuracy score | speed score | token efficiency score | plotted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 3 | 99.9967 | 0.0377 | 0.0000 | 5.9770 | 87.2146 | 9.999 | 9.999 | 10.000 | 2.968 | 4.186 | yes |
| Ministral | 3 | 100.0000 | 0.1525 | 0.0000 | 4.9210 | 81.4106 | 10.000 | 9.996 | 10.000 | 4.211 | 4.573 | yes |
| Granite | 2 | 99.9933 | 0.1432 | 0.0001 | 6.7387 | 70.0093 | 9.999 | 9.996 | 10.000 | 2.072 | 5.333 | yes |
| Phi | 2 | 99.9933 | 0.1812 | 0.0001 | 4.3392 | 70.0096 | 9.999 | 9.995 | 10.000 | 4.895 | 5.333 | no |
| Gemma | 2 | 99.9833 | 0.2939 | 0.0001 | 8.4101 | 88.6455 | 9.997 | 9.992 | 10.000 | 0.106 | 4.090 | no |
| SmolLM3 | 1 | 99.9900 | 0.3369 | 0.0002 | 5.5249 | 70.0099 | 9.998 | 9.990 | 9.999 | 3.500 | 5.333 | no |

## Reverse auction


### validation

decision agreement floor 35 %, price deviation reference 25 %, completion-time deviation reference 5 days

| model | epoch | decision agreement (%) | price deviation (%) | completion-time deviation (days) | decision agreement score | price accuracy score | time accuracy score | mean score | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 3 | 100.0000 | 0.0025 | 0.0000 | 10.000 | 9.999 | 10.000 | 10.000 | 1 |
| Phi | 3 | 99.9933 | 0.0104 | 0.0000 | 9.999 | 9.996 | 10.000 | 9.998 | 2 |
| SmolLM3 | 3 | 99.9933 | 0.0150 | 0.0000 | 9.999 | 9.994 | 10.000 | 9.998 | 3 |
| Ministral | 3 | 99.9967 | 0.0195 | 0.0000 | 9.999 | 9.992 | 10.000 | 9.997 | 4 |
| Gemma | 3 | 99.9933 | 0.0233 | 0.0000 | 9.999 | 9.991 | 10.000 | 9.997 | 5 |
| Granite | 3 | 99.9900 | 0.0252 | 0.0000 | 9.998 | 9.990 | 10.000 | 9.996 | 6 |

### test

decision agreement floor 35 %, price deviation reference 25 %, completion-time deviation reference 5 days, latency reference 8.5 s, output tokens reference 100 tokens

| model | epoch | decision agreement (%) | price deviation (%) | completion-time deviation (days) | latency (s) | output tokens (tokens) | decision agreement score | price accuracy score | time accuracy score | speed score | token efficiency score | plotted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 3 | 100.0000 | 0.0029 | 0.0000 | 5.6453 | 82.6859 | 10.000 | 9.999 | 10.000 | 3.358 | 1.731 | yes |
| Phi | 3 | 99.9900 | 0.0314 | 0.0000 | 4.2658 | 68.4797 | 9.998 | 9.987 | 10.000 | 4.981 | 3.152 | yes |
| SmolLM3 | 3 | 99.9933 | 0.0157 | 0.0000 | 5.7638 | 68.4789 | 9.999 | 9.994 | 10.000 | 3.219 | 3.152 | yes |
| Ministral | 3 | 100.0000 | 0.0190 | 0.0000 | 4.7506 | 78.7067 | 10.000 | 9.992 | 10.000 | 4.411 | 2.129 | no |
| Gemma | 3 | 99.9900 | 0.0232 | 0.0000 | 8.3445 | 87.0346 | 9.998 | 9.991 | 10.000 | 0.183 | 1.297 | no |
| Granite | 3 | 99.9900 | 0.0456 | 0.0000 | 6.6082 | 68.4791 | 9.998 | 9.982 | 10.000 | 2.226 | 3.152 | no |

## Supervisor


### validation

selection accuracy floor 10 %, cost deviation reference 7 %, completion-time deviation reference 35 %

| model | epoch | selection accuracy (%) | cost deviation (%) | completion-time deviation (%) | selection accuracy score | cost accuracy score | time accuracy score | mean score | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 3 | 98.8300 | 0.0440 | 0.2299 | 9.870 | 9.937 | 9.934 | 9.914 | 1 |
| Granite | 3 | 98.0000 | 0.0550 | 0.2912 | 9.778 | 9.921 | 9.917 | 9.872 | 2 |
| SmolLM3 | 3 | 97.7600 | 0.0588 | 0.3764 | 9.751 | 9.916 | 9.892 | 9.853 | 3 |
| Ministral | 3 | 97.8200 | 0.0649 | 0.3902 | 9.758 | 9.907 | 9.889 | 9.851 | 4 |
| Phi | 3 | 97.5100 | 0.0652 | 0.3504 | 9.723 | 9.907 | 9.900 | 9.843 | 5 |
| Gemma | 3 | 97.4000 | 0.0791 | 0.3699 | 9.711 | 9.887 | 9.894 | 9.831 | 6 |

### test

selection accuracy floor 10 %, cost deviation reference 7 %, completion-time deviation reference 35 %, latency reference 3 s, output tokens reference 50 tokens

| model | epoch | selection accuracy (%) | cost deviation (%) | completion-time deviation (%) | latency (s) | output tokens (tokens) | selection accuracy score | cost accuracy score | time accuracy score | speed score | token efficiency score | plotted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 3 | 98.6700 | 0.0343 | 0.2057 | 1.7964 | 25.0573 | 9.852 | 9.951 | 9.941 | 4.012 | 4.989 | yes |
| Granite | 3 | 97.9500 | 0.0524 | 0.2973 | 2.2727 | 22.9069 | 9.772 | 9.925 | 9.915 | 2.424 | 5.419 | yes |
| SmolLM3 | 3 | 97.5400 | 0.0573 | 0.3552 | 1.9660 | 22.9069 | 9.727 | 9.918 | 9.899 | 3.447 | 5.419 | yes |
| Ministral | 3 | 97.7200 | 0.0627 | 0.3543 | 1.6284 | 24.4593 | 9.747 | 9.910 | 9.899 | 4.572 | 5.108 | no |
| Phi | 3 | 97.1100 | 0.0751 | 0.3783 | 1.4804 | 22.9069 | 9.679 | 9.893 | 9.892 | 5.065 | 5.419 | no |
| Gemma | 3 | 97.5200 | 0.0670 | 0.3840 | 2.4598 | 24.6577 | 9.724 | 9.904 | 9.890 | 1.801 | 5.068 | no |
