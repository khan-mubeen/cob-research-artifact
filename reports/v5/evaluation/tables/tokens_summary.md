# Tokens per decision (summary)

Mean tokens per decision on the whole test split at batch size 1, for the base model and the selected fine-tuned epoch. The input is the same prompt for both arms.

| Mode | Model | Input (mean) | Output base | Output fine-tuned | Total base | Total fine-tuned |
|---|---|---|---|---|---|---|
| Sealed bid | gemma | 807.9 | 98.5 | 88.6 | 906.3 | 896.5 |
| Sealed bid | granite | 709.0 | 68.9 | 70.0 | 778.0 | 779.0 |
| Sealed bid | ministral | 769.7 | 83.8 | 81.4 | 853.5 | 851.1 |
| Sealed bid | phi | 699.8 | 68.0 | 70.0 | 767.8 | 769.8 |
| Sealed bid | qwen | 769.3 | 87.4 | 87.2 | 856.8 | 856.5 |
| Sealed bid | smollm3 | 747.0 | 148.8 | 70.0 | 895.9 | 817.0 |
| Reverse auction | gemma | 900.6 | 95.3 | 87.0 | 995.9 | 987.6 |
| Reverse auction | granite | 787.1 | 67.8 | 68.5 | 854.9 | 855.5 |
| Reverse auction | ministral | 859.5 | 79.3 | 78.7 | 938.7 | 938.2 |
| Reverse auction | phi | 779.8 | 69.5 | 68.5 | 849.4 | 848.3 |
| Reverse auction | qwen | 854.1 | 85.8 | 82.7 | 939.9 | 936.8 |
| Reverse auction | smollm3 | 825.1 | 78.3 | 68.5 | 903.4 | 893.5 |
| Supervisor | gemma | 1,643.5 | 41.9 | 24.7 | 1,685.5 | 1,668.2 |
| Supervisor | granite | 1,293.6 | 26.1 | 22.9 | 1,319.7 | 1,316.5 |
| Supervisor | ministral | 1,574.7 | 44.1 | 24.5 | 1,618.8 | 1,599.1 |
| Supervisor | phi | 1,285.6 | 26.6 | 22.9 | 1,312.2 | 1,308.5 |
| Supervisor | qwen | 1,530.3 | 47.5 | 25.1 | 1,577.8 | 1,555.4 |
| Supervisor | smollm3 | 1,331.6 | 29.2 | 22.9 | 1,360.8 | 1,354.5 |

Output is capped at 200 new tokens. Arms that hit the cap on at least 1 % of decisions: Sealed bid smollm3 base (54.8 %); Reverse auction smollm3 base (3.8 %).

Values are rounded independently, so a total can differ from input plus output by 0.1.
