# Figure and table captions

Captions for the charts in `evaluation/<mode>/` and the token summary table. Each dot-plot caption
covers the validation and the test chart of one metric, which share their axis and zoom ranges.

Shared definitions used below:
- **Dot plots.** Each row is one model. Gray diamonds are the base model; orange, green and blue circles are
  epochs 1, 2 and 3. The black ring marks the epoch selected on validation. The dashed line is the
  rule-based ground truth (100 % agreement or accuracy, 0 deviation). The strip beside the main plot
  magnifies the shaded range on the main plot; the data scale is unchanged. On the test chart, the base
  model and the epoch selected on validation are one-shot test reads; the other epochs are descriptive
  only and were not used for selection.
- **Macro.** Sealed-bid and reverse-auction values are, as the evaluation README defines them, "macro
  averages over the three stages (material, tailoring, logistics)". Supervisor values are means over all
  supervisor rows.

## Sealed-bid

**Decision agreement** (`sealed/validation/decision_agreement`, `sealed/test/decision_agreement`).
Macro decision agreement (%) of each model with the rule-based decisions, on validation and on test, for
the base model and epochs 1 to 3. The black ring marks the epoch selected on validation; the dashed line is
the rule-based ground truth at 100 %. The dotted line is the Always-BID baseline: the macro decision
agreement of a model that answers BID on every row of the split. The strip magnifies the shaded range
99.85 to 100.00 %; the data scale is unchanged. On the test chart, the base model and the epoch selected
on validation are one-shot test reads and the other epochs are descriptive only. On validation, the epoch
selected for Gemma, Granite, Phi and Qwen is not significantly different from the runner-up (exact
McNemar p >= 0.05: p = 0.69, 0.109, 0.0574 and 1).

**Price deviation** (`sealed/validation/price_deviation`, `sealed/test/price_deviation`).
Macro absolute price deviation (%) of the offered price from the rule-based price, on validation and on
test, for the base model and epochs 1 to 3. The black ring marks the epoch selected on validation; the
dashed line is the rule-based ground truth at 0. The strip magnifies the shaded range 0 to 1.5 %; the
data scale is unchanged. On the test chart, the base model and the epoch selected on validation are
one-shot test reads and the other epochs are descriptive only. The Phi base value covers the material
stage only: its tailoring and logistics rows carry no price value, on validation and on test.

**Completion-time deviation** (`sealed/validation/completion_time_deviation`,
`sealed/test/completion_time_deviation`). Macro absolute deviation of the estimated completion time from
the rule-based value, in days, on validation and on test, for the base model and epochs 1 to 3. The black
ring marks the epoch selected on validation; the dashed line is the rule-based ground truth at 0. The strip
magnifies the shaded range 0 to 0.0008 days; the data scale is unchanged. The values carry 4 decimals, so
the epoch differences inside the strip are at the precision limit of the stored values. On the test chart,
the base model and the epoch selected on validation are one-shot test reads and the other epochs are
descriptive only. The Phi base value covers the material stage only: its tailoring and logistics rows
carry no completion-time value, on validation and on test.

**Inference latency** (`sealed/latency`). Mean latency per decision in seconds on the whole test split,
one request at a time (batch size 1) on one GCD of an AMD MI250X, with warm-up requests excluded (the
first 3 requests of each shard and the first after a resume). Gray bars are the base model; coloured bars
are the fine-tuned model at the epoch selected on validation, timed with its LoRA adapter attached and not
merged. Whiskers show one standard deviation across requests. The wide whisker of SmolLM3
base comes from its spread in output length: 54.8 % of its decisions run to the 200-token cap. The wide
whiskers of Granite and Phi base come from a few requests whose first token took more than 1 s.

**Radar** (`sealed/radar`). Test scores of three fine-tuned models, each at the epoch selected on
validation, on five axes scored from 0 to 10, with outward better on every axis. Each axis is
scaled to the worst of the twelve arms of that mode (rounded), so scores compare within one chart and
not across modes. Decision agreement
scores 10 x (value - floor) / (100 - floor). Price deviation, completion-time deviation, latency and
output tokens are inverted, scoring 10 x (1 - value / reference). The floor and references are the worst
value among all twelve test arms of the mode (six models, base and selected), rounded to a clean number:
floor 35 %, price deviation 350 %, completion-time deviation 4 days, latency 8.5 s, output tokens 150. Base
arms set the scale only and are not drawn. The three plotted models were ranked by the mean of their
validation scores on accuracy, price accuracy and time accuracy; ties were broken by
accuracy score, then alphabetically. The validation references for the ranking are computed the same way
from the twelve validation arms: floor 35 %, price deviation 350 %, completion-time deviation 4 days.
Latency and output tokens come from the whole test split, one request at a time. Polygons that coincide at
the rim on the quality axes do so in the data.
The other three models are in tables/radar_values.md.

## Reverse auction

**Decision agreement** (`reverse_auction/validation/decision_agreement`,
`reverse_auction/test/decision_agreement`). Macro decision agreement (%) of each model with the rule-based
decisions, on validation and on test, for the base model and epochs 1 to 3. The black ring marks the epoch
selected on validation; the dashed line is the rule-based ground truth at 100 %. The dotted line is the
Always-BID baseline: the macro decision agreement of a model that answers BID on every row of the split.
The strip magnifies the shaded range 99.4 to 100.0 %; the data scale is unchanged. On the test chart, the
base model and the epoch selected on validation are one-shot test reads and the other epochs are
descriptive only.

**Price deviation** (`reverse_auction/validation/price_deviation`, `reverse_auction/test/price_deviation`).
Macro absolute price deviation (%) of the offered price from the rule-based price, on validation and on
test, for the base model and epochs 1 to 3. The black ring marks the epoch selected on validation; the
dashed line is the rule-based ground truth at 0. The strip magnifies the shaded range 0 to 0.6 %; the
data scale is unchanged. On the test chart, the base model and the epoch selected on validation are
one-shot test reads and the other epochs are descriptive only.

**Completion-time deviation** (`reverse_auction/validation/completion_time_deviation`,
`reverse_auction/test/completion_time_deviation`). Macro absolute deviation of the estimated completion
time from the rule-based value, in days, on validation and on test, for the base model and epochs 1 to 3.
The black ring marks the epoch selected on validation; the dashed line is the rule-based ground truth at 0.
The strip magnifies the shaded range 0 to 0.04 days; the data scale is unchanged. The values carry 4
decimals, so the epoch differences inside the strip are at the precision limit of the stored values. On
the test chart, the base model and the epoch selected on validation are one-shot test reads and the other
epochs are descriptive only.

**Inference latency** (`reverse_auction/latency`). Mean latency per decision in seconds on the whole test
split, one request at a time (batch size 1) on one GCD of an AMD MI250X, with warm-up requests excluded
(the first 3 requests of each shard and the first after a resume). Gray bars are the base model; coloured
bars are the fine-tuned model at the epoch selected on validation, timed with its LoRA adapter attached
and not merged. Whiskers show one standard deviation across requests. The whisker of SmolLM3 base comes
from its spread in output length and from a few requests with a slow first token. The wide whiskers of
Granite and Phi base come from a few requests whose first token took more than 1 s.

**Radar** (`reverse_auction/radar`). Test scores of three fine-tuned models, each at the epoch selected on
validation, on five axes scored from 0 to 10, with outward better on every axis. Each axis is
scaled to the worst of the twelve arms of that mode (rounded), so scores compare within one chart and
not across modes. Decision agreement
scores 10 x (value - floor) / (100 - floor). Price deviation, completion-time deviation, latency and
output tokens are inverted, scoring 10 x (1 - value / reference). The floor and references are the worst
value among all twelve test arms of the mode (six models, base and selected), rounded to a clean number:
floor 35 %, price deviation 25 %, completion-time deviation 5 days, latency 8.5 s, output tokens 100. Base
arms set the scale only and are not drawn. The three plotted models were ranked by the mean of their
validation scores on accuracy, price accuracy and time accuracy; ties were broken by
accuracy score, then alphabetically. The validation references for the ranking are computed the same way
from the twelve validation arms: floor 35 %, price deviation 25 %, completion-time deviation 5 days.
Latency and output tokens come from the whole test split, one request at a time. Polygons that coincide at
the rim on the quality axes do so in the data.
The other three models are in tables/radar_values.md.

## Supervisor

**Selection accuracy** (`supervisor/validation/selection_accuracy`, `supervisor/test/selection_accuracy`).
Selection accuracy (%): the share of decisions in which the model picks the optimal plan, on validation and
on test, for the base model and epochs 1 to 3. The black ring marks the epoch selected on validation; the
dashed line is the rule-based ground truth at 100 %. The strip magnifies the shaded range 92 to 100 %; the
data scale is unchanged. On the test chart, the base model and the epoch selected on validation are
one-shot test reads and the other epochs are descriptive only.

**Cost deviation** (`supervisor/validation/cost_deviation`, `supervisor/test/cost_deviation`). Mean
absolute deviation (%) of the chosen plan's total cost from the optimal plan's, on validation and on test,
for the base model and epochs 1 to 3. The black ring marks the epoch selected on validation; the dashed
line is the rule-based ground truth at 0. The strip magnifies the shaded range 0 to 0.20 %; the data scale
is unchanged. On the test chart, the base model and the epoch selected on validation are one-shot test
reads and the other epochs are descriptive only.

**Completion-time deviation** (`supervisor/validation/completion_time_deviation`,
`supervisor/test/completion_time_deviation`). Mean absolute deviation of the chosen plan's total
completion time from the optimal plan's, in percent, on validation and on test, for the base model and
epochs 1 to 3. The black ring marks the epoch selected on validation; the dashed line is the rule-based
ground truth at 0. The strip magnifies the shaded range 0 to 1.00 %; the data scale is unchanged. On the
test chart, the base model and the epoch selected on validation are one-shot test reads and the other
epochs are descriptive only.

**Inference latency** (`supervisor/latency`). Mean latency per decision in seconds on the whole test
split, one request at a time (batch size 1) on one GCD of an AMD MI250X, with warm-up requests excluded
(the first 3 requests of each shard and the first after a resume). Gray bars are the base model; coloured
bars are the fine-tuned model at the epoch selected on validation, timed with its LoRA adapter attached
and not merged. Whiskers show one standard deviation across requests.

**Radar** (`supervisor/radar`). Test scores of three fine-tuned models, each at the epoch selected on
validation, on five axes scored from 0 to 10, with outward better on every axis. Each axis is
scaled to the worst of the twelve arms of that mode (rounded), so scores compare within one chart and
not across modes. Selection accuracy
scores 10 x (value - floor) / (100 - floor). Cost deviation, completion-time deviation, latency and output
tokens are inverted, scoring 10 x (1 - value / reference). The floor and references are the worst value
among all twelve test arms of the mode (six models, base and selected), rounded to a clean number: floor
10 %, cost deviation 7 %, completion-time deviation 35 %, latency 3 s, output tokens 50. Base arms set the
scale only and are not drawn. The three plotted models were ranked by the mean of their validation scores
on accuracy, cost accuracy and time accuracy; ties were broken by accuracy score, then
alphabetically. The validation references for the ranking are computed the same way from the twelve
validation arms: floor 10 %, cost deviation 7 %, completion-time deviation 35 %. Latency and output tokens
come from the whole test split, one request at a time. Polygons that coincide at the rim on the quality
axes do so in the data.
The other three models are in tables/radar_values.md.

## Token summary table

**Tokens per decision** (`tables/tokens_summary.md`, `tables/tokens_summary.tex`). Mean input, output and
total tokens per decision on the whole test split, one request at a time (batch size 1), for the base
model and the fine-tuned model at the epoch selected on validation. The input prompt is identical for both
arms. Output tokens are counted from the generated token ids and include the stop token; output is capped
at 200 new tokens, which the Sealed-bid SmolLM3 base model reaches on 54.8 % of decisions and the Reverse
auction SmolLM3 base model on 3.8 %. Values are rounded independently, so a total can differ from input
plus output by 0.1.
