# Demo orders

Which real holdout orders to run live, and what each one shows. Every order here
was verified on all five configurations in both bidding modes, from a fresh
baseline each time.

## Run these one at a time, after a reset

The single orders below are **independent demonstrations, not a sequence**. Each
was verified against a freshly reset baseline, so run

```bash
make demo_reset
```

before each one. Running them back to back without a reset carries the previous
order's stock, workload and simulated clock into the next, and the numbers in
this file will not be the numbers on screen.

The one exception is the restock sequence at the bottom, which is deliberately
chained and must be run in its own order.

---

## ORD-34921 — completes everywhere

hoodie x262, cotton, normal, budget 4,318.06, deadline 2026-10-04, to Jyvaskyla

The safe opener: every configuration bids, every configuration finds a chain, in
both modes. 29.6% budget slack, so no configuration is priced out.

| config | sealed chain | EUR | auction chain | EUR |
|---|---|---|---|---|
| rule | M03 > T7 > L06 | 3,039.52 | M03 > T5 > L01 | 3,009.52 |
| qwen-base | M03 > T7 > L08 | 1,986.84 | M03 > T7 > L09 | 2,617.37 |
| qwen-tuned | M03 > T7 > L08 | 3,030.71 | M02 > T1 > L08 | 3,240.98 |
| granite-base | M02 > T5 > L08 | 187.63 | M02 > T7 > L08 | 2,655.28 |
| granite-tuned | M03 > T5 > L06 | 3,142.39 | M03 > T1 > L08 | 3,011.22 |

Both fine-tunes land within a few percent of the rule figure. Both base models
come in low — granite-base at 187.63 against 3,039.52 is the untrained model
inventing a price, which is the point of showing it.

Takes about 6 s on rule, 45-55 s on a model, and 18 s / 90-165 s in the auction.

## ORD-43006 — infeasible everywhere, and it is structural

jacket x1752, leather, normal, budget 129,269.78, deadline 2026-10-27

Infeasible on all five configurations in both modes, and not by a narrow margin:
the order needs **2,102.4 kg of leather** and the entire market holds **1,310 kg**
across six agents. Nine of the fifteen material agents do not carry leather at
all. No amount of budget or deadline fixes it.

Three different routes to the same answer, all worth watching:

- rule, qwen-tuned, granite-tuned: no material agent bids — *no material agent
  bid: 15 agents declined, 9 do not carry the material, 6 not enough stock even
  with a restock*
- qwen-base: the untrained model bids anyway on stock it does not have, and the
  reservation acknowledgment catches it — *every material bid was dropped after
  its agent could not honour the reservation*
- granite-base: bids too, sometimes with a 0.00 offer that never leaves the
  agent, and ends at *no combination of the bids fit both the budget and the
  deadline*

Fast to show: 2 s on rule, 15-19 s on the fine-tunes, up to 132 s on qwen-base.

## ORD-33452 — knife edge in sealed, fits in the auction

skirt x1630, cotton, urgent, budget 14,693.33, deadline 2026-09-05

The rule chain costs 14,684.64 against a 14,693.33 budget: **8.69 EUR of slack,
0.06%**. Anything priced a percent high does not fit — in sealed mode.

| config | sealed | auction |
|---|---|---|
| rule | completed, 14,684.64 | completed, 14,654.64 |
| qwen-tuned | completed, 14,667.32 | completed, 14,634.00 |
| granite-tuned | **infeasible**, its chain came to 14,766.03 | **completed**, 14,654.86 |
| qwen-base | infeasible | infeasible |
| granite-base | infeasible | infeasible |

The auction rescues granite-tuned, and the arithmetic is exact: three stages,
two improving rounds each, five euro a round is 30 EUR off the sealed price. That
is enough to bring granite's chain under the line. Good for a story about
margins, and about what the auction is actually for. Still a poor opening click
if you want everything to succeed.

It is also the first order of the restock sequence, where it runs on the rule
configuration and completes.

## ORD-08905 — completes in sealed, granite-base fails in the auction

skirt x331, polyester, normal, budget 2,510.46, deadline 2026-10-16

| config | sealed | auction |
|---|---|---|
| rule | completed, 1,711.57 | completed, 1,681.57 |
| qwen-base | completed, 1,497.07 | completed, 1,517.96 |
| qwen-tuned | completed, 1,702.83 | completed, 1,686.84 |
| granite-base | completed, 235.86 | **infeasible**, no feasible chain |
| granite-tuned | completed, 1,705.04 | completed, 1,681.68 |

Use this one to show that the mode matters. Untrained granite copies the
announced best bid instead of undercutting it, so its auction prices never come
down and the chain does not fit. Avoid it if the demo is meant to be smooth.

## The restock sequence — run in order, after a reset

`data/demo/restock_sequence.json`, three chained orders on one configuration:

```bash
make demo_reset
python -m src.api submit --file data/demo/restock_sequence.json --config rule
```

M02 starts at 800.0 kg of cotton, drops to 311.0, then to 139.4 which is under
its 160.0 kg threshold and triggers a 1500 kg restock, which lands before the
third order is priced. The numbers are in the file's own notes.

**It only tells the whole story on some configurations**, because the first order
is the knife-edge ORD-33452 and a configuration that cannot complete that one
never depletes M02 twice:

| config | sealed | auction |
|---|---|---|
| rule | restock triggers and lands | restock triggers and lands |
| qwen-tuned | restock triggers and lands | restock triggers and lands |
| granite-tuned | first order infeasible, no restock | restock triggers and lands |
| qwen-base | first order infeasible, no restock | first order infeasible, no restock |
| granite-base | all three infeasible, state untouched | all three infeasible, state untouched |

**Run it on rule or qwen-tuned**, in either mode, and it always works. On
granite-tuned it works in the auction only, for the ORD-33452 reason above.

---

## Two things to know before running any of this

**The rule configuration reproduces exactly in sealed; the auction's tie-break
does not, by design.** Rule chains and costs repeat to the cent across a reset in
sealed mode. In the auction, from round 2 on every surviving agent is required to
offer the same price, so the round is a tie and the winner is whoever's reply
reached the supervisor first. That is the documented rule, and message arrival
order varies run to run: two runs of ORD-34921 tailoring both closed at 2,170.52
with five agents tied, and the winner was T1 one time and T5 the next. **The
price is reproducible, the winning agent among tied bidders is not.**

**Model configurations vary occasionally even in sealed mode.** llama.cpp reuses
cache state between calls, so an untrained model sitting close to a decision
boundary can land either side of it: granite-base gave two different chains for
ORD-34921 across fresh resets, at 186.60 and 187.63. Do not promise an exact
number in advance on a base configuration.

**Verified on:** 29 August 2026, after the standing-bid fix in
`src/supervisor/auction.py`, five configurations, both modes, rule window 2.0 s,
qwen 18.0 s, granite 15.0 s.
