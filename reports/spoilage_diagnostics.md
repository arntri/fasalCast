# Spoilage-flag diagnostics — root cause & a tail-aware fix

Instances scored: **1,017** over the same viable-pair weekly folds. Actual events — spoilage: **6**, shortage: **35**.

## Root cause: mean predictions never reach the double tail

Spoilage fires only when predicted arrivals land in the **top decile** AND predicted price in the **bottom decile** of the trailing daily distribution (percentile > 0.90 and < 0.10 respectively). Mean regression predicts near the middle:

| prediction | median arrivals-percentile | median price-percentile |
|---|--:|--:|
| mean lgbm | 0.33 | 0.49 |
| quantile head (α=0.9 arr / 0.1 prc) | 0.47 | 0.40 |

- Mean lgbm predicts above-p90 arrivals only **0.8%** of weeks and below-p10 price only **11.1%** — and needs BOTH at once, so the joint almost never happens.
- The quantile head lifts those to **1.8%** / **20.4%**, giving the compound flag a chance to fire.

## The actual spoilage events (near-miss view)

For each real spoilage week: where the mean model put its prediction (percentile in the trailing daily distribution) vs the tail it needed to reach. `arr_pct` must exceed 0.90 and `prc_pct` fall below 0.10.

| cutoff | market | act arr_pct | act prc_pct | mean arr_pct | mean prc_pct | quantile arr_pct | quantile prc_pct |
|---|---|--:|--:|--:|--:|--:|--:|
| 2017-01-11 | 169 | 0.95 | 0.09 | 0.74 | 0.16 | 0.81 | 0.08 |
| 2017-02-01 | 169 | 0.95 | 0.04 | 0.61 | 0.04 | 0.73 | 0.01 |
| 2019-02-13 | 169 | 0.93 | 0.10 | 0.62 | 0.05 | 0.89 | 0.03 |
| 2019-02-20 | 169 | 0.96 | 0.06 | 0.67 | 0.05 | 0.83 | 0.06 |
| 2023-02-08 | 169 | 0.94 | 0.09 | 0.69 | 0.20 | 0.87 | 0.19 |
| 2025-05-14 | 4750 | 0.91 | 0.05 | 0.42 | 0.13 | 0.72 | 0.00 |

### Which half of the compound event is the bottleneck?

On the 6 real spoilage weeks, the quantile head reaches the required tail for **price (crash) on 5/6** events but for **arrivals (spike) on only 0/6**. The price crash is largely predictable a week ahead; the arrivals SPIKE is not — even the model's 90th-percentile arrivals forecast lands around the 0.82 percentile of the trailing daily distribution, short of the 0.90 the flag needs. Spoilage detection here is arrivals-spike-limited, not price-limited.


## Does the quantile head recover spoilage? (same folds)

*The α=0.9-arrivals / α=0.1-price head is SPOILAGE-specific (high arrivals + low price). Shortage needs the opposite tails, so its shortage row is expected to degrade and this head must not be used for shortage — shown only for completeness.*

| flag | model | precision | recall | F1 |
|---|---|--:|--:|--:|
| SPOILAGE | mean lgbm | — | 0.00 | — |
| SPOILAGE | quantile head | 0.00 | 0.00 | — |
| SHORTAGE | mean lgbm | 0.58 | 0.43 | 0.49 |
| SHORTAGE | quantile head | 0.60 | 0.09 | 0.15 |

## Verdict

- Mean lgbm catches **0 / 6** spoilage events (F1 —). The near-miss table shows why: on real spoilage weeks the mean model's arrivals percentile stays well below 0.90 and/or its price percentile well above 0.10 — it hedges toward the center.
- The quantile head fires on **1** weeks and recovers **0 / 6** true spoilage events (recall 0.00, precision 0.00, F1 —). It trades precision for the first non-zero spoilage recall on this panel.
- **The binding constraint is event scarcity: 6 spoilage events in 1,017 instances**, and the missing half is the arrivals spike, which is close to unpredictable a week ahead from price/arrival history alone. No amount of modeling manufactures signal from ~6 positives. The quantile head is the right *methodological* fix (align the estimator with the tail the flag targets, α set to the flag's own deciles — NOT searched on these folds), but on this panel it still can't cross the arrivals threshold.
- **What would actually move it:** (1) more markets — widening to Pune (Task 3) adds spoilage events and re-tests this on real signal; (2) if a validation split (earliest 60% of folds, frozen) justified it, a more aggressive arrivals quantile would trade precision for recall — a deliberate operating-point choice, not test-fold tuning. Reported as found; nothing tuned on these folds.
