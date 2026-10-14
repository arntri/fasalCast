# Limitations

What this project does and does not show. Written to be read before the results.

## Evidence base

- **One crop, two adjacent markets.** The forecasting results are onion at Lasalgaon
  and Lasalgaon-Vinchur. The two series are highly correlated, so the 750 weekly folds
  are far fewer than 750 independent trials.
- **Few events.** 35 shortage events and 6 spoilage events. Differences in flag
  precision/recall are sensitive to one or two events.
- **No confidence intervals** are reported on any metric.

## Models

- **The headline model does not beat a naive baseline on price.** LightGBM improves on
  seasonal-naive by 73% (price MAE), but a simple persistence baseline (next week = last
  week) reaches MAE 190 vs LightGBM's 250. LightGBM's genuine edge is the shortage
  flag (F1 0.55 vs 0.42).
- **`lgbm_residual` was added post hoc**, after seeing the persistence result. It is
  reported for transparency and did not close the gap (198 vs 190).
- **Spoilage flags are untrainable at this scale** (6 events). A quantile head fixes
  the price half of the compound flag but not the arrivals half.
- **Hyperparameters are fixed, not tuned.** This avoids overfitting to test folds, but
  it also means the models may be under-optimised.

## Stress index

- **Weights are a-priori judgement**, not estimated. Sensitivity analysis (vary weights,
  check episode stability) is planned but not done.
- **Episodes are a proxy**, defined mechanically as weekly stress above its trailing p95
  for 3 or more weeks. They are not a ground-truth record of food crises.
- **No early-warning evaluation yet.** Episode recall, median lead time and precision
  (`warning_eval.py`) are designed in [ARCHITECTURE.md](ARCHITECTURE.md) but not built.
  The index describes episodes; it is not yet shown to warn ahead of them.
- **Basket weights are approximate** staple-expenditure shares, not official CPI weights.

## Data

- **Backfill is about 10% complete** (about 1,037 of 10,564 planned chunks), limited by
  the API's 40 requests/hour/key cap and key expiry. The district basket therefore scores
  zero district-days: only wheat, rice and onion are present, below the 60% weight guard.
- **Coverage is uneven by state.** Bihar and Karnataka were dropped, and Odisha and
  Jharkhand failed the qualification probe. The states with the weakest market data are
  often the most food-insecure, so results describe monitorable states, not the country.
- **Red gram is deprioritised** after pathologically slow fetches.
- **Weather** is a single location per market from a reanalysis archive, used lagged only.

## Interpretation

Market stress is a proxy for deteriorating food access, not a measurement of hunger.
