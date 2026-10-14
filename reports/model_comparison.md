# Model comparison — weekly walk-forward

Harness restricted to pairs marked `VIABLE_FOR_MODELING` by `validate.py` (markets **169** and **4750**, Onion) BEFORE any model comparison: baseline-gating is only meaningful when every model faces an identical pair/fold population, and DROP pairs failed data-quality gates so scoring anything on them is unreliable. Each fold scores all models on the intersection of the instances they all predicted, so `seasonal_naive` and `lgbm` are compared on exactly the same folds. The earlier all-markets figures (906 folds, price MAE≈818) are **superseded** by the restricted-set baseline below.

Folds: **750** &nbsp;|&nbsp; spoilage events: **6** &nbsp;|&nbsp; shortage events: **35**

| model | price MAE | price sMAPE | arrivals MAE | arrivals sMAPE | spoilage P/R/F1 | shortage P/R/F1 |
|---|--:|--:|--:|--:|:--:|:--:|
| `seasonal_naive` | 935 | 57.1% | 38,670 | 65.8% | 0.00 / 0.00 / nan | 0.05 / 0.03 / 0.04 |
| `persistence` | 190 | 10.9% | 20,183 | 42.6% | 0.17 / 0.17 / 0.17 | 0.48 / 0.37 / 0.42 |
| `lgbm` | 250 | 14.5% | 20,067 | 44.9% | nan / 0.00 / nan | 0.63 / 0.49 / 0.55 |
| `lgbm_residual` | 198 | 12.2% | 19,965 | 44.8% | nan / 0.00 / nan | 0.53 / 0.46 / 0.49 |
| `lgbm_weather` | 254 | 14.8% | 20,003 | 44.6% | nan / 0.00 / nan | 0.58 / 0.43 / 0.49 |
| `lgbm_quantile_flags` | — | — | — | — | 0.00 / 0.00 / nan | n/a |

*`—` = regression MAE/sMAPE not meaningful for the quantile-flag head (its forecasts are intentionally biased to the flag's tail); `n/a` shortage = spoilage-specific operating point, see below.*

## Improvement vs baseline (seasonal_naive)

Negative Δ = lower error = better. Verdict is derived from the numbers, not asserted.

| model | Δ price MAE | Δ arrivals MAE | Δ shortage F1 | verdict |
|---|--:|--:|--:|:--|
| `lgbm` | -685 | -18,602 | +0.51 | **beats baseline** on both regression targets (price MAE -73%, arrivals MAE -48%; shortage F1 Δ +0.51) |
| `lgbm_residual` | -736 | -18,705 | +0.46 | **beats baseline** on both regression targets (price MAE -79%, arrivals MAE -48%; shortage F1 Δ +0.46) |
| `lgbm_weather` | -681 | -18,666 | +0.46 | **beats baseline** on both regression targets (price MAE -73%, arrivals MAE -48%; shortage F1 Δ +0.46) |

## Improvement vs baseline (persistence)

Negative Δ = lower error = better. Verdict is derived from the numbers, not asserted.

| model | Δ price MAE | Δ arrivals MAE | Δ shortage F1 | verdict |
|---|--:|--:|--:|:--|
| `lgbm` | +60 | -116 | +0.13 | mixed: beats baseline on one regression target (price MAE +31%, arrivals MAE -1%; shortage F1 Δ +0.13) |
| `lgbm_residual` | +9 | -218 | +0.07 | mixed: beats baseline on one regression target (price MAE +4%, arrivals MAE -1%; shortage F1 Δ +0.07) |
| `lgbm_weather` | +64 | -180 | +0.07 | mixed: beats baseline on one regression target (price MAE +34%, arrivals MAE -1%; shortage F1 Δ +0.07) |

## Does lagged weather help? (lgbm_weather vs lgbm)

**Not materially.** Trailing 7/14/28d rain sums, tmax/tmin means and days-since-heavy-rain move the numbers within noise (price MAE +4, arrivals MAE -64, shortage F1 -0.06 vs plain `lgbm`). For onion at a weekly horizon the price/arrival autoregressive features already capture the seasonal signal; recent local weather adds little. Reported as found — no result is discarded for being unflattering.


## Flag-focused operating point: `lgbm_quantile_flags`

The mean regressor minimizes MAE, so it predicts near the centre of the trailing distribution and **never reaches the double tail** the SPOILAGE flag asks for (arrivals > p90 AND price < p10). This model replaces the two mean regressors with two LightGBM **quantile** regressors — arrivals at α=0.90, price at α=0.10 — then thresholds those tail forecasts into flags with the *same* `flags()` rule. Its regression MAE/sMAPE are omitted (`—`): a quantile forecast is deliberately biased and a good MAE would be the wrong thing to want.

- **α is not tuned on test folds.** α_arr=0.90 / α_prc=0.10 are the flag's *own* defining deciles (the 0.90/0.10 `trailing_thresholds()` uses) — there is no free hyperparameter to overfit.
- **Frozen validation region, explicit boundary.** Whether this head reaches the tail was only ever inspected on the earliest **60%** of folds — cutoffs **≤ 2018-09-12** (558 folds). Every fold *after* 2018-09-12 is untouched test region that never informed α. Boundary printed by `evaluate.py`, not hand-set.
- **Spoilage flag, same 6 events, same folds:** mean `lgbm` P/R/F1 = — / 0.00 / — → quantile head = 0.00 / 0.00 / —. On the reported test region the head does **not** recover spoilage detection: it reaches far enough to *fire* flags (unlike the mean regressor, which fires none), but catches **0 of 6** real events — every firing is a false positive (precision 0.00). The tail-aware forecast fixes the price half of the flag, but the arrivals spike is near-unpredictable a week ahead, so the compound AND still never lands on a true event (see `reports/spoilage_diagnostics.md`). Reported as found.

*MAE/sMAPE lower is better. Flags: precision/recall/F1, higher is better. Numbers are produced by `python src/evaluate.py`; none are hand-entered.*
