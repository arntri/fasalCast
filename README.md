# Mandi Early-Warning System (FasalCast)

Forecasts weekly arrival volumes and modal prices for volatile staple crops
(onion/tomato/potato) in Indian wholesale markets (APMC mandis), and flags two
risks a week ahead: **spoilage/waste risk** (volume spike + price crash) and
**shortage/access risk** (near-zero arrivals + price spike).

Framed as an early-warning system for supply-demand imbalance — the headline
metric is flag precision/recall at lead time, not forecast RMSE.

## Design principles (enforced in code, not by memory)
- **No leakage.** Labels and features use trailing information only; weather
  features are lagged/recent actuals, never future or same-(target-)week.
  `features.py`'s self-check asserts `max(feature-source date − cutoff) ≤ 0`
  over every row.
- **Temporal validation only.** Walk-forward, expanding-window weekly folds
  (`src/evaluate.py`). Random splits are structurally impossible here.
- **Baseline first, baseline-gated.** Seasonal-naive is model #1. Any ML model
  must beat it **on the same folds** or it doesn't ship. The harness is
  restricted to data-quality-viable pairs and scores every model on an
  identical fold/instance population (see below).
- **No fabrication.** Every number in this README and in
  `reports/model_comparison.md` is produced by running the code; none is
  hand-entered or extrapolated.

## Results (real, walk-forward, `python src/evaluate.py`)

Restricted to pairs that pass `validate.py` (Onion @ **Lasalgaon 169** and
**Lasalgaon-Vinchur 4750**), scored on **750 identical weekly folds**
(2008→2025). Spoilage events: 6; shortage events: 35.

| model | price MAE | price sMAPE | arrivals MAE | arrivals sMAPE | spoilage P/R/F1 | shortage P/R/F1 |
|---|--:|--:|--:|--:|:--:|:--:|
| `seasonal_naive` (baseline) | 935 | 57.1% | 38,670 | 65.8% | 0.00/0.00/— | 0.05/0.03/0.04 |
| `persistence` (naive baseline) | **190** | **10.9%** | 20,183 | **42.6%** | 0.17/0.17/0.17 | 0.48/0.37/0.42 |
| `lgbm_residual` (post-hoc) | 198 | 12.2% | **19,965** | 44.8% | —/0.00/— | 0.53/0.46/0.49 |
| `lgbm` | 250 | 14.5% | 20,067 | 44.9% | —/0.00/— | **0.63/0.49/0.55** |
| `lgbm_weather` | 254 | 14.8% | 20,003 | 44.6% | —/0.00/— | 0.58/0.43/0.49 |
| `lgbm_quantile_flags` | — | — | — | — | 0.00/0.00/— | n/a |

(`lgbm_quantile_flags` is a spoilage-only operating point — two LightGBM
**quantile** regressors, not mean regressors — so its regression MAE/sMAPE are
deliberately biased and shown `—`, and it does not target shortage (`n/a`). See
the spoilage bullet below and `reports/model_comparison.md`.)

- **LightGBM beats the seasonal-naive baseline decisively** (price MAE −73%,
  arrivals MAE −48%, shortage F1 0.04 → **0.55**), but seasonal-naive is a weak
  baseline. Against the stronger **persistence** baseline (next week = last
  week) the picture is mixed, and reported as found: LightGBM **loses on price**
  (MAE 250 vs 190), is **level on arrivals** (20,067 vs 20,183, −1%), and wins
  only on the shortage flag (F1 0.55 vs 0.42, on 35 events). Persistence also
  catches 1 of 6 spoilage events (F1 0.17) where the mean LightGBM catches none.
  The honest claim is therefore *flag-quality* gains, not price-forecast gains.
  A **post-hoc** attempt to close the gap (`lgbm_residual`: LightGBM predicts the
  change from the persistence anchor; params unchanged, nothing tuned, added
  *after* seeing the persistence result) gets price MAE to 198 — still not below
  persistence's 190 — and arrivals 19,965 (−1%). It is disclosed as exploratory,
  not a headline model.
- **Weather does not materially help.** Lagged rain/temperature features move
  the numbers within noise (price MAE +4, arrivals MAE −64, shortage F1 −0.06
  vs plain `lgbm`). For onion at a weekly horizon the price/arrival
  autoregressive features already carry the seasonal signal. Reported as found.
- **Spoilage flags are not learnable here — and we know exactly why.** Only 6
  spoilage events occur in 750 folds. The two mean regressors (`lgbm`,
  `lgbm_weather`) predict no spoilage flag at all; the tail-aware
  `lgbm_quantile_flags` head *does* fire spoilage flags but catches **0 of 6**
  real events (precision/recall 0.00/0.00) — reported honestly as a
  panel-scored model, not a discarded prototype.
  `src/diagnostics.py` (→ `reports/spoilage_diagnostics.md`) traces the cause:
  spoilage is a compound tail (arrivals in the top decile AND price in the
  bottom decile), and a mean regressor predicts near the centre (median
  arrivals-percentile 0.33, price 0.49), so it never satisfies the double-AND.
  The tail-aware fix — a LightGBM **quantile head** (arrivals α=0.90,
  price α=0.10, α set to the flag's own deciles, not tuned) — fixes the **price
  half of the flag (crash right on 5/6 events) but not the arrivals half (spike
  on 0/6)**, so the compound AND still never lands on a true event and every
  flag it does fire is a false positive. The arrivals spike is close to
  unpredictable a week ahead from price/arrival history. So the binding
  constraints are (1) event scarcity (~6 positives) and (2) an unpredictable
  arrivals spike — the durable fix is **more markets** (Pune widening), not more
  model. Reported as found.

Historical label base rates (`python src/labels.py`, trailing-quantile labels):
Onion@169 spoilage 2.28% / shortage 2.24%; Onion@4750 spoilage 1.13% /
shortage 5.11%.

## Data quality findings (the hard-won ones)
1. **Unlabeled variety rows.** `/prices` returns multiple variety/grade rows
   per day with **no variety column**; `/quantities` returns one market-level
   row per day. They can't be reliably paired, so a volume-weighted price is
   not computable without guessing. Policy (in `clean.py`): arrivals = **sum**
   of quantity rows, modal = **median** of price rows, min/max = true daily
   envelope; `n_price_rows`/`n_qty_rows` kept so the aggregation is auditable.
2. **Tonnes vs quintals (a silent 10× trap).** The API reports `quantity` in
   **metric tonnes** while prices are **Rs/quintal** — verified against the
   agmarknet.gov.in portal header. `clean.py` converts arrivals ×10 to quintals
   so both targets share one basis. `validate.py` also scans for ~10× level
   shifts as a guard.
3. **Per-market coverage verdicts** (`min_coverage 0.60`, `min_rows 500`):
   Onion @ 169 (cov 0.698, OK), 4750 (0.832, OK), 2484/Niphad (0.589, DROP),
   2488/Pimpalgaon-Saykheda (0.584, DROP). Tomato @ 169 (cov **0.098**, DROP):
   tomato is barely traded at an onion market. DROP pairs are excluded from the
   modeling pool entirely — for the baseline as much as for any ML model.

## Pipeline
```
backfill.py -> clean.py -> validate.py -> labels.py -> features.py -> evaluate.py
(CEDA API,    (aggregate   (quality       (risk       (leakage-safe  (walk-forward,
 resumable     unlabeled    report,        flags,       weekly         seasonal_naive
 chunks)       variety      pair           base         supervised     vs lgbm vs
               rows, t/qtl) verdicts)      rates)       table)         lgbm+weather)
weather.py (Open-Meteo archive, no key) feeds lagged weather features into features.py
pairs.py   (single source of truth for VIABLE_FOR_MODELING pairs; shared by features+evaluate)
```

## Run instructions
```bash
pip install -r requirements.txt          # requests, pandas, pyyaml, pyarrow, lightgbm
export CEDA_API_KEY='...'                 # never commit; .env / *.env are gitignored

# 1. Data (resumable; skips completed chunks). --combine-only rebuilds parquets
#    from existing data/raw without re-fetching.
python src/backfill.py --from-year 2006
python src/clean.py
python src/validate.py data/panel_daily.parquet
python src/labels.py

# 2. Weather (free, no key; resumable, cached to data/weather/)
python src/weather.py --from-year 2006

# 3. Features + evaluation (writes reports/model_comparison.md)
python src/features.py                    # -> data/features_weekly.parquet + leakage self-check
python src/evaluate.py                    # seasonal_naive vs lgbm vs lgbm_weather

# 4. Optional: invariant tests + spoilage root-cause diagnostics
python tests/test_invariants.py           # leakage, scale, pairs, weather trailing-only
python src/diagnostics.py                 # -> reports/spoilage_diagnostics.md
```

Data source: CEDA (Ashoka University) Agri Market API — cleaned mirror of
Agmarknet (Ministry of Agriculture & Farmers Welfare). Used under CEDA's terms
(credit: Centre for Economic Data & Analysis, Ashoka University). Weather:
Open-Meteo historical archive.

## Status
- [x] Milestone 1: API integration; real schema handled (envelope unwrap, flat
      geographies, unlabeled variety rows, flaky endpoints → retries +
      resumable chunks + caching).
- [x] Historical backfill (2006→present) + quality report + label base rates.
- [x] Seasonal-naive baseline (walk-forward, restricted to viable pairs).
- [x] LightGBM forecaster beats baseline on identical folds (price MAE −73%,
      arrivals MAE −48%, shortage F1 0.04→0.55).
- [x] Weather features (lagged only) — added and evaluated; found not to help
      onion at a weekly horizon (reported honestly).
- [~] **Data widening (partial).** Tomato/Potato + Pune targets are configured
      in `config.yaml` and weather is already fetched for all six markets; the
      backfill is **resumable**. It fetched Tomato@169 (which validates DROP —
      too sparse) before CEDA returned sustained HTTP 429 rate-limits, which
      blocked the Pune markets (the ones most likely to add viable tomato/potato
      pairs). Re-run `python src/backfill.py --from-year 2006` once the CEDA
      quota resets; new pairs auto-join the pool **only if** `validate.py` marks
      them OK (`pairs.py` reads the verdicts — nothing hardcoded).
