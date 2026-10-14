# FasalCast

**A leakage-safe, walk-forward-validated early-warning pipeline for food-market stress in Indian wholesale mandis.**

FasalCast ingests daily price and arrival records for staple crops from India's APMC
mandis (via the [CEDA](https://ceda.ashoka.edu.in/) Agri Market API, a cleaned mirror
of Agmarknet), turns them into forecasts and risk flags, and builds a continuous
market-stress index with mechanically defined crisis episodes.

> **Status: working research prototype, not a finished product.** The forecasting
> and flag layers are complete and tested on **one crop at two adjacent markets
> (onion, Lasalgaon)**. The nationwide stress index is built and tested, but the
> data backfill that would feed it is only about 10% complete. See
> [Honest limitations](#honest-limitations) before drawing conclusions.

---

## Highlights

- **Leakage control that is machine-checked, not promised.** Every feature row is
  asserted to use only data at or before its cutoff; the stress index is verified by a
  truncation test (recompute on data cut at *T*, every value dated ≤ *T* must be
  identical). 8 invariant tests cover leakage, unit scale, pair eligibility, weather
  lag, the basket coverage guard and stress truncation.
- **Walk-forward only.** Expanding-window weekly folds; random splits are impossible by
  construction. Every model is scored on an identical fold and instance population.
- **Pre-registration.** Stress weights and the episode rule are committed to git
  before they are run on any data, and are never fit to the episodes they are scored on.
- **Found and fixed silent data traps.** The API reports arrivals in **tonnes** but
  prices in **Rs/quintal** (a silent 10x error); `/prices` returns unlabeled variety rows;
  Madhya Pradesh files rice under *Paddy*, not *Rice*. See [docs/DATA_NOTES.md](docs/DATA_NOTES.md).
- **Evidence-driven scope.** Phase-A states were chosen by a pre-registered density
  probe, not by hand. Bihar and Karnataka were dropped for measured structural reasons;
  Chhattisgarh and Rajasthan qualified. See [reports/state_qualification.md](reports/state_qualification.md).
- **Negative results reported as found**, including a stronger baseline that beats the
  headline model on price (below).

## Results

Onion at **Lasalgaon (169)** and **Lasalgaon-Vinchur (4750)**, **750 identical weekly
folds, 2008 to 2025**, 6 spoilage events, 35 shortage events. All numbers come from
`python src/evaluate.py` and are written to [reports/model_comparison.md](reports/model_comparison.md).

| model | price MAE | arrivals MAE | shortage P / R / F1 |
|---|--:|--:|:--:|
| `seasonal_naive` | 935 | 38,670 | 0.05 / 0.03 / 0.04 |
| `persistence` (next week = last week) | **190** | 20,183 | 0.48 / 0.37 / 0.42 |
| `lgbm` | 250 | 20,067 | **0.63 / 0.49 / 0.55** |
| `lgbm_weather` | 254 | 20,003 | 0.58 / 0.43 / 0.49 |
| `lgbm_residual` (post-hoc, exploratory) | 198 | **19,965** | 0.53 / 0.46 / 0.49 |

**How to read this honestly**

- LightGBM beats the *seasonal-naive* baseline by a wide margin (price MAE -73%,
  arrivals MAE -48%, shortage F1 0.04 to 0.55), but seasonal-naive is a weak baseline.
- Against the stronger **persistence** baseline, LightGBM **loses on price**
  (250 vs 190), is **level on arrivals** (-1%), and **wins only on the shortage flag**
  (F1 0.55 vs 0.42, on just 35 events).
- A post-hoc attempt to close the price gap (`lgbm_residual`, added *after* seeing the
  persistence result, nothing tuned) still does not beat persistence. It is disclosed as
  exploratory, not as a headline model.
- Lagged weather does not help onion at a weekly horizon.

**Spoilage flags are untrainable at this scale, and the repo explains why.** Only 6
spoilage events occur in 750 folds. A mean regressor predicts near the centre of the
distribution (median arrivals percentile 0.33, price 0.49), so the compound
"arrivals in the top decile AND price in the bottom decile" rule never fires. A
quantile-regression head (alpha set to the flag's own deciles, not tuned) repairs the
price half (crash caught on 5 of 6 events) but not the arrivals half (0 of 6).
Details: [reports/spoilage_diagnostics.md](reports/spoilage_diagnostics.md). This finding
motivated the move from discrete event flags to a continuous stress index.

## Stress index and episodes

`src/stress.py` builds a per-(market, crop) stress score from four trailing-only
components (price z-score, seasonal arrival deficit, momentum, cross-market
dispersion) with fixed a-priori weights. `src/episodes.py` defines a crisis episode as
weekly stress above its trailing p95 for 3 or more consecutive weeks. `src/basket.py`
aggregates crop groups into a district index and refuses to score a district-day unless
at least 60% of the basket weight is present.

First run on the data cached so far: **38 series, 81 market-crop episodes**. As a
descriptive sanity check, the Lasalgaon onion episodes fall on well-known spikes
(Aug to Oct 2013, Jul to Sep 2015, Nov to Dec 2017, Nov 2019). The rule was not tuned
to them. The district basket currently scores **zero** district-days because only 3 of
8 crop groups are backfilled; the coverage guard correctly refuses a thin index.

## Honest limitations

- **Narrow evidence base.** All forecast results are for onion at two *adjacent*
  Lasalgaon markets, so the 750 folds are not 750 independent trials, and the shortage
  result rests on 35 events.
- **Backfill is about 10% done.** The planned phase-A backfill is 10,564 chunks over 278
  market-crop pairs in 6 states; about 1,037 chunks are on disk. The API's hard limit
  of 40 requests per hour per key and its 504 storms are the bottleneck.
- **No early-warning headline yet.** Episode recall, lead time and precision
  (`warning_eval.py`) are designed but not built, so there is no claim that the stress
  index warns ahead of crises. It currently *describes* episodes.
- **Headline model does not beat a naive baseline on price.** See Results.
- **Red gram is deprioritised** after pathologically slow fetches, and the basket's
  pulse coverage is Bengal gram only for now.
- **Data blind spots track need.** The most food-insecure states often have the weakest
  market data (Bihar, Karnataka, Odisha, Jharkhand failed the qualification rule).
  Results describe monitorable states, not national coverage.
- **Market stress is a proxy**, not a measure of hunger.

Full list with reasoning: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Repository layout

```
src/
  ceda_client.py     API client: retries, backoff, Retry-After handling
  backfill.py        chunked, resumable price/arrival backfill (budget-capped)
  backfill_batched.py  multi-key fleet using adaptive windows
  pair_health.py     per-pair failure rates; troubled pairs sort last
  select_markets.py  density probe + state qualification
  clean.py           daily panel (tonnes->quintals, variety-row aggregation)
  validate.py        per-pair coverage verdicts that gate modelling
  pairs.py           single source of truth for viable pairs
  labels.py          trailing-quantile spoilage / shortage flags
  features.py        leakage-safe weekly feature table (+ self-check)
  weather.py         Open-Meteo archive, lagged features only
  evaluate.py        walk-forward harness, all models, report writer
  diagnostics.py     spoilage root-cause analysis
  stress.py          stress index (pre-registered)
  episodes.py        crisis-episode rule (pre-registered)
  basket.py          crop-group basket + 60% coverage guard
  run_stress.py      builds stress, episodes and basket from cached chunks
tests/test_invariants.py   8 invariant tests
scripts/             shell tooling for the multi-key fetch fleet
docs/                architecture, data notes, limitations
reports/             generated reports (kept in git for auditability)
config.yaml          crop groups, backfill targets, label thresholds
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CEDA_API_KEY='...'          # request a key from CEDA; never commit it

python src/backfill.py --from-year 2006     # resumable; skips completed chunks
python src/clean.py && python src/validate.py data/panel_daily.parquet
python src/weather.py --from-year 2006      # free, no key
python src/features.py && python src/evaluate.py    # writes reports/model_comparison.md
python tests/test_invariants.py             # needs the data above
python src/run_stress.py                 # stress index, episodes, basket
```

Raw data is not redistributed in this repository. The `data/` directory is git-ignored
and rebuilt by the commands above.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): full design, decisions and execution order
- [Data notes](docs/DATA_NOTES.md): API quirks and how each was handled
- [Limitations](docs/LIMITATIONS.md): what this does and does not show

## Data and credit

Price and arrival data: [CEDA, Ashoka University](https://ceda.ashoka.edu.in/) Agri
Market API, a cleaned mirror of Agmarknet (Ministry of Agriculture & Farmers Welfare).
Used under CEDA's terms; please credit the Centre for Economic Data & Analysis, Ashoka
University. Weather: [Open-Meteo](https://open-meteo.com/) historical archive.

## License

Code is released under the [MIT License](LICENSE). The data is not covered by it.
