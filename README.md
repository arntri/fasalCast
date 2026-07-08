# Mandi Early-Warning System

Forecasts weekly arrival volumes and modal prices for volatile staple crops
(onion/tomato/potato) in Indian wholesale markets (APMC mandis), and flags two
risks a week ahead: **spoilage/waste risk** (volume spike + price crash) and
**shortage/access risk** (near-zero arrivals + price spike).

Framed as an early-warning system for supply-demand imbalance — the headline
metric is flag precision/recall at lead time, not forecast RMSE.

## Design principles (enforced in code, not by memory)
- **No leakage.** Labels and features use trailing information only; weather
  features (phase 2) will be lagged/recent, never future actuals.
- **Temporal validation only.** Walk-forward, expanding-window weekly folds
  (`src/evaluate.py`). Random splits are structurally impossible here.
- **Baseline first.** Seasonal-naive is model #1 in the registry. Any ML model
  must beat it on the same folds or it doesn't ship.
- **Validated data.** Every source passes `src/validate.py` (coverage, gaps,
  duplicates, unit shifts, price sanity) plus a manual cross-check of random
  records against the raw Agmarknet portal (`src/crosscheck.py`).

## Pipeline
```
backfill.py  ->  clean.py  ->  validate.py  ->  labels.py  ->  evaluate.py
(CEDA API,      (aggregate     (quality        (risk flags,    (walk-forward,
 resumable       unlabeled      report,         base rates)     flag P/R/F1)
 chunks)         variety rows)  pair verdicts)
```

Data: CEDA (Ashoka University) Agri Market API — cleaned mirror of Agmarknet
(Ministry of Agriculture & Farmers Welfare). Historical backfill via CEDA;
live daily feed later via data.gov.in. CEDA data used under their terms
(credit: Centre for Economic Data & Analysis, Ashoka University).

## Status
- [x] Milestone 1: API integration proven; real schema discovered and handled
      (envelope unwrap, flat geographies, unlabeled variety rows, flaky
      endpoints -> retries + resumable chunks + caching)
- [ ] Historical backfill (2006->present) for the phase-1 slice
- [ ] Quality report + portal cross-check sign-off
- [ ] Label base rates on real data
- [ ] Seasonal-naive baseline numbers (walk-forward)
- [ ] Phase 2: weather features (lagged only), ML models vs baseline
