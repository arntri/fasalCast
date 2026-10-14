# India Food Access Watch — Full Architecture
### Nationwide early-warning system for local food-access shocks in Indian wholesale markets

Status: architecture v1 (pivot from FasalCast phase 1).
Everything below is grounded in the proven FasalCast pipeline and the verified
realities of the CEDA API. Sections marked [DECIDED] are settled; [DISCOVER]
requires a cheap probe before locking.

---

## 1. Objective and claims

**System:** monitor a staple food basket across major Indian mandis and emit a
continuous market-stress score per (market, crop) and an aggregated
food-access stress index per district/state, updated as data arrives.

**Primary claim to demonstrate (the headline):**
"Across N years and M markets, the system's stress index rose ahead of
mechanically-defined food-price crisis episodes with a median lead of X weeks,
at Y% episode recall and Z% precision — validated walk-forward with zero
lookahead."

**Secondary claim (credibility layer, onion only):** alignment of the index
with the documented record of government onion interventions (export bans,
stock limits, buffer releases), ~12 events over 20 years. Descriptive, not
statistical — stated as such.

**Retired finding (kept in README as motivation):** market-level compound
spoilage flags are untrainable at small scale (6 events; arrivals-spike half
unpredictable from price/arrival history). This motivated the continuous-index
design.

---

## 2. Data architecture

### 2.1 Sources
| Source | Content | Access | Reliability | Role |
|---|---|---|---|---|
| CEDA Agri API | daily mandi prices + arrivals, 453 commodities, ~2700 mandis, 2001→ | Bearer key (have) | slow/flaky, 1 req/s, resumable fetch proven | core panel |
| Open-Meteo archive API | daily rain/tmax/tmin by lat-long | free, no key | fast, reliable | weather features (lagged only) |
| DoCA/data.gov.in retail prices | daily retail staple prices, centre-level | api key (free) | [DISCOVER — 30-min probe] | optional: retail-side episode definition |
| Static reference table | onion intervention timeline (~12 dated events, sourced) | compiled once by hand | n/a | secondary validation only; never trained on |

### 2.2 Verified CEDA facts (do not rediscover)
- Envelope `{"output":{type,message,data}}`; fields `commodity_id/…_name`,
  `census_state_id/…_district_id`, `market_id/…_name`.
- `/prices` = variety-level rows (no variety key); `/quantities` =
  market-level. Aggregation policy: modal=median, min/max envelope,
  quantity=sum. UNITS: quantity is METRIC TONNES; prices Rs/QUINTAL; convert
  arrivals ×10 to quintals in clean.py (already implemented).
- Known commodity ids: Wheat=1, Paddy=2, Rice=3, Maize=4, Onion=23,
  Tomato=78. Others via one `/commodities` call.
- `/markets` and others hang for minutes; retries+backoff+caching required
  (already implemented).

### 2.3 Scope selection — driven by request budget, not appetite [DECIDED]
Backfill cost = pairs × years × 2 endpoints. At ~1 req/s with failures,
budget phase A at **≤ 12,000 chunks (~2–4 days of background fetching)**.

- **Basket (8 crops):** Rice, Wheat, Onion, Potato, Tomato, Red Gram (tur),
  Bengal Gram (chana), Mustard. Rationale: the actual poor-household plate —
  cereals + vegetables + pulses + oilseed. [ids: DISCOVER for the last 4]
- **Geography (phase A): 6 states, ~40–50 mandis total:** Maharashtra, Uttar
  Pradesh, Madhya Pradesh, West Bengal, Karnataka, Bihar. Selection of mandis
  is AUTOMATED, not hand-picked (see 3.2). Phase B expands after modeling
  starts.
- **Years:** 2008→present (older years proven sparse; validate.py decides
  per-pair usable spans anyway).
- Not every crop trades at every mandi — expect ~⅓–½ of the pair grid to be
  dense. That's fine; validation gates entry.

### 2.4 Storage layout (extends existing)
```
data/raw/            per-chunk JSON: {kind}_c{cid}_m{mid}_{year}.json  (append-only)
data/prices_raw.parquet, quantities_raw.parquet
data/panel_daily.parquet        one row per (date, market, commodity)
data/weather/                   per-location-year chunks -> weather_daily.parquet
data/stress_daily.parquet       stress components + composite per (date, market, crop)
data/basket_district.parquet    district/state basket index per date
reports/                        quality reports, episode tables, model comparisons
reference/interventions_onion.csv   static, hand-compiled, source URLs required
```

---

## 3. Pipeline stages

### 3.1 Ingestion (exists; scale it)
`backfill.py` unchanged in design: chunked, resumable, skip-if-exists,
throttled. Add: a `--budget N` guard that stops after N new requests per run
(protects against runaway costs), and a progress summary (chunks done/failed
per pair).

### 3.2 Market selection probe (new, small) [DISCOVER]
`src/select_markets.py`: for each (state, crop): fetch district list, fetch
markets per top districts (cached), then fetch ONE recent year of prices per
candidate market and rank by record count. Keep top-K mandis per state by
density, write `backfill_targets` into config automatically. Total probe cost
~1–2k requests. This removes all hand-picking and is itself a defensible
methodology ("markets selected by observed reporting density, not
convenience").

#### 3.2.1 As-built (updated 2026-07-13)

**Two data layers — keep distinct, they never overlap:**
- `data/raw/` = actual price+quantity **backfill** for chosen markets
  (`prices_c{cid}_m{mid}_{year}.json`, `quantities_…`). Pre-existing
  hand-picked coverage: Onion(23) 4 Maharashtra markets; Tomato(78) & Potato(24)
  1 MH market each; ~2006–2026. This is §3.1 output (the final model data).
- `data/probe/` = density-probe cache for market **selection**
  (`geo_c{cid}.json`, `mkts_c{cid}_s{sid}_d{did}.json`,
  `probe_c{cid}_s{sid}_d{did}_{year}.json`). Independent namespace — running the
  probe never re-downloads `data/raw`. The probe only *counts* records per
  market to rank density; it discards the price values (no price analysis here).

**Scope actually implemented in `select_markets.py`:**
- 9 crop-candidates: Wheat 1, Rice 3, Onion 23, Potato 24, Tomato 78,
  Bengal Gram(Gram)(Whole) 6, Mustard 12, **Red Gram 7 AND
  Arhar(Tur/Red Gram)(Whole) 49**. Ids 7 & 49 are the same crop (two catalogue
  entries); `resolve_redgram()` probes both and keeps whichever reports denser
  by probe-year record count, dropping the other — NOT double-counted.
- 6 states: Maharashtra 27, Uttar Pradesh 9, Madhya Pradesh 23,
  West Bengal 19, Karnataka 29, Bihar 10.
- Method per (crop, state): 1 `/geographies(crop)` (cached per crop) → districts
  filtered to the state → 1 `/markets` per district (cached) → ONE `/prices` per
  district passing all its market ids at once (probe year 2024) → count records
  per `market_id` → rank, keep top-K (default 6). All skip-if-exists in
  `data/probe/`; `--budget N` caps NEW api calls per run and is resumable.
- Real cost is well above the "~1–2k" estimate above (9 crops × 6 states ×
  dozens of districts) — handled by parallelising, not by shrinking scope.

**Parallelisation — sharded by crop across 8 API keys (embarrassingly parallel
fan-out/fan-in):**
- Flags added to `select_markets.py`: `--only-crops`, `--only-states`,
  `--skip-aggregate`. `run_shard.sh <crop-ids>` wraps one shard (reads
  `CEDA_API_KEY` from env, `--probe-year=2024 --budget=20000 --skip-aggregate`,
  tees to `logs/shard_c*.log`).
- Sharded BY CROP, not by state: every cache file is prefixed `c{crop}`, so
  disjoint crops never write the same file. (State-sharding would collide on the
  shared per-crop `geo_c*.json`.)
- 8-key ↔ crop map: k1→1(Wheat) k2→3(Rice) k3→23(Onion) k4→24(Potato)
  k5→78(Tomato) k6→6(Bengal Gram) k7→12(Mustard) k8→`7,49`(Red Gram).
- Workflow: **fan-out** 8 shards (cache-warm only) → wait until all exit
  (`pgrep -af select_markets.py` empty) → **fan-in**: ONE final *unfiltered*
  `select_markets.py` run reads the warm cache (~0 new calls), resolves Red Gram,
  and writes `reports/market_selection.json` with ranked targets +
  `backfill_chunk_budget` vs the 12k ceiling (the §5 T1 gate).
- Rules: one DISTINCT key per shard (shared key = shared rate limit →
  multi-minute retry stalls, observed on `/commodities`); crops disjoint across
  shards; run fan-in only after all shards finish.

**Progress @ 2026-07-13:** 8 shards running. Prior cache had Wheat only
(UP + MP complete and clean; MH ~done, 3 empty districts; WB barely started;
Karnataka + Bihar not started). Wheat shard resumes the missing states; the
other 7 crops are fresh. Key handling + the non-interactive-shell gotcha are in
`ceda_client.py` (env `CEDA_API_KEY`).

### 3.3 Clean + validate (exists; runs unchanged at scale)
clean.py aggregation + units conversion as-is. validate.py verdicts gate
every pair; only VIABLE pairs enter modeling. New at scale: per-state summary
of pass rates in the quality report.

### 3.4 Stress index construction (new — the core product)
`src/stress.py`, per (date, market, crop), all trailing-only:
- **price_z**: log(modal) minus trailing 365d median, scaled by trailing MAD
- **arrival_deficit**: −(arrivals vs trailing same-week-of-year median),
  scaled; positive = arrivals below seasonal norm
- **momentum**: 14d and 28d log-price change (z-scored trailing)
- **dispersion** (market-level, crop-level): cross-market price spread for the
  crop within state — widening spread = breakdown of arbitrage = stress
- **composite stress** = fixed documented weights over the above (weights
  chosen a priori, stated in the docstring; NEVER fit to episodes — that's
  the anti-dredging line)
Basket aggregation: `src/basket.py` — district/state index = CPI-style fixed
weights over crops (document the weight source; approximate CPI food weights
are fine if cited as approximations). Missing-crop handling: renormalize
weights over available crops, flag coverage % per district-day.

### 3.5 Episode definition (new — the evaluation ground truth)
`src/episodes.py`, mechanical and pre-registered BEFORE any evaluation runs:
- A crisis episode for (market, crop) = composite stress > T for ≥ k
  consecutive weeks (defaults: T = trailing p95, k = 3), merged if gaps < 2
  weeks. Same rule at basket level for district episodes.
- Episode table columns: level, id, start, peak, end, peak_stress, crops.
- Expected volume at scale: hundreds to low thousands of market-crop
  episodes; dozens of district-level ones. [DISCOVER — first run reports it]
- Onion intervention CSV joins here as a *separate* secondary table.

### 3.6 Forecasting layer (adapts existing)
Task: at cutoff t predict stress components (and composite) 1–4 weeks ahead
per (market, crop). Continuous targets, every row is signal.
- `features.py` (adapt): existing lags/rollings/seasonality + lagged weather
  + cross-market features (source-market stress for the crop, e.g. Lasalgaon
  for onion) + crop and state categoricals.
- Models in the SAME walk-forward harness (evaluate.py), expanding window,
  retrain every K folds: (1) persistence baseline, (2) seasonal-naive
  baseline, (3) pooled LightGBM. Baseline-gated as always.
- NO leakage: invariant tests extended to stress components and basket index
  (max source date − cutoff ≤ 0 for every feature and every index input).

### 3.7 Early-warning evaluation (new — the headline numbers)
`src/warning_eval.py`: walk-forward over cutoffs; at each cutoff the system
issues a warning if forecast composite stress crosses the pre-registered
threshold within horizon. Score against the episode table:
- **episode recall**: fraction of episodes preceded by a warning
- **lead time**: distribution of (episode start − first warning), report median
- **precision / false-alarm rate**: warnings not followed by an episode
  within horizon
- operating curve: sweep warning threshold ONLY on the earliest-60% frozen
  region; freeze; report the rest untouched.
Secondary: same lead-time readout against the onion intervention table
(descriptive).

---

## 4. Repo & module map (carry-over explicit)
| Module | Status |
|---|---|
| ceda_client.py, backfill.py, clean.py, validate.py | KEEP (backfill: add --budget) |
| weather.py, tests/test_invariants.py | KEEP, extend |
| features.py, evaluate.py | ADAPT (targets → stress components; models unchanged pattern) |
| labels.py | RETIRE into stress.py components (quantile machinery reused) |
| probe.py, crosscheck.py | KEEP (crosscheck rerun on 2–3 new states — units could differ by state; verify once) |
| select_markets.py, stress.py, basket.py, episodes.py, warning_eval.py | NEW |

---

## 5. Execution order (Claude Code task sequence)
1. **T1 — commodity/market discovery + budget plan:** /commodities ids for the
   8 staples; run select_markets.py probe; emit proposed backfill_targets +
   total chunk count. STOP and report if > 12k chunks.
   *[IN PROGRESS 2026-07-13 — probe sharded by crop across 8 API keys; see
   §3.2.1. Awaiting all 8 shards → then the fan-in run produces the chunk-count
   gate.]*
2. **T2 — launch scaled backfill** (nohup, background, --budget per run).
   Never block on it; report progress per session.
3. **T3 — stress.py + basket.py + episodes.py** on data as it lands
   (works on partial panel; re-run cheaply). Pre-register episode rule and
   composite weights in code BEFORE first evaluation. Commit that boundary.
4. **T4 — adapt features.py + evaluate.py** to stress-component targets;
   baselines first; LightGBM after. Extend invariant tests. Report the
   baseline-vs-model table.
5. **T5 — warning_eval.py**: episode recall / lead time / precision;
   operating point frozen on earliest-60%. This produces the headline.
6. **T6 — onion intervention CSV** (human task, ~a day; Claude assists with
   sourcing); secondary lead-time readout.
7. **T7 — results freeze + README**: architecture summary, findings (incl.
   spoilage negative result as motivation), real numbers only, run
   instructions, data citations (CEDA credit required).

Parallelism: T2 runs continuously in background; T3–T5 iterate on whatever
panel exists; re-run end-to-end when backfill completes.

## 6. Non-negotiables (unchanged from FasalCast)
No leakage · temporal validation only · baselines gate models · episode rule
and index weights pre-registered before evaluation · no fabricated or
extrapolated numbers, ever · every scope decision driven by measured request
budget · validate.py gates every pair · resumable everything.

## 7. Risks and honest limits
- **API throughput** is the schedule risk; mitigated by budget-capped
  resumable fetching and probe-based scope selection. Worst case: fewer
  states in phase A.
- **Data quality variance across states** — some states may mostly fail
  validation; the per-state pass-rate report makes this a finding, not a
  surprise.
- **Weights/threshold subjectivity** — mitigated by pre-registration and by
  reporting sensitivity (vary weights ±, show episode table stability) —
  cheap robustness check, big credibility gain.
- **Episodes ≠ hunger** — the index measures market stress as a *proxy* for
  food-access deterioration. State this plainly; do not claim to measure
  hunger directly.

## 8. Definition of done
1. Scaled panel (≥6 states, ≥8 crops attempted) with per-pair verdicts.
2. Pre-registered stress index + episode table generated walk-forward-safe.
3. Baseline-gated forecast comparison table (real numbers).
4. Early-warning headline: episode recall, median lead time, precision, with
   frozen operating point.
5. Onion-intervention secondary readout.
6. README that a skeptical ML interviewer can audit end to end.
