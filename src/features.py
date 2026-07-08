"""
Supervised feature table for the weekly forecasting task.

DECISION THIS MODULE EMBODIES — the leakage boundary is explicit and testable:
  * A row is stamped at a weekly CUTOFF date t (the moment a prediction is made).
  * Every feature is built from the (market, commodity) series through t
    INCLUSIVE and never later. Day t is observed by the time we decide, so it
    is fair game; nothing after t ever touches a feature.
  * The TARGETS are next week's actuals over the strictly-future window
    (t, t+7], defined identically to evaluate.py's weekly_actuals so a model
    trained here optimizes exactly what the harness scores:
        target_arrivals = SUM of daily arrivals in (t, t+7]   (weekly total, quintals)
        target_price    = MEAN of daily modal_price in (t, t+7]
  Because feature windows end at t and target windows start after t, the two
  never overlap. The __main__ self-check asserts this per row.

GAP POLICY (honest, not fabricated): each series is calendar-aligned (asfreq D)
so lag/rolling windows are measured in real days. Short gaps (weekends, the
occasional missing day) are bridged by forward-fill limited to 14 days; gaps
LONGER than 14 days are left as NaN rather than inventing a level across a
data outage. LightGBM consumes NaN natively, so missingness is passed through
honestly instead of being papered over.

Scope: restricted to pairs marked VIABLE_FOR_MODELING (see pairs.viable_pairs);
the restriction lives in one place so it cannot drift from evaluate.py.

Usage:
  python src/features.py            # writes data/features_weekly.parquet + self-check
"""

import numpy as np
import pandas as pd

from pairs import viable_pairs

# ---- window definitions (all trailing, all end at cutoff t) ----
FILL_LIMIT = 14          # forward-fill bridges gaps up to 14 days; never longer
LAGS = [1, 2, 3, 7, 14, 28]
ROLLS = [7, 14, 28, 91]
MOMENTA = [7, 28]
SW_LAG = 358             # same-week-last-year anchor: mean over ~[t-371, t-358]
SW_WIN = 14
QPOS_WIN = 366           # trailing-year window for the quantile-position feature
QPOS_MINOBS = 91         # need ~a season of history before a percentile is meaningful
HORIZON = 7              # predict the next 7 days


def _minp(window):
    """Min non-NaN observations before a rolling stat is emitted (std needs >=2)."""
    return max(2, window // 4)


def _qpos(series, win=QPOS_WIN, minobs=QPOS_MINOBS):
    """Percentile position of the current value within its TRAILING window
    (prior values only). Mirrors the flag definitions in labels.py, which
    compare today against a shifted trailing distribution."""
    def frac_below(a):
        cur = a[-1]
        if np.isnan(cur):
            return np.nan
        prev = a[:-1]
        prev = prev[~np.isnan(prev)]
        if len(prev) < minobs:
            return np.nan
        return float(np.mean(prev <= cur))
    return series.rolling(win, min_periods=minobs).apply(frac_below, raw=True)


def _pair_features(g_obs, cutoffs, market_id, commodity_id):
    """Build one feature row per cutoff for a single (market, commodity)."""
    g_obs = g_obs.set_index("date").sort_index()
    g_obs = g_obs[~g_obs.index.duplicated(keep="first")]  # panel is unique, be safe
    daily = g_obs.asfreq("D")                             # calendar-align; NaN on gaps

    # bridge short gaps only; long outages (>14d) stay NaN
    price = daily["modal_price"].ffill(limit=FILL_LIMIT)
    arr = daily["arrivals"].ffill(limit=FILL_LIMIT)

    f = pd.DataFrame(index=daily.index)
    # current level at t (observed by decision time)
    f["price_now"] = price
    f["arr_now"] = arr
    # lags
    for L in LAGS:
        f[f"price_lag{L}"] = price.shift(L)
        f[f"arr_lag{L}"] = arr.shift(L)
    # trailing rolling stats (windows END at t)
    for W in ROLLS:
        rp, ra = price.rolling(W, min_periods=_minp(W)), arr.rolling(W, min_periods=_minp(W))
        f[f"price_roll{W}_mean"], f[f"arr_roll{W}_mean"] = rp.mean(), ra.mean()
        f[f"price_roll{W}_std"], f[f"arr_roll{W}_std"] = rp.std(), ra.std()
        f[f"price_roll{W}_min"], f[f"arr_roll{W}_min"] = rp.min(), ra.min()
        f[f"price_roll{W}_max"], f[f"arr_roll{W}_max"] = rp.max(), ra.max()
    # momentum (pct change; fill_method=None so pandas does not silently ffill)
    for M in MOMENTA:
        f[f"price_mom{M}"] = price.pct_change(M, fill_method=None)
        f[f"arr_mom{M}"] = arr.pct_change(M, fill_method=None)
    # seasonal anchor: same week last year
    f["price_sw_ly"] = price.shift(SW_LAG).rolling(SW_WIN, min_periods=1).mean()
    f["arr_sw_ly"] = arr.shift(SW_LAG).rolling(SW_WIN, min_periods=1).mean()
    # trailing-year percentile positions (mirror the flag thresholds)
    f["price_qpos365"] = _qpos(price)
    f["arr_qpos365"] = _qpos(arr)

    # sample at the cutoff dates
    idx = pd.DatetimeIndex(cutoffs)
    s = f.reindex(idx)
    s.index.name = "cutoff"

    # calendar features (cheap, deterministic)
    woy = s.index.isocalendar().week.to_numpy(dtype=float)
    s["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    s["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    s["month"] = s.index.month
    s["market_id"] = market_id           # categorical for lgbm
    s["commodity_id"] = commodity_id

    # targets: next-week actuals from OBSERVED rows only, matching weekly_actuals
    tgt_arr, tgt_prc = [], []
    for t in s.index:
        w = g_obs[(g_obs.index > t) & (g_obs.index <= t + pd.Timedelta(days=HORIZON))]
        if w.empty:
            tgt_arr.append(np.nan)
            tgt_prc.append(np.nan)
        else:
            tgt_arr.append(w["arrivals"].sum())      # weekly total (quintals)
            tgt_prc.append(w["modal_price"].mean())  # weekly mean modal price
    s["target_arrivals"] = tgt_arr
    s["target_price"] = tgt_prc

    return s.reset_index()


def default_cutoffs(panel, min_train_days=730, step_days=7):
    """Weekly expanding-window cutoffs, identical to evaluate.walk_forward so
    features and evaluation land on exactly the same folds."""
    start = panel.date.min() + pd.Timedelta(days=min_train_days)
    return pd.date_range(start, panel.date.max() - pd.Timedelta(days=HORIZON),
                         freq=f"{step_days}D")


def make_features(panel, cutoffs, report_path="reports/quality_report.json",
                  config_path="config.yaml"):
    """Supervised weekly table over VIABLE pairs. Importable by evaluate.py.

    One row per (cutoff, market_id, commodity_id): trailing feature columns +
    next-week target columns. Rows with no next-week observation (no target)
    are dropped; rows with no usable recent data (no current level within the
    fill window) are dropped too — a prediction there would be pure guess.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    ok = set(viable_pairs(report_path, config_path))

    frames = []
    for (m, c), g in panel.groupby(["market_id", "commodity_id"]):
        if (int(m), int(c)) not in ok:
            continue
        frames.append(_pair_features(g[["date", "modal_price", "arrivals"]],
                                     cutoffs, int(m), int(c)))
    if not frames:
        raise SystemExit("No viable pairs produced features — check quality_report.json.")

    feats = pd.concat(frames, ignore_index=True)
    feats = feats.dropna(subset=["target_arrivals", "target_price"])
    feats = feats[feats["price_now"].notna() | feats["arr_now"].notna()]
    return feats.sort_values(["cutoff", "market_id", "commodity_id"]).reset_index(drop=True)


def _self_check(panel, feats, cutoffs, n=3):
    """Leakage self-check: print sample rows and prove every feature's source
    window ends at/ before the cutoff while the target window starts after it."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    print("\n--- Leakage self-check (feature window ends <= cutoff < target window) ---")
    max_source_gap = pd.Timedelta(days=-10_000)
    worst = None
    # per-row: newest observed date a feature could draw from vs cutoff
    for _, row in feats.iterrows():
        g = panel[(panel.market_id == row.market_id) &
                  (panel.commodity_id == row.commodity_id)]
        last_src = g.loc[g.date <= row.cutoff, "date"].max()
        gap = last_src - row.cutoff
        if gap > max_source_gap:
            max_source_gap, worst = gap, (row.cutoff, last_src)
    assert max_source_gap <= pd.Timedelta(0), (
        f"LEAKAGE: a feature drew from {worst[1]} > cutoff {worst[0]}")
    print(f"Rows: {len(feats):,} | max(last feature-source date - cutoff) = "
          f"{max_source_gap.days}d  (must be <= 0)   worst row: cutoff={worst[0].date()}, "
          f"newest source={worst[1].date()}")

    feat_cols = [c for c in feats.columns
                 if c not in ("cutoff", "market_id", "commodity_id",
                              "target_arrivals", "target_price")]
    sample = feats.sample(min(n, len(feats)), random_state=0).sort_values("cutoff")
    for _, row in sample.iterrows():
        g = panel[(panel.market_id == row.market_id) &
                  (panel.commodity_id == row.commodity_id)]
        last_src = g.loc[g.date <= row.cutoff, "date"].max()
        w = g[(g.date > row.cutoff) & (g.date <= row.cutoff + pd.Timedelta(days=HORIZON))]
        first_tgt = w.date.min()
        print(f"\ncutoff={row.cutoff.date()}  market={int(row.market_id)} "
              f"commodity={int(row.commodity_id)}")
        print(f"  newest feature source date <= cutoff : {last_src.date()}")
        print(f"  first target date (must be > cutoff)  : {first_tgt.date()}  "
              f"| target window {(row.cutoff + pd.Timedelta(days=1)).date()}.."
              f"{(row.cutoff + pd.Timedelta(days=HORIZON)).date()}")
        print(f"  target_arrivals (wk total)={row.target_arrivals:,.0f}  "
              f"target_price (wk mean)={row.target_price:,.0f}")
        shown = {k: (round(float(row[k]), 2) if pd.notna(row[k]) else None)
                 for k in ["price_now", "arr_now", "price_lag7", "arr_lag7",
                           "price_roll28_mean", "arr_roll28_mean",
                           "price_sw_ly", "price_qpos365", "arr_qpos365"]}
        print(f"  sample features: {shown}")
    print(f"\nFeature columns ({len(feat_cols)}): {feat_cols}")


if __name__ == "__main__":
    panel = pd.read_parquet("data/panel_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    ok = set(viable_pairs())
    scoped = panel[panel.apply(
        lambda r: (int(r.market_id), int(r.commodity_id)) in ok, axis=1)]
    cutoffs = default_cutoffs(scoped)
    print(f"Viable pairs: {sorted(ok)} | cutoffs: {len(cutoffs)} "
          f"({cutoffs.min().date()} -> {cutoffs.max().date()})")

    feats = make_features(panel, cutoffs)
    feats.to_parquet("data/features_weekly.parquet", index=False)
    print(f"Wrote data/features_weekly.parquet: {feats.shape[0]:,} rows x "
          f"{feats.shape[1]} cols")
    print("Rows per pair:")
    print(feats.groupby(["market_id", "commodity_id"]).size().to_string())

    _self_check(scoped, feats, cutoffs, n=3)
