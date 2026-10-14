"""
Market-stress index per (date, market, commodity)  (Architecture 3.4).

PRE-REGISTERED: every constant below is fixed a priori and committed BEFORE the
index is computed on any panel. Weights are NEVER fit to the episodes they are
later scored against (anti-dredging line). Everything is TRAILING-ONLY: the value
on day t uses data from days < t (or, for the seasonal reference, days <= t-357),
enforced by tests/test_invariants.py::test_stress_truncation_invariant.

Components (all signed so that POSITIVE = more food-access stress):
  price_z          log(modal_t) minus trailing-365d median, scaled by trailing MAD
  arrival_deficit  -(log1p arrivals_t minus seasonal reference), scaled by the
                   trailing MAD of that residual. Reference = median of the
                   +-7-day-smoothed log arrivals at the same day-of-year 1, 2 and
                   3 years earlier (all strictly past). Positive = arrivals below
                   seasonal norm.
  momentum         mean of the 14d and 28d log-price changes, each z-scored by its
                   own trailing-365d MAD. Positive = price rising.
  dispersion       crop-level, within state: cross-market std of log modal price
                   that day (needs >= 2 markets), z-scored against its own
                   trailing-365d history. Widening spread = arbitrage breakdown.

Composite = weighted mean of the AVAILABLE components with fixed weights
(price_z must be present), renormalised over those present. Components are
clipped to +-CLIP so one bad print cannot own the index.

Robust scale: MAD*1.4826 with a floor, so a flat stretch cannot blow up a z.
"""

import numpy as np
import pandas as pd

WINDOW = 365
MIN_OBS = 90
CLIP = 6.0
MAD_K = 1.4826
SCALE_FLOOR = 1e-3          # in log units
SEASONAL_LAGS = (364, 728, 1092)
SMOOTH_DAYS = 15            # centred +-7d, applied only to lagged (past) values
WEIGHTS = {"price_z": 0.40, "arrival_deficit": 0.25,
           "momentum": 0.20, "dispersion": 0.15}


def _roll_med(x, window=WINDOW, min_obs=MIN_OBS):
    return x.shift(1).rolling(window, min_periods=min_obs).median()


def _roll_mad(x, med, window=WINDOW, min_obs=MIN_OBS):
    """Trailing MAD of x around its (already trailing) median series."""
    dev = (x - med).abs()
    mad = dev.shift(1).rolling(window, min_periods=min_obs).median() * MAD_K
    return mad.clip(lower=SCALE_FLOOR)


def _z(x, window=WINDOW, min_obs=MIN_OBS):
    med = _roll_med(x, window, min_obs)
    return ((x - med) / _roll_mad(x, med, window, min_obs)).clip(-CLIP, CLIP)


def series_components(g):
    """g: one (market, commodity) daily frame with columns date, modal_price,
    arrivals (arrivals may be NaN). Returns frame indexed by calendar day with
    price_z, arrival_deficit, momentum, log_price."""
    g = g.set_index("date").sort_index()
    g = g[~g.index.duplicated()].asfreq("D")
    lp = np.log(g["modal_price"].where(g["modal_price"] > 0))
    out = pd.DataFrame(index=g.index)
    out["log_price"] = lp
    out["price_z"] = _z(lp)

    la = np.log1p(g["arrivals"].where(g["arrivals"] >= 0))
    smooth = la.rolling(SMOOTH_DAYS, center=True, min_periods=5).median()
    ref = pd.concat([smooth.shift(lag) for lag in SEASONAL_LAGS], axis=1).median(axis=1)
    resid = la - ref
    rmed = _roll_med(resid)
    out["arrival_deficit"] = (-(resid - rmed) / _roll_mad(resid, rmed)).clip(-CLIP, CLIP)

    m14 = lp - lp.shift(14)
    m28 = lp - lp.shift(28)
    out["momentum"] = pd.concat([_z(m14), _z(m28)], axis=1).mean(axis=1, skipna=False)
    return out


def dispersion_by_group(comp, meta):
    """comp: concat of series components with market_id/commodity_id/date columns;
    meta: market_id -> state_id. Cross-market std of log price per
    (state, commodity, date), z-scored on its own trailing history."""
    d = comp[["date", "market_id", "commodity_id", "log_price"]].merge(
        meta[["market_id", "state_id"]], on="market_id", how="left")
    d = d.dropna(subset=["log_price", "state_id"])
    grp = d.groupby(["state_id", "commodity_id", "date"])["log_price"]
    sp = pd.concat([grp.std(), grp.size().rename("n")], axis=1).reset_index()
    sp.loc[sp["n"] < 2, "log_price"] = np.nan
    sp = sp.rename(columns={"log_price": "spread"})
    out = []
    for (s, c), g in sp.groupby(["state_id", "commodity_id"]):
        g = g.set_index("date").sort_index().asfreq("D")
        g["dispersion"] = _z(g["spread"])
        g["state_id"], g["commodity_id"] = s, c
        out.append(g[["state_id", "commodity_id", "dispersion"]].reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["date", "state_id", "commodity_id", "dispersion"])


def composite(df, weights=WEIGHTS):
    comps = list(weights)
    w = pd.Series(weights)
    vals = df[comps]
    present = vals.notna() * w
    num = (vals.fillna(0) * w).sum(axis=1)
    den = present.sum(axis=1)
    out = num / den
    return out.where(df["price_z"].notna() & (den > 0))


def build_stress(panel, meta, min_rows=MIN_OBS):
    """panel: daily panel (date, market_id, commodity_id, modal_price, arrivals).
    meta: DataFrame market_id, state_id, district_id. Returns tidy stress frame."""
    frames = []
    for (m, c), g in panel.groupby(["market_id", "commodity_id"]):
        if g["modal_price"].notna().sum() < min_rows:
            continue
        comp = series_components(g[["date", "modal_price", "arrivals"]])
        comp["market_id"], comp["commodity_id"] = m, c
        frames.append(comp.reset_index(names="date"))
    if not frames:
        return pd.DataFrame()
    comp = pd.concat(frames, ignore_index=True)
    disp = dispersion_by_group(comp, meta)
    comp = comp.merge(meta[["market_id", "state_id", "district_id"]],
                      on="market_id", how="left")
    comp = comp.merge(disp, on=["date", "state_id", "commodity_id"], how="left")
    comp["stress"] = composite(comp)
    return comp.dropna(subset=["stress"]).reset_index(drop=True)
