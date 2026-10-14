"""
Crisis episodes  (Architecture 3.5) -- mechanical, PRE-REGISTERED, committed
before any evaluation.

For a series (a (market, commodity) pair, or a district basket index):
  1. Weekly stress = mean of the daily composite over each W-SUN week
     (needs >= MIN_DAYS_PER_WEEK observed days).
  2. Threshold T_w = trailing p95 of weekly stress over the PRIOR 104 weeks
     (shifted one week; needs >= MIN_WEEKS), so the threshold never sees the
     week it judges.
  3. A week is "hot" if stress_w > T_w. An episode = >= K_WEEKS consecutive hot
     weeks; two runs separated by a gap of <= GAP_WEEKS cool weeks are merged.
The rule is never fit to anything. Output columns:
level, id, start, peak, end, peak_stress, n_weeks.
"""

import numpy as np
import pandas as pd

THRESH_Q = 0.95
THRESH_WINDOW_WEEKS = 104
MIN_WEEKS = 52
K_WEEKS = 3
GAP_WEEKS = 1               # "merge if gaps < 2 weeks"
MIN_DAYS_PER_WEEK = 3


def weekly(daily):
    """daily: Series indexed by date -> weekly mean (week ending Sunday)."""
    r = daily.resample("W-SUN")
    w = r.mean()
    return w.where(r.count() >= MIN_DAYS_PER_WEEK)


def hot_weeks(w):
    thr = w.shift(1).rolling(THRESH_WINDOW_WEEKS, min_periods=MIN_WEEKS).quantile(THRESH_Q)
    return (w > thr).fillna(False), thr


def find_episodes(w, level, ident):
    hot, _ = hot_weeks(w)
    runs, start = [], None
    idx = list(w.index)
    for i, h in enumerate(hot.values):
        if h and start is None:
            start = i
        if (not h or i == len(hot) - 1) and start is not None:
            end = i if h else i - 1
            runs.append([start, end])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= GAP_WEEKS:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    rows = []
    for s, e in merged:
        # a merged run can be K long only counting its hot weeks
        if int(hot.values[s:e + 1].sum()) < K_WEEKS:
            continue
        seg = w.iloc[s:e + 1]
        rows.append(dict(level=level, id=ident, start=idx[s], peak=seg.idxmax(),
                         end=idx[e], peak_stress=float(seg.max()), n_weeks=e - s + 1))
    return rows


def episode_table(stress, key_cols=("market_id", "commodity_id"), level="market-crop",
                  value="stress"):
    rows = []
    for key, g in stress.groupby(list(key_cols)):
        ident = "|".join(map(str, key)) if isinstance(key, tuple) else str(key)
        s = g.set_index("date")[value].sort_index()
        rows += find_episodes(weekly(s), level, ident)
    return pd.DataFrame(rows, columns=["level", "id", "start", "peak", "end",
                                       "peak_stress", "n_weeks"])
