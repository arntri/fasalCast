"""
Walk-forward evaluation harness. Built BEFORE any ML model, so temporal
validation and baseline-first are structural, not remembered.

Design:
- Weekly task: at each fold's cutoff date t, models see data up to t and
  predict, for each (market, commodity), the NEXT week's (t+1..t+7)
  total arrivals and mean modal price.
- Expanding-window folds, stepped weekly. Never a random split.
- Metrics: MAE + sMAPE for both targets, PLUS precision/recall/F1 on
  predicted risk flags vs actual-outcome flags (thresholds from trailing
  history at time t only — no leakage into either side).
- Every model implements: predict(history_panel, targets_index) -> DataFrame
  with columns [market_id, commodity_id, pred_arrivals, pred_price].
- Model #1 is seasonal-naive. Any ML model must beat it here, on the same
  folds, or it doesn't count.

Usage:
  python src/evaluate.py                # runs all registered models
"""

import numpy as np
import pandas as pd


# ---------------- models ----------------

def seasonal_naive(history, pairs, cutoff):
    """Next week = same calendar week last year; fallback = last 4 weeks' mean."""
    preds = []
    for (m, c) in pairs:
        g = history[(history.market_id == m) & (history.commodity_id == c)]
        last_year = g[(g.date >= cutoff - pd.Timedelta(days=371)) &
                      (g.date <= cutoff - pd.Timedelta(days=358))]
        recent = g[g.date > cutoff - pd.Timedelta(days=28)]
        src = last_year if len(last_year) >= 3 else recent
        if src.empty:
            src = g.tail(14)
        preds.append({"market_id": m, "commodity_id": c,
                      "pred_arrivals": src["arrivals"].mean() * 6,  # ~6 trading days
                      "pred_price": src["modal_price"].mean()})
    return pd.DataFrame(preds)


MODELS = {"seasonal_naive": seasonal_naive}
# Register future ML models here. They face the same folds and metrics.


# ---------------- harness ----------------

def weekly_actuals(panel, cutoff):
    wk = panel[(panel.date > cutoff) & (panel.date <= cutoff + pd.Timedelta(days=7))]
    return wk.groupby(["market_id", "commodity_id"]).agg(
        act_arrivals=("arrivals", "sum"),
        act_price=("modal_price", "mean")).reset_index()


def trailing_thresholds(history, cutoff, hi=0.90, lo=0.10):
    h = history[history.date > cutoff - pd.Timedelta(days=365)]
    g = h.groupby(["market_id", "commodity_id"]).agg(
        arr_hi=("arrivals", lambda s: s.quantile(hi)),
        arr_lo=("arrivals", lambda s: s.quantile(lo)),
        prc_hi=("modal_price", lambda s: s.quantile(hi)),
        prc_lo=("modal_price", lambda s: s.quantile(lo)),
        arr_med=("arrivals", "median")).reset_index()
    return g


def flags(df, arr_col, prc_col, th):
    d = df.merge(th, on=["market_id", "commodity_id"], how="left")
    # weekly totals vs daily thresholds: compare on daily-average scale
    daily = d[arr_col] / 6.0
    spoil = (daily > d["arr_hi"]) & (d[prc_col] < d["prc_lo"])
    short = (daily < d["arr_lo"]) & (d[prc_col] > d["prc_hi"])
    return spoil.fillna(False), short.fillna(False)


def smape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    denom = (np.abs(a) + np.abs(p)) / 2
    ok = denom > 0
    return float(np.mean(np.abs(a[ok] - p[ok]) / denom[ok]))


def prf(actual, pred):
    tp = int((actual & pred).sum()); fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if prec == prec and rec == rec and (prec + rec) else float("nan")
    return prec, rec, f1, tp + fn  # last = positives count


def walk_forward(panel, min_train_days=730, step_days=7):
    panel = panel.sort_values("date")
    start = panel.date.min() + pd.Timedelta(days=min_train_days)
    cutoffs = pd.date_range(start, panel.date.max() - pd.Timedelta(days=7),
                            freq=f"{step_days}D")
    pairs = list(panel.groupby(["market_id", "commodity_id"]).groups)

    results = {name: [] for name in MODELS}
    for cutoff in cutoffs:
        history = panel[panel.date <= cutoff]
        actual = weekly_actuals(panel, cutoff)
        if actual.empty:
            continue
        th = trailing_thresholds(history, cutoff)
        a_spoil, a_short = flags(actual, "act_arrivals", "act_price", th)
        for name, model in MODELS.items():
            pred = model(history, pairs, cutoff)
            j = actual.merge(pred, on=["market_id", "commodity_id"], how="inner")
            if j.empty:
                continue
            p_spoil, p_short = flags(j, "pred_arrivals", "pred_price", th)
            aj_spoil, aj_short = flags(j, "act_arrivals", "act_price", th)
            results[name].append({
                "cutoff": cutoff,
                "mae_price": float((j.act_price - j.pred_price).abs().mean()),
                "smape_price": smape(j.act_price, j.pred_price),
                "mae_arr": float((j.act_arrivals - j.pred_arrivals).abs().mean()),
                "smape_arr": smape(j.act_arrivals, j.pred_arrivals),
                "spoil_actual": aj_spoil.values, "spoil_pred": p_spoil.values,
                "short_actual": aj_short.values, "short_pred": p_short.values,
            })
    return results


def report(results):
    print("\n--- Walk-forward results (weekly folds, expanding window) ---")
    for name, folds in results.items():
        if not folds:
            print(f"{name}: no folds — not enough history yet.")
            continue
        df = pd.DataFrame(folds)
        sa = np.concatenate(df.spoil_actual); sp = np.concatenate(df.spoil_pred)
        ha = np.concatenate(df.short_actual); hp = np.concatenate(df.short_pred)
        p1, r1, f1, n1 = prf(pd.Series(sa), pd.Series(sp))
        p2, r2, f2, n2 = prf(pd.Series(ha), pd.Series(hp))
        print(f"\n[{name}] {len(df)} folds")
        print(f"  price:    MAE={df.mae_price.mean():,.0f}  sMAPE={df.smape_price.mean():.1%}")
        print(f"  arrivals: MAE={df.mae_arr.mean():,.0f}  sMAPE={df.smape_arr.mean():.1%}")
        print(f"  SPOILAGE flags ({n1} actual events): "
              f"precision={p1:.2f} recall={r1:.2f} F1={f1:.2f}")
        print(f"  SHORTAGE flags ({n2} actual events): "
              f"precision={p2:.2f} recall={r2:.2f} F1={f2:.2f}")
    print("\nAny ML model must beat seasonal_naive on THESE numbers, same folds.")


if __name__ == "__main__":
    panel = pd.read_parquet("data/panel_daily.parquet")
    report(walk_forward(panel))
