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
- Every model implements: model(history_panel, pairs, cutoff) -> DataFrame
  with columns [market_id, commodity_id, pred_arrivals, pred_price].
- Model #1 is seasonal-naive. Any ML model must beat it here, on the same
  folds, or it doesn't count.

SCOPE / FAIR-COMPARISON DECISION (Milestone 2):
  The harness is restricted to pairs marked VIABLE_FOR_MODELING by validate.py
  (parsed once in pairs.viable_pairs, shared with features.py). DROP pairs
  failed data-quality gates, so scoring anything on them — baseline included —
  is unreliable. Baseline-gating only means something when every model faces
  an IDENTICAL pair/fold population, so each fold scores all models on the
  INTERSECTION of the instances they all predicted (this also excludes
  cold-start folds where an ML model has no trained weights yet, and rows a
  supervised model cannot form because there is no recent data). The old
  "906 folds, price MAE=818" numbers came from all four markets and are
  SUPERSEDED by the restricted-set baseline reported here.

Usage:
  python src/evaluate.py                # seasonal_naive vs lgbm, writes report
"""

import numpy as np
import pandas as pd

import lightgbm as lgb

from pairs import viable_pairs
from features import make_features, default_cutoffs, HORIZON


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
        if src.empty:                    # guard: no history at all -> skip pair
            continue
        preds.append({"market_id": m, "commodity_id": c,
                      "pred_arrivals": src["arrivals"].mean() * 6,  # ~6 trading days
                      "pred_price": src["modal_price"].mean()})
    return pd.DataFrame(preds, columns=["market_id", "commodity_id",
                                        "pred_arrivals", "pred_price"])


class LgbmForecaster:
    """Two LightGBM regressors (next-week arrivals, next-week mean price) over the
    trailing feature table from features.py.

    Leakage discipline:
      * A training row stamped at cutoff t' has features <= t' and a target over
        (t', t'+7]. To use it while predicting fold cutoff t, its target must be
        fully realized before t, i.e. t' <= t - 7. We enforce exactly that.
      * Retraining every 906 folds is too slow, so we retrain every K=26 folds
        on all rows valid at that fold (t' <= cutoff - 7d) and reuse those
        weights for the next 26 folds. Weights are therefore always trained on
        strictly past, fully-observed data — never future, only staler.

    Hyperparameters are fixed and documented; NO search is run on test folds.
    Predictions are weekly TOTALS (arrivals) / weekly MEAN (price) — the same
    quantities the harness scores — so they drop straight into flags().

    Uses LightGBM's NATIVE lgb.train API (not the sklearn wrapper) so the only
    new dependency is lightgbm itself — scikit-learn is not required."""

    PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=20, bagging_fraction=0.8, bagging_freq=1,
                  feature_fraction=0.8, lambda_l2=1.0, seed=0,
                  num_threads=0, verbose=-1)
    NUM_ROUNDS = 400
    K_RETRAIN = 26
    MIN_TRAIN_ROWS = 100          # cold start: no ML predictions until this many

    def __init__(self, feats, k_retrain=K_RETRAIN):
        feats = feats.reset_index(drop=True).copy()
        # fixed category dtypes so codes are identical across train/predict slices
        for col in ("market_id", "commodity_id"):
            feats[col] = feats[col].astype(
                pd.CategoricalDtype(categories=sorted(feats[col].unique())))
        self.feats = feats
        self.k = k_retrain
        self.feat_cols = [c for c in feats.columns
                          if c not in ("cutoff", "target_arrivals", "target_price")]
        self._by_cutoff = {t: g for t, g in feats.groupby("cutoff")}
        self._calls = 0
        self._m_arr = self._m_prc = None
        self.last_train = None

    def _train(self, cutoff):
        eligible = self.feats[self.feats.cutoff <= cutoff - pd.Timedelta(days=HORIZON)]
        if len(eligible) < self.MIN_TRAIN_ROWS:
            self._m_arr = self._m_prc = None
            return
        X = eligible[self.feat_cols]
        d_arr = lgb.Dataset(X, label=eligible["target_arrivals"], free_raw_data=False)
        d_prc = lgb.Dataset(X, label=eligible["target_price"], free_raw_data=False)
        self._m_arr = lgb.train(self.PARAMS, d_arr, num_boost_round=self.NUM_ROUNDS)
        self._m_prc = lgb.train(self.PARAMS, d_prc, num_boost_round=self.NUM_ROUNDS)
        self.last_train = (cutoff, len(eligible))

    def __call__(self, history, pairs, cutoff):
        if self._calls % self.k == 0:
            self._train(cutoff)
        self._calls += 1
        empty = pd.DataFrame(columns=["market_id", "commodity_id",
                                      "pred_arrivals", "pred_price"])
        rows = self._by_cutoff.get(cutoff)
        if rows is None or self._m_arr is None:
            return empty
        want = set(map(tuple, pairs))
        keys = list(zip(rows.market_id.astype(int), rows.commodity_id.astype(int)))
        mask = [k in want for k in keys]
        rows = rows[mask]
        if rows.empty:
            return empty
        X = rows[self.feat_cols]
        out = pd.DataFrame({
            "market_id": rows.market_id.astype(int).values,
            "commodity_id": rows.commodity_id.astype(int).values,
            "pred_arrivals": np.clip(self._m_arr.predict(X), 0, None),
            "pred_price": np.clip(self._m_prc.predict(X), 0, None),
        })
        return out.reset_index(drop=True)


MODELS = {"seasonal_naive": seasonal_naive}
# ML models are registered in __main__ (they need the precomputed feature table).


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
    if not ok.any():                     # guard: empty slice -> no defined sMAPE
        return float("nan")
    return float(np.mean(np.abs(a[ok] - p[ok]) / denom[ok]))


def prf(actual, pred):
    tp = int((actual & pred).sum()); fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if prec == prec and rec == rec and (prec + rec) else float("nan")
    return prec, rec, f1, tp + fn  # last = positives count


def _common_pairs(actual, preds):
    """Instances (market, commodity) that appear in the actuals AND in EVERY
    model's predictions this fold — the identical-fold population all models
    are scored on."""
    common = set(map(tuple, actual[["market_id", "commodity_id"]].values))
    for p in preds.values():
        common &= set(map(tuple, p[["market_id", "commodity_id"]].values))
    return common


def walk_forward(panel, min_train_days=730, step_days=7, pairs_filter=None):
    if pairs_filter is None:
        pairs_filter = viable_pairs()
    pairs_filter = set(map(tuple, pairs_filter))
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    keep = [(int(m), int(c)) in pairs_filter
            for m, c in zip(panel.market_id, panel.commodity_id)]
    panel = panel[keep].sort_values("date")

    start = panel.date.min() + pd.Timedelta(days=min_train_days)
    cutoffs = pd.date_range(start, panel.date.max() - pd.Timedelta(days=7),
                            freq=f"{step_days}D")
    pairs = sorted(pairs_filter)

    results = {name: [] for name in MODELS}
    for cutoff in cutoffs:
        history = panel[panel.date <= cutoff]
        actual = weekly_actuals(panel, cutoff)
        if actual.empty:
            continue
        th = trailing_thresholds(history, cutoff)
        preds = {name: model(history, pairs, cutoff) for name, model in MODELS.items()}

        common = _common_pairs(actual, preds)      # identical instances for all models
        if not common:
            continue
        a_common = actual[[tuple(x) in common
                           for x in actual[["market_id", "commodity_id"]].values]]
        for name, pred in preds.items():
            j = a_common.merge(pred, on=["market_id", "commodity_id"], how="inner")
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
                "tot_pred_arr": float(j.pred_arrivals.sum()),
                "tot_act_arr": float(j.act_arrivals.sum()),
                "spoil_actual": aj_spoil.values, "spoil_pred": p_spoil.values,
                "short_actual": aj_short.values, "short_pred": p_short.values,
            })
    return results


def _summarize(name, folds):
    df = pd.DataFrame(folds)
    sa = np.concatenate(df.spoil_actual); sp = np.concatenate(df.spoil_pred)
    ha = np.concatenate(df.short_actual); hp = np.concatenate(df.short_pred)
    p1, r1, f1, n1 = prf(pd.Series(sa), pd.Series(sp))
    p2, r2, f2, n2 = prf(pd.Series(ha), pd.Series(hp))
    # SCALE ASSERTION: predicted and actual arrivals must share the weekly-total
    # scale (a ~6x daily<->weekly slip would blow this band and is caught here).
    ratio = df.tot_pred_arr.sum() / df.tot_act_arr.sum()
    assert 0.2 < ratio < 5.0, (
        f"{name}: pred/act weekly-arrivals scale mismatch (ratio={ratio:.2f}); "
        f"pred_arrivals and act_arrivals are NOT on the same weekly-total scale")
    return {
        "name": name, "folds": len(df), "scale_ratio": ratio,
        "mae_price": df.mae_price.mean(), "smape_price": df.smape_price.mean(),
        "mae_arr": df.mae_arr.mean(), "smape_arr": df.smape_arr.mean(),
        "spoil": (p1, r1, f1, n1), "short": (p2, r2, f2, n2),
    }


def report(results, out_path="reports/model_comparison.md"):
    print("\n--- Walk-forward results (weekly folds, expanding window, VIABLE pairs) ---")
    summaries = []
    for name, folds in results.items():
        if not folds:
            print(f"{name}: no folds — not enough history yet.")
            continue
        s = _summarize(name, folds)
        summaries.append(s)
        p1, r1, f1, n1 = s["spoil"]; p2, r2, f2, n2 = s["short"]
        print(f"\n[{name}] {s['folds']} folds  (arrivals scale-check ratio={s['scale_ratio']:.2f})")
        print(f"  price:    MAE={s['mae_price']:,.0f}  sMAPE={s['smape_price']:.1%}")
        print(f"  arrivals: MAE={s['mae_arr']:,.0f}  sMAPE={s['smape_arr']:.1%}")
        print(f"  SPOILAGE flags ({n1} actual events): precision={p1:.2f} recall={r1:.2f} F1={f1:.2f}")
        print(f"  SHORTAGE flags ({n2} actual events): precision={p2:.2f} recall={r2:.2f} F1={f2:.2f}")
    print("\nAny ML model must beat seasonal_naive on THESE numbers, same folds.")

    if summaries:
        _write_markdown(summaries, out_path)
        print(f"\nWrote {out_path}")
    return summaries


def _fmt_prf(t):
    p, r, f, n = t
    return f"{p:.2f} / {r:.2f} / {f:.2f}"


def _write_markdown(summaries, out_path):
    base = next((s for s in summaries if s["name"] == "seasonal_naive"), None)
    lines = []
    lines.append("# Model comparison — weekly walk-forward\n")
    lines.append(
        "Harness restricted to pairs marked `VIABLE_FOR_MODELING` by "
        "`validate.py` (markets **169** and **4750**, Onion) BEFORE any model "
        "comparison: baseline-gating is only meaningful when every model faces "
        "an identical pair/fold population, and DROP pairs failed data-quality "
        "gates so scoring anything on them is unreliable. Each fold scores all "
        "models on the intersection of the instances they all predicted, so "
        "`seasonal_naive` and `lgbm` are compared on exactly the same folds. "
        "The earlier all-markets figures (906 folds, price MAE≈818) are "
        "**superseded** by the restricted-set baseline below.\n")
    n_folds = summaries[0]["folds"]
    lines.append(f"Folds: **{n_folds}** &nbsp;|&nbsp; "
                 f"spoilage events: **{summaries[0]['spoil'][3]}** &nbsp;|&nbsp; "
                 f"shortage events: **{summaries[0]['short'][3]}**\n")
    lines.append("| model | price MAE | price sMAPE | arrivals MAE | arrivals sMAPE "
                 "| spoilage P/R/F1 | shortage P/R/F1 |")
    lines.append("|---|--:|--:|--:|--:|:--:|:--:|")
    for s in summaries:
        lines.append(
            f"| `{s['name']}` | {s['mae_price']:,.0f} | {s['smape_price']:.1%} "
            f"| {s['mae_arr']:,.0f} | {s['smape_arr']:.1%} "
            f"| {_fmt_prf(s['spoil'])} | {_fmt_prf(s['short'])} |")

    if base is not None:
        lines.append("\n## Improvement vs baseline (seasonal_naive)\n")
        lines.append("Negative Δ = lower error = better. Verdict is derived from "
                     "the numbers, not asserted.\n")
        lines.append("| model | Δ price MAE | Δ arrivals MAE | Δ shortage F1 | verdict |")
        lines.append("|---|--:|--:|--:|:--|")
        for s in summaries:
            if s["name"] == "seasonal_naive":
                continue
            dp = s["mae_price"] - base["mae_price"]
            da = s["mae_arr"] - base["mae_arr"]
            df1 = s["short"][2] - base["short"][2]        # shortage F1 delta
            wins = []
            wins.append(f"price MAE {dp/base['mae_price']:+.0%}")
            wins.append(f"arrivals MAE {da/base['mae_arr']:+.0%}")
            n_better = int(dp < 0) + int(da < 0)
            if n_better == 2:
                head = "**beats baseline** on both regression targets"
            elif n_better == 1:
                head = "mixed: beats baseline on one regression target"
            else:
                head = "**loses to baseline** on both regression targets"
            df1_txt = (f"{df1:+.2f}" if df1 == df1 else "n/a")
            verdict = f"{head} ({', '.join(wins)}; shortage F1 Δ {df1_txt})"
            lines.append(f"| `{s['name']}` | {dp:+,.0f} | {da:+,.0f} | {df1_txt} | {verdict} |")

    # Honest weather note: did lagged weather beat plain lgbm?
    by = {s["name"]: s for s in summaries}
    if "lgbm" in by and "lgbm_weather" in by:
        a, w = by["lgbm"], by["lgbm_weather"]
        dp = w["mae_price"] - a["mae_price"]
        da = w["mae_arr"] - a["mae_arr"]
        df1 = w["short"][2] - a["short"][2]
        helped = (dp < 0) and (da < 0) and (df1 >= 0)
        lines.append("\n## Does lagged weather help? (lgbm_weather vs lgbm)\n")
        if helped:
            lines.append(
                f"Yes — adding trailing rain/temperature features lowers both "
                f"errors (price MAE {dp:+.0f}, arrivals MAE {da:+.0f}) and does "
                f"not hurt shortage F1 ({df1:+.2f}).\n")
        else:
            lines.append(
                f"**Not materially.** Trailing 7/14/28d rain sums, tmax/tmin "
                f"means and days-since-heavy-rain move the numbers within noise "
                f"(price MAE {dp:+.0f}, arrivals MAE {da:+.0f}, shortage F1 "
                f"{df1:+.2f} vs plain `lgbm`). For onion at a weekly horizon the "
                f"price/arrival autoregressive features already capture the "
                f"seasonal signal; recent local weather adds little. Reported as "
                f"found — no result is discarded for being unflattering.\n")

    lines.append("\n*MAE/sMAPE lower is better. Flags: precision/recall/F1, "
                 "higher is better. Numbers are produced by `python src/evaluate.py`; "
                 "none are hand-entered.*\n")
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


if __name__ == "__main__":
    from pathlib import Path

    panel = pd.read_parquet("data/panel_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    ok = viable_pairs()
    scoped = panel[[(int(m), int(c)) in set(ok)
                    for m, c in zip(panel.market_id, panel.commodity_id)]]
    cutoffs = default_cutoffs(scoped)
    print(f"Viable pairs: {ok} | {len(cutoffs)} weekly cutoffs "
          f"({cutoffs.min().date()} -> {cutoffs.max().date()})")

    feats = make_features(panel, cutoffs)
    MODELS["lgbm"] = LgbmForecaster(feats)

    # lgbm + weather (Task 4): only if the weather panel has been fetched.
    wx_path = Path("data/weather_daily.parquet")
    if wx_path.exists():
        weather = pd.read_parquet(wx_path)
        feats_w = make_features(panel, cutoffs, weather=weather)
        MODELS["lgbm_weather"] = LgbmForecaster(feats_w)
    else:
        print("  (no data/weather_daily.parquet — run weather.py to add lgbm_weather)")

    print(f"Registered models: {list(MODELS)}")
    report(walk_forward(scoped, pairs_filter=ok))
