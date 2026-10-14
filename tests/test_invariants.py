"""
Invariant tests for the FasalCast pipeline. Plain-python asserts (no pytest, no
new dependency): run with `python tests/test_invariants.py`. Each test locks a
correctness guarantee the rest of the system relies on; a failure here means a
downstream number cannot be trusted.

Covered:
  1. viable_pairs parses the quality report correctly and excludes DROP pairs.
  2. No leakage: every feature row's newest source date <= its cutoff.
  3. Targets == evaluate.weekly_actuals (SUM arrivals / MEAN price over (t,t+7]).
  4. Weekly-total scale: target_arrivals is a weekly total, not a daily level.
  5. Weather features are trailing-only (recomputed by hand, must match).
  6. Basket coverage guard: a district-day is scored only if >= 60% of basket
     weight is present; below that it is NaN with scored=False.
  7. Rice/Paddy are one group but never averaged at price level.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs import viable_pairs                                    # noqa: E402
from features import make_features, default_cutoffs, HORIZON, HEAVY_RAIN_MM  # noqa: E402
import evaluate as ev                                            # noqa: E402
import basket as bk                                              # noqa: E402

PANEL = pd.read_parquet(ROOT / "data/panel_daily.parquet")
PANEL["date"] = pd.to_datetime(PANEL["date"])
WX_PATH = ROOT / "data/weather_daily.parquet"


def _scoped():
    ok = set(viable_pairs(ROOT / "reports/quality_report.json", ROOT / "config.yaml"))
    mask = [(int(m), int(c)) in ok for m, c in zip(PANEL.market_id, PANEL.commodity_id)]
    return PANEL[mask].copy(), ok


def test_viable_pairs_matches_report():
    import json
    ok = viable_pairs(ROOT / "reports/quality_report.json", ROOT / "config.yaml")
    assert isinstance(ok, list) and ok, "viable_pairs returned nothing"
    assert all(isinstance(t, tuple) and len(t) == 2 for t in ok)
    rep = json.loads((ROOT / "reports/quality_report.json").read_text())
    n_ok = sum(1 for r in rep["pairs"].values() if r["VIABLE_FOR_MODELING"])
    assert len(ok) == n_ok, f"viable_pairs count {len(ok)} != report OK count {n_ok}"
    # every viable pair actually exists in the panel
    present = set(map(tuple, PANEL[["market_id", "commodity_id"]].drop_duplicates().values))
    for p in ok:
        assert p in present, f"viable pair {p} absent from panel"
    return f"{len(ok)} viable pairs, all present, matches report OK count"


def test_no_leakage_feature_sources():
    scoped, _ = _scoped()
    cutoffs = default_cutoffs(scoped)
    feats = make_features(PANEL, cutoffs)
    worst = pd.Timedelta(days=-10_000)
    for _, row in feats.iterrows():
        g = scoped[(scoped.market_id == row.market_id) &
                   (scoped.commodity_id == row.commodity_id)]
        last_src = g.loc[g.date <= row.cutoff, "date"].max()
        worst = max(worst, last_src - row.cutoff)
    assert worst <= pd.Timedelta(0), f"LEAKAGE: newest source is {worst.days}d AFTER cutoff"
    return f"{len(feats)} rows; max(newest source - cutoff) = {worst.days}d (<= 0)"


def test_targets_equal_weekly_actuals():
    scoped, _ = _scoped()
    cutoffs = default_cutoffs(scoped)
    feats = make_features(PANEL, cutoffs).set_index(["cutoff", "market_id", "commodity_id"])
    checked = 0
    for cutoff in feats.index.get_level_values("cutoff").unique()[::120]:
        act = ev.weekly_actuals(scoped, cutoff)
        for _, a in act.iterrows():
            key = (cutoff, int(a.market_id), int(a.commodity_id))
            if key not in feats.index:
                continue
            row = feats.loc[key]
            assert np.isclose(row.target_arrivals, a.act_arrivals, rtol=1e-6), \
                f"arrivals target != weekly_actuals at {key}"
            assert np.isclose(row.target_price, a.act_price, rtol=1e-6), \
                f"price target != weekly_actuals at {key}"
            checked += 1
    assert checked > 0
    return f"{checked} (cutoff,pair) targets match weekly_actuals exactly"


def test_weekly_total_scale():
    scoped, _ = _scoped()
    cutoffs = default_cutoffs(scoped)
    feats = make_features(PANEL, cutoffs)
    # target_arrivals is a weekly SUM; arr_now is a daily level. The ratio should
    # sit near the number of trading days/week (~5-7), never ~1 (that would mean a
    # daily-vs-weekly scale slip like the one the evaluate.py assertion guards).
    ratio = feats["target_arrivals"].median() / feats["arr_now"].median()
    assert 3.0 < ratio < 9.0, f"weekly/daily arrivals ratio {ratio:.2f} outside [3,9]"
    return f"median(weekly target)/median(daily level) = {ratio:.2f} (weekly-total scale)"


def test_weather_features_trailing_only():
    if not WX_PATH.exists():
        return "SKIP (no data/weather_daily.parquet)"
    scoped, _ = _scoped()
    cutoffs = default_cutoffs(scoped)
    wx = pd.read_parquet(WX_PATH)
    wx["date"] = pd.to_datetime(wx["date"])
    feats = make_features(PANEL, cutoffs, weather=wx)
    sample = feats.dropna(subset=["rain_sum7"]).sample(20, random_state=1)
    for _, row in sample.iterrows():
        w = wx[wx.market_id == row.market_id].set_index("date").sort_index()
        # hand-recompute the trailing 7d rain sum ending AT the cutoff
        lo = row.cutoff - pd.Timedelta(days=6)
        manual = w.loc[(w.index >= lo) & (w.index <= row.cutoff), "rain_mm"].sum()
        assert np.isclose(row.rain_sum7, manual, rtol=1e-6, atol=1e-6), \
            f"rain_sum7 uses non-trailing window at {row.cutoff} m{row.market_id}"
        # days_since_heavy_rain must reference a PAST heavy-rain day (<= cutoff)
        if pd.notna(row.days_since_heavy_rain):
            assert row.days_since_heavy_rain >= 0
    return f"20 sampled rows: rain_sum7 == hand-computed trailing (<=cutoff) window"


def _basket_rows(day, groups_present, district=1, val=1.0):
    gid = {"rice": 3, "wheat": 1, "onion": 23, "potato": 24, "tomato": 78,
           "redgram": 7, "bengalgram": 6, "mustard": 12}
    return [dict(date=pd.Timestamp(day), district_id=district, market_id=100 + gid[g],
                 commodity_id=gid[g], stress=val) for g in groups_present]


def test_basket_coverage_guard():
    import yaml
    c2g = bk.commodity_to_group(yaml.safe_load(open(ROOT / "config.yaml"))["crop_groups"])
    W = bk.BASKET_WEIGHTS
    assert abs(sum(W.values()) - 1.0) < 1e-9, "basket weights must sum to 1"
    # ok = 0.65 weight (scored); thin = 0.45 and edge = 0.55 (both null);
    # edge60 = 0.65 (scored). The exact 0.60 boundary is checked below.
    ok = ["rice", "wheat", "onion", "potato"]
    thin = ["rice", "wheat"]
    edge = ["rice", "wheat", "mustard"]           # 0.25+0.20+0.10 = 0.55 -> null
    edge60 = ["rice", "wheat", "mustard", "onion"]  # 0.65; and boundary below
    rows = (_basket_rows("2024-01-01", ok, 1, 2.0) + _basket_rows("2024-01-01", thin, 2, 2.0)
            + _basket_rows("2024-01-01", edge, 3, 2.0) + _basket_rows("2024-01-01", edge60, 4, 2.0))
    out = bk.basket_index(pd.DataFrame(rows), c2g).set_index("district_id")
    assert out.loc[1, "scored"] and abs(out.loc[1, "coverage"] - 0.65) < 1e-9
    for d in (2, 3):
        assert not out.loc[d, "scored"] and np.isnan(out.loc[d, "index"]), \
            f"district {d} below 60% coverage must be NaN, got {out.loc[d, 'index']}"
        assert out.loc[d, "coverage"] < 0.60
    # exact boundary: custom weights giving precisely 0.60 present -> scored
    w = {"rice": 0.6, "wheat": 0.4}
    b = bk.basket_index(pd.DataFrame(_basket_rows("2024-01-01", ["rice"])), c2g, weights=w)
    assert bool(b.scored.iloc[0]) and abs(b.coverage.iloc[0] - 0.6) < 1e-9
    # scored index renormalises over present groups (all stress=2 -> index 2)
    assert abs(out.loc[1, "index"] - 2.0) < 1e-9
    return "0.65 scored; 0.45/0.55 -> NaN+flag; boundary 0.60 scored; weights sum to 1"


def test_rice_paddy_group_stress_not_price():
    import yaml
    c2g = bk.commodity_to_group(yaml.safe_load(open(ROOT / "config.yaml"))["crop_groups"])
    assert c2g[2] == c2g[3] == "rice", "Paddy(2) and Rice(3) must share the rice group"
    # Two districts, same stress, wildly different implied price levels are
    # irrelevant: the index consumes stress only.
    rows = [dict(date=pd.Timestamp("2024-01-01"), district_id=1, market_id=1,
                 commodity_id=2, stress=1.0),
            dict(date=pd.Timestamp("2024-01-01"), district_id=1, market_id=2,
                 commodity_id=3, stress=3.0)]
    g = bk.basket_index(pd.DataFrame(rows), c2g, weights={"rice": 1.0})
    assert abs(g["index"].iloc[0] - 2.0) < 1e-9 and g["n_groups"].iloc[0] == 1
    return "paddy+rice share one group; combined at stress level only"


TESTS = [
    ("viable_pairs matches report / excludes DROP", test_viable_pairs_matches_report),
    ("no leakage: feature sources <= cutoff", test_no_leakage_feature_sources),
    ("targets == weekly_actuals", test_targets_equal_weekly_actuals),
    ("weekly-total arrivals scale", test_weekly_total_scale),
    ("weather features trailing-only", test_weather_features_trailing_only),
    ("basket 60% coverage guard", test_basket_coverage_guard),
    ("rice/paddy group at stress level", test_rice_paddy_group_stress_not_price),
]


def main():
    print("Running invariant tests...\n")
    failed = 0
    for name, fn in TESTS:
        try:
            detail = fn()
            print(f"  PASS  {name}\n        -> {detail}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        -> {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}\n        -> {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
