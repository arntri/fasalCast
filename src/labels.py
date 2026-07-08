"""
Risk-flag label definitions + historical base-rate report.

The flags ARE the product; the forecaster is machinery. Definitions use
TRAILING quantiles only (shifted by 1 day) so a label on day t depends solely
on information available before t — the same leakage discipline the model
will face.

  SPOILAGE risk on day t:
      arrivals_t  > trailing p90 of arrivals   AND
      modal_t     < trailing p10 of modal price
  SHORTAGE risk on day t:
      arrivals_t  < trailing p10 of arrivals (or ~zero)   AND
      modal_t     > trailing p90 of modal price

Trailing window: 365 days, min 90 observations. Thresholds in config.yaml.

Usage:
  python src/labels.py            # prints base rates per (market, commodity)
"""

import pandas as pd
import yaml


def add_labels(panel, window=365, min_obs=90,
               hi_q=0.90, lo_q=0.10):
    """Adds spoilage_risk / shortage_risk boolean columns. Trailing-only."""
    panel = panel.sort_values(["market_id", "commodity_id", "date"]).copy()

    def per_series(g):
        g = g.set_index("date").asfreq("D")  # calendar-align so windows are in days
        roll_arr = g["arrivals"].shift(1).rolling(f"{window}D", min_periods=min_obs)
        roll_prc = g["modal_price"].shift(1).rolling(f"{window}D", min_periods=min_obs)
        g["arr_hi"] = roll_arr.quantile(hi_q)
        g["arr_lo"] = roll_arr.quantile(lo_q)
        g["prc_hi"] = roll_prc.quantile(hi_q)
        g["prc_lo"] = roll_prc.quantile(lo_q)
        g["spoilage_risk"] = (g["arrivals"] > g["arr_hi"]) & (g["modal_price"] < g["prc_lo"])
        g["shortage_risk"] = (g["arrivals"] < g["arr_lo"]) & (g["modal_price"] > g["prc_hi"])
        return g.dropna(subset=["arrivals", "modal_price"], how="all").reset_index()

    out = []
    for (m, c), g in panel.groupby(["market_id", "commodity_id"]):
        r = per_series(g.drop(columns=["market_id", "commodity_id"]))
        r["market_id"], r["commodity_id"] = m, c
        out.append(r)
    return pd.concat(out, ignore_index=True)


def base_rate_report(labeled):
    print("\n--- Risk-flag base rates (historical, trailing-quantile labels) ---")
    for (m, c), g in labeled.groupby(["market_id", "commodity_id"]):
        scored = g["arr_hi"].notna()
        n = int(scored.sum())
        if n == 0:
            continue
        sp = g.loc[scored, "spoilage_risk"].mean()
        sh = g.loc[scored, "shortage_risk"].mean()
        print(f"  market {m} / commodity {c}: {n:,} scored days | "
              f"spoilage {sp:.2%} ({int(sp*n)} days) | "
              f"shortage {sh:.2%} ({int(sh*n)} days)")
    print("\nIf base rates are <1%, evaluation MUST use precision/recall on flags,")
    print("not accuracy — and RMSE alone can hide total failure on the tails.")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    lc = cfg.get("labels", {})
    panel = pd.read_parquet("data/panel_daily.parquet")
    labeled = add_labels(panel,
                         window=lc.get("trailing_window_days", 365),
                         min_obs=lc.get("min_obs", 90),
                         hi_q=lc.get("hi_quantile", 0.90),
                         lo_q=lc.get("lo_quantile", 0.10))
    labeled.to_parquet("data/panel_labeled.parquet", index=False)
    base_rate_report(labeled)
