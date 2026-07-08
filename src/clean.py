"""
Clean + aggregate raw panels into one tidy daily series per (market, commodity).

Key decision (documented because interviewers will ask):
The API returns multiple unlabeled rows per (date, market) — variety/grade
level records with no variety column. Price rows and quantity rows can't be
reliably paired, so a volume-weighted price is NOT computable without
guessing. Policy:
  arrivals   = SUM of quantity rows        (total volume is well-defined)
  modal      = MEDIAN of modal_price rows  (robust, no fake precision)
  min / max  = MIN of mins / MAX of maxes  (true daily envelope)
  n_price_rows, n_qty_rows kept as columns (aggregation is auditable)

Output: data/panel_daily.parquet — one row per (date, market_id, commodity_id).
Run validate.py afterwards for the quality report.
"""

import pandas as pd
from pathlib import Path


def build_panel(prices_path="data/prices_raw.parquet",
                quantities_path="data/quantities_raw.parquet",
                out_path="data/panel_daily.parquet"):
    px = pd.read_parquet(prices_path)
    qt = pd.read_parquet(quantities_path)
    for df in (px, qt):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    keys = ["date", "market_id", "commodity_id"]

    px_daily = px.groupby(keys).agg(
        modal_price=("modal_price", "median"),
        min_price=("min_price", "min"),
        max_price=("max_price", "max"),
        n_price_rows=("modal_price", "size"),
    ).reset_index()

    # SOURCE UNIT NOTE: CEDA/Agmarknet reports arrivals in METRIC TONNES
    # (verified against agmarknet.gov.in portal header, which reads
    # "Metric Tonnes"). Prices are Rs/quintal. We standardize arrivals to
    # quintals (1 tonne = 10 quintals) so both targets share the quintal basis.
    qt_daily = qt.groupby(keys).agg(
        arrivals_tonnes=("quantity", "sum"),
        n_qty_rows=("quantity", "size"),
    ).reset_index()
    qt_daily["arrivals"] = qt_daily["arrivals_tonnes"] * 10  # tonnes -> quintals

    panel = px_daily.merge(qt_daily, on=keys, how="outer", indicator=True)
    both = (panel["_merge"] == "both").mean()
    print(f"rows: {len(panel):,} | price-only: {(panel['_merge']=='left_only').sum():,} "
          f"| qty-only: {(panel['_merge']=='right_only').sum():,} "
          f"| matched: {both:.1%}")
    panel = panel.drop(columns="_merge").sort_values(keys)

    Path(out_path).parent.mkdir(exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return panel


if __name__ == "__main__":
    build_panel()
