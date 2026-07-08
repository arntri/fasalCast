"""
Cross-check CEDA's mirror against the source of truth (raw Agmarknet portal).

Samples N random (date, market, commodity) rows from the cleaned panel and
writes a checklist CSV. You look each one up on https://agmarknet.gov.in
(Price & Arrival reports) and fill in the portal's values by hand. ~10 rows,
~15 minutes, and it catches systematic issues (unit conversions, date
offsets, dropped varieties) that internal validation cannot.

NOTE: portal shows variety-level rows; our panel is aggregated per day.
Compare arrivals to the SUM of portal varieties and modal price to the
portal's per-variety modal range (our median should sit inside it).

Usage: python src/crosscheck.py --n 10
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    panel = pd.read_parquet("data/panel_daily.parquet")
    scored = panel.dropna(subset=["modal_price", "arrivals"])
    sample = scored.sample(min(args.n, len(scored)), random_state=args.seed)

    out = sample[["date", "market_id", "commodity_id",
                  "arrivals", "modal_price", "min_price", "max_price",
                  "n_price_rows", "n_qty_rows"]].copy()
    out["portal_arrivals_sum"] = ""
    out["portal_modal_range"] = ""
    out["match"] = ""
    out["notes"] = ""
    path = "reports/crosscheck_checklist.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} with {len(out)} rows.")
    print("Fill portal_* columns from agmarknet.gov.in reports, mark match=Y/N.")
    print("Any systematic mismatch (all rows off by 10x, dates shifted by 1) "
          "-> stop and investigate before modeling.")


if __name__ == "__main__":
    main()
