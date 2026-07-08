"""
Chunked, resumable historical backfill.

Pulls prices AND quantities per (commodity, market, year) and saves each chunk
to data/raw/ immediately. Re-running skips completed chunks, so API flakiness
never costs finished work. Combine at the end into two parquet panels.

Usage:
  export CEDA_API_KEY='...'
  python src/backfill.py --from-year 2006            # uses config.yaml slice
  python src/backfill.py --from-year 2020 --only-market-id 2484
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from ceda_client import CedaClient, CedaError

RAW = Path("data/raw")


def chunk_path(kind, commodity_id, market_id, year):
    return RAW / f"{kind}_c{commodity_id}_m{market_id}_{year}.json"


def fetch_chunk(client, kind, commodity_id, state_id, district_id, market_id, year):
    """One (endpoint, commodity, market, year) chunk. Returns record count."""
    path = chunk_path(kind, commodity_id, market_id, year)
    if path.exists():
        return None  # already done — resume behavior
    fn = client.prices if kind == "prices" else client.quantities
    frm, to = f"{year}-01-01", f"{year}-12-31"
    try:
        data = fn(commodity_id, state_id, [district_id], [market_id], frm, to)
    except CedaError as e:
        print(f"    FAILED {kind} c{commodity_id} m{market_id} {year}: {str(e)[:120]}")
        return -1
    path.write_text(json.dumps(data, default=str))
    return len(data)


def combine(kind):
    frames = []
    for p in sorted(RAW.glob(f"{kind}_*.json")):
        recs = json.loads(p.read_text())
        if recs:
            frames.append(pd.DataFrame(recs))
    if not frames:
        print(f"No {kind} chunks to combine.")
        return
    df = pd.concat(frames, ignore_index=True)
    # normalize the UTC-timestamp dates seen in the live API
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    out = Path(f"data/{kind}_raw.parquet")
    df.to_parquet(out, index=False)
    print(f"Combined {len(df):,} {kind} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--from-year", type=int, default=2006)
    ap.add_argument("--to-year", type=int, default=date.today().year)
    ap.add_argument("--only-market-id", type=int, default=None)
    ap.add_argument("--combine-only", action="store_true",
                    help="Skip fetching; just rebuild the parquet panels.")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.combine_only:
        combine("prices"); combine("quantities")
        return

    cfg = yaml.safe_load(open(args.config))
    targets = cfg.get("backfill_targets", [])
    if args.only_market_id:
        targets = [t for t in targets if t["market_id"] == args.only_market_id]
    if not targets:
        raise SystemExit("No backfill_targets in config.yaml (or filter matched none).")

    client = CedaClient()
    years = range(args.from_year, args.to_year + 1)
    total = len(targets) * len(list(years)) * 2
    done = 0
    print(f"Backfill plan: {len(targets)} (commodity, market) pairs x "
          f"{args.to_year - args.from_year + 1} years x 2 endpoints = {total} chunks.")

    for t in targets:
        label = f"{t['commodity_name']} @ {t['market_name']}"
        for year in range(args.from_year, args.to_year + 1):
            for kind in ("prices", "quantities"):
                n = fetch_chunk(client, kind, t["commodity_id"], t["state_id"],
                                t["district_id"], t["market_id"], year)
                done += 1
                if n is None:
                    continue  # cached
                status = "FAIL (will retry on next run)" if n == -1 else f"{n} recs"
                print(f"  [{done}/{total}] {label} {year} {kind}: {status}")

    combine("prices")
    combine("quantities")
    print("\nDone. Failed chunks (if any) retry automatically on re-run.")


if __name__ == "__main__":
    main()
