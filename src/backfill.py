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


def load_targets(cfg):
    """Prefer the probe-selected backfill_targets_auto (written by
    select_markets.py --write-config); fall back to the hand-picked
    backfill_targets. Returns (targets, source_key)."""
    auto = cfg.get("backfill_targets_auto")
    if auto:
        return auto, "backfill_targets_auto"
    return cfg.get("backfill_targets", []), "backfill_targets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--from-year", type=int, default=2006)
    ap.add_argument("--to-year", type=int, default=date.today().year)
    ap.add_argument("--only-market-id", type=int, default=None)
    ap.add_argument("--only-crops", default=None,
                    help="comma-separated commodity ids to fetch (default: all). "
                         "Use with --shard to fan the backfill across API keys.")
    ap.add_argument("--shard", default=None,
                    help="i/N — process only targets where index %% N == i "
                         "(0-based i). Even fan-out across N keys, one key per i. "
                         "Combine is SKIPPED under --shard; run --combine-only once "
                         "all shards finish.")
    ap.add_argument("--budget", type=int, default=None,
                    help="stop after N NEW chunk fetches this run (cached chunks "
                         "are free). Resumable — re-run to continue.")
    ap.add_argument("--targets-key", default=None,
                    help="force a specific config key for targets "
                         "(default: backfill_targets_auto if present, else "
                         "backfill_targets).")
    ap.add_argument("--combine-only", action="store_true",
                    help="Skip fetching; just rebuild the parquet panels.")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.combine_only:
        combine("prices"); combine("quantities")
        return

    cfg = yaml.safe_load(open(args.config))
    if args.targets_key:
        targets, source = cfg.get(args.targets_key, []), args.targets_key
    else:
        targets, source = load_targets(cfg)
    if args.only_market_id:
        targets = [t for t in targets if t["market_id"] == args.only_market_id]
    if args.only_crops:
        want = {int(x) for x in args.only_crops.split(",")}
        targets = [t for t in targets if t["commodity_id"] in want]
    sharded = args.shard is not None
    if sharded:
        i, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= i < n):
            raise SystemExit(f"--shard i/N needs 0 <= i < N (got {args.shard}).")
        targets = [t for idx, t in enumerate(targets) if idx % n == i]
    if not targets:
        raise SystemExit(f"No targets from '{source}' (or filters matched none).")

    client = CedaClient()
    years = range(args.from_year, args.to_year + 1)
    total = len(targets) * len(list(years)) * 2
    done = fetched = 0
    shard_note = f"  shard {args.shard}" if sharded else ""
    print(f"Backfill plan [{source}{shard_note}]: {len(targets)} (commodity, "
          f"market) pairs x {args.to_year - args.from_year + 1} years x 2 "
          f"endpoints = {total} chunks."
          + (f"  budget={args.budget} new fetches." if args.budget else ""))

    stop = False
    for t in targets:
        if stop:
            break
        label = f"{t['commodity_name']} @ {t['market_name']}"
        for year in range(args.from_year, args.to_year + 1):
            if stop:
                break
            for kind in ("prices", "quantities"):
                n = fetch_chunk(client, kind, t["commodity_id"], t["state_id"],
                                t["district_id"], t["market_id"], year)
                done += 1
                if n is None:
                    continue  # cached — free, not counted against budget
                fetched += 1
                status = "FAIL (will retry on next run)" if n == -1 else f"{n} recs"
                print(f"  [{done}/{total}] {label} {year} {kind}: {status}")
                if args.budget is not None and fetched >= args.budget:
                    print(f"\n!! BUDGET {args.budget} new fetches reached — "
                          f"pausing. Re-run the same command to continue.")
                    stop = True
                    break

    if sharded:
        print(f"\nShard {args.shard} done ({fetched} new fetches). Combine is "
              f"skipped under --shard; run `--combine-only` once every shard "
              f"finishes.")
    else:
        combine("prices")
        combine("quantities")
    print("\nDone. Failed chunks (if any) retry automatically on re-run.")


if __name__ == "__main__":
    main()
