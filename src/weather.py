"""
Historical daily weather per market, from Open-Meteo's free archive API
(archive-api.open-meteo.com — no API key). Pulls rainfall + tmax + tmin so
features.py can build LAGGED weather features.

LEAKAGE NOTE (the decision this module supports): weather enters the model only
as TRAILING/recent actuals (see features.py: trailing 7/14/28d rain sums, temp
means, days-since-heavy-rain). We never use same-week (the target week) or
future weather — that would be forecasting with information a real operator
would not have at decision time. This module just fetches the raw daily
history; the trailing-window discipline is enforced in features.py.

Market coordinates are APPROXIMATE market-town locations (not exact APMC yard
GPS); daily rain/temperature at ~10km resolution is not sensitive to the last
few km, and Open-Meteo snaps to its nearest grid cell anyway.

Fetching mirrors backfill.py: chunked by (market, year), each chunk cached to
data/weather/ immediately, re-runs skip finished chunks. Open-Meteo is far more
reliable than CEDA, but we keep the same resumable discipline and a light
retry/backoff so a transient error never costs finished work.

Usage:
  python src/weather.py                       # fetch config markets, 2006..present
  python src/weather.py --from-year 2010 --to-year 2025
"""

import argparse
import json
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"]
WEATHER_DIR = Path("data/weather")

# market_id -> (lat, lon). APPROXIMATE market-town coords (see module docstring).
# Nashik-district onion markets sit within a ~30km cluster; Pune markets share
# the Pune city approximation.
MARKET_COORDS = {
    169:  (20.14, 74.23),   # Lasalgaon
    2484: (20.14, 74.23),   # Lasalgaon(Niphad) — same Lasalgaon cluster
    4750: (20.11, 74.28),   # Lasalgaon(Vinchur)
    2488: (20.17, 73.98),   # Pimpalgaon Baswant(Saykheda)
    172:  (18.52, 73.86),   # Pune (main market yard)
    4752: (18.52, 73.86),   # Pune(Moshi) — Pune city approximation
}
# General fallbacks named in the brief (kept for reference / new markets):
NASHIK = (20.00, 73.79)
PUNE = (18.52, 73.86)


def chunk_path(market_id, year):
    return WEATHER_DIR / f"weather_m{market_id}_{year}.json"


def fetch_chunk(market_id, lat, lon, year, max_retries=4, timeout=60):
    """One (market, year) daily-weather chunk. Cached; returns day count or -1."""
    path = chunk_path(market_id, year)
    if path.exists():
        return None  # already done — resume behavior
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "daily": ",".join(DAILY_VARS), "timezone": "auto",
    }
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
            if r.status_code == 200:
                daily = r.json().get("daily", {})
                path.write_text(json.dumps(daily, default=str))
                return len(daily.get("time", []))
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            print(f"    FAILED weather m{market_id} {year}: {r.status_code} {r.text[:120]}")
            return -1
        except (requests.RequestException,) as e:
            if attempt == max_retries:
                print(f"    FAILED weather m{market_id} {year} after retries: {str(e)[:120]}")
                return -1
            time.sleep((2 ** attempt) + random.random())
    return -1


def combine(out_path="data/weather_daily.parquet"):
    """Fold all cached chunks into one tidy daily frame:
    [date, market_id, rain_mm, tmax, tmin]."""
    frames = []
    for p in sorted(WEATHER_DIR.glob("weather_m*.json")):
        market_id = int(p.stem.split("_")[1][1:])   # weather_m169_2010 -> 169
        d = json.loads(p.read_text())
        if not d.get("time"):
            continue
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(d["time"]),
            "market_id": market_id,
            "rain_mm": d.get("precipitation_sum"),
            "tmax": d.get("temperature_2m_max"),
            "tmin": d.get("temperature_2m_min"),
        }))
    if not frames:
        print("No weather chunks to combine.")
        return None
    df = pd.concat(frames, ignore_index=True).sort_values(["market_id", "date"])
    df.to_parquet(out_path, index=False)
    print(f"Combined {len(df):,} weather rows -> {out_path}")
    return df


def markets_from_config(config_path="config.yaml"):
    """Distinct market_ids that appear in backfill_targets (so weather covers
    exactly the markets we might model)."""
    import yaml
    cfg = yaml.safe_load(open(config_path))
    return sorted({t["market_id"] for t in cfg.get("backfill_targets", [])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2006)
    ap.add_argument("--to-year", type=int, default=date.today().year)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    if args.combine_only:
        combine()
        return

    market_ids = markets_from_config(args.config)
    missing = [m for m in market_ids if m not in MARKET_COORDS]
    if missing:
        print(f"  WARNING: no coords for markets {missing} — add them to "
              f"MARKET_COORDS. Skipping those.")
    market_ids = [m for m in market_ids if m in MARKET_COORDS]

    years = range(args.from_year, args.to_year + 1)
    total = len(market_ids) * len(list(years))
    print(f"Weather plan: {len(market_ids)} markets x {len(list(years))} years "
          f"= {total} chunks -> {WEATHER_DIR}/")
    done = 0
    for mid in market_ids:
        lat, lon = MARKET_COORDS[mid]
        for year in years:
            n = fetch_chunk(mid, lat, lon, year)
            done += 1
            if n is None:
                continue  # cached
            status = "FAIL (retry on re-run)" if n == -1 else f"{n} days"
            print(f"  [{done}/{total}] m{mid} {year}: {status}")

    combine()
    print("\nDone. Feed data/weather_daily.parquet to features.make_features(weather=...).")


if __name__ == "__main__":
    main()
