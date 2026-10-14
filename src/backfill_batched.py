"""
Batched, key-parallel, resumable backfill.

WHY THIS EXISTS (measured 2026-07-31, supersedes the per-market/per-year path in
backfill.py):

  * The gateway 504s on long windows regardless of how little data is asked for.
    Unbatched probes: 7d/14d -> 200 in ~1-2s; 30d/60d/90d/180d/365d -> 504 @60s.
  * /prices and /quantities accept district_id and market_id as ARRAYS. One call
    can therefore cover every market in a state instead of one market.
  * Batched by (state, commodity), response time scales ~linearly with the window
    (~0.23s/day) and the 504 cliff sits between 90d and 180d:
        30d -> 200 @7.7s | 60d -> 200 @14.0s | 90d -> 200 @21.3s
        180d -> 504      | 365d -> 504
  * Rate limit is a hard 40 requests/hour PER KEY (burst of 60: first 40 -> 200,
    requests 41-60 -> 429).

Consequence: fetching per (market, year) needs ~286k requests (~42 days at 7
keys). Batching by (state, commodity) over 90-day windows needs ~9.1k (~32h).
That is the whole reason this module exists.

Cache: one JSON per (kind, state, commodity, window) under data/raw_batched/.
Re-running skips cached windows, so a kill/restart never loses finished work.

The safe window is per-group, not global: 90d succeeds for Maharashtra but 504s
for denser Rajasthan. So each (kind, state, commodity) group walks its own
timeline with a LEARNED span — start at WINDOW_DAYS, halve on 504, and keep the
span that worked for every later window in that group. Discovery therefore costs
~log2(90/7) wasted requests per group ONCE, instead of one wasted request on
every window (which is what a fixed 90d + per-window split would do).

Usage:
  python src/backfill_batched.py --keyfile backfill_keys_7.env
  python src/backfill_batched.py --keyfile backfill_keys_7.env --kinds prices
  python src/backfill_batched.py --combine-only
"""

import argparse
import json
import queue
import sys
import threading
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ceda_client import CedaClient, CedaError  # noqa: E402

RAW_BATCHED = Path("data/raw_batched")
RAW_LEGACY = Path("data/raw")

WINDOW_DAYS = 90          # safe: 200 @~21s; the 504 cliff is between 90 and 180
MIN_SPAN_DAYS = 7         # floor for adaptive splitting
PER_KEY_INTERVAL = 92.0   # 40 req/hr/key measured -> 90s; 92s keeps us under it
REQ_TIMEOUT = 75          # a 90d window answers in ~21s; 75s means "gateway gave up"

_print_lock = threading.Lock()


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)


def windows(from_year, to_year, span_days=WINDOW_DAYS, clamp_today=True):
    """Contiguous [start, end] date pairs covering the year range."""
    start = date(from_year, 1, 1)
    last = date(to_year, 12, 31)
    if clamp_today:
        last = min(last, date.today())
    out = []
    while start <= last:
        end = min(start + timedelta(days=span_days - 1), last)
        out.append((start, end))
        start = end + timedelta(days=1)
    return out


def cache_path(kind, state_id, commodity_id, frm, to):
    return RAW_BATCHED / f"{kind}_s{state_id}_c{commodity_id}_{frm}_{to}.json"


def build_groups(cfg, targets_key=None):
    """Collapse per-(commodity, market) targets into per-(state, commodity)
    batches. District/market id lists are the union across ALL commodities in
    that state: asking for a market that does not trade the commodity costs
    nothing extra and keeps one id list per state."""
    targets = cfg.get(targets_key) if targets_key else (
        cfg.get("backfill_targets_auto") or cfg.get("backfill_targets", []))
    if not targets:
        raise SystemExit("No backfill targets found in config.")
    by_state_districts = defaultdict(set)
    by_state_markets = defaultdict(set)
    commodities_per_state = defaultdict(set)
    names = {}
    for t in targets:
        s = t["state_id"]
        by_state_districts[s].add(t["district_id"])
        by_state_markets[s].add(t["market_id"])
        commodities_per_state[s].add(t["commodity_id"])
        names[(s, t["commodity_id"])] = (t.get("state_name", s),
                                         t.get("commodity_name", t["commodity_id"]))
    groups = []
    for s, comms in sorted(commodities_per_state.items()):
        for c in sorted(comms):
            groups.append({
                "state_id": s,
                "commodity_id": c,
                "district_ids": sorted(by_state_districts[s]),
                "market_ids": sorted(by_state_markets[s]),
                "state_name": names[(s, c)][0],
                "commodity_name": names[(s, c)][1],
            })
    return groups


class Fleet:
    """One worker thread per API key; each key paced independently at its own
    40/hr budget.

    The queue holds JOBS, not fixed windows. A job is a (kind, group) timeline
    with a cursor and a learned span: a worker fetches [cursor, cursor+span),
    then requeues the job with the cursor advanced. Jobs are therefore walked
    sequentially per group but many groups run concurrently, so all 7 keys stay
    busy while each group keeps its own hard-won span."""

    def __init__(self, keys, jobs, total_days):
        self.keys = keys
        self.q = queue.Queue()
        for j in jobs:
            self.q.put(j)
        self.total_days = max(total_days, 1)
        self.days_done = 0
        self.requests = 0
        self.wasted = 0
        self.failed = []
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def _advance(self, days, note, wasted=False):
        with self.lock:
            self.days_done += days
            self.requests += 1
            if wasted:
                self.wasted += 1
            pct = 100.0 * self.days_done / self.total_days
            n = self.requests
        log(f"  {pct:5.1f}% ({n:,} req) {note}")

    def worker(self, idx, key):
        client = CedaClient(api_key=key, min_interval=0.0,
                            max_retries=0, timeout=REQ_TIMEOUT)
        next_ok = 0.0
        while not self.stop.is_set():
            try:
                job = self.q.get(timeout=5)
            except queue.Empty:
                return

            kind, g = job["kind"], job["group"]
            cursor, end, span = job["cursor"], job["end"], job["span"]
            if cursor > end:
                self.q.task_done()
                continue

            to = min(cursor + timedelta(days=span - 1), end)
            path = cache_path(kind, g["state_id"], g["commodity_id"], cursor, to)
            tag = f"{g['state_name'][:4]}/{g['commodity_name'][:10]}"

            if path.exists():
                # Free: advance without spending a request or the pacing wait.
                self.q.put({**job, "cursor": to + timedelta(days=1)})
                self.q.task_done()
                with self.lock:
                    self.days_done += (to - cursor).days + 1
                continue

            wait = next_ok - time.monotonic()
            if wait > 0:
                time.sleep(wait)

            fn = client.prices if kind == "prices" else client.quantities
            t0 = time.monotonic()
            # Pace from request START, not completion: the hourly quota counts
            # requests issued, so time spent waiting on a slow (or 60s-timing-out)
            # call already counts toward the interval. Adding the interval after
            # the fact would stretch a 504 cycle to 152s and silently cut each
            # key from ~39 req/hr to ~24.
            next_ok = t0 + PER_KEY_INTERVAL
            try:
                data = fn(g["commodity_id"], g["state_id"], g["district_ids"],
                          g["market_ids"], str(cursor), str(to))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, default=str))
                dt = time.monotonic() - t0
                self.q.put({**job, "cursor": to + timedelta(days=1)})
                self.q.task_done()
                self._advance((to - cursor).days + 1,
                              f"key{idx} {kind[:5]} {tag} {cursor}..{to} "
                              f"{len(data)} recs {dt:.0f}s")
            except CedaError as e:
                msg = str(e)  # next_ok already set from t0 above

                if "429" in msg:
                    # Hourly budget exhausted on this key: hand the job back so
                    # another key can take it, and idle this one.
                    self.q.put(job)
                    self.q.task_done()
                    log(f"  key{idx} 429 — cooling down 10min")
                    next_ok = time.monotonic() + 600
                    continue

                heavy = ("504" in msg or "502" in msg or "503" in msg
                         or "timeout" in msg.lower())
                if heavy and span > MIN_SPAN_DAYS:
                    # Too heavy for the gateway. Halve the span and REMEMBER it
                    # for the rest of this group's timeline.
                    new_span = max(span // 2, MIN_SPAN_DAYS)
                    self.q.put({**job, "span": new_span})
                    self.q.task_done()
                    self._advance(0, f"key{idx} 504 {tag} {span}d -> "
                                     f"{new_span}d (learned)", wasted=True)
                    continue

                # Unrecoverable for this window: record and step over it so one
                # bad window cannot stall the whole group's timeline.
                self.q.put({**job, "cursor": to + timedelta(days=1)})
                self.q.task_done()
                with self.lock:
                    self.failed.append((kind, g["state_id"], g["commodity_id"],
                                        str(cursor), str(to), msg[:120]))
                self._advance((to - cursor).days + 1,
                              f"key{idx} FAIL {tag} {cursor}..{to}: {msg[:70]}",
                              wasted=True)

    def run(self):
        threads = [threading.Thread(target=self.worker, args=(i, k), daemon=True)
                   for i, k in enumerate(self.keys)]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            log("interrupted — cached windows are kept; re-run to resume")
            self.stop.set()
            raise


def _records(payload):
    """Cache files hold whatever _unwrap returned — normally a list of records."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "records"):
            if isinstance(payload.get(k), list):
                return payload[k]
    return []


def combine(kind):
    """Merge legacy per-(market, year) chunks with the new batched windows.
    Both describe the same rows, so dedup on the natural key."""
    frames = []
    for root, pattern in ((RAW_LEGACY, f"{kind}_*.json"),
                          (RAW_BATCHED, f"{kind}_*.json")):
        for p in sorted(root.glob(pattern)):
            try:
                recs = _records(json.loads(p.read_text()))
            except json.JSONDecodeError:
                log(f"  skipping unreadable {p.name}")
                continue
            if recs:
                frames.append(pd.DataFrame(recs))
    if not frames:
        log(f"No {kind} chunks to combine.")
        return
    df = pd.concat(frames, ignore_index=True)
    df["date"] = (pd.to_datetime(df["date"], errors="coerce", utc=True)
                    .dt.tz_localize(None).dt.normalize())
    keys = [c for c in ("date", "commodity_id", "market_id") if c in df.columns]
    before = len(df)
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    df = df.sort_values(keys or ["date"]).reset_index(drop=True)
    out = Path(f"data/{kind}_raw.parquet")
    df.to_parquet(out, index=False)
    log(f"Combined {before:,} -> {len(df):,} deduped {kind} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--keyfile", default="backfill_keys_7.env")
    ap.add_argument("--from-year", type=int, default=2006)
    ap.add_argument("--to-year", type=int, default=date.today().year)
    ap.add_argument("--kinds", default="prices,quantities",
                    help="comma-separated: prices,quantities. Prices are "
                         "queued first so a truncated run still yields a "
                         "complete price panel.")
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--targets-key", default=None)
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the request budget and exit")
    args = ap.parse_args()

    if args.combine_only:
        for k in args.kinds.split(","):
            combine(k.strip())
        return

    cfg = yaml.safe_load(open(args.config))
    groups = build_groups(cfg, args.targets_key)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]

    start = date(args.from_year, 1, 1)
    end = min(date(args.to_year, 12, 31), date.today())
    span_days = (end - start).days + 1

    jobs = []
    for kind in kinds:                      # prices first, by argument order
        for g in groups:
            jobs.append({"kind": kind, "group": g, "cursor": start,
                         "end": end, "span": args.window_days})
    total_days = span_days * len(jobs)

    keys = [k.strip() for k in Path(args.keyfile).read_text().splitlines()
            if k.strip() and not k.startswith("#")]
    if not keys:
        raise SystemExit(f"no keys in {args.keyfile}")

    rate = len(keys) * 3600.0 / PER_KEY_INTERVAL
    est = total_days / args.window_days
    log(f"Groups: {len(groups)} (state,commodity) x {len(kinds)} kinds "
        f"= {len(jobs)} timelines | {start}..{end} | start span {args.window_days}d")
    log(f"Best case ~{est:,.0f} requests | keys: {len(keys)} | ~{rate:.0f} req/hr "
        f"| ETA ~{est/rate:.1f}h (more if dense groups learn a smaller span)")
    if args.dry_run:
        return

    RAW_BATCHED.mkdir(parents=True, exist_ok=True)
    fleet = Fleet(keys, jobs, total_days)
    t0 = time.monotonic()
    fleet.run()
    log(f"Made {fleet.requests:,} requests in {(time.monotonic()-t0)/3600:.2f}h "
        f"({fleet.wasted} wasted on 504-discovery/failures, "
        f"{len(fleet.failed)} windows unrecoverable)")
    if fleet.failed:
        Path("logs").mkdir(exist_ok=True)
        Path("logs/batched_failures.tsv").write_text(
            "\n".join("\t".join(map(str, f)) for f in fleet.failed))
        log("  failures -> logs/batched_failures.tsv (re-run to retry them)")

    for kind in kinds:
        combine(kind)


if __name__ == "__main__":
    main()
