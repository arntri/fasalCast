"""
Per-pair fetch health for the backfill (Scope-Correction Task 5).

Two jobs:
  1. ORDER targets so slow/troubled pairs sort LAST -- they can never starve the
     healthy pairs. Rank 0 = normal, 1 = troubled (observed failure rate
     >= TROUBLED_RATE over >= MIN_ATTEMPTS fetches), 2 = de-prioritised by
     policy (red gram, ids 7/49: its own background run, see architecture 3.2).
  2. REPORT per-pair failure rates, so the next pathological commodity/state is
     caught from numbers, not noticed by vibe.

Stats persist in logs/pair_stats.json (cumulative across runs/shards); writes
are read-merge-write under an atomic rename so concurrent shards lose at worst
a few increments, never corrupt the file. Stats only steer ORDER -- they never
skip a pair, so resumability and completeness are unchanged.
"""

import json
import os
from pathlib import Path

STATS_PATH = Path("logs/pair_stats.json")
TROUBLED_RATE = 0.25
MIN_ATTEMPTS = 4
DEPRIORITISED_COMMODITIES = {7, 49}  # Red Gram -- phase B unless coverage proves usable


def load(path=STATS_PATH):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def pair_key(commodity_id, scope_id):
    """scope_id = market_id (backfill.py) or state_id (backfill_batched.py)."""
    return f"c{commodity_id}|{scope_id}"


def failure_rate(entry):
    a = entry.get("attempts", 0)
    return entry.get("failures", 0) / a if a else 0.0


def rank(commodity_id, key, stats):
    """0 normal, 1 troubled by observation, 2 de-prioritised by policy."""
    if commodity_id in DEPRIORITISED_COMMODITIES:
        return 2
    e = stats.get(key, {})
    if e.get("attempts", 0) >= MIN_ATTEMPTS and failure_rate(e) >= TROUBLED_RATE:
        return 1
    return 0


def order_last(items, commodity_of, key_of, stats):
    """Stable sort: healthy first, troubled next, de-prioritised last. Within a
    rank the original (config) order is preserved."""
    return sorted(items, key=lambda x: rank(commodity_of(x), key_of(x), stats))


def merge_save(delta, path=STATS_PATH):
    """delta = {key: {"attempts": n, "failures": n, "label": str}} for this run."""
    if not delta:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = load(path)
    for k, d in delta.items():
        e = cur.setdefault(k, {"attempts": 0, "failures": 0})
        e["attempts"] += d["attempts"]
        e["failures"] += d["failures"]
        if d.get("label"):
            e["label"] = d["label"]
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(cur, indent=1, sort_keys=True))
    os.replace(tmp, path)


def summary(stats, top=10):
    """Text table of the worst pairs by failure rate (needs MIN_ATTEMPTS)."""
    rows = [(failure_rate(e), k, e) for k, e in stats.items()
            if e.get("attempts", 0) >= MIN_ATTEMPTS]
    rows.sort(key=lambda r: (-r[0], -r[2]["attempts"]))
    n_bad = sum(1 for r in rows if r[0] >= TROUBLED_RATE)
    lines = [f"Pair health: {len(stats)} pairs tracked, {n_bad} troubled "
             f"(>= {TROUBLED_RATE:.0%} failures over >= {MIN_ATTEMPTS} fetches)."]
    for rate, k, e in rows[:top]:
        if rate == 0:
            break
        lines.append(f"  {rate:5.0%}  {e['failures']:>4}/{e['attempts']:<4} "
                     f"{e.get('label', k)}")
    return "\n".join(lines)
