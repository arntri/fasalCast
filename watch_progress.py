#!/usr/bin/env python3
"""Per-second PROGRESS report — districts probed / districts total, per shard.

    watch -n 1 python3 ~/mandi-ews/watch_progress.py

Why this and not the log: select_markets.py only prints a summary line when a
whole (crop, state) cell finishes, so the log looks frozen for minutes. But the
denominator is knowable up front — each crop's cached /geographies lists every
district it trades in. So:

  total  = districts of this crop across the 6 target states  (from geo cache)
  done   = districts that already have a cached /markets response
  NOW    = the newest file decoded -> which (state, district, call) is live now

Every /markets request writes one mkts_* file, so `done` climbs by one per
district and the bar reaches 100% when the shard is finished.
"""
import json
import re
from pathlib import Path

PROBE = Path(__file__).resolve().parent / "data" / "probe"
# Phase-A geography after Scope-Correction Task 3: dropped Bihar(10) + Karnataka(29)
# for structural data reasons, added qualified Chhattisgarh(22) + Rajasthan(8).
# See reports/state_qualification.md.
TARGET_STATES = {27: "MH", 9: "UP", 23: "MP", 19: "WB", 22: "CG", 8: "RJ"}
SHARDS = [  # (log-label, [crop ids in this shard], display name)
    ("c1", [1], "Wheat"), ("c3", [3, 2], "Rice+Paddy"), ("c23", [23], "Onion"),
    ("c24", [24], "Potato"), ("c78", [78], "Tomato"),
    ("c6", [6], "BengalGram"), ("c12", [12], "Mustard"),
    ("c7_49", [7, 49], "RedGram"),
]
FNAME = re.compile(r"(mkts|probe)_c(\d+)_s(\d+)_d(\d+)")


def districts_total(crop_ids):
    """Districts this shard must probe = geo rows in the 6 target states.
    Returns (total, complete): complete=False when some crop's /geographies
    isn't cached yet, so `total` is a lower bound (shown with a trailing '+')."""
    total, complete = 0, True
    for cid in crop_ids:
        geo = PROBE / f"geo_c{cid}.json"
        if not geo.exists():
            complete = False  # not fetched yet -> denominator is a lower bound
            continue
        rows = json.loads(geo.read_text())
        total += sum(1 for r in rows if r["census_state_id"] in TARGET_STATES)
    return total, complete


def bar(done, total, width=22):
    if not total:
        return "[" + "?" * width + "]   --"
    frac = min(done / total, 1.0)
    fill = int(frac * width)
    return "[" + "#" * fill + "." * (width - fill) + f"] {frac*100:3.0f}%"


def decode_newest(crop_ids):
    """Newest file across this shard -> 'UP d197 prices' (what's live now)."""
    files = []
    for cid in crop_ids:
        files += PROBE.glob(f"*_c{cid}_*")
    if not files:
        return "-"
    newest = max(files, key=lambda p: p.stat().st_mtime)
    m = FNAME.search(newest.name)
    if not m:
        return newest.name
    kind, cid, sid, did = m.groups()
    call = "markets" if kind == "mkts" else "prices "
    st = TARGET_STATES.get(int(sid), f"s{sid}")
    return f"{call} {st} d{did}"


def districts_done(crop_ids):
    """Districts probed = mkts_ files IN THE TARGET STATES only (files for the
    dropped states still sit in the cache, so glob-all would over-count)."""
    done = 0
    for cid in crop_ids:
        for p in PROBE.glob(f"mkts_c{cid}_s*_d*.json"):
            m = FNAME.search(p.name)
            if m and int(m.group(3)) in TARGET_STATES:
                done += 1
    return done


def main():
    print(f"{'SHARD':<9} {'CROP':<11} {'PROGRESS':<30} {'DIST':<9} NOW")
    print("-" * 92)
    for label, crop_ids, name in SHARDS:
        total, complete = districts_total(crop_ids)
        done = districts_done(crop_ids)
        dist = f"{done}/{total}" + ("" if complete else "+")
        print(f"{label:<9} {name:<11} {bar(done, total):<30} {dist:<9} "
              f"{decode_newest(crop_ids)}")


if __name__ == "__main__":
    main()
