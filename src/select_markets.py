"""
Automated market selection by observed reporting density (Architecture 3.2 / T1).

Goal: replace all hand-picking of mandis with a measured, defensible rule —
"markets are chosen by how densely they actually report, not by convenience."

Method, per (crop, state):
  1. /geographies(crop)  -> the districts where that crop is known (cached once
     per crop). Filter to the target state's census districts.
  2. /markets(crop, state, district) -> candidate market ids (cached per district).
  3. ONE /prices call per district for a single recent probe year, passing ALL
     the district's market ids at once, then count records PER market_id in the
     response. One request prices a whole district instead of one-per-market,
     which is what keeps the probe inside its ~1-2k request budget.
  4. Rank markets within (crop, state) by probe-year record count; keep top-K.

Everything is resumable (skip-if-exists per-call JSON cache in data/probe/) and
budget-capped (--budget stops after N *new* API calls this run, then you re-run).
The probe's own request cost is SEPARATE from the backfill chunk budget; this
script's final job is to emit the proposed backfill_targets and print that
backfill chunk count so T1 can gate on the 12k ceiling before T2 launches.

Usage:
  export CEDA_API_KEY='...'
  python src/select_markets.py --probe-year 2024 --top-k 6 --budget 1500
  python src/select_markets.py --write-config        # append targets to config.yaml
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from ceda_client import CedaClient, CedaError

PROBE = Path("data/probe")

# The staple basket (Architecture 2.3). ids verified via /commodities discovery.
# Some physical commodities are split across several catalogue labels; we tag
# the candidates with a shared `group` and probe ALL of them, then keep — per
# state — whichever id reports denser (see resolve_groups). Two such splits:
#   * Red Gram:  7 "Red Gram"  vs  49 "Arhar (Tur/Red Gram)(Whole)".
#   * Rice:      3 "Rice"       vs  2 "Paddy(Dhan)(Common)".  Measured 2026-07:
#     id 3 has ~0 records in Madhya Pradesh (and Bihar) while id 2 has thousands
#     there — MP files paddy, not "Rice" — yet id 3 is denser in MH/UP/WB, so the
#     winner must be chosen state-by-state, not nationally.
CROPS = [
    {"id": 1,  "name": "Wheat"},
    {"id": 3,  "name": "Rice", "group": "rice"},
    {"id": 2,  "name": "Paddy(Dhan)(Common)", "group": "rice"},
    {"id": 23, "name": "Onion"},
    {"id": 24, "name": "Potato"},
    {"id": 78, "name": "Tomato"},
    {"id": 6,  "name": "Bengal Gram(Gram)(Whole)"},   # chana (whole grain)
    {"id": 12, "name": "Mustard"},
    {"id": 7,  "name": "Red Gram", "group": "redgram"},
    {"id": 49, "name": "Arhar (Tur/Red Gram)(Whole)", "group": "redgram"},
]

# 6 states (Architecture 2.3). census_state_id — verified equal to the state_id
# the /markets and /prices endpoints expect (Maharashtra=27, Nashik=516 matched
# the already-working config).
STATES = [
    {"id": 27, "name": "Maharashtra"},
    {"id": 9,  "name": "Uttar Pradesh"},
    {"id": 23, "name": "Madhya Pradesh"},
    {"id": 19, "name": "West Bengal"},
    {"id": 29, "name": "Karnataka"},
    {"id": 10, "name": "Bihar"},
]


class Budget:
    """Caps NEW API calls this run so the probe can never run away. Cache hits
    (skip-if-exists) do not count — only live requests do."""
    def __init__(self, limit):
        self.limit = limit
        self.spent = 0

    def check(self):
        if self.limit is not None and self.spent >= self.limit:
            raise StopIteration(f"probe budget of {self.limit} new calls reached")

    def tick(self):
        self.spent += 1


def cached_call(path, budget, fn):
    """Return cached JSON at `path`, else spend one budget unit, call fn(), cache
    it, and return it. Raises StopIteration (caught by main) when budget is out."""
    if path.exists():
        return json.loads(path.read_text())
    budget.check()
    budget.tick()
    data = fn()
    path.write_text(json.dumps(data, default=str))
    return data


def find_key(record, *fragments):
    for k in record:
        lk = k.lower()
        if all(f in lk for f in fragments):
            return k
    return None


def districts_for(client, crop, budget):
    """Cached /geographies(crop), grouped by census_state_id -> [(did, dname)]."""
    path = PROBE / f"geo_c{crop['id']}.json"
    geo = cached_call(path, budget, lambda: client.geographies(crop["id"]))
    by_state = defaultdict(list)
    for r in geo:
        by_state[r["census_state_id"]].append(
            (r["census_district_id"], r["census_district_name"]))
    return by_state


def markets_in(client, crop, state, did, budget):
    """Cached /markets(crop, state, district) -> list of {market_id, name}."""
    path = PROBE / f"mkts_c{crop['id']}_s{state['id']}_d{did}.json"
    try:
        mk = cached_call(path, budget,
                         lambda: client.markets(crop["id"], state["id"], did))
    except CedaError as e:
        # A crop that does not trade in a district often 400s here; treat as
        # "no markets" rather than aborting the whole probe.
        print(f"      (markets c{crop['id']} s{state['id']} d{did}: {str(e)[:80]})")
        path.write_text("[]")
        return []
    return mk or []


def probe_district(client, crop, state, did, market_ids, year, budget):
    """Cached ONE /prices call for a whole district+year; return {market_id: count}."""
    path = PROBE / f"probe_c{crop['id']}_s{state['id']}_d{did}_{year}.json"
    try:
        recs = cached_call(
            path, budget,
            lambda: client.prices(crop["id"], state["id"], [did], market_ids,
                                  f"{year}-01-01", f"{year}-12-31"))
    except CedaError as e:
        print(f"      (prices c{crop['id']} s{state['id']} d{did} {year}: {str(e)[:80]})")
        path.write_text("[]")
        return {}
    if not recs:
        return {}
    mkey = find_key(recs[0], "market", "id") or "market_id"
    counts = defaultdict(int)
    for r in recs:
        counts[r.get(mkey)] += 1
    return counts


def run_probe(client, year, budget, crops=CROPS, states=STATES):
    """Returns density[(crop_id, crop_name, group)][state_id] -> ranked
    [(market_id, market_name, count)]. Fully resumable; stops cleanly on budget.
    `crops`/`states` may be subsets (for parallel sharding across API keys)."""
    density = {}
    try:
        for crop in crops:
            by_state = districts_for(client, crop, budget)
            for state in states:
                dists = by_state.get(state["id"], [])
                if not dists:
                    continue
                name_by_mid = {}
                counts = defaultdict(int)
                for did, dname in dists:
                    mkts = markets_in(client, crop, state, did, budget)
                    if not mkts:
                        continue
                    mid_key = find_key(mkts[0], "market", "id") or "market_id"
                    nm_key = find_key(mkts[0], "market", "name") or "market_name"
                    mids = [m[mid_key] for m in mkts]
                    for m in mkts:
                        name_by_mid[m[mid_key]] = m.get(nm_key)
                    c = probe_district(client, crop, state, did, mids, year, budget)
                    for mid, n in c.items():
                        counts[mid] += n
                ranked = sorted(
                    ((mid, name_by_mid.get(mid), n) for mid, n in counts.items()
                     if n > 0),
                    key=lambda x: x[2], reverse=True)
                density.setdefault(
                    (crop["id"], crop["name"], crop.get("group")), {})[state["id"]] = ranked
                if ranked:
                    print(f"  {crop['name']:<28} @ {state['name']:<16} "
                          f"{len(ranked)} reporting markets "
                          f"(top={ranked[0][2]} recs)  [probe calls: {budget.spent}]")
    except StopIteration as e:
        print(f"\n!! PROBE PAUSED: {e}. Re-run the same command to continue "
              f"(cached calls are free).")
        return density, False
    return density, True


def resolve_groups(density):
    """For each commodity `group` (several catalogue ids for one physical
    commodity — Rice id 3 vs Paddy id 2, Red Gram 7 vs 49), keep, PER STATE,
    only the candidate id that reported the most probe-year records in that
    state; drop the others there. Per-state (not one national winner) because a
    split can flip by state — Rice is denser in MH/UP/WB, Paddy in MP. Ungrouped
    crops pass through untouched."""
    cands_by_group = defaultdict(list)          # group -> [(cid, cname), ...]
    for (cid, cname, group) in density:
        if group:
            cands_by_group[group].append((cid, cname))
    if not cands_by_group:
        return density

    def recs(cid, cname, group, sid):
        return sum(n for _, _, n in density[(cid, cname, group)].get(sid, []))

    winner = {}                                 # (group, sid) -> (cid, cname)
    for group, cands in cands_by_group.items():
        states = set().union(*(density[(c[0], c[1], group)].keys() for c in cands))
        for sid in states:
            best = max(cands, key=lambda c: recs(c[0], c[1], group, sid))
            winner[(group, sid)] = best
            if len(cands) > 1:
                print(f"  group '{group}' @ state {sid}: kept {best[1]} "
                      f"(id {best[0]}, {recs(best[0], best[1], group, sid)} recs) "
                      f"over {[c[0] for c in cands if c != best]}")

    out = {}
    for (cid, cname, group), states in density.items():
        if not group:
            out[(cid, cname, group)] = states
            continue
        kept = {sid: ranked for sid, ranked in states.items()
                if winner.get((group, sid)) == (cid, cname)}
        if kept:
            out[(cid, cname, group)] = kept
    return out


def build_targets(density, top_k):
    """Flatten ranked density into config backfill_targets (top_k markets per
    crop-state that actually reported), with district ids recovered from cache."""
    targets = []
    state_name = {s["id"]: s["name"] for s in STATES}
    for (cid, cname, _group), states in density.items():
        for sid, ranked in states.items():
            for mid, mname, n in ranked[:top_k]:
                did = _district_of(cid, sid, mid)
                targets.append({
                    "commodity_id": cid, "commodity_name": cname,
                    "state_id": sid, "state_name": state_name[sid],
                    "district_id": did, "market_id": mid,
                    "market_name": mname, "probe_recs": n,
                })
    return targets


def _district_of(cid, sid, mid):
    """Recover a market's district id from the cached /markets responses."""
    for p in PROBE.glob(f"mkts_c{cid}_s{sid}_d*.json"):
        try:
            mk = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not mk:
            continue
        mid_key = find_key(mk[0], "market", "id") or "market_id"
        if any(m.get(mid_key) == mid for m in mk):
            return int(p.stem.split("_d")[-1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-year", type=int, default=2024)
    ap.add_argument("--top-k", type=int, default=6,
                    help="markets kept per (crop, state) by density")
    ap.add_argument("--budget", type=int, default=1500,
                    help="max NEW api calls this run (resumable; re-run to continue)")
    ap.add_argument("--from-year", type=int, default=2008)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--write-config", action="store_true",
                    help="append proposed backfill_targets into config.yaml")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports/market_selection.json")
    ap.add_argument("--only-crops", default=None,
                    help="comma-separated commodity ids to probe (default: all). "
                         "Use for parallel sharding, one key per crop.")
    ap.add_argument("--only-states", default=None,
                    help="comma-separated state ids to probe (default: all).")
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="only warm the cache for the selected shard; do NOT write "
                         "market_selection.json. Run one final unfiltered command to "
                         "aggregate once all shards finish.")
    args = ap.parse_args()

    PROBE.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    client = CedaClient()
    budget = Budget(args.budget)

    crops, states = CROPS, STATES
    if args.only_crops:
        want = {int(x) for x in args.only_crops.split(",")}
        crops = [c for c in CROPS if c["id"] in want]
    if args.only_states:
        want = {int(x) for x in args.only_states.split(",")}
        states = [s for s in STATES if s["id"] in want]

    print(f"Market-selection probe: {len(crops)} crop-candidates x {len(states)} "
          f"states, probe year {args.probe_year}, top-{args.top_k} per cell, "
          f"budget {args.budget} new calls.\n")
    density, complete = run_probe(client, args.probe_year, budget, crops, states)

    if args.skip_aggregate:
        print(f"\n[shard] cache warmed for crops={[c['id'] for c in crops]} "
              f"states={[s['id'] for s in states]} — {budget.spent} new calls, "
              f"complete={complete}. Skipping aggregation; run the full unfiltered "
              f"command once every shard is done.")
        return
    density = resolve_groups(density)
    targets = build_targets(density, args.top_k)

    n_years = args.to_year - args.from_year + 1
    chunk_budget = len(targets) * n_years * 2

    # Persist the machine-readable selection + budget.
    Path(args.out).write_text(json.dumps({
        "probe_year": args.probe_year, "top_k": args.top_k,
        "from_year": args.from_year, "to_year": args.to_year,
        "probe_calls_spent": budget.spent, "probe_complete": complete,
        "n_target_pairs": len(targets), "n_years": n_years,
        "backfill_chunk_budget": chunk_budget,
        "targets": targets,
    }, indent=2, default=str))

    print("\n" + "=" * 66)
    print(f"PROBE {'COMPLETE' if complete else 'INCOMPLETE (budget paused)'} "
          f"— {budget.spent} new api calls this run.")
    print(f"Proposed backfill: {len(targets)} (crop, market) pairs "
          f"x {n_years} years x 2 endpoints")
    print(f"  => BACKFILL CHUNK BUDGET = {chunk_budget:,} chunks")
    ceiling = 12000
    if chunk_budget > ceiling:
        print(f"  !! OVER the {ceiling:,}-chunk ceiling by {chunk_budget-ceiling:,}. "
              f"Do NOT launch T2. Lower --top-k or narrow scope and re-run.")
    else:
        print(f"  OK — within the {ceiling:,}-chunk ceiling "
              f"({ceiling-chunk_budget:,} headroom).")
    print(f"Wrote {args.out}")
    if not complete:
        print("NOTE: probe incomplete — budget is a LOWER BOUND; re-run to finish "
              "before trusting the number.")

    if args.write_config and complete:
        _write_config(args.config, targets)
        print(f"Appended {len(targets)} targets to {args.config}.")
    elif args.write_config:
        print("Refusing --write-config on an incomplete probe.")


def _write_config(config_path, targets):
    cfg = yaml.safe_load(open(config_path)) or {}
    clean = [{k: t[k] for k in ("commodity_id", "commodity_name", "state_id",
                                "state_name", "district_id", "market_id",
                                "market_name")} for t in targets]
    cfg["backfill_targets_auto"] = clean
    Path(config_path).write_text(yaml.safe_dump(cfg, sort_keys=False, width=100))


if __name__ == "__main__":
    main()
