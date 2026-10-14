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
# Phase-A geography after Scope-Correction Task 3 (reports/state_qualification.md):
# dropped Karnataka(29) + Bihar(10) for structural under-reporting, added the two
# qualified poverty-relevant replacements Chhattisgarh(22) + Rajasthan(8). This is
# the set the write-config aggregation iterates; it must match watch_progress.py's
# TARGET_STATES and the qualification report.
STATES = [
    {"id": 27, "name": "Maharashtra"},
    {"id": 9,  "name": "Uttar Pradesh"},
    {"id": 23, "name": "Madhya Pradesh"},
    {"id": 19, "name": "West Bengal"},
    {"id": 22, "name": "Chhattisgarh"},
    {"id": 8,  "name": "Rajasthan"},
]

# ---- Scope-Correction Task 3: state qualification (PRE-REGISTERED) -----------
# Phase-A geography is chosen by MEASURED reporting density, not hand-picking.
# Two states are DROPPED from phase A for STRUCTURAL data reasons (not appetite),
# and poverty-relevant candidates are probed to replace them. Every threshold
# below is fixed BEFORE the probe runs — it is NOT tuned to the outcome.
PHASE_A_EXCLUDE = {
    10: "Bihar — APMC (regulated mandi) Act repealed 2006; structurally few "
        "reporting markets. Measured: 4 of 8 crop groups absent, e.g. wheat "
        "reported by only 6 mandis vs 286 in MP.",
    29: "Karnataka — most trade flows through the state's own e-market platform "
        "(ReMS / Unified Market Platform), bypassing central Agmarknet. Measured: "
        "~37-68 reporting days per mandi per year vs 150-290 in the kept states.",
}

# Candidates chosen for HIGH poverty / food-insecurity relevance — the mission
# constraint is to keep poor states in scope, NOT to swap them for data-rich
# richer ones. Poverty tags are qualitative (NITI National MPI ranking), stated
# without fabricated precise figures.
QUALIFY_CANDIDATES = [
    {"id": 21, "name": "Odisha",       "poverty": "high — persistently poor, eastern rice belt"},
    {"id": 22, "name": "Chhattisgarh", "poverty": "high — tribal-majority, among India's poorest by MPI"},
]
QUALIFY_FALLBACKS = [   # probed ONLY if a primary candidate fails to qualify
    {"id": 20, "name": "Jharkhand", "poverty": "very high — consistently among the poorest by MPI"},
    {"id": 8,  "name": "Rajasthan", "poverty": "moderate — large arid agrarian population"},
]

# Hard probe caps (cost control) + the pre-registered qualification rule.
QUAL_TOP_DISTRICTS      = 5       # per state, ranked by cereal market count
QUAL_TOP_MANDIS         = 3       # per district, ranked by cross-group presence
QUAL_DENSE_RECORDS      = 150     # a mandi is "dense" for a group at >= this many probe-year records
QUAL_MIN_DENSE_MANDIS   = 5       # a state QUALIFIES with >= this many dense mandis...
QUAL_MIN_GROUPS_COVERED = 5       # ...that together cover >= this many of the 8 crop groups
QUAL_REF_CROPS          = [3, 2, 1]   # rank districts by rice(3)+paddy(2)+wheat(1) market count
                                      # (cereal infrastructure, robust to rice/paddy labeling)


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
        msg = str(e)
        # 400 = crop genuinely does not trade in this district: a real "no
        # markets" answer, safe to cache [] and move on.
        if "400" in msg:
            print(f"      (markets c{crop['id']} s{state['id']} d{did}: no markets)")
            path.write_text("[]")
            return []
        # 401/403 = auth broken: systemic, abort the probe loudly. NEVER cache []
        # for it — the cache is skip-if-exists, so a poisoned cell is permanent.
        # (A dead key once wrote 175 empty Paddy cells exactly this way.)
        if "401" in msg or "403" in msg:
            raise
        # Anything else (504/5xx/429/timeout) is a transient server glitch: do
        # NOT cache (would poison) and do NOT crash the whole probe — skip this
        # district so a re-run retries it. CEDA's gateway 504s intermittently.
        print(f"      (markets c{crop['id']} s{state['id']} d{did}: transient "
              f"{msg[:70]} — skipped, retry on re-run)")
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
    crop-state that actually reported), with district ids recovered from cache.
    States in PHASE_A_EXCLUDE (Bihar, Karnataka) are dropped for the structural
    reasons recorded there — see Scope-Correction Task 3 / state_qualification."""
    targets = []
    # Name map spans the full pool so the phase-A qualified candidates
    # (Chhattisgarh 22, Rajasthan 8 — which live in QUALIFY_*, not STATES) are
    # resolvable at aggregation time. Without this the fan-in KeyErrors / silently
    # drops the two newly-qualified states from backfill targets.
    state_name = {s["id"]: s["name"]
                  for s in STATES + QUALIFY_CANDIDATES + QUALIFY_FALLBACKS}
    for (cid, cname, _group), states in density.items():
        for sid, ranked in states.items():
            if sid in PHASE_A_EXCLUDE:
                continue          # dropped from phase A (structural, recorded)
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


def load_crop_groups(config_path):
    """The 8 crop groups (basket unit) from config.yaml -> {group: [ids]}."""
    cfg = yaml.safe_load(open(config_path)) or {}
    groups = cfg.get("crop_groups")
    if not groups:
        raise SystemExit(f"config {config_path} has no crop_groups (Task 1).")
    return groups


def _qual_prices(client, cid, sid, did, market_ids, year, budget):
    """Qualification prices probe: ONE cached /prices for a district's chosen
    mandis, prices only, one year. Separate cache namespace (qual_*) so it never
    collides with the density probe's whole-district probe_* files. Returns
    {market_id: record_count}."""
    path = PROBE / f"qual_c{cid}_s{sid}_d{did}_{year}.json"
    try:
        recs = cached_call(
            path, budget,
            lambda: client.prices(cid, sid, [did], market_ids,
                                  f"{year}-01-01", f"{year}-12-31"))
    except CedaError as e:
        print(f"      (qual prices c{cid} s{sid} d{did}: {str(e)[:80]})")
        path.write_text("[]")
        return {}
    if not recs:
        return {}
    mkey = find_key(recs[0], "market", "id") or "market_id"
    counts = defaultdict(int)
    for r in recs:
        counts[r.get(mkey)] += 1
    return counts


def qualify_one_state(client, st, groups, year, budget):
    """Run the pre-registered qualification probe for ONE state under the hard
    caps. Returns a verdict dict. May raise StopIteration if budget runs out
    (caught upstream; everything cached, so re-run resumes)."""
    sid, sname = st["id"], st["name"]
    state = {"id": sid, "name": sname}

    # 1. Rank districts by cereal (ref-crop) market count; keep the top-K.
    mkts_by_did = defaultdict(dict)          # did -> {mid: mname}
    for cid in QUAL_REF_CROPS:
        crop = {"id": cid, "name": f"c{cid}"}
        by_state = districts_for(client, crop, budget)
        for did, _dname in by_state.get(sid, []):
            for m in markets_in(client, crop, state, did, budget):
                mid = m[find_key(m, "market", "id") or "market_id"]
                mkts_by_did[did][mid] = m.get(find_key(m, "market", "name") or "market_name")
    top_dids = sorted(mkts_by_did, key=lambda d: len(mkts_by_did[d]),
                      reverse=True)[:QUAL_TOP_DISTRICTS]

    # 2+3. For each top district: pick top-K mandis by cross-GROUP presence,
    #      then price them (one year) and tally records per (mandi, group).
    dense_mandis = {}                        # mid -> mname  (>=150 recs for >=1 group)
    groups_hit = set()
    probed_mandis = set()
    for did in top_dids:
        present = defaultdict(int)           # mid -> # groups listing it
        grp_mkts = defaultdict(dict)         # group -> {mid: mname} in this district
        for gname, ids in groups.items():
            gmids = {}
            for cid in ids:
                crop = {"id": cid, "name": f"c{cid}"}
                for m in markets_in(client, crop, state, did, budget):
                    mid = m[find_key(m, "market", "id") or "market_id"]
                    gmids[mid] = m.get(find_key(m, "market", "name") or "market_name")
            for mid, nm in gmids.items():
                grp_mkts[gname][mid] = nm
                present[mid] += 1            # +1 per GROUP (not per member id)
        top_mids = sorted(present, key=present.get, reverse=True)[:QUAL_TOP_MANDIS]
        if not top_mids:
            continue
        probed_mandis.update(top_mids)
        for gname, ids in groups.items():
            got = defaultdict(int)           # mid -> records for this group
            for cid in ids:
                counts = _qual_prices(client, cid, sid, did, top_mids, year, budget)
                for mid, n in counts.items():
                    if mid in top_mids:
                        got[mid] += n
            for mid, n in got.items():
                if n >= QUAL_DENSE_RECORDS:
                    dense_mandis[mid] = grp_mkts[gname].get(mid) or mkts_by_did[did].get(mid)
                    groups_hit.add(gname)

    qualified = (len(dense_mandis) >= QUAL_MIN_DENSE_MANDIS
                 and len(groups_hit) >= QUAL_MIN_GROUPS_COVERED)
    print(f"  {sname}: {len(probed_mandis)} mandis probed, "
          f"{len(dense_mandis)} dense, {len(groups_hit)}/8 groups "
          f"-> {'QUALIFIED' if qualified else 'NOT qualified'}")
    return {
        "state_id": sid, "state_name": sname, "poverty": st["poverty"],
        "role": st.get("role", "candidate"),
        "districts_probed": len(top_dids), "mandis_probed": len(probed_mandis),
        "dense_mandis": len(dense_mandis),
        "groups_covered": sorted(groups_hit), "qualified": qualified,
    }


def run_qualification(client, budget, groups, year, out_md,
                      candidates=None, fallbacks=None):
    """Probe primary candidates; probe a fallback only when a primary fails, in
    order, until the primary shortfall is covered or fallbacks run out. Writes
    reports/state_qualification.md. Resumable: budget-pause writes a partial
    report and asks for a re-run.

    candidates/fallbacks default to the module lists. Pass a filtered candidate
    list (and fallbacks=[]) to warm ONE state's cache in a parallel process on a
    separate key; the orchestrating full run then sails through that state on
    cache hits and writes the real combined report with fallback logic intact."""
    candidates = QUALIFY_CANDIDATES if candidates is None else candidates
    fallbacks = QUALIFY_FALLBACKS if fallbacks is None else fallbacks
    results, complete = [], True
    try:
        for st in candidates:
            print(f"\n[qualify] {st['name']} (s{st['id']}) ...")
            results.append(qualify_one_state(client, st, groups, year, budget))
        shortfall = sum(1 for r in results if not r["qualified"])
        fbs = iter(fallbacks)
        while shortfall > 0:
            st = next(fbs, None)
            if st is None:
                break
            print(f"\n[qualify] primary shortfall -> fallback {st['name']} ...")
            r = qualify_one_state(client, dict(st, role="fallback"), groups, year, budget)
            results.append(r)
            if r["qualified"]:
                shortfall -= 1
    except StopIteration as e:
        complete = False
        print(f"\n!! QUALIFY PAUSED: {e}. Re-run the same command to continue "
              f"(cached calls are free).")
    _write_qualification_report(results, year, out_md, complete)
    return results, complete


def _write_qualification_report(results, year, out_md, complete):
    n_groups = 8
    lines = [
        "# State Qualification Probe — Food Access Watch (Scope-Correction Task 3)",
        "",
        f"Probe year **{year}**. Status: "
        f"**{'COMPLETE' if complete else 'INCOMPLETE (budget-paused; re-run)'}**.",
        "",
        "**Pre-registered rule (fixed before probing):** a state QUALIFIES for "
        f"phase A if **>= {QUAL_MIN_DENSE_MANDIS} mandis** show dense reporting "
        f"(**>= {QUAL_DENSE_RECORDS} price records** in the probe year) together "
        f"covering **>= {QUAL_MIN_GROUPS_COVERED} of {n_groups} crop groups**.",
        "",
        f"**Hard caps:** top {QUAL_TOP_DISTRICTS} districts by cereal market "
        f"count, top {QUAL_TOP_MANDIS} mandis per district, one year, PRICES "
        "only, every /markets response cached.",
        "",
        "## Dropped from phase A (structural, not appetite)",
        "",
    ]
    for sid, reason in PHASE_A_EXCLUDE.items():
        lines.append(f"- **{reason.split(' — ')[0]} (state {sid})** — "
                     f"{reason.split(' — ', 1)[1]}")
    lines += [
        "",
        "## Candidates probed (poverty-relevant replacements)",
        "",
        "| State | Role | Poverty relevance | Districts | Mandis probed | "
        "Dense mandis | Groups covered | QUALIFIED |",
        "|---|---|---|---:|---:|---:|---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['state_name']} ({r['state_id']}) | {r['role']} | {r['poverty']} "
            f"| {r['districts_probed']} | {r['mandis_probed']} | {r['dense_mandis']} "
            f"| {len(r['groups_covered'])} ({', '.join(r['groups_covered']) or '—'}) "
            f"| {'✅ YES' if r['qualified'] else '❌ no'} |")
    qualified = [r["state_name"] for r in results if r["qualified"]]
    lines += [
        "",
        f"**Phase-A geography decision:** kept states = the 4 dense originals "
        f"(Maharashtra, Uttar Pradesh, Madhya Pradesh, West Bengal) + qualified "
        f"candidates = **{', '.join(qualified) if qualified else '(none yet)'}**.",
        "",
        "_Scope is evidence, not opinion: this table — not a hand-picked list — "
        "decides phase-A geography._",
    ]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_md}")


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
    ap.add_argument("--qualify-states", action="store_true",
                    help="Run the pre-registered state-qualification probe (Task 3) "
                         "over the poverty-relevant candidate states instead of the "
                         "density probe; write reports/state_qualification.md.")
    ap.add_argument("--qual-out", default="reports/state_qualification.md")
    args = ap.parse_args()

    PROBE.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    client = CedaClient()
    budget = Budget(args.budget)

    if args.qualify_states:
        groups = load_crop_groups(args.config)
        candidates, fallbacks = QUALIFY_CANDIDATES, QUALIFY_FALLBACKS
        if args.only_states:
            want = {int(x) for x in args.only_states.split(",")}
            pool = QUALIFY_CANDIDATES + QUALIFY_FALLBACKS
            candidates = [s for s in pool if s["id"] in want]
            fallbacks = []   # cache-warm mode: no fallback orchestration here —
                             # the full run owns that and the combined report.
            print(f"[qualify] cache-warm mode: states {sorted(want)} only, "
                  f"fallbacks suppressed (a full run writes the real report).")
        print(f"State-qualification probe (Task 3): candidates "
              f"{[s['name'] for s in candidates]}, fallbacks "
              f"{[s['name'] for s in fallbacks]}, probe year "
              f"{args.probe_year}, budget {args.budget} new calls.\n"
              f"Rule: >={QUAL_MIN_DENSE_MANDIS} dense mandis "
              f"(>={QUAL_DENSE_RECORDS} recs) over >={QUAL_MIN_GROUPS_COVERED}/8 groups.")
        run_qualification(client, budget, groups, args.probe_year, args.qual_out,
                          candidates=candidates, fallbacks=fallbacks)
        return

    crops, states = CROPS, STATES
    if args.only_crops:
        want = {int(x) for x in args.only_crops.split(",")}
        crops = [c for c in CROPS if c["id"] in want]
    if args.only_states:
        want = {int(x) for x in args.only_states.split(",")}
        # Resolve from the full pool so qualified phase-A candidates (Chhattisgarh,
        # Rajasthan) — which live in the QUALIFY_* lists, not STATES — are probeable
        # by id. STATES stays the master list the qualification report is built on.
        pool = STATES + QUALIFY_CANDIDATES + QUALIFY_FALLBACKS
        states = [s for s in pool if s["id"] in want]

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
