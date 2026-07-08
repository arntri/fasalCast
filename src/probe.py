"""
Milestone 1 probe (v2 — adapted to the real API envelope).

The live API wraps everything as {"output": {"type","message","data"}} and uses
field names like commodity_id/commodity_name. Since the rest of the schema may
also differ from the docs, this probe DISCOVERS field names at runtime and
dumps the raw first record whenever a step sees an unexpected shape.

Usage:
  export CEDA_API_KEY='...'
  python src/probe.py [--month 2025-03] [--district Nashik] [--market Lasalgaon]
"""

import argparse
import calendar
import json
import sys

import pandas as pd

from ceda_client import CedaClient, CedaError


def dump(label, obj, n=2):
    print(f"  RAW {label} (first {n}):")
    print(json.dumps(obj[:n] if isinstance(obj, list) else obj,
                     indent=2, default=str)[:1500])


def find_key(record, *fragments):
    """Return the first key of `record` containing all fragments (case-insens)."""
    for k in record:
        lk = k.lower()
        if all(f in lk for f in fragments):
            return k
    return None


def pick(items, name_key, target, label):
    hits = [x for x in items if target.lower() in str(x.get(name_key, "")).lower()]
    if not hits:
        print(f"  !! No {label} matching '{target}'. Available: "
              f"{sorted(str(x.get(name_key)) for x in items)[:40]}")
        return None
    if len(hits) > 1:
        print(f"  note: multiple {label} matches: {[x[name_key] for x in hits]} — using first.")
    return hits[0]


def quality_report(df, from_date, to_date):
    print("\n--- Mini quality report (one-month slice) ---")
    print(f"records: {len(df)}")
    if df.empty:
        print("EMPTY response — pair may not report, or params/date range wrong.")
        return
    date_col = find_key(df.iloc[0].to_dict(), "date") or "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    print(f"unparseable dates: {int(df[date_col].isna().sum())}")
    print(f"date span returned: {df[date_col].min().date()} -> {df[date_col].max().date()}")

    span = pd.date_range(from_date, to_date, freq="D")
    non_sun = span[span.dayofweek != 6]
    covered = df[date_col].dt.normalize().nunique()
    print(f"days reported: {covered} / {len(non_sun)} non-Sundays ({covered/len(non_sun):.0%})")

    mkt_col = find_key(df.iloc[0].to_dict(), "market")
    dup_keys = [c for c in (date_col, mkt_col) if c]
    dups = int(df.duplicated(subset=dup_keys, keep=False).sum())
    print(f"duplicate {tuple(dup_keys)} rows: {dups}")

    price_cols = [c for c in df.columns if "price" in c.lower()]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"{col}: nulls={int(df[col].isna().sum())}, "
              f"min={df[col].min()}, max={df[col].max()}")

    mn = find_key(df.iloc[0].to_dict(), "min", "price")
    mx = find_key(df.iloc[0].to_dict(), "max", "price")
    md = find_key(df.iloc[0].to_dict(), "modal")
    if mn and mx:
        print(f"min>max rows: {int((df[mn] > df[mx]).sum())}")
    if mn and mx and md:
        outside = int(((df[md] < df[mn]) | (df[md] > df[mx])).sum())
        print(f"modal outside [min,max]: {outside} | modal<=0: {int((df[md] <= 0).sum())}")

    print("\nsample rows:")
    print(df.head(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2025-03", help="YYYY-MM")
    ap.add_argument("--district", default="Nashik")
    ap.add_argument("--market", default="Lasalgaon")
    ap.add_argument("--market-id", type=int, default=None,
                    help="Skip the /markets call entirely and use this id.")
    args = ap.parse_args()

    y, m = map(int, args.month.split("-"))
    from_date = f"{y}-{m:02d}-01"
    to_date = f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]}"

    client = CedaClient()

    print("[1/7] GET /agmarknet/commodities (auth check)...")
    commodities = client.commodities()
    print(f"  OK — {len(commodities)} commodities. Auth works.")

    print("[2/7] Locating Onion...")
    c0 = commodities[0]
    c_name = find_key(c0, "name") or "commodity_name"
    c_id = find_key(c0, "commodity", "id") or find_key(c0, "id") or "commodity_id"
    onion = pick(commodities, c_name, "onion", "commodity")
    if not onion:
        dump("commodities", commodities)
        sys.exit(1)
    onion_id = onion[c_id]
    print(f"  Onion: {c_id}={onion_id} ({c_name}='{onion[c_name]}')")

    print("[3/7] GET /agmarknet/geographies...")
    geos = client.geographies(onion_id)
    if not geos:
        sys.exit("  Empty geographies response.")
    g0 = geos[0]
    s_name = find_key(g0, "state", "name")
    s_id = find_key(g0, "state", "id")
    d_name = find_key(g0, "district", "name")
    d_id = find_key(g0, "district", "id")

    if s_name and s_id and d_name and d_id:
        # FLAT shape (real API): one row per (state, district)
        mh_rows = [g for g in geos
                   if "maharashtra" in str(g[s_name]).lower()]
        if not mh_rows:
            print(f"  !! No Maharashtra rows. States available: "
                  f"{sorted({str(g[s_name]) for g in geos})}")
            sys.exit(1)
        dist = pick(mh_rows, d_name, args.district, "district")
        if not dist:
            sys.exit(1)
        state_id, district_id = dist[s_id], dist[d_id]
        print(f"  Maharashtra {s_id}={state_id}, "
              f"{dist[d_name]} {d_id}={district_id}")
    else:
        # NESTED shape (documented): state -> districts[]
        d_list_key = next((k for k in g0 if isinstance(g0[k], list)), None)
        if not (s_name and s_id and d_list_key):
            dump("geographies", geos)
            sys.exit("  Unexpected geographies shape — paste the RAW dump above to Claude.")
        mh = pick(geos, s_name, "maharashtra", "state")
        if not mh:
            sys.exit(1)
        dd0 = mh[d_list_key][0]
        dn = find_key(dd0, "district", "name")
        di = find_key(dd0, "district", "id")
        if not (dn and di):
            dump("districts", mh[d_list_key])
            sys.exit("  Unexpected district shape — paste the RAW dump above to Claude.")
        dist = pick(mh[d_list_key], dn, args.district, "district")
        if not dist:
            sys.exit(1)
        state_id, district_id = mh[s_id], dist[di]
        print(f"  Maharashtra {s_id}={state_id}, {dist[dn]} {di}={district_id}")

    print("[4/7] Resolving market...")
    m_name, m_id = "market_name", "market_id"
    cache_path = f"reports/markets_{onion_id}_{district_id}.json"
    if args.market_id is not None:
        mkt = {m_name: args.market, m_id: args.market_id}
        print(f"  Using --market-id {args.market_id} ('{args.market}'), "
              f"skipping /markets call.")
    else:
        import os.path
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                mkts = json.load(f)
            print(f"  Loaded {len(mkts)} markets from cache ({cache_path}).")
        else:
            print("  Calling POST /agmarknet/markets (this endpoint can be "
                  "slow — retries wait up to 2 min each)...")
            mkts = client.markets(onion_id, state_id, district_id)
            if not mkts:
                sys.exit("  Empty markets response.")
            with open(cache_path, "w") as f:
                json.dump(mkts, f, indent=2)
            print(f"  Cached markets list to {cache_path}.")
        m0 = mkts[0]
        m_name = find_key(m0, "market", "name")
        m_id = find_key(m0, "market", "id")
        if not (m_name and m_id):
            dump("markets", mkts)
            sys.exit("  Unexpected markets shape — paste the RAW dump above to Claude.")
        print(f"  {len(mkts)} markets: {[x[m_name] for x in mkts]}")
        mkt = pick(mkts, m_name, args.market, "market") or mkts[0]
    print(f"  Using: {mkt[m_name]} ({m_id}={mkt[m_id]})")

    print(f"[5/7] POST /agmarknet/prices, {from_date} -> {to_date}...")
    data = client.prices(onion_id, state_id, [district_id], [mkt[m_id]],
                         from_date, to_date)
    if data:
        dump("prices", data, n=1)
    df = pd.DataFrame(data)
    quality_report(df, from_date, to_date)

    print(f"\n[6/7] POST /agmarknet/quantities, same market & range...")
    qdata = client.quantities(onion_id, state_id, [district_id], [mkt[m_id]],
                              from_date, to_date)
    if qdata:
        dump("quantities", qdata, n=1)
    qdf = pd.DataFrame(qdata)

    print("\n[7/7] Structural comparison (the variety/duplicate question)...")
    if qdf.empty:
        print("  quantities EMPTY — arrivals may not be reported for this market.")
    else:
        qd = find_key(qdf.iloc[0].to_dict(), "date") or "date"
        qdf[qd] = pd.to_datetime(qdf[qd], errors="coerce")
        pd_date = find_key(df.iloc[0].to_dict(), "date") or "date"
        p_per_day = df.groupby(df[pd_date].dt.normalize()).size()
        q_per_day = qdf.groupby(qdf[qd].dt.normalize()).size()
        print(f"  prices:     {len(df)} rows over {p_per_day.size} days "
              f"(rows/day: min={p_per_day.min()}, max={p_per_day.max()})")
        print(f"  quantities: {len(qdf)} rows over {q_per_day.size} days "
              f"(rows/day: min={q_per_day.min()}, max={q_per_day.max()})")
        qty_col = find_key(qdf.iloc[0].to_dict(), "quantity")
        if qty_col:
            qdf[qty_col] = pd.to_numeric(qdf[qty_col], errors="coerce")
            print(f"  {qty_col}: nulls={int(qdf[qty_col].isna().sum())}, "
                  f"min={qdf[qty_col].min()}, max={qdf[qty_col].max()}, "
                  f"zero-days={int((qdf[qty_col]==0).sum())}")
        both = p_per_day.index.intersection(q_per_day.index)
        if len(both):
            same = int((p_per_day[both] == q_per_day[both]).sum())
            print(f"  days present in both: {len(both)}; days with SAME row-count "
                  f"in both: {same}")
            print("  -> if row-counts match daily, rows are parallel variety/grade "
                  "records; if quantities is 1/day, prices must be aggregated.")

    out = f"reports/probe_{args.month}_{str(mkt[m_name]).replace(' ', '_')}.json"
    with open(out, "w") as f:
        json.dump({"params": vars(args),
                   "ids": {"commodity_id": onion_id, "state_id": state_id,
                           "district_id": district_id, "market": mkt},
                   "prices": data, "quantities": qdata}, f, indent=2, default=str)
    print(f"\nRaw probe response saved to {out}")
    print("Next: same probe on /quantities, then the chunked backfill.")


if __name__ == "__main__":
    try:
        main()
    except CedaError as e:
        sys.exit(f"CEDA API error: {e}")
