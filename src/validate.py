"""
Milestone 1 validation harness.

Takes raw mandi CSV(s), applies the column map from config.yaml, and produces
a data-quality report answering: what's missing, inconsistent, or dirty —
per crop-market pair — plus a verdict on which pairs are viable for modeling.

Usage:
    python src/validate.py data/raw/*.csv --config config.yaml --out reports/

No modeling here. This must pass before anything else gets built.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------- loading ----------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(paths, column_map):
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={k.lower(): v for k, v in column_map.items()})
        df["__source_file"] = Path(p).name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw


def coerce(df):
    issues = {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    issues["unparseable_dates"] = int(df["date"].isna().sum())
    for col in ["arrivals_qtl", "min_price", "max_price", "modal_price"]:
        if col in df:
            before = df[col].copy()
            df[col] = pd.to_numeric(
                before.astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            issues[f"non_numeric_{col}"] = int(df[col].isna().sum() - before.isna().sum())
    for col in ["state", "market", "commodity", "variety"]:
        if col in df:
            df[col] = df[col].astype(str).str.strip()
    return df, issues


# ---------- checks ----------

def name_variants(df):
    """Spot near-duplicate market/commodity spellings (case/whitespace variants)."""
    out = {}
    for col in ["market", "commodity"]:
        if col not in df:
            continue
        norm = df[col].str.lower().str.replace(r"[^a-z]", "", regex=True)
        groups = df.groupby(norm)[col].unique()
        variants = {k: list(v) for k, v in groups.items() if len(v) > 1}
        out[col] = variants
    return out


def duplicates(df):
    keys = [k for k in ["date", "market", "commodity", "variety"] if k in df]
    dup_mask = df.duplicated(subset=keys, keep=False)
    dups = df[dup_mask]
    conflicting = 0
    if not dups.empty and "modal_price" in df:
        conflicting = int(
            dups.groupby(keys)["modal_price"].nunique().gt(1).sum()
        )
    return {
        "duplicate_rows": int(dup_mask.sum()),
        "duplicate_keys_with_conflicting_prices": conflicting,
    }


def pair_report(g, cfg):
    """Quality report for one (market, commodity) pair."""
    v = cfg["validation"]
    g = g.sort_values("date")
    start, end = g["date"].min(), g["date"].max()
    span = pd.date_range(start, end, freq="D")
    non_sun = span[span.dayofweek != 6]  # markets typically closed Sundays
    reported = g["date"].dt.normalize().nunique()
    coverage = reported / max(len(non_sun), 1)

    # longest gap in days
    gaps = g["date"].dt.normalize().drop_duplicates().diff().dt.days
    longest_gap = int(gaps.max()) if len(gaps) > 1 else 0

    # zero vs missing arrivals
    zero_arrivals = int((g["arrivals_qtl"] == 0).sum()) if "arrivals_qtl" in g else None
    null_arrivals = int(g["arrivals_qtl"].isna().sum()) if "arrivals_qtl" in g else None

    # price sanity
    lo, hi = v["price_bounds"].get(g["commodity"].iloc[0], [0, np.inf])
    out_of_bounds = int(((g["modal_price"] < lo) | (g["modal_price"] > hi)).sum())
    min_gt_max = int((g["min_price"] > g["max_price"]).sum()) if {"min_price", "max_price"} <= set(g) else None
    modal_outside = None
    if {"min_price", "max_price", "modal_price"} <= set(g):
        modal_outside = int(
            ((g["modal_price"] < g["min_price"]) | (g["modal_price"] > g["max_price"])).sum()
        )

    # day-over-day price jumps
    daily = g.groupby(g["date"].dt.normalize())["modal_price"].median()
    jumps = daily.pct_change().abs()
    big_jumps = int((jumps > v["max_daily_price_jump"]).sum())

    # possible unit shift in arrivals (quintal/tonne mix-up shows as ~10x level shift)
    unit_shift = False
    if "arrivals_qtl" in g and g["arrivals_qtl"].notna().sum() > 60:
        arr = g.groupby(g["date"].dt.normalize())["arrivals_qtl"].sum()
        rm = arr.rolling(30, min_periods=15).median()
        ratio = (rm / rm.shift(30)).replace([np.inf, -np.inf], np.nan).dropna()
        f = v["unit_shift_factor"]
        unit_shift = bool(((ratio > f) | (ratio < 1 / f)).any())

    viable = coverage >= v["min_coverage"] and len(g) >= v["min_rows"]
    return {
        "rows": int(len(g)),
        "date_start": str(start.date()),
        "date_end": str(end.date()),
        "coverage_ex_sunday": round(float(coverage), 3),
        "longest_gap_days": longest_gap,
        "zero_arrival_days": zero_arrivals,
        "null_arrivals": null_arrivals,
        "modal_price_out_of_bounds": out_of_bounds,
        "min_gt_max": min_gt_max,
        "modal_outside_minmax": modal_outside,
        "daily_jumps_gt_threshold": big_jumps,
        "possible_unit_level_shift": unit_shift,
        "VIABLE_FOR_MODELING": viable,
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = [p for pat in args.inputs for p in glob.glob(pat)]
    if not paths:
        sys.exit("No input files matched.")

    if paths[0].endswith(".parquet"):
        # Panel mode: cleaned daily panel from clean.py (ids, not names)
        df = pd.read_parquet(paths[0])
        df["date"] = pd.to_datetime(df["date"])
        id_names = {int(k): v for k, v in
                    (cfg.get("commodity_names") or {23: "Onion"}).items()}
        df["commodity"] = df["commodity_id"].map(id_names).fillna(
            df["commodity_id"].astype(str))
        df["market"] = "market_" + df["market_id"].astype(str)
        df = df.rename(columns={"arrivals": "arrivals_qtl"})
        coercion_issues = {}
    else:
        df = load_raw(paths, cfg["column_map"])
        df, coercion_issues = coerce(df)

        # restrict to scope
        scope = cfg["scope"]
        df = df[df["commodity"].str.lower().isin([c.lower() for c in scope["crops"]])]
        df = df[df["market"].str.lower().str.contains(
            "|".join(m.lower().split()[0] for m in scope["markets"]), na=False
        )]

    report = {
        "files": [Path(p).name for p in paths],
        "rows_in_scope": int(len(df)),
        "coercion_issues": coercion_issues,
        "name_variants": name_variants(df),
        "duplicates": duplicates(df),
        "pairs": {},
    }
    for (mkt, com), g in df.groupby(["market", "commodity"]):
        report["pairs"][f"{mkt} | {com}"] = pair_report(g, cfg)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    df.to_parquet(outdir / "clean_candidate.parquet", index=False)

    # console summary
    print(f"\nRows in scope: {report['rows_in_scope']}")
    print(f"Duplicates: {report['duplicates']}")
    print("\nPair verdicts:")
    for pair, r in sorted(report["pairs"].items()):
        flag = "OK " if r["VIABLE_FOR_MODELING"] else "DROP"
        print(f"  [{flag}] {pair}: {r['rows']} rows, "
              f"coverage={r['coverage_ex_sunday']}, "
              f"longest_gap={r['longest_gap_days']}d, "
              f"jumps>{cfg['validation']['max_daily_price_jump']*100:.0f}%={r['daily_jumps_gt_threshold']}")
    print(f"\nFull report: {outdir/'quality_report.json'}")


if __name__ == "__main__":
    main()
