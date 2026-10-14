"""
District/state food-access basket index (Architecture 3.4, Scope-Correction T1+T4).

INPUT is per-series STRESS (already unit-free z-scores from stress.py), never
prices. Per (district, date):

  1. CROP GROUPS are the basket unit. Each group maps to one or more commodity
     ids (config.yaml `crop_groups`):
         rice      = {Rice 3, Paddy(Dhan)(Common) 2}   <- the only multi-member
         redgram   = {Red Gram 7, Arhar(Tur) 49}          group besides red gram
         wheat, onion, potato, tomato, bengalgram, mustard = 1:1
     A state contributes whichever member it reports (MP files rice under
     Paddy, MH/UP/WB under Rice).
  2. Stress is computed per SERIES (market, commodity) against that series' OWN
     trailing history, so a Paddy series and a Rice series are never compared
     on level. Group stress = mean of its available series' stress.
     RAW PADDY AND RICE PRICES ARE NEVER AVERAGED. Caveat: paddy is unmilled
     (milling margin, ~2/3 rice-equivalent by weight), so their price LEVELS
     differ structurally; z-scoring each series against itself removes the
     level, and aggregation happens only at that unit-free stress level.
  3. Basket index = CPI-style fixed-weight average of group stress,
     RENORMALISED over the groups present that day.

COVERAGE GUARD (T4): renormalising over a thin set of groups would silently
turn "the basket" into one or two crops. So a district-day is SCORED only if
the groups present carry >= MIN_COVERAGE (60%) of the full-basket weight.
Below that the index is NaN with scored=False -- null with a flag, never a
silently thin number. `coverage` is always reported.

WEIGHTS are fixed a priori and NEVER fit to episodes (anti-dredging line).
They are APPROXIMATE staple-expenditure shares for a poor-household plate
(cereals dominant, then vegetables, pulses, oilseed), not official CPI
weights. Sensitivity (+/- perturbation) is the intended robustness check.
"""

import numpy as np

MIN_COVERAGE = 0.60

BASKET_WEIGHTS = {  # approximate, pre-registered, sums to 1.0
    "rice": 0.25, "wheat": 0.20, "onion": 0.10, "potato": 0.10,
    "tomato": 0.10, "redgram": 0.08, "bengalgram": 0.07, "mustard": 0.10,
}


def commodity_to_group(crop_groups):
    """config `crop_groups` {group: [commodity ids]} -> {commodity id: group}."""
    return {int(c): g for g, ids in crop_groups.items() for c in ids}


def basket_index(stress, c2g, weights=BASKET_WEIGHTS, min_coverage=MIN_COVERAGE):
    """stress: DataFrame with columns date, district_id, commodity_id, market_id,
    stress (per-series, unit-free). Returns one row per (district_id, date):
    index (NaN unless scored), coverage (present weight / full weight),
    n_groups, scored (bool)."""
    df = stress.dropna(subset=["stress"]).copy()
    df["group"] = df["commodity_id"].map(c2g)
    df = df[df["group"].isin(weights)]
    # series -> group: mean of member series' stress (unit-free by construction)
    grp = (df.groupby(["district_id", "date", "group"])["stress"]
             .mean().rename("g_stress").reset_index())
    grp["w"] = grp["group"].map(weights)
    grp["ws"] = grp["w"] * grp["g_stress"]
    agg = grp.groupby(["district_id", "date"]).agg(
        w_present=("w", "sum"), ws=("ws", "sum"), n_groups=("group", "nunique"))
    full = float(sum(weights.values()))
    agg["coverage"] = agg["w_present"] / full
    agg["scored"] = agg["coverage"] >= min_coverage - 1e-12
    agg["index"] = np.where(agg["scored"], agg["ws"] / agg["w_present"], np.nan)
    return agg[["index", "coverage", "n_groups", "scored"]].reset_index()
