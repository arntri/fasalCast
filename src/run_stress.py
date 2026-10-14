"""
Build the wide panel from ALL cached raw chunks (data/raw), then stress index,
episode table and basket index. Writes into data/ and reports/ WITHOUT touching
the frozen onion panel (data/panel_daily.parquet) used by evaluate.py.

  python src/run_stress.py
"""
import json
from pathlib import Path

import pandas as pd
import yaml

import basket
import clean
import episodes
import stress

RAW = Path("data/raw")
WIDE = Path("data/wide")


def combine_raw(kind):
    frames = []
    for p in sorted(RAW.glob(f"{kind}_*.json")):
        recs = json.loads(p.read_text())
        if recs:
            frames.append(pd.DataFrame(recs))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    return df


def main():
    WIDE.mkdir(parents=True, exist_ok=True)
    px, qt = combine_raw("prices"), combine_raw("quantities")
    px.to_parquet(WIDE / "prices_raw.parquet", index=False)
    qt.to_parquet(WIDE / "quantities_raw.parquet", index=False)
    panel = clean.build_panel(WIDE / "prices_raw.parquet", WIDE / "quantities_raw.parquet",
                              WIDE / "panel_daily.parquet")
    meta = (px.rename(columns={"census_state_id": "state_id", "census_district_id": "district_id"})
              [["market_id", "state_id", "district_id"]].drop_duplicates("market_id"))

    st = stress.build_stress(panel, meta)
    st.to_parquet("data/stress_daily.parquet", index=False)
    ep = episodes.episode_table(st)
    ep.to_csv("reports/episodes_market_crop.csv", index=False)

    cfg = yaml.safe_load(open("config.yaml"))
    c2g = basket.commodity_to_group(cfg["crop_groups"])
    bk = basket.basket_index(st[["date", "district_id", "commodity_id", "market_id", "stress"]], c2g)
    bk.to_parquet("data/basket_district.parquet", index=False)
    scored = bk[bk.scored].rename(columns={"index": "stress"})
    dep = episodes.episode_table(scored, key_cols=("district_id",), level="district")
    dep.to_csv("reports/episodes_district.csv", index=False)

    lines = ["# Stress index & episodes -- first run", "",
             f"Series scored: **{st.groupby(['market_id','commodity_id']).ngroups}** "
             f"(market, commodity) over **{st.date.min().date()} -> {st.date.max().date()}**; "
             f"{len(st):,} series-days. Components and weights are pre-registered in `src/stress.py`.", "",
             f"Market-crop episodes: **{len(ep)}** | district-day basket rows: {len(bk):,} "
             f"(scored {int(bk.scored.sum()):,}, coverage<60% -> null {int((~bk.scored).sum()):,}) | "
             f"district episodes: **{len(dep)}**", "",
             "Episodes per commodity:", ""]
    if len(ep):
        ep["commodity_id"] = ep["id"].str.split("|").str[1].astype(int)
        for c, n in ep.groupby("commodity_id").size().items():
            lines.append(f"- commodity {c}: {n}")
    Path("reports/stress_run.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
