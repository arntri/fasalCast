# Data notes

Quirks of the CEDA Agri Market API and the source data, each discovered the hard way,
with how the pipeline handles it.

| # | Finding | Handling |
|---|---|---|
| 1 | **Tonnes vs quintals.** `quantity` is in metric tonnes; prices are Rs/quintal. Mixing them is a silent 10x error. Verified against the agmarknet.gov.in portal header. | `clean.py` multiplies arrivals by 10. `validate.py` scans for ~10x level shifts as a standing guard. |
| 2 | **Unlabeled variety rows.** `/prices` returns several variety/grade rows per day with no variety column, while `/quantities` is market-level. They cannot be paired, so a volume-weighted price is not computable. | arrivals = sum of quantity rows; modal = median of price rows; min/max = true daily envelope. Row counts are kept so the aggregation is auditable. |
| 3 | **Rice is filed as Paddy in Madhya Pradesh.** Commodity id 3 (Rice) has about zero MP records; MP reports under id 2 (Paddy(Dhan)(Common)). | Crop *groups* in `config.yaml`: `rice = {3, 2}`. Stress is computed per series, and groups combine only at the unit-free stress level. Raw paddy and rice prices are never averaged (paddy is unmilled). |
| 4 | **Red gram has two catalogue ids** (7 and 49) for one crop. | Group `redgram = {7, 49}`; the denser member is chosen per state. |
| 5 | **Hard rate limit: 40 requests/hour per key**, returned as HTTP 429 with `Retry-After`. Throughput scales with the number of keys, not parallelism. | The client honours `Retry-After`; the fleet paces each key independently. |
| 6 | **Keys expire about 6 days after issue**, which is the real deadline for a backfill. | Plan work to finish inside the window; everything is resumable. |
| 7 | **The heavy `/prices` endpoint intermittently 504s**, while cheap endpoints stay up. Short date windows can succeed while year-long ones fail. | Health gating probes a full-year window; the batched backfill learns a safe window span per (state, commodity) group. |
| 8 | **Karnataka reports ~37-68 days per mandi per year** (vs 150-290 elsewhere) because much trade runs through the state's own e-market platform. | Dropped from phase A with the reason recorded. |
| 9 | **Bihar has very few reporting mandis** since its APMC Act was repealed in 2006 (e.g. wheat: 6 mandis vs 286 in MP). | Dropped from phase A with the reason recorded. |

## Quality gates

`validate.py` issues a per-(market, commodity) verdict: minimum coverage 0.60 and
minimum 500 rows. Pairs that fail are excluded from the modelling pool entirely, for
the baselines as much as for the ML models. Example: Onion @ Niphad (0.589) and
Pimpalgaon-Saykheda (0.584) are dropped; Tomato @ Lasalgaon (0.098) is dropped because
tomato is barely traded at an onion market.
