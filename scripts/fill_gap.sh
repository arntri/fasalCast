#!/usr/bin/env bash
# Fill the density-probe gap for the two newly-qualified phase-A states
# (Chhattisgarh 22, Rajasthan 8). The original 8 crop shards warmed only the OLD
# 6 states; red gram (7,49) was already re-probed for 22/8 by the corrected
# shard, so this covers the remaining 8 crop ids only.
#
# Shard ONE STATE PER KEY (collision-free: all geo_* caches already exist and
# probe_*/mkts_* files are disjoint by state id). Run in two terminals:
#   export CEDA_API_KEY='<key A>'; scripts/fill_gap.sh 22
#   export CEDA_API_KEY='<key B>'; scripts/fill_gap.sh 8
set -uo pipefail

sid="${1:?usage: scripts/fill_gap.sh <state-id>   e.g.  scripts/fill_gap.sh 22   (CG)   or   scripts/fill_gap.sh 8 (RJ)}"
: "${CEDA_API_KEY:?CEDA_API_KEY not exported — run: export CEDA_API_KEY='...'}"

cd "$(dirname "$0")/.."
mkdir -p logs
log="logs/gap_s${sid}.log"

echo "key fp ${CEDA_API_KEY:0:4}...${CEDA_API_KEY: -4}   state=${sid}   -> ${log}"
.venv/bin/python -u src/select_markets.py \
    --probe-year=2024 --only-crops=1,3,2,23,24,78,6,12 --only-states="${sid}" \
    --budget=20000 --skip-aggregate \
    2>&1 | tee "${log}"
