#!/usr/bin/env bash
# Run one crop-shard of the market-selection density probe.
# Usage:
#   export CEDA_API_KEY='<your-key>'
#   ./run_shard.sh 3        # one crop id
#   ./run_shard.sh 7,49     # Red Gram: both candidate ids
set -uo pipefail

crops="${1:?usage: ./run_shard.sh <crop-ids>   e.g.  ./run_shard.sh 3   or   ./run_shard.sh 7,49}"
: "${CEDA_API_KEY:?CEDA_API_KEY not exported — run: export CEDA_API_KEY='...'}"

cd "$(dirname "$0")"
mkdir -p logs
log="logs/shard_c${crops//,/_}.log"

echo "key fp ${CEDA_API_KEY:0:4}...${CEDA_API_KEY: -4}   crops=${crops}   -> ${log}"
.venv/bin/python -u src/select_markets.py \
    --probe-year=2024 --only-crops="${crops}" --budget=20000 --skip-aggregate \
    2>&1 | tee "${log}"
