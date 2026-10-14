#!/usr/bin/env bash
# Fire the full price backfill, sharded one-key-per-shard across every CEDA key
# in the keyfile. Safe to run the moment the density probe has finished.
#
# It performs the whole handoff:
#   1. If config.yaml has no backfill_targets_auto yet, aggregate the warm probe
#      caches and write it (select_markets.py --write-config; this REFUSES on an
#      incomplete probe, so a premature run aborts cleanly instead of backfilling
#      a half-built target set).
#   2. Enforce the 12,000-chunk ceiling from reports/market_selection.json.
#   3. Fan backfill.py across the keys (--shard i/N), one key each.
#   4. Combine the parquet panels once every shard finishes.
#
# Usage:
#   ./run_backfill_fleet.sh                 # keyfile defaults to ./backfill_keys.env
#   ./run_backfill_fleet.sh mykeys.env      # one CEDA key per line; # and blank lines ignored
#   FORCE_OVER_CEILING=1 ./run_backfill_fleet.sh   # backfill even if over the 12k ceiling
set -uo pipefail
cd "$(dirname "$0")"

keyfile="${1:-backfill_keys.env}"
[[ -f "$keyfile" ]] || { echo "keyfile not found: $keyfile  (one CEDA key per line)"; exit 1; }
mapfile -t KEYS < <(grep -vE '^\s*(#|$)' "$keyfile")
n=${#KEYS[@]}
(( n > 0 )) || { echo "no keys in $keyfile"; exit 1; }
mkdir -p logs
PY=.venv/bin/python

# 1. Ensure the probe-selected target set exists.
if ! grep -q '^backfill_targets_auto:' config.yaml; then
  echo "backfill_targets_auto not in config.yaml — aggregating probe caches + writing config…"
  CEDA_API_KEY="${KEYS[0]}" "$PY" -u src/select_markets.py \
      --probe-year=2024 --write-config 2>&1 | tee logs/write_config.log
  grep -q '^backfill_targets_auto:' config.yaml || {
    echo "!! write-config produced no backfill_targets_auto — probe is incomplete."
    echo "   Let the probe shards finish, then re-run this script."; exit 1; }
fi

# 2. Honor the chunk ceiling recorded by select_markets.py.
budget=$("$PY" - <<'PY'
import json, pathlib
p = pathlib.Path("reports/market_selection.json")
print(json.loads(p.read_text()).get("backfill_chunk_budget", -1) if p.exists() else -1)
PY
)
echo "Backfill chunk budget = ${budget} (ceiling 12000)."
if [[ "$budget" -gt 12000 && "${FORCE_OVER_CEILING:-0}" != "1" ]]; then
  echo "!! Over the 12,000-chunk ceiling. Narrow scope / lower --top-k, or re-run with"
  echo "   FORCE_OVER_CEILING=1 to override. Aborting."; exit 1
fi

# 3. Fan the backfill across the keys, one shard per key.
echo "Launching $n backfill shards (2006→present), one key each…"
pids=()
for i in "${!KEYS[@]}"; do
  log="logs/backfill_shard_${i}_of_${n}.log"
  echo "  shard ${i}/${n}  key ${KEYS[$i]:0:4}…${KEYS[$i]: -4}  -> $log"
  CEDA_API_KEY="${KEYS[$i]}" "$PY" -u src/backfill.py \
      --from-year 2006 --shard "${i}/${n}" 2>&1 | tee "$log" &
  pids+=($!)
done

# 4. Wait for all shards, then combine.
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "All shards finished (fail=${fail}). Combining parquet panels…"
CEDA_API_KEY="${KEYS[0]}" "$PY" -u src/backfill.py --combine-only 2>&1 | tee logs/backfill_combine.log
echo "Backfill fleet complete. Failed chunks (if any) retry on a plain re-run of this script."
exit "$fail"
