#!/usr/bin/env bash
# Drive the price backfill to completion under a FLAKY/OSCILLATING gateway.
#
# Strategy: the CEDA API swings between healthy (200 in ~1s) and 504-storming.
#   * Fast-fail (few retries, short timeout) so a 504'd chunk costs ~3s not ~31s.
#   * Health-GATE every pass: only launch shards when the API answers 200, so we
#     don't grind the whole target list into a dead endpoint.
#   * Re-run full passes (each skips cached chunks) until a pass has ZERO
#     failures  ->  every target chunk is on disk = truly complete.
#   * Give up gracefully after N zero-progress passes (persistent no-data years
#     or a long outage) or a wall-clock cap, then combine whatever landed.
# Fully resumable: kill it anytime and re-run; cached chunks are free.
#
# Usage:  ./run_to_completion.sh backfill_keys.env
set -uo pipefail
cd "$(dirname "$0")"

keyfile="${1:-backfill_keys.env}"
[[ -f "$keyfile" ]] || { echo "keyfile not found: $keyfile"; exit 1; }
mapfile -t KEYS < <(grep -vE '^\s*(#|$)' "$keyfile")
n=${#KEYS[@]}; (( n > 0 )) || { echo "no keys"; exit 1; }
PY=.venv/bin/python
BASE="https://api.ceda.ashoka.edu.in/v1"
export CEDA_MAX_RETRIES="${CEDA_MAX_RETRIES:-2}"
export CEDA_TIMEOUT="${CEDA_TIMEOUT:-30}"
MAX_PASSES="${MAX_PASSES:-40}"
MAX_HOURS="${MAX_HOURS:-6}"
STALL_LIMIT="${STALL_LIMIT:-4}"     # consecutive zero-gain passes -> give up
mkdir -p logs
STATUS=logs/backfill_completion.status
raw_count(){ ls data/raw/*.json 2>/dev/null | wc -l; }
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))

# Block until the PRICE-QUERY backend is healthy (or the deadline passes).
# NOTE: gate on /prices, not /geographies — the cheap read endpoint stays up
# during outages while the heavy year-long /prices aggregation 504s.
# CRITICAL: the probe window must match what backfill.py actually requests (a
# full year). Degraded CEDA serves short windows fine while 504ing long ones —
# measured 2026-07-28: 7/8/10/14-day windows all 200 in ~1s, while 21-day,
# 1-month and 1-year windows every one 504'd at exactly 60s. A 1-week probe
# therefore reads "healthy" during an outage the shards cannot survive, opens
# the gate, and burns the whole rate-limit budget on guaranteed failures (this
# is what happened on the Jul 27 run: 8 hours of shards, +9 files). Probe with
# a year so the gate opens only when a real chunk can complete.
wait_for_api(){
  local tries=0 code probekey
  while :; do
    # Rotate the probe key every cycle. At ~1 probe/105s a single key would be
    # hit >40/hr and self-exhaust (permanent 429 -> gate never opens); spreading
    # probes across all n keys keeps each probe key well under its hourly budget.
    probekey="${KEYS[$(( tries % n ))]}"
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 70 \
           -X POST "$BASE/agmarknet/prices" \
           -H "Authorization: Bearer ${probekey}" -H "Content-Type: application/json" \
           -d '{"commodity_id":1,"state_id":27,"district_id":[505],"market_id":[163],"from_date":"2012-01-01","to_date":"2012-12-31"}' 2>/dev/null)
    [[ "$code" == "200" ]] && { [[ $tries -gt 0 ]] && echo "  /prices healthy again after $tries probes $(date '+%T')" >> "$STATUS"; return 0; }
    tries=$((tries+1))
    echo "  /prices unhealthy ($code) $(date '+%T') — waiting 45s (probe $tries)" >> "$STATUS"
    [[ $(date +%s) -ge $deadline ]] && return 1
    sleep 45
  done
}

echo "START $(date '+%F %T')  keys=$n retries=$CEDA_MAX_RETRIES timeout=$CEDA_TIMEOUT baseline=$(raw_count) max_hours=$MAX_HOURS" > "$STATUS"
stall=0
for pass in $(seq 1 "$MAX_PASSES"); do
  [[ $(date +%s) -ge $deadline ]] && { echo "WALLCLOCK cap hit before pass $pass" >> "$STATUS"; break; }
  wait_for_api || { echo "GAVE UP waiting for API $(date '+%T')" >> "$STATUS"; break; }
  before=$(raw_count)
  pids=()
  for i in "${!KEYS[@]}"; do
    CEDA_API_KEY="${KEYS[$i]}" "$PY" -u src/backfill.py --from-year 2006 --shard "${i}/${n}" \
        > "logs/pass${pass}_shard${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  after=$(raw_count); gained=$((after - before))
  fails=$(grep -h FAILED logs/pass${pass}_shard*.log 2>/dev/null | wc -l)
  echo "PASS $pass $(date '+%T'): +$gained new (total $after), $fails failed-this-pass" >> "$STATUS"
  if [[ "$fails" -eq 0 ]]; then echo "COMPLETE at pass $pass — 0 failures, all target chunks on disk" >> "$STATUS"; break; fi
  if [[ "$gained" -eq 0 ]]; then stall=$((stall+1)); else stall=0; fi
  if [[ "$stall" -ge "$STALL_LIMIT" ]]; then echo "STALLED — $stall zero-gain passes; $fails chunks persistently failing (likely no-data years or sustained outage)" >> "$STATUS"; break; fi
  sleep 10
done
echo "COMBINING $(date '+%T')..." >> "$STATUS"
CEDA_API_KEY="${KEYS[0]}" "$PY" -u src/backfill.py --combine-only > logs/backfill_combine.log 2>&1
echo "DONE $(date '+%F %T')  total_chunks=$(raw_count)" >> "$STATUS"
