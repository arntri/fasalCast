#!/usr/bin/env bash
# Live status of all 8 probe shards. Use with:  watch -n 2 bash scripts/watch_shards.sh
cd "$(dirname "$0")/.." || exit 1
printf '%-14s  %-6s  %-6s  %s\n' "SHARD" "STATUS" "FILES" "LATEST"
printf '%.0s-' {1..90}; echo
for f in logs/shard_c*.log; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .log)               # e.g. shard_c23
    cid=${name#shard_c}                        # e.g. 23  (or 7_49)
    cid=${cid%%_*}                             # first id only, for file glob
    if grep -q "warmed" "$f"; then status="DONE"; else status="run"; fi
    files=$(ls data/probe/*_c${cid}_* 2>/dev/null | wc -l)
    latest=$(grep -E "@|warmed" "$f" 2>/dev/null | tail -1 | sed -E 's/ +/ /g; s/^ //')
    printf '%-14s  %-6s  %-6s  %s\n' "$name" "$status" "$files" "${latest:0:60}"
done
