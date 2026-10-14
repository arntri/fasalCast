#!/usr/bin/env bash
# Per-commodity live download feed. Every cached_call() writes ONE file into
# data/probe/ per NEW api request, so file count = downloads landed and the
# newest file's age = "is this crop pulling right now".
#   watch -n 1 -d bash ~/mandi-ews/watch_downloads.sh
cd "$(dirname "$0")" || exit 1
now=$(date +%s)

# commodity id -> name (mirrors CROPS in src/select_markets.py)
names="1:Wheat 3:Rice 23:Onion 24:Potato 78:Tomato 6:BengalGram 12:Mustard 7:RedGram 49:Arhar"

printf '%-12s  %-5s  %-5s  %-6s  %s\n' "CROP" "FILES" "MKTS" "AGE" "NEWEST"
printf '%.0s-' {1..92}; echo
for pair in $names; do
    cid=${pair%%:*}; nm=${pair#*:}
    files=$(ls data/probe/*_c${cid}_* 2>/dev/null | wc -l)
    [ "$files" -eq 0 ] && continue
    mkts=$(ls data/probe/mkts_c${cid}_* 2>/dev/null | wc -l)
    newest=$(ls -t data/probe/*_c${cid}_* 2>/dev/null | head -1)
    mtime=$(stat -c %Y "$newest" 2>/dev/null)
    age=$(( now - ${mtime:-now} ))
    # flag anything touched in the last 3s as actively downloading
    tag=""; [ "$age" -le 3 ] && tag="  <== LIVE"
    printf '%-12s  %-5s  %-5s  %3ss   %s%s\n' \
        "$nm" "$files" "$mkts" "$age" "$(basename "$newest")" "$tag"
done
