#!/usr/bin/env bash
# Parse the raw per-leg logs. Kept separate from the runner so a parsing bug
# never costs a re-run -- this project has lost matrices to collector bugs.
OUT=${1:-/tmp/twinpar}; STEPS=${2:-10}
for name in dp1 fsdp2 pp2 cp2 tp2 fsdp2_tp2_pp2 fsdp2_tp2_cp2 tp2_pp2_cp2 \
            fsdp2_pp2_cp2 ep2_fsdp2 ep2_fsdp2_tp2_pp2 ep2_fsdp2_tp2_cp2 \
            ep2_fsdp2_pp2_cp2; do
  f="$OUT/$name.log"
  [ -f "$f" ] || { printf "%-22s MISSING\n" "$name"; continue; }
  uniq=$(sed -E 's/\x1b\[[0-9;]*m//g' "$f" | grep -E "step: +[0-9]+ +loss" \
    | sed -E 's/.*step: +([0-9]+) +loss: +([-0-9.]+).*/\1 \2/' \
    | grep -vE ' -' | sort -u -n -k1,1)
  n=$(echo "$uniq" | grep -c .)
  if [ "$n" -eq "$STEPS" ]; then
    printf "%-22s %s\n" "$name" "$(echo "$uniq" | awk '{print $2}' | tr '\n' ' ')"
  else
    err=$(sed -E 's/\x1b\[[0-9;]*m//g' "$f" | grep -oiE \
      "(RuntimeError|ValueError|AssertionError|InternalError|OutOfMemoryError): .{0,60}" | head -1)
    printf "%-22s FAIL (%d/%d) %s\n" "$name" "$n" "$STEPS" "$err"
  fi
done
