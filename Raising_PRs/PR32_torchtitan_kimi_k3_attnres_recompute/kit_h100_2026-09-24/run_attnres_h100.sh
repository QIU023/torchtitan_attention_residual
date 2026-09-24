#!/bin/bash
# Training-level peak memory and throughput, eager aggregation vs the fla-fused override, on one flavor.
# Usage: TITAN=<checkout of attnres_review1> VENV=<venv with torch, fla>=0.6.0 (git main)> \
#        FLAVOR=attnres_debug|attnres_wide NGPU=1 TOKENS="4096 16384" STEPS=20 OUT=<dir> bash run_attnres_h100.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${TITAN:?}" "${VENV:?}" "${OUT:?}"
FLAVOR=${FLAVOR:-attnres_debug}; NGPU=${NGPU:-1}; TOKENS=${TOKENS:-"4096 16384"}; STEPS=${STEPS:-20}
mkdir -p "$OUT"
for tokens in $TOKENS; do
  # eager twice: the second run is the noise floor the fused row is read against.
  for form in eager eager2 fused; do
    name="${FLAVOR}_t${tokens}_${form}"
    extra=""
    [ "$form" = fused ] && extra="--override.imports torchtitan.overrides.fused_attnres.fused_attnres"
    echo "=== $name"
    ( cd "$TITAN" && PYTHONPATH="$HERE:${PYTHONPATH:-}" TRITON_CACHE_DIR="$OUT/tc_$name" TORCHINDUCTOR_CACHE_DIR="$OUT/ic_$name" \
      "$VENV/bin/torchrun" --nproc_per_node="$NGPU" --rdzv_backend c10d --rdzv_endpoint=localhost:0 --local-ranks-filter 0 --role rank --tee 3 \
      -m torchtitan.train --module probe_attnres --config "$FLAVOR" \
      --training.steps "$STEPS" --debug.seed 42 --debug.deterministic \
      --training.num-tokens-per-train-step "$((tokens * NGPU))" --training.num-tokens-per-microbatch-per-dp-rank "$tokens" \
      $extra --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1; echo "rc=$?" >> "$OUT/$name.log" )
    tail -1 "$OUT/$name.log"
  done
done
"$VENV/bin/python" "$HERE/tables_attnres.py" "$OUT"
