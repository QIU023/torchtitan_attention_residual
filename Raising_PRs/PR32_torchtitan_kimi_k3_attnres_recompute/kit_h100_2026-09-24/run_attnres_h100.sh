#!/bin/bash
# PR 4780 on the H100: upstream main against attnres_review1, AC off and selective AC.
# Usage: MAIN=<upstream main checkout> HEAD=<attnres_review1 checkout> VENV=<venv with torch, torch_remat, attn-gym> \
#        OUT=<dir> FLAVOR=attnres_debug|attnres_wide TOKENS="512 4096" STEPS=20 GPUS="0 1 2 3" bash run_attnres_h100.sh
# One GPU per cell (dp1). For each token count every cell runs on its own copy of ONE warm cache
# (inductor and Triton), filled by main with AC off; main_none_rerun is the noise floor on that lineage. The KDA capability guard is lifted for SM90 in both checkouts
# (undo with `git -C <tree> checkout -- torchtitan/models/kimi_k3/kda.py`).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${MAIN:?}" "${HEAD:?}" "${VENV:?}" "${OUT:?}"
FLAVOR=${FLAVOR:-attnres_debug}; TOKENS=${TOKENS:-"512 4096"}; STEPS=${STEPS:-20}; GPUS=(${GPUS:-0 1 2 3})
mkdir -p "$OUT"
"$VENV/bin/python" "$HERE/lift_kda_guard.py" "$MAIN" "$HEAD" || exit 1

cell() {  # cell <name> <tree> <gpu> <cache> <steps> <tokens> <ac...>
  local name=$1 tree=$2 gpu=$3 cache=$4 steps=$5 tokens=$6; shift 6
  ( cd "$tree" && PYTHONPATH="$HERE:." CUDA_VISIBLE_DEVICES=$gpu \
    TORCHINDUCTOR_CACHE_DIR="$cache/ic" TRITON_CACHE_DIR="$cache/tc" \
    "$VENV/bin/torchrun" --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:$((29700 + gpu)) \
      --local-ranks-filter 0 --role rank --tee 3 \
      -m torchtitan.train --module probe_attnres --config "$FLAVOR" \
      --training.steps "$steps" --debug.seed 42 --debug.deterministic \
      --training.num-tokens-per-train-step "$((4 * tokens))" --training.num-tokens-per-microbatch-per-dp-rank "$tokens" \
      --dump-folder "$OUT/$name/dump" "$@" > "$OUT/$name/train.log" 2>&1
    echo "rc=$?" >> "$OUT/$name/train.log" )
}

for tokens in $TOKENS; do
  W="$OUT/cache_t${tokens}_warm"
  rm -rf "$W"; mkdir -p "$W" "$OUT/t${tokens}_warm_main"
  echo "=== t$tokens: warming the shared cache"
  cell "t${tokens}_warm_main" "$MAIN" "${GPUS[0]}" "$W" "$STEPS" "$tokens" activation-checkpoint:none
  specs=(
    "main_none $MAIN none" "main_none_rerun $MAIN none" "head_none $HEAD none"
    "main_sel $MAIN selective" "head_sel $HEAD selective"
  )
  i=0
  for spec in "${specs[@]}"; do
    read -r base tree ac <<< "$spec"
    name="t${tokens}_$base"
    rm -rf "$OUT/$name" "$OUT/cache_$name"; mkdir -p "$OUT/$name"
    cp -r "$W" "$OUT/cache_$name"
    cell "$name" "$tree" "${GPUS[$((i % ${#GPUS[@]}))]}" "$OUT/cache_$name" "$STEPS" "$tokens" "activation-checkpoint:$ac" &
    i=$((i + 1))
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
  done
  wait
  "$VENV/bin/python" "$HERE/tables_attnres.py" "$OUT" \
    "t${tokens}_main_none:t${tokens}_main_none_rerun,t${tokens}_head_none" \
    "t${tokens}_main_sel:t${tokens}_head_sel" | tee "$OUT/table_t${tokens}.md"
done
