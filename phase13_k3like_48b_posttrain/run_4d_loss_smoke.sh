#!/bin/bash
# Multi-step loss comparison across 4D parallelism combinations.
#
# Distinct from the per-parameter gradient work: that needs identical weights, so
# it compares ONE step. A smoke test asks the opposite question -- does this
# combination still train the same way over time -- and for that the loss curve
# over several steps is exactly the right signal, because a small per-step error
# shows up as a curve that peels away.
#
# All legs share one seed checkpoint and differ only in how the same global batch
# is split, so the curves are directly comparable.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/smoke4d}
SEED=$OUT/seed; STEPS=${STEPS:-8}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=${FLAVOR:-kimi_linear_k3mini_diag_4l_moe_depth}
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1 --training.steps $STEPS"

rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 1800 torchrun --nproc_per_node=1 --master_port=52000 \
  -m torchtitan.train $BASE --training.steps 1 --training.global-batch-size 1 \
  --training.local-batch-size 1 --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint --dump-folder "$SEED" >/dev/null 2>&1
SP=$(find "$SEED" -maxdepth 3 -type d -name step-0 | head -1)
[ -z "${SP:-}" ] && { echo "ABORT: no seed"; exit 1; }
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SP \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=52100
declare -A CURVE
run() {
  local name="$1" ngpu="$2" lbs="$3"; shift 3
  PORT=$((PORT+1)); rm -rf "$OUT/$name"
  local out
  out=$(timeout 2400 torchrun --nproc_per_node=$ngpu --master_port=$PORT \
    -m torchtitan.train $BASE $LOAD --training.global-batch-size 8 \
    --training.local-batch-size $lbs --parallelism.data_parallel_shard_degree 1 \
    "$@" --dump-folder "$OUT/$name" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g')
  # Every rank prints the loss, so dedupe by step number before building the
  # curve -- otherwise a tp2 run yields 16 values for 8 steps and any positional
  # comparison against the reference is meaningless.
  CURVE[$name]=$(echo "$out" | grep -oE "step: +[0-9]+ +loss: +[0-9.]+" \
    | sed -E 's/step: +([0-9]+) +loss: +([0-9.]+)/\1 \2/' | sort -k1,1n -u \
    | awk '{print $2}' | tr '\n' ' ')
  if [ -z "${CURVE[$name]}" ]; then
    echo "$name: FAILED -- $(echo "$out" | grep -oE '[A-Za-z_.]*Error: .*' | head -1)"
  else
    echo "$name: ${CURVE[$name]}"
  fi
}

echo "########## reference and 4D combinations, $STEPS steps ##########"
run ref            1 2
run tp2            2 2 --parallelism.tensor_parallel_degree 2
run tp2_pp2        4 4 --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2
run tp2_cp2        4 2 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2
run tp2_pp2_cp2    8 4 --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 \
                       --parallelism.context_parallel_degree 2
run fsdp2_tp2_cp2  8 2 --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 \
                       --parallelism.context_parallel_degree 2
# EP is carved out of the data-parallel axes rather than costing extra ranks, so
# these fit in 8 GPUs. Full 5D (dp_shard x tp x pp x cp all >= 2) needs 16.
run ep2_tp2_pp2    8 4 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
                       --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2
run ep2_tp2_cp2    8 2 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
                       --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2

echo; echo "########## max |loss - ref| over the curve ##########"
python3 - <<PY
ref="${CURVE[ref]}".split()
import sys
legs = {
 "tp2": "${CURVE[tp2]}", "tp2_pp2": "${CURVE[tp2_pp2]}",
 "tp2_cp2": "${CURVE[tp2_cp2]}", "tp2_pp2_cp2": "${CURVE[tp2_pp2_cp2]}",
 "fsdp2_tp2_cp2": "${CURVE[fsdp2_tp2_cp2]}",
 "ep2_tp2_pp2": "${CURVE[ep2_tp2_pp2]}", "ep2_tp2_cp2": "${CURVE[ep2_tp2_cp2]}",
}
r=[float(x) for x in ref]
print(f"ref curve: {' '.join(f'{v:.5f}' for v in r)}")
for k,v in legs.items():
    xs=[float(x) for x in v.split()]
    if not xs: print(f"{k:<16} NO CURVE"); continue
    n=min(len(xs),len(r))
    d=[abs(xs[i]-r[i]) for i in range(n)]
    print(f"{k:<16} steps={n}  max|dloss|={max(d):.5f}  final={xs[n-1]:.5f} vs {r[n-1]:.5f}")
PY
