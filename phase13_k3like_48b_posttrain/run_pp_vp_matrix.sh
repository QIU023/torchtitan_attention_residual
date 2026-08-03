#!/bin/bash
# PP re-examination on all 8 GPUs, including VP != 1.
#
# The earlier claim "PP is numerically transparent" rested on pp=2 with the
# default VP=1, on 2-4 GPUs. That is too narrow: this repo's own history cites
# PP8 x VP4 validation, so VP is a known-load-bearing axis, and a 2-stage split
# exercises one boundary while pp8 exercises seven.
#
# Every leg uses all 8 GPUs and loads the same seed checkpoint, so losses are
# directly comparable to the fsdp8 reference.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/matrix_ppvp}
SEED=$OUT/seed
STEPS=3
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

BASE="--module kimi_k3 --config kimi_k3_mini_block_attn_res \
 --training.seq_len 512 --debug.seed 42 --debug.deterministic \
 --metrics.log_freq 1 --training.steps $STEPS --training.global-batch-size 8"

losses() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+ +grad_norm: +[-0-9.]+).*/\1/' \
  | grep -vE 'loss: +-'; }

echo "########## seed ##########"
rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 1800 torchrun --nproc_per_node=1 --master_port=36000 \
  -m torchtitan.train $BASE --training.steps 1 --training.global-batch-size 1 \
  --training.local-batch-size 1 --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint --dump-folder "$SEED" 2>&1 | tail -1
SEED_PATH=$(find "$SEED" -maxdepth 3 -type d -name step-0 | head -1)
echo "seed: ${SEED_PATH:-MISSING}"; [ -z "${SEED_PATH:-}" ] && exit 1
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SEED_PATH \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=36100
run() {
  local name="$1"; shift
  PORT=$((PORT+1))
  echo "=== $name (8 GPU) ==="
  local out uniq n
  out=$(CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2400 torchrun \
        --nproc_per_node=8 --master_port=$PORT -m torchtitan.train \
        $BASE $LOAD "$@" --dump-folder "$OUT/$name" 2>&1)
  uniq=$(echo "$out" | losses | sort -u); echo "$uniq"
  n=$(echo "$uniq" | grep -c "step:")
  if [ "$n" -eq 0 ]; then
    echo "  -> FAIL"
    echo "$out" | sed -E 's/\x1b\[[0-9;]*m//g' \
      | grep -oE "[A-Za-z_.]+Error[^\"]{0,100}|AssertionError.{0,100}" | head -2
  elif [ "$n" -eq "$STEPS" ]; then echo "  -> PASS rank-identical"
  else echo "  -> DIVERGENT ($n distinct lines, expected $STEPS)"; fi
}

echo; echo "########## reference: no PP ##########"
run fsdp8 --parallelism.data_parallel_shard_degree 8

echo; echo "########## PP degree sweep, VP=1 ##########"
run dp4_pp2 --training.local-batch-size 2 \
    --parallelism.data_parallel_shard_degree 4 --parallelism.pipeline_parallel_degree 2
run dp2_pp4 --training.local-batch-size 1 \
    --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 4
run dp1_pp8 --training.local-batch-size 1 \
    --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 8

echo; echo "########## VP != 1 (looped Interleaved1F1B) ##########"
# stages_per_rank > 1 is what makes VP > 1. layers_per_stage sets the split; with
# 21 layers the split is necessarily uneven, which is itself worth testing since
# K3's 93 layers over 8 ranks is also uneven.
run dp2_pp4_vp2 --training.local-batch-size 1 \
    --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 4 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_layers_per_stage 3
run dp1_pp8_vp2 --training.local-batch-size 1 \
    --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 8 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_layers_per_stage 2
run dp1_pp4_vp3 --training.local-batch-size 1 \
    --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 4 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_layers_per_stage 2

echo; echo "########## PP + CP + EP at 8 GPUs ##########"
run dp2_pp2_cp2_ep2 --training.local-batch-size 2 \
    --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 2 \
    --parallelism.context_parallel_degree 2 --parallelism.expert_parallel_degree 2
echo "########## DONE ##########"
