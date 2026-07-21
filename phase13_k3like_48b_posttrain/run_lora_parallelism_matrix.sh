#!/bin/bash
# Phase 2: adapter-LoRA (bf16) x parallelism matrix on the 8-card box, via
# the REAL torchtitan.train path (mesh/parallelize/PP handled by titan).
# Closes the flagged gap: LoRA post-training only ever ran under pure FSDP.
# Model: kimi_linear_debugmodel_gated_lora (AttnRes graft + LoRA rank-8,
# 4 layers, 8 experts). Each cell = 3 steps; PASS = finite loss + grad_norm
# (grad_norm>0 under PP proves the cross-stage adapter routes the LoRA/graft
# skip-edge gradients -- the correctness-critical cell).
set -u
cd /workspace/torchtitan_attention_residual/torchtitan
CFG="--module kimi_k3 --config kimi_linear_debugmodel_gated_lora --training.steps 3 --checkpoint.no-enable"
PORT=29541
run_cell() {
  local name="$1"; shift
  PORT=$((PORT+1))
  echo "=================== CELL: $name ==================="
  local out
  out=$(CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        --master_port=$PORT -m torchtitan.train $CFG "$@" 2>&1)
  # strip ANSI color codes before extracting.
  local clean=$(echo "$out" | sed -E 's/\x1b\[[0-9;]*m//g')
  # under PP the first-stage rank logs a negative placeholder loss; the
  # real loss is on the last stage (positive). Grab the positive one.
  local last=$(echo "$clean" | grep -oE "step:  3  loss:  [0-9][0-9.]+  grad_norm: *[0-9.]+" | tail -1)
  if [ -n "$last" ]; then
    echo "  PASS  $last"
  else
    echo "  FAIL/ERROR:"; echo "$clean" | grep -iE "traceback|error:|assertionerror|cuda out of memory|runtimeerror|!= WORLD" | grep -viE "error-prone" | tail -3
  fi
}

run_cell "FSDP (dp8)"          --parallelism.data_parallel_shard_degree 8
run_cell "FSDP+EP (dp8,ep2)"   --parallelism.data_parallel_shard_degree 8 --parallelism.expert_parallel_degree 2
run_cell "FSDP+PP (dp4,pp2)"   --parallelism.data_parallel_shard_degree 4 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B
run_cell "FSDP+PP+EP (dp4,pp2,ep2)" --parallelism.data_parallel_shard_degree 4 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2
run_cell "wide-EP (dp8,ep8)"    --parallelism.data_parallel_shard_degree 8 --parallelism.expert_parallel_degree 8
echo "=================== MATRIX DONE ==================="
