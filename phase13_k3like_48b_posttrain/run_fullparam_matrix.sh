#!/bin/bash
# Full-param K3 parallelism matrix on 8 cards, adding the TP axis (the LoRA
# matrix covered FSDP/EP/PP; this adds TP + the 4D FSDP+TP+EP+PP combo).
# Model: kimi_k3_debugmodel (full-param, KDA+MLA+MoE, 4 layers 8 experts).
# Reports step-1 and step-3 loss per cell; PASS = finite loss on the last
# stage. Cross-cell loss varies within the bf16/KDA-nondeterminism band.
set -u
cd /workspace/torchtitan_attention_residual/torchtitan
CFG="--module kimi_k3 --config kimi_k3_debugmodel --training.steps 3 --checkpoint.no-enable"
PORT=29590
run_cell() {
  local name="$1"; shift
  PORT=$((PORT+1))
  echo "=================== CELL: $name ==================="
  local out clean l1 l3
  out=$(CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        --master_port=$PORT -m torchtitan.train $CFG "$@" 2>&1)
  clean=$(echo "$out" | sed -E 's/\x1b\[[0-9;]*m//g')
  l1=$(echo "$clean" | grep -oE "step:  1  loss:  [0-9][0-9.]+" | tail -1)
  l3=$(echo "$clean" | grep -oE "step:  3  loss:  [0-9][0-9.]+  grad_norm: *[0-9.]+" | tail -1)
  if [ -n "$l3" ]; then
    echo "  PASS  $l1 | $l3"
  else
    echo "  FAIL/ERROR:"; echo "$clean" | grep -iE "traceback|error:|assertionerror|runtimeerror|out of memory|not.*support|!= WORLD" | grep -viE "error-prone" | tail -3
  fi
}

run_cell "FSDP (dp8)"               --parallelism.data_parallel_shard_degree 8
run_cell "TP (dp4,tp2)"             --parallelism.data_parallel_shard_degree 4 --parallelism.tensor_parallel_degree 2
run_cell "EP (dp8,ep2)"             --parallelism.data_parallel_shard_degree 8 --parallelism.expert_parallel_degree 2
run_cell "PP (dp4,pp2)"             --parallelism.data_parallel_shard_degree 4 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B
run_cell "TP+EP (dp4,tp2,ep2)"      --parallelism.data_parallel_shard_degree 4 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2
run_cell "TP+PP (dp2,tp2,pp2)"      --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B
run_cell "4D FSDP+TP+EP+PP (dp2,tp2,pp2,ep2)" --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2
echo "=================== MATRIX DONE ==================="
