#!/bin/bash
# pp4 of the mm_lora arm again, on the tree with the frozen-payload fix (4716f8ee6); the first pass
# ran before it and died at the first backward.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel_lora PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_int18l
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CUDA_VISIBLE_DEVICES=2,3,4,5 TRITON_CACHE_DIR=/workspace/.triton_mm18l_b
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"; MB="--parallelism.num-pp-microbatches 8"; S="--parallelism.spmd_backend spmd_types"
CFG=kimi_k3_debugmodel_lora CELLS="pp4|4|$D 1 $P 4 $MB $S" $MX mm18l_pp4r
echo "MM18L PP4 RERUN DONE"
