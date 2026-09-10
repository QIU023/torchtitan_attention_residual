#!/bin/bash
# Build the text arm's seed once (one dp1 cell, one stream) before the three streams start, so no two
# mx3 invocations build the same seed at once.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_text18b SEED_CFG=kimi_k3_debugmodel_text PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_text18
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_text18_a MEASURE_STEPS=1
CFG=kimi_k3_debugmodel_text CELLS="seed_dp1|1|--parallelism.data_parallel_shard_degree 1 --parallelism.spmd_backend spmd_types" $MX text18_seed
unset MEASURE_STEPS CUDA_VISIBLE_DEVICES TRITON_CACHE_DIR
bash /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/text18_matrix.sh
