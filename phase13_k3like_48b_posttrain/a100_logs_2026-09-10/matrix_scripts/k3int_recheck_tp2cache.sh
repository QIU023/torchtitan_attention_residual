#!/bin/bash
# The mm18 rows on the integration tree after the MTP port and the adapter fix (7d3ba3553):
# dp1 12.41967 / 7.49054 and tp2 12.40921 / 7.11185 expected bitwise.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_mtpport MEASURE_STEPS=3
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"; S="--parallelism.spmd_backend spmd_types"
export CUDA_VISIBLE_DEVICES=2,3 TRITON_CACHE_DIR=/workspace/.triton_k3int_a INDUCTOR_SEED_CACHE=$(ls -d /workspace/mx3_k3int_a_*/inductor | head -1)
CELLS="tp2|2|$D 1 $T 2 $S" $MX k3int_recheck_tp2cache
cat /workspace/mx3_k3int_recheck_tp2cache_*/results.txt
echo "TP2CACHE DONE"
