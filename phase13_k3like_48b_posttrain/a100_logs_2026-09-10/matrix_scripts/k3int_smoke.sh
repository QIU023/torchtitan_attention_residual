#!/bin/bash
# Smoke of the 2026-09-06 integration branch (k3_int_20260906): the mm18 cells' step 1 must come back
# bitwise (dp1 12.41967, tp2 12.40921, pp2 12.41967, ep2_fsdp2 12.40257, cp2 12.39522); 3 measured steps.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_k3int MEASURE_STEPS=3
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"; P="--parallelism.pipeline_parallel_degree"; E="--parallelism.expert_parallel_degree"
MB="--parallelism.num-pp-microbatches 8"; S="--parallelism.spmd_backend spmd_types"
( export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_k3int_a; CFG=kimi_k3_debugmodel CELLS="dp1|1|$D 1 $S
tp2|2|$D 1 $T 2 $S
pp2|2|$D 1 $P 2 $MB $S
ep2_fsdp2|2|$D 2 $E 2 $S" $MX k3int_a ) &
( export CUDA_VISIBLE_DEVICES=2,3 TRITON_CACHE_DIR=/workspace/.triton_k3int_b; CFG=kimi_k3_debugmodel_cp2 CELLS="cp2|2|$D 1" $MX k3int_cp ) &
wait
echo "K3INT SMOKE DONE"
