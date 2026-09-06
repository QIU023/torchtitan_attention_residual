#!/bin/bash
# cp_review5 (edc4cd71b, /tmp/wt_cprobe) against Attention Gym upstream/main 7418529 (2026-09-05,
# 17 commits past our b19162e checkout: routing as a batch input, consolidated CP summaries, FP32
# state summaries): the same dp1 and cp2 cells as the CP matrix, to see whether the kernels still
# run and where step 1 lands (b19162e: dp1 12.52977, cp2 12.53972).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up2 TITAN=/tmp/wt_cprobe
D="--parallelism.data_parallel_shard_degree"; S="--parallelism.spmd_backend spmd_types"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
( export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/workspace/.triton_gymlatest_dp1; CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1|1|$D 1 $S" $MX gymlatest_dp1 ) &
( export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_gymlatest_cp2; CFG=kimi_k3_debugmodel_cp2 BATCH="$B" CELLS="cp2|2|$D 1" $MX gymlatest_cp2 ) &
wait
echo "GYM LATEST DONE"
