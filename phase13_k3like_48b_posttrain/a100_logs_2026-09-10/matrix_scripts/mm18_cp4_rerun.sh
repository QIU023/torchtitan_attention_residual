#!/bin/bash
# cp4 again: the first pass failed before loading (the cp2 recipe flavor is reached through the
# run worktree's local registry alias, the same uncommitted hack the CP matrices used; it was missing).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_int18
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CUDA_VISIBLE_DEVICES=2,3,4,5 TRITON_CACHE_DIR=/workspace/.triton_mm18_b
D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"
CFG=kimi_k3_debugmodel_cp2 CELLS="cp4|4|$D 1 $C 4" $MX mm18_b4cp2
