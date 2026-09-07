#!/bin/bash
# Is the torch upgrade itself numerically neutral? The same tree (qb_review2 = aecbb8199 + QB, control flavor)
# and seed that gave dp1 12.52977 / 7.36833 / 2.91045 in /venv/main (torch 2.14.0.dev20260802), now in
# venv_bfx9 (torch 2.15.0.dev20260906); one cell, own cache.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_venvneutral
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
PD="--parallelism.spmd_backend partial_dtensor"
TITAN=/tmp/wt_qbrebase CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1|1|$D 1 $PD
dp2_ep2|2|$D 2 $E 2 $PD" $MX venvneutral
grep -v "^seed\|^tree" $(ls -td /workspace/mx3_venvneutral_* | head -1)/results.txt
echo "VENV NEUTRALITY DONE"
