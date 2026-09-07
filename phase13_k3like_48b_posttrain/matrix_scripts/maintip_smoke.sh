#!/bin/bash
# First run of upstream main tip (d263ca0a1 + the kda.py guard lift) on this box, in venv_bfx9 (torch
# 2.15.0.dev20260906+cu130, BFX9 accepted): dp1 and dp2 x ep2 of the declarations protocol, partial_dtensor.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_maintip
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
PD="--parallelism.spmd_backend partial_dtensor"
TITAN=/tmp/wt_maintip CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1|1|$D 1 $PD
dp2_ep2|2|$D 2 $E 2 $PD" $MX maintip
grep -v "^seed\|^tree" $(ls -td /workspace/mx3_maintip_* | head -1)/results.txt
echo "MAINTIP SMOKE DONE"
