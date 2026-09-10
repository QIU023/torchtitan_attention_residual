#!/bin/bash
# MTP port smoke on the integration tree (fa5117ac2): weight-0 cell must be bitwise the
# kimi_k3_debugmodel dp1 row (12.41967); dp1 / fsdp2 / fsdp4 carry the composite loss.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export CFG=kimi_k3_debugmodel_mtp SEED_CFG=kimi_k3_debugmodel_mtp PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_mtpport MEASURE_STEPS=3
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
D="--parallelism.data_parallel_shard_degree"; S="--parallelism.spmd_backend spmd_types"
( export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_mtp_a SEED_ROOT=/workspace/.mx3_seeds_mtp_a; CELLS="mtp_w0|1|$D 1 $S --loss.mtp_weight 0.0
mtp_dp1|1|$D 1 $S
mtp_fsdp2|2|$D 2 $S" $MX mtp_a ) &
( export CUDA_VISIBLE_DEVICES=2,3,4,5 TRITON_CACHE_DIR=/workspace/.triton_mtp_b SEED_ROOT=/workspace/.mx3_seeds_mtp_b; CELLS="mtp_fsdp4|4|$D 4 $S" $MX mtp_b ) &
wait
for f in /workspace/mx3_mtp_a_*/results.txt /workspace/mx3_mtp_b_*/results.txt; do echo "== $f"; cat $f; done
echo "MTP SMOKE DONE"
