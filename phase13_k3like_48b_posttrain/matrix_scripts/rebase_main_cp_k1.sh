#!/bin/bash
# Where the CP cells' step-1 forward difference comes from (cp2 12.53972 against dp1 12.52977, same
# samples): the cp2 recipe flavor at context_parallel_degree 1, so the CP kernels (KCP on KDA, the
# packed Ulysses on MLA) run on the whole sequence with a group of one. Equal to dp1 -> the difference
# is the sharded execution; equal to cp2 -> the kernels themselves. Tree /tmp/wt_cprobe (edc4cd71b +
# the registry alias), Attention Gym b19162e, the CP matrix's seed and batch.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
export CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/workspace/.triton_cp_k1
D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprobe CFG=kimi_k3_debugmodel_cp2 BATCH="$B" CELLS="cp1_kernels|1|$D 1 $C 1" $MX cp_k1
TITAN=/tmp/wt_cprobe CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1_ref|1|$D 1 --parallelism.spmd_backend spmd_types" $MX cp_k1_ref
echo "CP K1 DONE"
