#!/bin/bash
# The declarations PR (spmd_decl_review1 96eeb51c1 on upstream/main 390e2985b with 4446): the multimodal
# debug flavor under spmd_types (the default now) against partial_dtensor, on the dp / ep meshes, and
# the way 4446's B200 cell runs it (spmd_types with typechecking, AC off).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
S="--parallelism.spmd_backend spmd_types"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_declrun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1_pd|1|$D 1 $PD
dp1|1|$D 1 $S
dp2|2|$D 2 $S
dp2_ep2|2|$D 2 $E 2 $S" $MX decl
TITAN=/tmp/wt_declrun CFG=kimi_k3_debugmodel_tc BATCH="$B" \
CELLS="dp1_tc|1|$D 1
dp2_tc|2|$D 2
dp2_ep2_tc|2|$D 2 $E 2" $MX decl_tc
echo "DECL MATRIX DONE"
