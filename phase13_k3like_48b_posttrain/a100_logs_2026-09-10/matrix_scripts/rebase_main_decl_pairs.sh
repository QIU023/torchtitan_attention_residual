#!/bin/bash
# The like-for-like rows 4492's review asked for. The same degree under both backends reads the
# same samples (the loader shards documents by dp rank, so dp1 and dp2 do not), and every cell here
# shares ONE inductor cache (one mx3 invocation): a fresh cache picks different autotuned kernels and
# moves step 3 / 10 on this flavor by itself (the two declarations matrices' dp1 rows: 7.27107 / 2.98077
# against 7.36833 / 2.91045 from the same tree and seed). Tree /tmp/wt_declrun (dbc60701d plus the local
# SM120 guard lift and the registry alias), seed checkpoint and batch of the declarations matrix.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_decl_pairs
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
S="--parallelism.spmd_backend spmd_types"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_declrun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1_spmd|1|$D 1 $S
dp1_pd|1|$D 1 $PD
dp2_spmd|2|$D 2 $S
dp2_pd|2|$D 2 $PD
dp2_ep2_spmd|2|$D 2 $E 2 $S
dp2_ep2_pd|2|$D 2 $E 2 $PD" $MX decl_pairs
echo "DECL PAIRS DONE"
