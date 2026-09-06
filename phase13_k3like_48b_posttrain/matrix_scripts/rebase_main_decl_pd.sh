#!/bin/bash
# The like-for-like rows 4492's review asked for: the same degree under both backends reads the
# same samples (the loader shards documents by dp rank, so dp1 and dp2 do not). partial_dtensor
# twins of the declarations matrix's dp2 and dp2 x ep2 rows, same tree (/tmp/wt_declrun, dbc60701d),
# same seed checkpoint and batch.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_decl_pd
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_declrun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp2_pd|2|$D 2 $PD
dp2_ep2_pd|2|$D 2 $E 2 $PD" $MX decl_pd
echo "DECL PD DONE"
