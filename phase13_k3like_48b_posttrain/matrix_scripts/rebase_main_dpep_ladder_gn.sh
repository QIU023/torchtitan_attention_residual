#!/bin/bash
# Pure data parallel and data x expert parallel at 1 / 2 / 4 / 8 on the 33-layer model with the
# float32 grad-norm reduction (wt_pprun33gn); dp1, dp2 and dp2 x ep2 are the fp32 table's rows.
# Same seed, 4096 tokens per step; the loader shards the dataset by dp rank, so the pure-dp rows
# change the batch composition with the degree, and EP is read against the same-dp row.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun33gn CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp4|4|$D 4
dp8|8|$D 8
dp4_ep2|4|$D 4 $E 2
dp4_ep4|4|$D 4 $E 4
dp8_ep2|8|$D 8 $E 2
dp8_ep4|8|$D 8 $E 4
dp8_ep8|8|$D 8 $E 8" $MX ladder_gn
echo "DPEP LADDER DONE"
