#!/bin/bash
# The no-parallelism perturbation on the fp32 grad-norm tree: dp1 with 512-token micro-batches
# (same data, same seed, a different accumulation order), against the fp32 dp1 row.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 512"
TITAN=/tmp/wt_pprun33gn CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1_mb512|1|$D 1" $MX pp33gn_mb512
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun33 CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1_again|1|$D 1" $MX pp33_dp1_again
echo "MB512 DONE"
