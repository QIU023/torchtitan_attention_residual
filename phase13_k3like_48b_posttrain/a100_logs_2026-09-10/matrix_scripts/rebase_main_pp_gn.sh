#!/bin/bash
# The pp x vp cells with the grad-norm reduction carried in float32 (PR 4135's change applied
# to the run worktree, not on the branch): same seed, same cells as the bf16 matrix.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1
pp2_vp4|2|$D 1 $P 2 $L 4 $IL
pp4_vp4|4|$D 1 $P 4 $L 2 $IL
pp8_vp4|8|$D 1 $P 8 $L 1 $IL" $MX main30gn_pp
TITAN=/tmp/wt_pprun CFG=kimi_k3_debugmodel_pp_naive BATCH="$B" \
CELLS="pp8_vp4n|8|$D 1 $P 8 $L 1 $IL" $MX main30gn_ppn
echo "PP MAIN30 GN DONE"
