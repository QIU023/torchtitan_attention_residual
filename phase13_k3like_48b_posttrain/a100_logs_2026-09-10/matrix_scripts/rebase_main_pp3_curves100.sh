#!/bin/bash
# 100-step loss curves on the rebased pp_review3 (0e7cc5ea1), bf16 as the branch is: dp1, pp2 x vp4,
# pp8 x vp4 (delta transport) and pp8 x vp4 with the whole stack every hop. Same seed and batch as
# the 10-step matrix; the schedule stretches with training.steps (warmup 2, linear decay over 80%).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30 MEASURE_STEPS=100
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun3 CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1
pp2_vp4|2|$D 1 $P 2 $L 4 $IL
pp8_vp4|8|$D 1 $P 8 $L 1 $IL" $MX pp3c100_pp
TITAN=/tmp/wt_pprun3 CFG=kimi_k3_debugmodel_pp_naive BATCH="$B" \
CELLS="pp8_vp4n|8|$D 1 $P 8 $L 1 $IL" $MX pp3c100_ppn
echo "PP CURVES100 DONE"
