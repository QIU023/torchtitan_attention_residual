#!/bin/bash
# Addendum to ctl_pp_matrix.sh: pp8 on the six-layer flavors needs the single-stage schedule
# (the flavors default to Interleaved1F1B, which wants two stages per rank and refuses 16 > 8 units).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_ctl
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
MB="--parallelism.num-pp-microbatches 8"; ONE="--parallelism.pipeline_parallel_schedule 1F1B"
PD="--parallelism.spmd_backend partial_dtensor --training.disable_cuda_graphs"
export SEED_EXTRA="$PD"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
for m in llama3 deepseek_v3; do
MODULE=$m SEED_CFG=${m}_debugmodel TITAN=/tmp/wt_main4446 CFG=${m}_debugmodel BATCH="$B" \
CELLS="pp8_1f1b|8|$D 1 $P 8 $ONE $MB $PD
pp2_1f1b|2|$D 1 $P 2 $ONE $MB $PD" $MX ctl8_$m
done
echo "CTL PP8 MATRIX DONE"
