#!/bin/bash
# After the SP splice fix: tp2 (SP) and tp4 (SP) on the new spmd_review2 tip, dp2 x tp2 too.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
S="--parallelism.spmd_backend spmd_types"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
cd /tmp/wt_spmdrun && git checkout -q --detach 86f46dee3 && git status --short | tr '\n' ' '; echo
TITAN=/tmp/wt_spmdrun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="tp2|2|$D 1 $T 2 $S
tp4|4|$D 1 $T 4 $S
dp2_tp2|4|$D 2 $T 2 $S" $MX spmdtp_fix
echo "SPMDTP FIX DONE"
