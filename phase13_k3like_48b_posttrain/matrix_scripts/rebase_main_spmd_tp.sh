#!/bin/bash
# spmd_review2 (TP/SP + the declarations, on post-EP main): every cell under spmd_types.
# The partial_dtensor references for dp1/dp2 are the QB control rows (same seed key).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
S="--parallelism.spmd_backend spmd_types"; NOSP="--parallelism.no-enable-sequence-parallel"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_spmdrun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1 $S
dp2|2|$D 2 $S
tp2|2|$D 1 $T 2 $S
tp2_nosp|2|$D 1 $T 2 $S $NOSP
tp4|4|$D 1 $T 4 $S
dp2_tp2|4|$D 2 $T 2 $S" $MX spmdtp
echo "SPMDTP DONE"
