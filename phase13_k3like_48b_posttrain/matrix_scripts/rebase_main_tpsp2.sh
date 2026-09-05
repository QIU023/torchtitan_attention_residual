#!/bin/bash
# TP/SP + spmd_types on tpsp_spmd_review1 2e2230cbb (upstream/main 390e2985b with 4446 merged; the
# default backend is spmd_types now, so the partial_dtensor control names its backend).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"; E="--parallelism.expert_parallel_degree"
S="--parallelism.spmd_backend spmd_types"; PD="--parallelism.spmd_backend partial_dtensor"; NOSP="--parallelism.no-enable-sequence-parallel"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_tpsprun2 CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1_pd|1|$D 1 $PD
dp1_spmd|1|$D 1 $S
dp2_spmd|2|$D 2 $S
dp2_ep2_spmd|2|$D 2 $E 2 $S
tp2|2|$D 1 $T 2 $S
tp2_nosp|2|$D 1 $T 2 $S $NOSP
tp4|4|$D 1 $T 4 $S
dp2_tp2|4|$D 2 $T 2 $S
dp2_ep2_tp2|4|$D 2 $E 2 $T 2 $S" $MX tpsp2
echo "TPSP2 MATRIX DONE"
