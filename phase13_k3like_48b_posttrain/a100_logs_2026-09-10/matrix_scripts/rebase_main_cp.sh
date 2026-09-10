#!/bin/bash
# CP sanity on cp_review3 (post-EP main): dp1 / tp2 / tp2 x cp2 under spmd_types on the
# base flavor, cp2 (Ulysses) and cp2 (all-gather KV) on the recipe flavors.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
S="--parallelism.spmd_backend spmd_types"; NOSP="--parallelism.no-enable-sequence-parallel"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1 $S
tp2|2|$D 1 $T 2 $S" $MX cpmain_base
TITAN=/tmp/wt_cprun CFG=kimi_k3_debugmodel_cp2 BATCH="$B" \
CELLS="cp2|2|$D 1
tp2cp2_nosp|4|$D 1 $T 2 $NOSP" $MX cpmain_cp2
TITAN=/tmp/wt_cprun CFG=kimi_k3_debugmodel_cp2_allgather BATCH="$B" \
CELLS="cp2_ag|2|$D 1" $MX cpmain_cp2ag
echo "CP MAIN MATRICES DONE"
