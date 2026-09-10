#!/bin/bash
# Text arm on the integration tree (fa5117ac2 + the kimi_k3_debugmodel_text alias): dp1, fsdp2, tp2.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export CFG=kimi_k3_debugmodel_text SEED_CFG=kimi_k3_debugmodel_text PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_textrun MEASURE_STEPS=3
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"; S="--parallelism.spmd_backend spmd_types"
export CUDA_VISIBLE_DEVICES=2,3 TRITON_CACHE_DIR=/workspace/.triton_text_a SEED_ROOT=/workspace/.mx3_seeds_text
CELLS="text_dp1|1|$D 1 $S
text_fsdp2|2|$D 2 $S
text_tp2|2|$D 1 $T 2 $S" $MX text_a
for f in /workspace/mx3_text_a_*/results.txt; do echo "== $f"; cat $f; done
echo "TEXT SMOKE DONE"
