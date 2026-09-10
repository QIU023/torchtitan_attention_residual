#!/bin/bash
# The one text cell that lost torchrun's random master port (EADDRINUSE); waits for the matrix to end.
set -uo pipefail
until grep -q "TEXT18 DONE" /workspace/text18_matrix.log 2>/dev/null; do sleep 30; done
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_text18b SEED_CFG=kimi_k3_debugmodel_text PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_text18
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_text18_c
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"; P="--parallelism.pipeline_parallel_degree"
MB="--parallelism.num-pp-microbatches 8"; S="--parallelism.spmd_backend spmd_types"
CFG=kimi_k3_debugmodel_text CELLS="fsdp2_tp2_pp2|8|$D 2 $T 2 $P 2 $MB $S" $MX text18_c8fix
grep -v "^seed\|^tree" $(ls -td /workspace/mx3_text18_c8fix_* | head -1)/results.txt
echo "TEXT18 C8FIX DONE"
