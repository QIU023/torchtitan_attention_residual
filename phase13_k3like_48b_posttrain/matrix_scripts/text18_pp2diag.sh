#!/bin/bash
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
export SEED_ROOT=/workspace/.mx3_seeds_text18 SEED_CFG=kimi_k3_debugmodel_text PYPRE=$SP/site_fsdpdiag:/tmp/attn_gym_up TITAN=/tmp/wt_text18
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CUDA_VISIBLE_DEVICES=6,7 TRITON_CACHE_DIR=/workspace/.triton_text18_diag MEASURE_STEPS=3 WARM_STEPS=1
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"; MB="--parallelism.num-pp-microbatches 8"; S="--parallelism.spmd_backend spmd_types"
CFG=kimi_k3_debugmodel_text CELLS="pp2|2|$D 1 $P 2 $MB $S" $MX text18_pp2diag
d=$(ls -td /workspace/mx3_text18_pp2diag_* | head -1); grep -v "^seed\|^tree" $d/results.txt; grep -ah "fsdp-diag" $d/*.log | sort | uniq -c | head -12
echo "PP2 DIAG DONE"
