#!/bin/bash
# The qlora_mxfp4 cells again, after the packed-experts fix on lora_review1 (the seed build had
# raised on main's SpmdType). dp1 / dp2 / dp2 x ep2, same seed key as the lora rows.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_lora CFG=kimi_k3_debugmodel_qlora_mxfp4 BATCH="$B" CELLS="dp1|1|$D 1
dp2|2|$D 2
dp2_ep2|2|$D 2 $E 2" $MX lora2_kimi_k3_debugmodel_qlora_mxfp4
echo "LORA QLORA RERUN DONE"
