#!/bin/bash
# lora_review1 on the latest main: the LoRA and QLoRA-MXFP4 flavors at dp1 and dp2, against the
# plain flavor's dp1 / dp2 (the QB control rows, same seed key) -- the LoRA rows for the draft body.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main
D="--parallelism.data_parallel_shard_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
for cfg in kimi_k3_debugmodel_lora kimi_k3_debugmodel_qlora_mxfp4; do
TITAN=/tmp/wt_lora CFG=$cfg BATCH="$B" CELLS="dp1|1|$D 1
dp2|2|$D 2" $MX lora_$cfg
done
echo "LORA MATRICES DONE"
