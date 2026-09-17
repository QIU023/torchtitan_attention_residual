#!/bin/bash
# Direct evidence that the vision tower reaches the policy: the same image cell twice on
# two GPUs, once normally and once with every vision tensor dropped before the forward
# (KIMI_GRPO_DROP_IMAGES, verl_drop_images.patch). Step-1 loss and log-prob must differ.
# Usage: GPUS=2,3 bash verl_img_dropcheck.sh
set -uo pipefail
GPUS=${GPUS:-2,3}
V=/tmp/wt_verl_0915
S=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
OUT=/workspace/.imgdrop_$(date +%m%d_%H%M%S); mkdir -p $OUT; R=$OUT/results.txt; : > $R
git -C $V apply --check $S/verl_drop_images.patch || { echo "patch does not apply; is it already in?" >> $R; }
git -C $V apply $S/verl_drop_images.patch && echo "diagnostic patch applied to $V" >> $R
run() {  # name, env
  local nm=$1 drop=$2
  ( export CUDA_VISIBLE_DEVICES=$GPUS NUM_GPUS=2 VERL_EXP_NAME=imgdrop-$nm \
      TORCHINDUCTOR_CACHE_DIR=/workspace/.ind_imgdrop TRITON_CACHE_DIR=/workspace/.tri_imgdrop \
      ATTN_GYM_CUTE_CACHE_DIR=/workspace/.cute_imgdrop KIMI_GRPO_DROP_IMAGES=$drop
    FSDP_SIZE=2 CELL_TIMEOUT=3600 bash $S/verl_grpo_k3_image.sh > $OUT/$nm.out 2>&1 )
  local L=/workspace/imgdrop-$nm.log
  local s1=$(sed 's/\x1b\[[0-9;]*m//g' $L | grep -aoE "step:1 - .*" | head -1 | grep -oE "actor/pg_loss:[0-9.e-]+|training/rollout_logprobs_diff_mean:[0-9.]+|critic/score/mean:[0-9.]+" | tr '\n' ' ')
  echo "$nm (drop=$drop): $s1" >> $R; tail -1 $R
}
run with_images ""
run without_images 1
git -C $V checkout -- verl/workers/engine/torchtitan/transformer_impl.py && echo "diagnostic patch reverted" >> $R
echo "DONE $OUT" >> $R
