#!/bin/bash
# Image GRPO cells across the parallelism axes, one chain per GPU set (two Ray clusters at most).
# Usage: CHAIN=A|B|C bash verl_img5d.sh
set -uo pipefail
CHAIN=${CHAIN:?A, B or C}
OUT=/workspace/.img5d_$(date +%m%d)_$CHAIN; mkdir -p $OUT; R=$OUT/results.txt; : > $R
S=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
case "$CHAIN" in
  A) GPUS=0,1,2,3; CELLS="img_tp2|2|FSDP_SIZE=1 TP_SIZE=2
img_ep2|2|FSDP_SIZE=2 EP_SIZE=2
img_cp2tp2|4|FSDP_SIZE=1 CP_SIZE=2 TP_SIZE=2
img_fsdp2pp2ep2|4|FSDP_SIZE=2 PP_SIZE=2 EP_SIZE=2";;
  B) GPUS=4,5,6,7; CELLS="img_cp2|2|FSDP_SIZE=1 CP_SIZE=2
img_pp2|2|FSDP_SIZE=1 PP_SIZE=2
img_pp2cp2|4|FSDP_SIZE=1 PP_SIZE=2 CP_SIZE=2
img_tp2ep2|4|FSDP_SIZE=1 TP_SIZE=2 EP_SIZE=2";;
  C) GPUS=0,1,2,3,4,5,6,7; CELLS="img_dp2cp2tp2ep2|8|FSDP_SIZE=2 CP_SIZE=2 TP_SIZE=2 EP_SIZE=2
img_pp2tp2cp2|8|FSDP_SIZE=1 PP_SIZE=2 TP_SIZE=2 CP_SIZE=2";;
esac
while IFS='|' read -r nm ngpu envs; do
  [ -z "${nm// }" ] && continue
  dev=$(echo $GPUS | cut -d, -f1-$ngpu)
  L=/workspace/img5d-$nm.log
  ( export CUDA_VISIBLE_DEVICES=$dev NUM_GPUS=$ngpu VERL_EXP_NAME=img5d-$nm \
      TORCHINDUCTOR_CACHE_DIR=/workspace/.ind_img5d_$CHAIN TRITON_CACHE_DIR=/workspace/.tri_img5d_$CHAIN \
      ATTN_GYM_CUTE_CACHE_DIR=/workspace/.cute_img5d_$CHAIN RAY_TMPDIR=/tmp/ray_$CHAIN
    env $envs CELL_TIMEOUT=3600 bash $S/verl_grpo_k3_image.sh > $OUT/$nm.out 2>&1 )
  rc=$?
  steps=$(grep -aoE "step:[0-9]+ - " $L 2>/dev/null | wc -l)
  d=$(grep -ao "rollout_logprobs_diff_mean:[0-9.]*" $L 2>/dev/null | tail -1)
  # "Error" also appears in the trainer's configuration dump ('truncation': 'error'), so
  # only a traceback or a raised exception counts. The cell script ends with an echo, so
  # rc is always 0: the step count is the verdict.
  err=$(grep -aE "^Traceback|AssertionError|ValueError|RuntimeError:|NotImplementedError" $L 2>/dev/null | grep -v "initial_load\|deprecat\|select_algorithm\|triton_bundler\|lspci\|not available\|jax profiler\|truncation" | head -1 | cut -c1-150)
  printf "%-18s %-26s rc=%-3s steps=%-3s %s %s\n" "$nm" "$envs" "$rc" "$steps" "$d" "$err" >> $R
  tail -1 $R
done <<< "$CELLS"
echo "DONE $OUT" >> $R
