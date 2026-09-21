#!/bin/bash
# The veRL engine cells on the 09-22 tree (/tmp/wt_int0922b = k3_int_20260922b) on GPUs 4-7, while the
# matrix's part A runs on 0-3: fsdp2 x pp2, pp2 x cp2, fsdp2 x pp2 x ep2, the save / resume pair (fsdp2),
# one image cell (fsdp2 x pp2 x ep2). Each cell's verdict: the runner's rc and the rollout-vs-actor
# log-prob diff at every step, in $R.
set -uo pipefail
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
R=/workspace/verl_cells_int0922b_results.txt; : > $R
export CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_TMPDIR=/tmp/ray_int0922b TOTAL_TRAIN_STEPS=3
cell(){ local name=$1; shift
  echo "== $name $(date -u +%H:%M)" >> $R
  VERL_EXP_NAME=grpo-k3-int0922b-$name env "$@" > /workspace/verl_cell_int0922b_${name}_driver.log 2>&1
  local log=/workspace/grpo-k3-int0922b-$name.log
  local diffs; diffs=$(grep -aoE "step:[0-9]+ - .*rollout_logprobs_diff_mean:[0-9.]+" "$log" | grep -oE "step:[0-9]+|training/rollout_logprobs_diff_mean:[0-9.e-]+" | paste - - | tr '\n' ' ')
  printf "%-16s %s %s\n" "$name" "$(tail -n 1 /workspace/verl_cell_int0922b_${name}_driver.log)" "$diffs" >> $R
}
cell fsdp2pp2       NUM_GPUS=4 FSDP_SIZE=2 PP_SIZE=2 bash $M/verl_grpo_int0922b.sh
cell pp2cp2         NUM_GPUS=4 FSDP_SIZE=1 PP_SIZE=2 CP_SIZE=2 bash $M/verl_grpo_int0922b.sh
cell fsdp2pp2ep2    NUM_GPUS=4 FSDP_SIZE=2 PP_SIZE=2 EP_SIZE=2 bash $M/verl_grpo_int0922b.sh
cell resumeA        NUM_GPUS=2 FSDP_SIZE=2 TOTAL_TRAIN_STEPS=2 bash $M/verl_grpo_int0922b.sh trainer.save_freq=2 trainer.experiment_name=grpo-k3-int0922b-resume
cell resumeB        NUM_GPUS=2 FSDP_SIZE=2 TOTAL_TRAIN_STEPS=4 bash $M/verl_grpo_int0922b.sh trainer.resume_mode=auto trainer.experiment_name=grpo-k3-int0922b-resume
cell img_fsdp2pp2ep2 NUM_GPUS=4 FSDP_SIZE=2 PP_SIZE=2 EP_SIZE=2 bash $M/verl_grpo_k3_image_int0922b.sh
echo "CELLS DONE" >> $R
