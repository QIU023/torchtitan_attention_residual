#!/bin/bash
# Save / resume through the engine: run A saves at step 2 and stops; run B resumes from it and runs to step 4.
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
export VERL_TREE=${VERL_TREE:-/tmp/wt_verl_new}
# Dated names: the fixed ones overwrote an earlier run's archive once.
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d)}
export CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 FSDP_SIZE=2 CP_SIZE=1 PPO_MBS=1 LOGP_MBS=1 VERL_TORCHTITAN_FLAVOR=rl
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_verl_resume TRITON_CACHE_DIR=/workspace/.triton_verl_resume ATTN_GYM_CUTE_CACHE_DIR=/workspace/.cute_verl_resume
rm -rf "${VERL_TREE}"/checkpoints/verl_grpo_example_gsm8k_0217/grpo-k3-resume
TOTAL_TRAIN_STEPS=2 VERL_EXP_NAME=grpo-k3-resume bash $M/verl_grpo_int0916_nd.sh trainer.save_freq=2 > /workspace/.smoke_int0916/verl_grpo_resumeA.out 2>&1
echo "A rc=$?"; ls "${VERL_TREE}"/checkpoints/verl_grpo_example_gsm8k_0217/grpo-k3-resume/ 2>/dev/null
cp /workspace/grpo-k3-resume.log /workspace/.smoke_int0916/grpo-k3-resumeA-${RUN_TAG}.log
TOTAL_TRAIN_STEPS=4 VERL_EXP_NAME=grpo-k3-resume bash $M/verl_grpo_int0916_nd.sh trainer.save_freq=2 trainer.resume_mode=auto > /workspace/.smoke_int0916/verl_grpo_resumeB.out 2>&1
echo "B rc=$?"; cp /workspace/grpo-k3-resume.log /workspace/.smoke_int0916/grpo-k3-resumeB-${RUN_TAG}.log
echo "RESUME PAIR DONE"
