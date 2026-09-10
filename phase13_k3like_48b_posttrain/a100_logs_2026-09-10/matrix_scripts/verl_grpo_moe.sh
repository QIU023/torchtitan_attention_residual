#!/bin/bash
# GRPO on the K3 debug model WITH its routed experts, vLLM 6dc76a9ad in venv_verl.
# First cell: fsdp2 only. The MoE rollout was recorded as blocked on SITU
# coverage; that was the older vLLM. This is the run that retires the note.
set -uo pipefail
source /workspace/venv_verl/bin/activate
export HF_HOME=/workspace/.hf_home
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The source build reports 0.1.dev1+g<sha>; verl gates on >= 0.7.0 and offers this override.
export VERL_VLLM_VERSION=${VERL_VLLM_VERSION:-0.11.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
cd /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/verl_src
NUM_GPUS=${NUM_GPUS:-2} FSDP_SIZE=${FSDP_SIZE:-2} CP_SIZE=${CP_SIZE:-1} SPMD_BACKEND=partial_dtensor MODEL_ID=kimi-k3-debug MODEL_PATH=/root/models/kimi-k3-debug TP_SIZE=${TP_SIZE:-1} EP_SIZE=${EP_SIZE:-1} TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3} VERL_EXP_NAME=${VERL_EXP_NAME:-grpo-k3-moe} \
timeout 5400 bash tests/special_e2e/run_ppo_trainer_torchtitan.sh \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  actor_rollout_ref.actor.torchtitan.param_offload=True \
  actor_rollout_ref.actor.torchtitan.optimizer_offload=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  "$@" > /workspace/${VERL_EXP_NAME:-grpo-k3-moe}.log 2>&1
rc=$?
grep -aoE "step:[0-9]+ - .*(train/loss|rollout_actor_probs_pearson_corr):[0-9.]+" /workspace/${VERL_EXP_NAME:-grpo-k3-moe}.log | tail -4
grep -aiE "Error|Traceback" /workspace/${VERL_EXP_NAME:-grpo-k3-moe}.log | grep -v "ERROR:root:initial_load\|deprecat" | tail -5
# Last, so a tail of this output still carries it: the chained cells wait on it.
echo "rc=$rc"
