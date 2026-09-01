#!/bin/bash
set -uo pipefail
source /workspace/venv_verl/bin/activate
export HF_HOME=/workspace/.hf_home
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/verl_src
# GRPO with the actor sharded fsdp2 x cp2 (the CP path unblocked on 08-31).
NUM_GPUS=4 FSDP_SIZE=2 CP_SIZE=2 SPMD_BACKEND=partial_dtensor MODEL_ID=kimi-k3-debug MODEL_PATH=/root/models/kimi-k3-debug TP_SIZE=1 EP_SIZE=1 TOTAL_TRAIN_STEPS=4 VERL_EXP_NAME=stage2-k3-cp2 \
timeout 5400 bash tests/special_e2e/run_ppo_trainer_torchtitan.sh \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  actor_rollout_ref.actor.torchtitan.context_parallel_size=2 \
  actor_rollout_ref.ref.torchtitan.context_parallel_size=2 \
  actor_rollout_ref.actor.torchtitan.param_offload=True \
  actor_rollout_ref.actor.torchtitan.optimizer_offload=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  > /workspace/verl_stage2_cp.log 2>&1
echo "STAGE2CP rc=$?"
grep -aoE "step:[0-9]+ - .*(train/loss|rollout_actor_probs_pearson_corr):[0-9.]+" /workspace/verl_stage2_cp.log | tail -4
