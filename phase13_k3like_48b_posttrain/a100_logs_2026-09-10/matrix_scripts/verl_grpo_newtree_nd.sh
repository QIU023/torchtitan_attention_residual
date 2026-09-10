#!/bin/bash
# Multi-axis GRPO cells on the NEW tree with the synthetic reward as the sync instrument (the actor must move
# for steps 2-3 to test the sync of updated weights). Every parallel size comes from the environment:
#   NUM_GPUS FSDP_SIZE EP_SIZE CP_SIZE PP_SIZE TP_SIZE, flavor via VERL_TORCHTITAN_FLAVOR
#   (kimi_k3_rl_cp2 / kimi_k3_rl_cp2_mx_qat for CP cells; rl / kimi_k3_rl_mx_qat otherwise; all run-worktree aliases).
# Contracts: CP runs one sequence per micro-batch (PPO_MBS=1, LOGP_MBS=1), PP pads to VERL_PP_TOKEN_BUDGET
# (2048 with micro-batch 1-2), offload off under PP. Judged by training/rollout_logprobs_diff_mean at steps 1 and 3.
set -uo pipefail
source /workspace/venv_verl/bin/activate
export PYTHONPATH=/tmp/wt_verl_cp:/tmp/wt_k3int_cp:/tmp/attn_gym_up
export VERL_TORCHTITAN_FLAVOR=${VERL_TORCHTITAN_FLAVOR:-rl}
export HF_HOME=/workspace/.hf_home
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VERL_VLLM_VERSION=${VERL_VLLM_VERSION:-0.18.0}
export VERL_PP_TOKEN_BUDGET=${VERL_PP_TOKEN_BUDGET:-2048}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/workspace/.inductor_verl_nd} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/workspace/.triton_verl_nd}
export ATTN_GYM_CUTE_CACHE_DIR=${ATTN_GYM_CUTE_CACHE_DIR:-/workspace/.cute_verl_nd}
export TORCHINDUCTOR_COMPILE_THREADS=1
cd /tmp/wt_verl_cp
NUM_GPUS=${NUM_GPUS:-8} FSDP_SIZE=${FSDP_SIZE:-2} EP_SIZE=${EP_SIZE:-1} TP_SIZE=${TP_SIZE:-1} SPMD_BACKEND=${SPMD_BACKEND:-spmd_types} MODEL_ID=kimi-k3-debug-nt MODEL_PATH=/root/models/kimi-k3-debug-nt \
timeout ${CELL_TIMEOUT:-7200} bash tests/special_e2e/run_ppo_trainer_torchtitan.sh \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  actor_rollout_ref.actor.torchtitan.param_offload=False \
  actor_rollout_ref.actor.torchtitan.optimizer_offload=False \
  actor_rollout_ref.actor.torchtitan.context_parallel_size=${CP_SIZE:-1} \
  actor_rollout_ref.ref.torchtitan.context_parallel_size=${CP_SIZE:-1} \
  actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=${PP_SIZE:-1} \
  actor_rollout_ref.ref.torchtitan.pipeline_parallel_size=${PP_SIZE:-1} \
  actor_rollout_ref.ref.torchtitan.expert_parallel_size=${EP_SIZE:-1} \
  actor_rollout_ref.ref.torchtitan.tensor_parallel_size=${TP_SIZE:-1} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MBS:-1} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PPO_MBS:-1} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MBS:-1} \
  reward.custom_reward_function.path=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/synthetic_reward.py \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  ray_kwargs.ray_init.num_cpus=${RAY_CPUS:-24} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  "$@" > /workspace/${VERL_EXP_NAME:-grpo-k3-newtree-nd}.log 2>&1
rc=$?
grep -aoE "step:[0-9]+ - .*rollout_logprobs_diff_mean:[0-9.]+" /workspace/${VERL_EXP_NAME:-grpo-k3-newtree-nd}.log | grep -oE "step:[0-9]+|training/rollout_logprobs_diff_(max|mean):[0-9.e-]+" | paste - - - | tail -4
grep -aiE "Error|Traceback" /workspace/${VERL_EXP_NAME:-grpo-k3-newtree-nd}.log | grep -v "ERROR:root:initial_load\|deprecat\|select_algorithm\|triton_bundler\|lspci\|not available" | tail -3
echo "rc=$rc"
