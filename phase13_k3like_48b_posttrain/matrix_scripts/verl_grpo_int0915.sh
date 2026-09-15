#!/bin/bash
# GRPO on the 09-15 integration tree (/tmp/wt_int0915_rl = k3_on_4025 rebuild + the local rl flavor) with verl kimi_k3_integration_rebased on upstream main (/tmp/wt_verl_0915).
# The `rl` flavor (uncommitted in the run worktree) matches the export's shape; venv_verl is the vLLM
# source build with spmd_types 0.2.5, torch_remat and renderers 0.1.11, Attention Gym from /tmp/attn_gym_up.
# First cell: fsdp2 only, spmd_types backend. Derived from verl_grpo_moe.sh (09-02).
set -uo pipefail
source /workspace/venv_verl/bin/activate
export PYTHONPATH=/tmp/wt_verl_0915:/tmp/wt_int0915_rl:/tmp/attn_gym_up
export VERL_TORCHTITAN_FLAVOR=${VERL_TORCHTITAN_FLAVOR:-rl}
export HF_HOME=/workspace/.hf_home
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The source build reports 0.1.dev1+g6dc76a9ad; the rebased verl gates on >= 0.18.0.
export VERL_VLLM_VERSION=${VERL_VLLM_VERSION:-0.18.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_verl_newtree TRITON_CACHE_DIR=/workspace/.triton_verl_newtree
# The container caps pids at 15616 (threads count): one Ray instance per cell pre-starts num_cpus idle
# workers and every rank a compile-worker pool, so keep both small.
export TORCHINDUCTOR_COMPILE_THREADS=1
cd /tmp/wt_verl_0915
NUM_GPUS=${NUM_GPUS:-2} FSDP_SIZE=${FSDP_SIZE:-2} SPMD_BACKEND=${SPMD_BACKEND:-spmd_types} MODEL_ID=kimi-k3-debug-nt MODEL_PATH=${MODEL_PATH:-/root/models/kimi-k3-debug-nt-rel} TP_SIZE=1 \
timeout 5400 bash tests/special_e2e/run_ppo_trainer_torchtitan.sh \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  actor_rollout_ref.actor.torchtitan.param_offload=True \
  actor_rollout_ref.actor.torchtitan.optimizer_offload=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MBS:-8} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  ray_kwargs.ray_init.num_cpus=${RAY_CPUS:-24} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  "$@" > /workspace/${VERL_EXP_NAME:-grpo-k3-int0915}.log 2>&1
rc=$?
grep -aoE "step:[0-9]+ - .*(train/loss|rollout_actor_probs_pearson_corr):[0-9.]+" /workspace/${VERL_EXP_NAME:-grpo-k3-int0915}.log | tail -4
grep -aiE "Error|Traceback" /workspace/${VERL_EXP_NAME:-grpo-k3-int0915}.log | grep -v "ERROR:root:initial_load\|deprecat" | tail -5
echo "rc=$rc"
