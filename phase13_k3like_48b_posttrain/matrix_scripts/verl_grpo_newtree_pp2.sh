#!/bin/bash
# GRPO on the NEW tree's 12-layer debug export (/root/models/kimi-k3-debug-nt, export_rl_newtree.py) with the torchtitan engine on the
# NEW tree: torchtitan = the 2026-09-06 integration branch (/tmp/wt_k3int, with the run-worktree `rl`
# flavor that matches the export's shape), verl = kimi_k3_integration_rebased + the core-LoRA port
# (/tmp/wt_verl_new), venv_verl (vLLM source build) with spmd_types 0.2.5 and Attention Gym b19162e.
# First cell: fsdp2 only, spmd_types backend. Derived from verl_grpo_moe.sh (09-02).
set -uo pipefail
source /workspace/venv_verl/bin/activate
export PYTHONPATH=/tmp/wt_verl_cp:/tmp/wt_k3int_cp:/tmp/attn_gym_up
export VERL_TORCHTITAN_FLAVOR=${VERL_TORCHTITAN_FLAVOR:-rl}
# Stages size their P2P buffers once: every micro-batch is padded to this many tokens.
export VERL_PP_TOKEN_BUDGET=${VERL_PP_TOKEN_BUDGET:-2048}
export HF_HOME=/workspace/.hf_home
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# pp2 cell on the new tree: the rl model split over two stages (the tree's own layout), offload off,
# token budget 2048 with micro-batch 2 (the 09-02 contract), log-prob micro batch 4.
# The source build reports 0.1.dev1+g6dc76a9ad; the rebased verl gates on >= 0.18.0.
export VERL_VLLM_VERSION=${VERL_VLLM_VERSION:-0.18.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
# In-process inductor compiles: the compile-worker subprocess pool came back "closed" in the pipeline's log-prob step.
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_verl_pp2 TRITON_CACHE_DIR=/workspace/.triton_verl_pp2
cd /tmp/wt_verl_cp
NUM_GPUS=${NUM_GPUS:-2} FSDP_SIZE=${FSDP_SIZE:-1} SPMD_BACKEND=${SPMD_BACKEND:-spmd_types} MODEL_ID=kimi-k3-debug-nt MODEL_PATH=/root/models/kimi-k3-debug-nt TP_SIZE=1 \
timeout 5400 bash tests/special_e2e/run_ppo_trainer_torchtitan.sh \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  actor_rollout_ref.actor.torchtitan.param_offload=${OFFLOAD:-False} \
  actor_rollout_ref.actor.torchtitan.optimizer_offload=${OFFLOAD:-False} \
  actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=${PP_SIZE:-2} \
  actor_rollout_ref.ref.torchtitan.pipeline_parallel_size=${PP_SIZE:-2} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MBS:-2} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PPO_MBS:-2} \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MBS:-8} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  "$@" > /workspace/${VERL_EXP_NAME:-grpo-k3-newtree-pp2}.log 2>&1
rc=$?
grep -aoE "step:[0-9]+ - .*(train/loss|rollout_actor_probs_pearson_corr):[0-9.]+" /workspace/${VERL_EXP_NAME:-grpo-k3-newtree-pp2}.log | tail -4
grep -aiE "Error|Traceback" /workspace/${VERL_EXP_NAME:-grpo-k3-newtree-pp2}.log | grep -v "ERROR:root:initial_load\|deprecat" | tail -5
echo "rc=$rc"
