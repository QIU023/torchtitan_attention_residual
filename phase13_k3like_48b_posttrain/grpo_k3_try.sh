#!/usr/bin/env bash
set -uo pipefail
cd /workspace/torchtitan_attention_residual/verl
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan
export VERL_VLLM_VERSION=0.27.0
export VERL_TORCHTITAN_FLAVOR=kimi_k3_k3mini_block_attn_res
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
/venv/vllm_k3/bin/python -m verl.trainer.main_ppo \
  --config-name=ppo_trainer \
  model_engine=torchtitan \
  algorithm.adv_estimator=grpo \
  data.train_files=/root/data/gsm8k/train.parquet \
  data.val_files=/root/data/gsm8k/test.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=128 \
  data.max_response_length=32 \
  actor_rollout_ref.model.path=/workspace/k3mini_text_hf \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.max_model_len=256 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.torchtitan.data_parallel_shard_size=2 \
  actor_rollout_ref.ref.torchtitan.data_parallel_shard_size=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.n_gpus_per_node=2 trainer.nnodes=1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  'trainer.logger=[console]' \
  trainer.val_before_train=false \
  trainer.use_v1=false 2>&1
echo "=== exit=$? ==="
