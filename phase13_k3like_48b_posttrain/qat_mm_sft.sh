#!/usr/bin/env bash
# veRL SFT on the multimodal QAT flavor (MXFP4 weights / MXFP8 activations).
set -uo pipefail
cd /workspace/torchtitan_attention_residual/verl
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan
export VERL_VLLM_VERSION=0.27.0
export VERL_TORCHTITAN_FLAVOR=kimi_k3_debugmodel_report_arch_qat
export CUDA_VISIBLE_DEVICES=0,1
/venv/vllm_k3/bin/torchrun --nproc_per_node=2 --master_port=50011 -m verl.trainer.sft_trainer \
  engine=torchtitan optim=torchtitan \
  engine.data_parallel_shard_size=2 engine.spmd_backend=default \
  model.path=/workspace/k3qat_mm_hf \
  model.trust_remote_code=true \
  data.train_files=/root/data/gsm8k_sft/train.parquet \
  data.val_files=/root/data/gsm8k_sft/test.parquet \
  data.train_batch_size=4 data.micro_batch_size_per_gpu=1 \
  data.max_length=1024 data.max_token_len_per_gpu=4096 data.truncation=right \
  trainer.total_epochs=1 trainer.total_training_steps=2 \
  'trainer.logger=[console]' 2>&1
echo "=== exit=$? ==="
