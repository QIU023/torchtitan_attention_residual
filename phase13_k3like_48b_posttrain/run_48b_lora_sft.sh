#!/bin/bash
set -eo pipefail
source /venv/verl/bin/activate
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan:$PYTHONPATH
MODEL_DIR=/workspace/.hf_home/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc
CKPT=/workspace/ckpt_48b_lora_gsm8k
VERL_TORCHTITAN_FLAVOR=kimi_linear_48b_block_attn_res_gated_lora \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29200 -m verl.trainer.sft_trainer \
  engine=torchtitan optim=torchtitan \
  engine.data_parallel_shard_size=8 engine.spmd_backend=default \
  model.path=$MODEL_DIR model.trust_remote_code=true \
  data.train_files=/workspace/fake_hf/gsm8k_sft_2k.parquet \
  data.train_batch_size=48 data.micro_batch_size_per_gpu=6 \
  data.max_length=2048 data.max_token_len_per_gpu=4096 \
  trainer.total_training_steps=120 \
  trainer.save_freq=40 \
  trainer.max_ckpt_to_keep=1 \
  trainer.default_local_dir=$CKPT \
  'trainer.logger=[console]'
echo "SFT_48B_DONE"
