#!/usr/bin/env bash
# GRPO with the ADAPTER-ONLY weight sync, and a merged arm to compare it against.
#
# The engine used to return peft_config=None, so engine_workers never took its
# base-then-adapter sequence and a run asking for model.lora.merge=False still received
# merged full weights. This runs both modes back to back on the same flavor.
#
# What each arm has to show, from the log rather than by inference:
#   merged   SYNC-CHECKSUM digests CHANGE across steps, one sync per step
#   adapter  "adapter mode, base_sync_done=False" exactly once (the base half is sent
#            only until base_sync_done sticks), then base_sync_done=True every step,
#            and the digests change
# An adapter arm with no base_sync_done=False line means the sequence never engaged;
# an adapter arm whose digests never change means the adapters are not reaching the
# rollout even though the path was taken. Those are different failures and the log has
# to separate them.
set -uo pipefail
# TEXT stack on purpose. The multimodal report_arch flavors cannot serve here: vLLM
# rejects their MLA dimensions with "No valid MLA prefill backend found". The k3mini
# text flavor is the one every working GRPO script here uses, and its _gated_lora suffix
# gives rank-16 LoRA over the same architecture the text weights were exported from.
cd /workspace/torchtitan_attention_residual/verl
OUT=${OUT:-/workspace/grpo_adapter}
mkdir -p "$OUT"
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan
export VERL_VLLM_VERSION=0.27.0
export VERL_TORCHTITAN_FLAVOR=${FLAVOR:-kimi_k3_k3mini_block_attn_res_gated_lora}
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1

arm() {
  local tag="$1"; shift
  echo "########## arm $tag ##########"
  /venv/vllm_k3/bin/python -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/data/gsm8k/train.parquet \
    data.val_files=/root/data/gsm8k/test.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=128 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path=${MODEL_PATH:-/workspace/k3mini_text_hf} \
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
    trainer.total_epochs=1 trainer.total_training_steps=3 \
    'trainer.logger=[console]' \
    trainer.val_before_train=false \
    reward.custom_reward_function.path=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/grpo_variance_reward.py \
    reward.custom_reward_function.name=surface_form_reward \
    trainer.use_v1=false "$@" > "$OUT/$tag.log" 2>&1
  echo "  exit=$?"
  echo "  --- sync checksums ---"
  grep -o "SYNC-CHECKSUM [0-9a-f]* [^ ]*" "$OUT/$tag.log" | sort -u | head -8
  echo "  --- adapter-mode lines ---"
  grep -o "adapter mode, base_sync_done=[A-Za-z]*, shipping [0-9]* tensors[^\"]*" \
    "$OUT/$tag.log" | sort | uniq -c | head -6
  echo "  --- merge lines ---"
  grep -c "folding LoRA adapters into base weights" "$OUT/$tag.log"
  echo "  --- errors ---"
  grep -oiE "(RuntimeError|ValueError|KeyError|AssertionError): .{0,110}" \
    "$OUT/$tag.log" | sort -u | head -3
}

arm merged  '+actor_rollout_ref.model.lora={rank:8,merge:true}'
arm adapter '+actor_rollout_ref.model.lora={rank:8,merge:false}'
echo "########## DONE ##########"
