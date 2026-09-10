#!/bin/bash
set -uo pipefail
source /workspace/venv_verl/bin/activate
export HF_HOME=/workspace/.hf_home FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/verl_src
OUT=/workspace/verl_ctx_ladder; mkdir -p $OUT
# One sequence per microbatch, so the ctx is the only thing that grows.
rung(){ ctx=$1; fused=$2; tag=ctx${ctx}_fused${fused}
  half=$((ctx/2))
  NUM_GPUS=4 FSDP_SIZE=4 SPMD_BACKEND=partial_dtensor MODEL_ID=kimi-k3-debug MODEL_PATH=/root/models/kimi-k3-debug \
  TP_SIZE=1 EP_SIZE=1 TOTAL_TRAIN_STEPS=1 VERL_EXP_NAME=$tag \
  timeout 3600 bash tests/special_e2e/run_ppo_trainer_torchtitan.sh \
    data.train_batch_size=8 \
    data.max_prompt_length=$half \
    data.max_response_length=$half \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.trust_remote_code=True data.trust_remote_code=True \
    actor_rollout_ref.model.use_fused_kernels=$fused \
    actor_rollout_ref.actor.torchtitan.param_offload=True \
    actor_rollout_ref.actor.torchtitan.optimizer_offload=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ctx \
    actor_rollout_ref.rollout.max_model_len=$ctx \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    > $OUT/$tag.log 2>&1
  rc=$?
  peak=$(grep -aoE "max_memory_reserved_gb:[0-9.]+" $OUT/$tag.log | tail -1)
  oom=$(grep -ac "OutOfMemory\|out of memory" $OUT/$tag.log)
  echo "$tag rc=$rc peak=${peak:-none} oom_lines=$oom" >> $OUT/rc.txt
}
for c in 2048 4096 8192; do rung $c False; done
for c in 8192 16384; do rung $c True; done
echo DONE >> $OUT/rc.txt
