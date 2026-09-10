#!/bin/bash
# QAT (rl_mx_qat fake-quant actor) under each parallelism, then all of them at once.
# Micro-batch 2 throughout (the STE's dequantized copies cost memory on 16 GB cards);
# PP cells pad to a fixed token budget and keep offload off.
set -uo pipefail
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/verl_grpo_moe.sh
R=/workspace/verl_grpo_qat_ladder.txt; : > $R
MB="actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2"
PPX="actor_rollout_ref.actor.torchtitan.param_offload=False actor_rollout_ref.actor.torchtitan.optimizer_offload=False"
run() { local name=$1; shift; echo "### $name $(date +%H:%M)" | tee -a $R; VERL_EXP_NAME=$name bash $M "$@" 2>&1 | tail -4 | tee -a $R; }
export VERL_TORCHTITAN_FLAVOR=kimi_k3_debugmodel_rl_mx_qat
CUDA_VISIBLE_DEVICES=4,5,6,7 EP_SIZE=2 run qat-ep2 $MB actor_rollout_ref.actor.torchtitan.expert_parallel_size=2 actor_rollout_ref.ref.torchtitan.expert_parallel_size=2
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 FSDP_SIZE=2 CP_SIZE=2 run qat-cp2 $MB actor_rollout_ref.actor.torchtitan.context_parallel_size=2 actor_rollout_ref.ref.torchtitan.context_parallel_size=2
CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_SIZE=1 VERL_PP_TOKEN_BUDGET=2048 VERL_TORCHTITAN_FLAVOR=kimi_k3_debugmodel_rl_mx_qat_vit1 run qat-pp2 $MB $PPX actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=2 actor_rollout_ref.ref.torchtitan.pipeline_parallel_size=2
# everything at once: fsdp2 x ep2 x cp2 x pp2 on all eight cards
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 FSDP_SIZE=2 EP_SIZE=2 CP_SIZE=2 VERL_PP_TOKEN_BUDGET=2048 VERL_TORCHTITAN_FLAVOR=kimi_k3_debugmodel_rl_mx_qat_vit1 run qat-ep2-cp2-pp2 $MB $PPX actor_rollout_ref.actor.torchtitan.expert_parallel_size=2 actor_rollout_ref.ref.torchtitan.expert_parallel_size=2 actor_rollout_ref.actor.torchtitan.context_parallel_size=2 actor_rollout_ref.ref.torchtitan.context_parallel_size=2 actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=2 actor_rollout_ref.ref.torchtitan.pipeline_parallel_size=2
echo "QAT LADDER DONE" | tee -a $R
