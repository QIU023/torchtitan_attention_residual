#!/bin/bash
# After the MoE GRPO cell, the parallelism ladder on the same model and vLLM:
# ep2, cp2, pp2, then QAT (the rl_mx_qat flavor) at dp2. Each cell has its own log.
set -uo pipefail
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/verl_grpo_moe.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
R=/workspace/verl_grpo_chain.txt; : > $R
run() { local name=$1; shift; echo "### $name $(date +%H:%M)" | tee -a $R; VERL_EXP_NAME=$name bash $M "$@" 2>&1 | tail -6 | tee -a $R; }
EP_SIZE=2 run grpo-k3-ep2 actor_rollout_ref.actor.torchtitan.expert_parallel_size=2 actor_rollout_ref.ref.torchtitan.expert_parallel_size=2
run grpo-k3-cp2  actor_rollout_ref.actor.torchtitan.context_parallel_size=2 actor_rollout_ref.ref.torchtitan.context_parallel_size=2
FSDP_SIZE=1 run grpo-k3-pp2 actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=2 actor_rollout_ref.ref.torchtitan.pipeline_parallel_size=2
VERL_TORCHTITAN_FLAVOR=kimi_k3_debugmodel_rl_mx_qat run grpo-k3-qat
echo "CHAIN DONE" | tee -a $R
