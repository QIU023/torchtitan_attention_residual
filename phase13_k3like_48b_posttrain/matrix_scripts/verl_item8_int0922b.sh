#!/bin/bash
# Item 8 on the 09-22 tree: GRPO with the rl QLoRA flavor (packed MXFP4 bases, the fused w13 stacked as
# main stores it), the merged weight sync, eight cards (FSDP_SIZE=8: dp_shard must multiply to the world
# size). The packed DCP comes from make_qlora_src_0922b.py + scripts/quantize_lora_dcp.py (engine venv's torch).
set -uo pipefail
M=$(cd "$(dirname "$0")" && pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_TMPDIR=/tmp/ray_int0922b_item8 TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3} RAY_CPUS=${RAY_CPUS:-48}
PACKED=${PACKED:-/workspace/.qlora_rl_packed_0922b}
VERL_EXP_NAME=${VERL_EXP_NAME:-grpo-k3-int0922b-item8-qlora} VERL_TORCHTITAN_FLAVOR=kimi_k3_rl_qlora_mxfp4 NUM_GPUS=8 FSDP_SIZE=8 \
  bash $M/verl_grpo_int0922b.sh \
  actor_rollout_ref.actor.torchtitan.initial_load_path=$PACKED \
  actor_rollout_ref.ref.torchtitan.initial_load_path=$PACKED \
  actor_rollout_ref.model.lora.merge=True "$@"
