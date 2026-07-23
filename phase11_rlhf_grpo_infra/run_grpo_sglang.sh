#!/bin/bash
source /venv/main/bin/activate
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan:$PYTHONPATH
cd /workspace/torchtitan_attention_residual/phase11_rlhf_grpo_infra
python grpo_titan_sglang_rollout.py
echo "GRPO_SGLANG_EXIT=$?"
