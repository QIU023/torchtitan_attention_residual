#!/usr/bin/env bash
# Resume stage 0 (paperalign_C) from latest checkpoint (auto-resume).
set -euo pipefail
cd /workspace/torchtitan_attention_residual

export LOCAL_BS=4
export GLOBAL_BS=384
export LR=1.5e-3
export WARMUP=150
export STEPS=12750
export SAVE_FREQ=100
export OUT_DIR=/workspace/torchtitan_attention_residual/phase4_kimi_attnres_lm_pretrain/runs/lm_447m_fp8_paperalign_C

exec bash phase4_kimi_attnres_lm_pretrain/launch_redo_paperalign_10B.sh
