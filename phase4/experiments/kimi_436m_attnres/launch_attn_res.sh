#!/usr/bin/env bash
# Problem A — Kimi Linear 436M Block AttnRes FSDP run.
# See README.md in this directory for the experiment plan.
#
# Identical configuration to launch_baseline.sh except --config switches
# to kimi_linear_436m_block_attn_res (num_blocks=8, 2 layers per block,
# paper N=8 recipe). Run AFTER the baseline finishes — the 4× RTX 5090
# box does one job at a time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="${SCRIPT_DIR}/../.."

OUT_DIR="${OUT_DIR:-${PHASE4_DIR}/runs/kimi_436m_block_attn_res_fsdp_overnight}" \
MODULE=kimi_linear \
CONFIG=kimi_linear_436m_block_attn_res \
NGPU=4 \
STEPS=12500 \
LOCAL_BS=3 \
GLOBAL_BS=12 \
SEQ_LEN=2048 \
COMPILE=1 \
bash "${PHASE4_DIR}/launch_fsdp_small.sh"
