#!/usr/bin/env bash
# Problem B — Kimi Linear PP=4 V=2 lps=2 + AttnRes, with the
# cross-stage cache adapter ON. This is the contribution: only delta
# blocks ship between stages, the rest is reconstructed from the
# per-rank cache. Loss MUST match naive_pp within bf16+NCCL noise.
# See README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="${SCRIPT_DIR}/../.."

OUT_DIR="${OUT_DIR:-${PHASE4_DIR}/runs/kimi_pp_adapter_bench/adapter_pp}" \
MODULE=kimi_linear \
CONFIG=kimi_linear_436m_block_attn_res \
NGPU=4 \
STEPS=12500 \
LOCAL_BS=1 \
GLOBAL_BS=12 \
SEQ_LEN=2048 \
CACHE=1 \
bash "${PHASE4_DIR}/launch_pp4_kimi.sh"
