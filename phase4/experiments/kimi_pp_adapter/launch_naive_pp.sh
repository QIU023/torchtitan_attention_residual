#!/usr/bin/env bash
# Problem B — Kimi Linear PP=4 V=2 lps=2 + AttnRes, NAIVE (no adapter).
# This is the upper-bound-on-comm reference: every stage transition
# ships the full block stack instead of the constant-size delta the
# adapter would send. See README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="${SCRIPT_DIR}/../.."

OUT_DIR="${OUT_DIR:-${PHASE4_DIR}/runs/kimi_pp_adapter_bench/naive_pp}" \
MODULE=kimi_linear \
CONFIG=kimi_linear_436m_block_attn_res \
NGPU=4 \
STEPS=1000 \
LOCAL_BS=1 \
GLOBAL_BS=4 \
SEQ_LEN=2048 \
CACHE=0 \
bash "${PHASE4_DIR}/launch_pp4_kimi.sh"
