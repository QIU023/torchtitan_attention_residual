#!/usr/bin/env bash
# Problem A — Kimi Linear 436M baseline FSDP run.
# See README.md in this directory for the experiment plan.
#
# Reproduces the run kicked off 2026-04-23 22:49 UTC, currently writing
# to phase4/runs/kimi_436m_baseline_fsdp_overnight/. If the original
# run completes cleanly, no need to re-run; this script exists for
# reproducibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="${SCRIPT_DIR}/../.."

# Hand-off to the standard FSDP launcher with Problem A's pinned config.
OUT_DIR="${OUT_DIR:-${PHASE4_DIR}/runs/kimi_436m_baseline_fsdp_overnight}" \
MODULE=kimi_linear \
CONFIG=kimi_linear_436m_baseline \
NGPU=4 \
STEPS=12500 \
LOCAL_BS=3 \
GLOBAL_BS=12 \
SEQ_LEN=2048 \
COMPILE=1 \
bash "${PHASE4_DIR}/launch_fsdp_small.sh"
