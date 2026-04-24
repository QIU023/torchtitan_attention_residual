#!/usr/bin/env bash
# Problem B — Kimi Linear FSDP=4 reference (no PP). Same model,
# same hyperparameters as the two PP arms, just different
# parallelism. Used as the loss-target baseline that the two PP arms
# must match (within bf16+NCCL nondeterminism). See README.md.
#
# COMPILE=0 is set explicitly so the comparison isolates parallelism
# strategy — torch.compile would interact with PP scheduling
# differently than with FSDP-only, biasing the throughput numbers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="${SCRIPT_DIR}/../.."

OUT_DIR="${OUT_DIR:-${PHASE4_DIR}/runs/kimi_pp_adapter_bench/fsdp}" \
MODULE=kimi_linear \
CONFIG=kimi_linear_528m_l16_block_attn_res \
NGPU=4 \
STEPS=1000 \
LOCAL_BS=1 \
GLOBAL_BS=4 \
SEQ_LEN=2048 \
COMPILE=0 \
bash "${PHASE4_DIR}/launch_fsdp_small.sh"
