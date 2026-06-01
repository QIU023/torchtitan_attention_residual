#!/usr/bin/env bash
# Phase 4: from-scratch AttnRes-Kimi-436M with paper-faithful hparams.
#
# Replaces the failed continuation attempt. Key differences from
# launch_continuation_100k.sh:
#
# * FROM SCRATCH (no --checkpoint.initial_load_path) — fresh weights
# * Paper LR (2.2e-3, paper Table 2 for 436M)
# * Paper warmup + cosine decay (warmup=500, decay_ratio=0.8 cosine,
#   min_lr_factor=0.1) — config defaults, no override
# * GRAD ACCUMULATION 8x via global_batch_size: torchtitan splits a
#   global_batch_size of 96 into local_batch_size=3 x num_ranks=4 = 12
#   per-microbatch x 8 grad accum steps. Effective bs=96 reduces Adam
#   gradient noise by sqrt(8)=2.83x vs the original Phase 4 (bs=12),
#   approaching the noise/signal ratio paper bs=384 had at LR=2.2e-3.
# * Memory: grad accumulation does NOT increase memory (gradients
#   accumulate in same buffer); LBS=3 SEQ=2048 stays at ~26 GiB peak.
# * Wallclock: each effective optimizer step takes ~8x longer (~15s).
#   In 20h: ~4,800 effective steps -> ~940M tokens (3x current Phase 4
#   baseline of 320M tokens).
# * Checkpoint footprint: KEEP_K=2 + SAVE_FREQ=5000 -> 30 GiB ongoing
#   (vs the failed run's 75 GiB which filled disk).
#
# Why grad_accum=8 specifically (not 32 to match paper exactly):
#   At grad_accum=32 (effective bs=384, paper-exact), step time = 60s,
#   yielding only ~1,200 effective steps in 20h. With warmup=500 that's
#   half the run on warmup -> no time to actually learn. grad_accum=8
#   is the sweet spot: significant noise reduction without losing
#   step granularity.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/kimi_436m_block_attn_res_fsdp_paperhparams}"

MODULE="${MODULE:-attention_residual}" \
CONFIG="${CONFIG:-kimi_linear_436m_block_attn_res_n4}" \
NGPU="${NGPU:-4}" \
STEPS="${STEPS:-12500}" \
LOCAL_BS="${LOCAL_BS:-3}" \
GLOBAL_BS="${GLOBAL_BS:-96}" \
SEQ_LEN="${SEQ_LEN:-2048}" \
LR="" \
COMPILE="${COMPILE:-1}" \
VAL="${VAL:-1}" \
VAL_FREQ="${VAL_FREQ:-1000}" \
VAL_STEPS="${VAL_STEPS:-100}" \
OUT_DIR="${OUT_DIR}" \
EXTRA_ARGS_APPEND="\
--checkpoint.enable \
--checkpoint.interval ${SAVE_FREQ:-2000} \
--checkpoint.keep_latest_k ${KEEP_K:-2} \
" \
bash "${SCRIPT_DIR}/launch_fsdp_small.sh"
