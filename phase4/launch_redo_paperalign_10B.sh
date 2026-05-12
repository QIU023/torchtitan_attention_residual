#!/usr/bin/env bash
# Phase 4 REDO — paper-aligned LM AttnRes 447M from-scratch, 10B tokens.
#
# Why this exists:
#   The original phase4 run (12500 steps, lbs=3, gbs=96, 4 GPU, ~28h) only
#   produced 2.46B tokens — 1/36 of paper Table 2 (87.9B for 436M). That
#   undertrained LM is the upstream cause of VLM caption collapse and
#   GRPO failing to learn. To make the AttnRes story load-bearing for
#   downstream (image VLM → video VLM → GRPO), the LM backbone has to be
#   at least Chinchilla-saturated (~9B tokens for 447M).
#
# Compared to phase4 launch_from_scratch_paperhparams.sh:
#   NGPU       4   → 8           (2× throughput)
#   GLOBAL_BS  96  → 192         (2× toward paper bs=384, halves grad noise)
#   STEPS      12500 → 25500     (=> 10B tokens at seq=2048)
#   LR         2.2e-3 → 1.5e-3   (sqrt-rule for bs=192 vs paper 384)
#   FP8        off → rowwise_with_gw_hp + auto_filter_small_kn
#                                (1.3-1.5× speedup on dense MLA/lm_head;
#                                 KDA Triton and MoE experts stay bf16)
#   VAL_FREQ   1000 → 5000       (less validation overhead during ramp)
#   keep_latest_k 2              (same — guarded against disk blowup)
#
# Expected wall-clock:
#   phase4: 23k tok/s @ 4 GPU
#   redo:   50k tok/s @ 8 GPU base × 1.4 FP8 = 70k tok/s ≈ 252M tok/h
#   10B / 252M ≈ 40h ≈ 1.7 days
#
# Disk:
#   2 ckpts × ~4 GB = 8 GB ongoing footprint. Auto-guard aborts if
#   workspace free < 30 GB (matches phase11 sft pattern).
#
# Gate (mid-training):
#   step  2000 expected loss <  3.5  (else lr still too hot)
#   step 10000 expected loss <  2.7
#   step 25500 expected loss <  2.45 (val C4 < 2.8) → ready for phase5/11 SFT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/lm_447m_redo_10B_fp8}"
mkdir -p "${OUT_DIR}"

# Disk guard — refuse to launch if < 30 GB free.
free_gb=$(df -BG --output=avail "${WORKSPACE_DIR}" | tail -1 | tr -d 'G ')
if [[ "${free_gb}" -lt 30 ]]; then
    echo "ERROR: workspace free ${free_gb}GB < 30GB; refusing to launch." >&2
    exit 2
fi
echo "[$(date)] starting phase4 redo (free ${free_gb}GB)"

# bf16 baseline for stage 0. FP8 flavor
# ``kimi_linear_447m_aligned_block_attn_res_n4_fp8`` exists (rowwise
# recipe via Float8LinearConverter) but currently fails verify_module_protocol
# because the kimi_linear model uses plain nn.Linear instead of the
# torchtitan.models.common.linear.Linear subclass introduced in PR 2527.
# Switching to the Linear protocol is a follow-up: numerically identical
# rename in ~20 sites of kimi_linear/model.py, isolated from training.
# Stage 0 runs bf16 to start collecting tokens immediately; FP8 ablation
# can attach to stage 0-redo-B after the Linear-protocol PR lands.

# Training knobs — paper-faithful for 447M aligned variant.
MODULE="kimi_linear" \
CONFIG="${CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4}" \
NGPU=8 \
STEPS=25500 \
LOCAL_BS=3 \
GLOBAL_BS=192 \
SEQ_LEN=2048 \
LR=1.5e-3 \
COMPILE=1 \
VAL=1 \
VAL_FREQ=5000 \
VAL_STEPS=100 \
OUT_DIR="${OUT_DIR}" \
EXTRA_ARGS_APPEND="\
--checkpoint.enable \
--checkpoint.interval 2500 \
--checkpoint.keep_latest_k 2 \
--lr_scheduler.warmup_steps 500 \
--lr_scheduler.decay_ratio 0.8 \
--lr_scheduler.min_lr_factor 0.1 \
" \
bash "${SCRIPT_DIR}/launch_fsdp_small.sh"
