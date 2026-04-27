#!/usr/bin/env bash
# Phase 4 continuation: AttnRes-Kimi-436M from step-12500 -> step-100000.
#
# Why:
# Phase 4 overnight only got to step 12500 (val ~3.7 on C4) — far from
# chinchilla-optimal for a 436M model. Phase 5 multimodal showed the LM
# is the bottleneck for caption quality. This run continues pretrain on
# C4 for 87500 more steps.
#
# Key design choices for SAFE continuation at small bs (12):
#
# 1. --checkpoint.initial_load_model_only: load weights only, NOT
#    optimizer state / LR scheduler step counter. Fresh Adam state.
#
# 2. Re-warmup 500 steps from 0 -> peak. Required because Adam moments
#    are 0 at step 1; without warmup the first updates degenerate to
#    sign(grad) * lr and oscillate.
#
# 3. peak LR = 3e-4, ~14% of original Phase4 peak (2.2e-3). Higher than
#    original Phase4 *final* LR (2.2e-4) so the model has room to
#    escape the local min it settled into (grad_norm 0.08 at end of
#    Phase4 = strong evidence of local-min trapping).
#
# 4. decay_ratio=0.0 (constant LR after warmup). Cosine decay at small
#    bs locks the model into whatever it found early; constant LR keeps
#    stochastic exploration available throughout. Trade-off: no
#    refinement-stage low LR. Acceptable for continuation pretraining;
#    a final cosine cooldown can be added in a separate run if needed.
#
# 5. Same shape as original Phase4: LOCAL_BS=3, GLOBAL_BS=12,
#    SEQ_LEN=2048, NGPU=4, FSDP=4. Re-using the proven shape avoids
#    introducing new variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INIT_CKPT="${INIT_CKPT:-${SCRIPT_DIR}/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/kimi_436m_block_attn_res_fsdp_100k}"

if [[ ! -d "${INIT_CKPT}" ]]; then
    echo "ERROR: init ckpt not found: ${INIT_CKPT}" >&2; exit 1
fi

# Pass-through to the existing FSDP launcher with continuation knobs.
MODULE="kimi_linear" \
CONFIG="kimi_linear_436m_block_attn_res_n4" \
NGPU="${NGPU:-4}" \
STEPS="${STEPS:-87500}" \
LOCAL_BS="${LOCAL_BS:-3}" \
GLOBAL_BS="${GLOBAL_BS:-12}" \
SEQ_LEN="${SEQ_LEN:-2048}" \
LR="${LR:-3e-4}" \
COMPILE="${COMPILE:-1}" \
VAL="${VAL:-1}" \
VAL_FREQ="${VAL_FREQ:-2500}" \
VAL_STEPS="${VAL_STEPS:-100}" \
OUT_DIR="${OUT_DIR}" \
EXTRA_ARGS_APPEND="\
--checkpoint.enable \
--checkpoint.initial_load_path ${INIT_CKPT} \
--checkpoint.initial_load_model_only \
--checkpoint.interval ${SAVE_FREQ:-2500} \
--checkpoint.keep_latest_k ${KEEP_K:-5} \
--lr_scheduler.warmup_steps 500 \
--lr_scheduler.decay_ratio 0.0 \
" \
bash "${SCRIPT_DIR}/launch_fsdp_small.sh"
