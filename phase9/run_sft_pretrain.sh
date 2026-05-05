#!/usr/bin/env bash
# Phase 9-A SFT — visual instruction tuning on LLaVA-Instruct-150K.
#
# Resumes from v11 step-5000 ckpt (4D pretrain). Same mesh as v11
# preserves the parallelism investment + lets phase 7 ext compare
# pre/post-train traces directly. Dataset is conversation-format,
# loss only on assistant ("gpt") turns.
#
# Recipe:
#   * LR = 2e-5 (10× v11's 1e-5; SFT standard for full-fine-tune)
#   * micro=10 LBS=200 (half of v11's micro=20 because SFT seq_len
#     is ~2× pretrain's 260 → halve micro to keep activation memory)
#   * GBS=400 (= LBS × dp_world=2)
#   * 1 epoch over 150K samples ≈ 375 steps at GBS=400 (LLaVA-1.5
#     trained 1 epoch at GBS=128 on 8×A100, ~3-4h)
#   * SAVE_FREQ=100, KEEP_K=2 — finer ckpt for the shorter run
#   * TRACE_TIER=tier_b on first attempt (post-train fabric pattern
#     for phase 7 ext)
#
# Pre-req:
#   * v11 step-5000 ckpt at $V11_CKPT
#   * LLaVA-Instruct-150K JSON + COCO train2017 images
#
# Disk discipline + retry-loop pattern from run_v11_pretrain.sh.
set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6/launch_8gpu_mm.sh"
V11_CKPT="$WORKSPACE_DIR/phase5/runs/v11_4d_fsdp2_pp2_tp2_ep2_continue_8gpu_from_p4_step8000/checkpoint/step-5000"
OUT_DIR="$WORKSPACE_DIR/phase5/runs/sft_v11_llava_instruct_150k_4d"

# SFT data
SFT_JSON="/workspace/.hf_home/LLaVA-Instruct-150K/llava_instruct_150k.json"
SFT_IMAGES="/workspace/.hf_home/coco_train2017/train2017"

mkdir -p "$OUT_DIR"
LOG="$WORKSPACE_DIR/phase9/sft_orchestrator.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] phase 9-A SFT START (LLaVA-Instruct-150K)"
echo "==============================================================="

# Pre-flight: data + ckpt presence
if [[ ! -f "$SFT_JSON" ]]; then
    echo "ERROR: SFT JSON not found at $SFT_JSON"; exit 1
fi
if [[ ! -d "$SFT_IMAGES" ]]; then
    echo "ERROR: COCO images not found at $SFT_IMAGES"; exit 1
fi
if [[ ! -d "$V11_CKPT" ]]; then
    echo "ERROR: v11 ckpt not found at $V11_CKPT"; exit 1
fi

MAX_RETRIES=15
attempt=0
while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    free_gb=$(df -BG --output=avail "$WORKSPACE_DIR" | tail -1 | tr -d 'G ')
    if [[ "$free_gb" -lt 32 ]]; then
        echo "[$(date)] SFT DISK ABORT: ${free_gb}GB < 32GB"; break
    fi
    echo "[$(date)] SFT attempt #$attempt (disk free: ${free_gb}GB)"
    if [[ "$attempt" -gt 1 ]]; then
        rm -f "$OUT_DIR/tier_b_trace/nccl-rank-"*.log 2>/dev/null
    fi
    if [[ "$attempt" -eq 1 ]]; then
        TRACE_ENV_TIER=tier_b
    else
        TRACE_ENV_TIER=
    fi

    # Same mesh as v11. SFT-specific overrides:
    #   - smaller micro because SFT seq_len ~2× pretrain
    #   - higher LR (SFT standard)
    #   - shorter STEPS (1 epoch on 150K samples at GBS=400 ≈ 375)
    #
    # JSON / IMAGES paths are passed via mm.json / mm.images that
    # phase5/train_mm.py reads.
    OUT_DIR="$OUT_DIR" \
    FSDP=2 DP_REP=1 PP=2 TP=2 CP=1 EP=2 V=2 ADAPTER=1 \
    PP_MICROBATCH=8 \
    STEPS=400 LOCAL_BS=160 GLOBAL_BS=320 SEQ_LEN=580 \
    MM_GLOBAL_SEQ_LEN=580 \
    FLAVOR=kimi_linear_436m_block_attn_res_n4 \
    STUDENT_CKPT="$V11_CKPT" \
    SEED=42 DETERMINISTIC=0 COMPILE=0 \
    LR=2e-5 WARMUP=20 \
    CHECKPOINT_ENABLED=1 SAVE_FREQ=100 KEEP_K=2 \
    TRACE_TIER="$TRACE_ENV_TIER" TRACE_STEPS=50 \
    JSON="$SFT_JSON" IMAGES="$SFT_IMAGES" \
    MM_LAYOUT=sft \
    bash "$LAUNCHER"
    rc=$?
    last_step=$(grep -aoE "step:\s*[0-9]+" "$OUT_DIR/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step=${last_step:-0}
    echo "[$(date)] SFT attempt #$attempt rc=$rc last_step=$last_step"
    if [[ "$last_step" -ge 400 ]]; then
        echo "[$(date)] SFT done at step $last_step"; break
    fi
    if [[ "$rc" -eq 0 ]]; then break; fi
    sleep 30
done

echo "[$(date)] phase 9-A SFT DONE (attempt=$attempt)"
