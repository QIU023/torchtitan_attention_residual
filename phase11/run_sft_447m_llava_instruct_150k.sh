#!/usr/bin/env bash
# Phase 11 VLM SFT — 447M aligned AttnRes ckpt on LLaVA-Instruct-150K.
#
# Mirror of phase9/run_sft_pretrain.sh swapping in:
#   * 447M aligned flavor (kimi_linear_447m_aligned_block_attn_res_n4)
#     instead of 436M (which had unservable head dims under flashinfer)
#   * 447M base ckpt at phase4/runs/.../step-12500
#   * Pure FSDP=8 mesh (matches the v_fsdp8_447m continued-pretrain run;
#     the 4D mesh hit "FSDP requires DP and TP/EP same parent mesh"
#     under torch 2.9 stable per PHASE9_10_11_SUMMARY 9-B)
#
# Recipe matches LLaVA-1.5 stage-1 hyperparameters scaled for our
# smaller backbone.
set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6/launch_8gpu_mm.sh"
BASE_CKPT="$WORKSPACE_DIR/phase4/runs/kimi_447m_aligned_block_attn_res_fsdp_paperhparams/checkpoint/step-12500"
OUT_DIR="$WORKSPACE_DIR/phase5/runs/sft_v_fsdp8_447m_aligned_llava_instruct_150k"

SFT_JSON="/workspace/.hf_home/LLaVA-Instruct-150K/llava_instruct_150k.json"
SFT_IMAGES="/workspace/.hf_home/coco_train2017/train2017"

mkdir -p "$OUT_DIR"
LOG="$WORKSPACE_DIR/phase11/sft_447m_orchestrator.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] phase 11 447M SFT START (LLaVA-Instruct-150K)"
echo "==============================================================="

if [[ ! -f "$SFT_JSON" ]]; then
    echo "ERROR: SFT JSON not found at $SFT_JSON"; exit 1
fi
if [[ ! -d "$SFT_IMAGES" ]]; then
    echo "ERROR: COCO images not found at $SFT_IMAGES"; exit 1
fi
if [[ ! -d "$BASE_CKPT" ]]; then
    echo "ERROR: 447M base ckpt not found at $BASE_CKPT"; exit 1
fi

MAX_RETRIES=10
attempt=0
while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    free_gb=$(df -BG --output=avail "$WORKSPACE_DIR" | tail -1 | tr -d 'G ')
    if [[ "$free_gb" -lt 24 ]]; then
        echo "[$(date)] SFT DISK ABORT: ${free_gb}GB < 24GB"; break
    fi
    echo "[$(date)] SFT 447M attempt #$attempt (disk free: ${free_gb}GB)"

    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    OUT_DIR="$OUT_DIR" \
    FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
    STEPS=2344 LOCAL_BS=4 GLOBAL_BS=64 SEQ_LEN=512 \
    MM_GLOBAL_SEQ_LEN=512 \
    FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
    STUDENT_CKPT="$BASE_CKPT" \
    SEED=43 DETERMINISTIC=0 COMPILE=0 \
    AC=full \
    LR=2e-5 WARMUP=20 \
    PROJ_LR_MULT=50.0 \
    CHECKPOINT_ENABLED=1 SAVE_FREQ=200 KEEP_K=2 \
    JSON="$SFT_JSON" IMAGES="$SFT_IMAGES" \
    MM_LAYOUT=sft \
    bash "$LAUNCHER"
    rc=$?
    last_step=$(grep -aoE "step:\s*[0-9]+" "$OUT_DIR/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step=${last_step:-0}
    echo "[$(date)] SFT 447M attempt #$attempt rc=$rc last_step=$last_step"
    # Don't declare done until we hit at least 1100 (close to STEPS=1200).
    # The 436M template's threshold (490) was too low for our retrying
    # CUDA-assert MoE OOMs that re-trigger after a hundred steps.
    if [[ "$last_step" -ge 2300 ]]; then
        echo "[$(date)] SFT 447M done at step $last_step"; break
    fi
    if [[ "$rc" -eq 0 ]]; then break; fi
    sleep 30
done

echo "[$(date)] phase 11 447M SFT DONE (attempt=$attempt)"
