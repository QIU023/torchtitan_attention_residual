#!/usr/bin/env bash
# Overnight orchestrator: pretrain → SFT (re-uses LLaVA-Pretrain).
#
# 1. Wait for the FSDP=8 multimodal pretrain (already started by
#    run_v_fsdp8_447m_pretrain.sh) to reach step-2500.
# 2. Kick SFT: same LLaVA-Pretrain data, fine-tune-style hparams
#    (LR 2e-5 from 1e-5, smaller batch, 500 steps). NO COCO —
#    aligns with what 436m phase 9-A actually used: same data
#    head, different LR/schedule.
# 3. PPO is downstream of SFT and needs SGLang rollout bridge —
#    deferred to next session, not in this overnight chain.
#
# Run via:
#   nohup bash phase6/orchestrate_overnight.sh > phase6/overnight.log 2>&1 &
set -u
WS=/root/torchtitan_attention_residual
PRETRAIN_OUT=$WS/phase5/runs/vlm_447m_pretrain
SFT_OUT=$WS/phase5/runs/vlm_447m_sft_pretrain
PRETRAIN_DATA_DIR=/workspace/.hf_home/LLaVA-Pretrain
PRETRAIN_JSON=$PRETRAIN_DATA_DIR/blip_laion_cc_sbu_558k.json

mkdir -p "$SFT_OUT"
echo "[$(date)] orchestrator START"

while true; do
    last=$(grep -oE "step:\s*[0-9]+" "$PRETRAIN_OUT/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last=${last:-0}
    if [[ "$last" -ge 2500 ]]; then
        echo "[$(date)] pretrain reached step $last → proceeding"
        break
    fi
    sleep 300
    echo "[$(date)] pretrain at step $last / 2500 ..."
done

PRETRAIN_CKPT=$(ls -d "$PRETRAIN_OUT/checkpoint/step-"* 2>/dev/null | sort -V | tail -1)
if [[ -z "$PRETRAIN_CKPT" ]]; then
    echo "[$(date)] ERROR: no pretrain ckpt found at $PRETRAIN_OUT/checkpoint/"
    exit 1
fi
echo "[$(date)] using pretrain ckpt: $PRETRAIN_CKPT"

LOG="$WS/phase6/overnight_sft.log"

OUT_DIR="$SFT_OUT" \
LOG_RANK=0 \
FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
PP_MICROBATCH=8 \
STEPS=500 LOCAL_BS=8 GLOBAL_BS=64 SEQ_LEN=260 \
FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
STUDENT_CKPT="$PRETRAIN_CKPT" \
SEED=42 DETERMINISTIC=0 COMPILE=0 \
LR=2e-5 WARMUP=50 \
CHECKPOINT_ENABLED=1 SAVE_FREQ=100 KEEP_K=2 \
TRACE_TIER= \
DATA_DIR="$PRETRAIN_DATA_DIR" \
JSON="$PRETRAIN_JSON" IMAGES="$PRETRAIN_DATA_DIR" \
bash "$WS/phase6/launch_8gpu_mm.sh" >>"$LOG" 2>&1
sft_rc=$?
echo "[$(date)] SFT done rc=$sft_rc"
echo "[$(date)] orchestrator END"
