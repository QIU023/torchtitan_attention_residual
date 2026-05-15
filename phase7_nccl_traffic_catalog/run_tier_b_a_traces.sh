#!/usr/bin/env bash
# Phase 7 Tier B (production-realistic, GBS=120 50 steps) and
# Tier A (production-standardized, GBS=384 100 steps) trace recording
# for each of the 6 valid 3D configs (CP=2 excluded — see phase 6 CP
# branch in parallelize.py for the fla-core blocker).
#
# Tier C traces were already collected during the phase 6 alignment runs.
# This script re-launches each config WITHOUT --debug.deterministic
# (production load), with NCCL tracing on, and dumps per-tier dirs
# under phase5_vlm_multimodal_sft/runs/8gpu_<cfg>_<tier>/.

set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6_upstream_pr_prep/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"
ORCH_LOG="$WORKSPACE_DIR/phase7_nccl_traffic_catalog/orchestrator_tiers.log"

run_tier() {
    local cfg="$1"; local fsdp="$2"; local pp="$3"; local tp="$4"; local cp="$5"; local ep="$6"
    local v="$7"; local local_bs="$8"; local flavor="$9"
    local tier="${10}"; local steps="${11}"; local gbs="${12}"
    local out="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_${cfg}_${tier}"

    if [[ -f "$out/train.log" ]] && grep -qE "step: ${steps} " "$out/train.log" 2>/dev/null; then
        echo "[$(date)] [skip] $cfg/$tier already complete" >> "$ORCH_LOG"
        return 0
    fi
    rm -rf "$out"
    echo "" >> "$ORCH_LOG"
    echo "===============================================================" >> "$ORCH_LOG"
    echo "[$(date)] config=$cfg/$tier mesh=FSDP${fsdp}xPP${pp}xTP${tp}xCP${cp}xEP${ep} V=$v LBS=$local_bs GBS=$gbs steps=$steps" >> "$ORCH_LOG"
    echo "===============================================================" >> "$ORCH_LOG"

    OUT_DIR="$out" \
    FSDP="$fsdp" PP="$pp" TP="$tp" CP="$cp" EP="$ep" \
    V="$v" \
    STEPS="$steps" LOCAL_BS="$local_bs" GLOBAL_BS="$gbs" SEQ_LEN=260 \
    FLAVOR="$flavor" \
    STUDENT_CKPT="$PHASE4_CKPT" \
    SEED=42 DETERMINISTIC=0 COMPILE=1 \
    TRACE_TIER="$tier" TRACE_STEPS="$steps" \
    bash "$LAUNCHER" >> "$ORCH_LOG" 2>&1 || {
        echo "[$(date)] [ERROR] $cfg/$tier failed; continuing" >> "$ORCH_LOG"
    }
    if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
        pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true; sleep 20
        pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true; sleep 5
    fi
}

# Tier B: 50 steps, GBS=120 (matches v8/v9 production multimodal recipe).
# LBS chosen for correct microbatch count under PP V=2: GBS/dp/LBS must
# be ≥ V*PP. Picks satisfy each config's microbatch constraint.
TB_STEPS=50; TB_GBS=120
run_tier "b0_fsdp8"                  8 1 1 1 1   1 15  kimi_linear_436m_block_attn_res_n4 tier_b "$TB_STEPS" "$TB_GBS"
run_tier "a2_fsdp2_pp4"              2 4 1 1 1   2 5   kimi_linear_436m_block_attn_res_n4 tier_b "$TB_STEPS" "$TB_GBS"
run_tier "a3_fsdp2_pp2_tp2"          2 2 2 1 1   2 5   kimi_linear_436m_block_attn_res_n4 tier_b "$TB_STEPS" "$TB_GBS"
run_tier "a6_fsdp4_pp2_ep2"          4 2 1 1 2   2 5   kimi_linear_436m_block_attn_res_n4 tier_b "$TB_STEPS" "$TB_GBS"
run_tier "all4d_fsdp2_pp2_tp2_ep2"   2 2 2 1 2   2 5   kimi_linear_436m_block_attn_res_n4 tier_b "$TB_STEPS" "$TB_GBS"
run_tier "noppc_fsdp4_tp2_ep2"       4 1 2 1 2   1 15  kimi_linear_436m_block_attn_res_n4 tier_b "$TB_STEPS" "$TB_GBS"

# Tier A: 100 steps, GBS=384 (paper Table 2 standardized batch size).
# LBS=8 works across all configs; memory ~5GB/rank at SEQ=260, plenty of slack.
TA_STEPS=100; TA_GBS=384
run_tier "b0_fsdp8"                  8 1 1 1 1   1 8   kimi_linear_436m_block_attn_res_n4 tier_a "$TA_STEPS" "$TA_GBS"
run_tier "a2_fsdp2_pp4"              2 4 1 1 1   2 8   kimi_linear_436m_block_attn_res_n4 tier_a "$TA_STEPS" "$TA_GBS"
run_tier "a3_fsdp2_pp2_tp2"          2 2 2 1 1   2 8   kimi_linear_436m_block_attn_res_n4 tier_a "$TA_STEPS" "$TA_GBS"
run_tier "a6_fsdp4_pp2_ep2"          4 2 1 1 2   2 8   kimi_linear_436m_block_attn_res_n4 tier_a "$TA_STEPS" "$TA_GBS"
run_tier "all4d_fsdp2_pp2_tp2_ep2"   2 2 2 1 2   2 8   kimi_linear_436m_block_attn_res_n4 tier_a "$TA_STEPS" "$TA_GBS"
run_tier "noppc_fsdp4_tp2_ep2"       4 1 2 1 2   1 8   kimi_linear_436m_block_attn_res_n4 tier_a "$TA_STEPS" "$TA_GBS"

echo "" >> "$ORCH_LOG"
echo "[$(date)] phase 7 tier B + tier A trace recording done" >> "$ORCH_LOG"
