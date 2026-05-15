#!/usr/bin/env bash
# Run all remaining 8-GPU alignment configs back-to-back after B0 finishes.
# Polls for B0 train.log "Training completed" or step:500 marker, then
# launches A2 → A3 → A6 → phase7 controls in sequence. Each run produces
# an alignment loss curve in tb/ + Tier C NCCL trace in tier_c_trace/.

set -u  # don't -e — let individual failures continue to next config

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6_upstream_pr_prep/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"

ALIGN_STEPS="${ALIGN_STEPS:-500}"
# GBS=16 satisfies the largest 3D mesh's microbatch constraint:
# Interleaved1F1B V=2 PP=4 with FSDP=2 LBS=1 needs >= V*PP = 8 mb per dp rank,
# so GBS = FSDP*V*PP = 2*2*4 = 16 minimum. All other 3D combos here have
# slack at GBS=16. The B0 anchor matches by using LBS=2 on FSDP=8.
GBS_ALIGN="${GBS_ALIGN:-16}"
SEED=42
ORCH_LOG="$WORKSPACE_DIR/phase6_upstream_pr_prep/orchestrator_8gpu.log"

# B0 anchor and all 3D configs run via the same launch path so they
# share GBS=16 (the cross-mesh minimum). B0 uses LOCAL_BS=2 on FSDP=8;
# all other configs use LOCAL_BS=1.
echo "[$(date)] orchestrator START at GBS=$GBS_ALIGN" >> "$ORCH_LOG"

# Cleanup any stragglers
if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
    pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 30
    pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 10
fi

run_config() {
    local cfg="$1"; local fsdp="$2"; local pp="$3"; local tp="$4"; local cp="$5"; local ep="$6"
    local v="$7"; local local_bs="$8"; local flavor="$9"
    local out="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_${cfg}_seed${SEED}"

    if [[ -f "$out/train.log" ]] && grep -qE "(- step:[ ]+500[ ]+|Training completed)" "$out/train.log" 2>/dev/null; then
        echo "[$(date)] [skip] $cfg already complete" >> "$ORCH_LOG"
        return 0
    fi
    rm -rf "$out"

    echo "" >> "$ORCH_LOG"
    echo "===============================================================" >> "$ORCH_LOG"
    echo "[$(date)] config=$cfg mesh=FSDP${fsdp}xPP${pp}xTP${tp}xCP${cp}xEP${ep} V=$v LBS=$local_bs flavor=$flavor" >> "$ORCH_LOG"
    echo "===============================================================" >> "$ORCH_LOG"

    OUT_DIR="$out" \
    FSDP="$fsdp" PP="$pp" TP="$tp" CP="$cp" EP="$ep" \
    V="$v" \
    STEPS="$ALIGN_STEPS" LOCAL_BS="$local_bs" GLOBAL_BS="$GBS_ALIGN" SEQ_LEN=260 \
    FLAVOR="$flavor" \
    STUDENT_CKPT="$PHASE4_CKPT" \
    SEED="$SEED" DETERMINISTIC=1 COMPILE=1 \
    TRACE_TIER="tier_c" TRACE_STEPS=50 \
    bash "$LAUNCHER" >> "$ORCH_LOG" 2>&1 || {
        echo "[$(date)] [ERROR] $cfg failed; continuing" >> "$ORCH_LOG"
    }
    echo "[$(date)] $cfg done" >> "$ORCH_LOG"

    # Cleanup any leftover ranks before next config
    if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
        pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
        sleep 30
        pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
        sleep 10
    fi
}

# Mesh constraint on 8 GPUs: dense_mesh = FSDP*PP*TP*CP must equal 8.
# EP borrows from FSDP×TP and does not enter the dense product.
#
# B0 anchor — FSDP=8 with LBS=2 to match GBS=16
# (skip if already complete from prior run with valid step:500)
run_config "b0_fsdp8"            8 1 1 1 1   1 2  kimi_linear_436m_block_attn_res_n4

# A2 — FSDP=2 × PP=4 V=2 + adapter (deep PP, no TP/EP)
run_config "a2_fsdp2_pp4"        2 4 1 1 1   2 1  kimi_linear_436m_block_attn_res_n4

# A3 — FSDP=2 × PP=2 × TP=2 V=2 + adapter (TP plan applied; dense MLP TP only)
run_config "a3_fsdp2_pp2_tp2"    2 2 2 1 1   2 1  kimi_linear_436m_block_attn_res_n4

# A6 — FSDP=4 × PP=2 V=2 + EP=2 + adapter
# EP=2 borrows from FSDP=4 (since FSDP*TP=4 ≥ EP*ETP=2). dense product = 4*2*1*1 = 8.
run_config "a6_fsdp4_pp2_ep2"    4 2 1 1 2   2 1  kimi_linear_436m_block_attn_res_n4

# 4D — FSDP=2 × PP=2 × TP=2 V=2 + EP=2 + adapter (full 4-axis composition)
# Dense product 2*2*2*1=8; EP=2 borrows from FSDP*TP=4 ≥ 2.
run_config "all4d_fsdp2_pp2_tp2_ep2"  2 2 2 1 2  2 1  kimi_linear_436m_block_attn_res_n4

# No-PP control — FSDP=4 × TP=2 + EP=2 (PP=1, exercises EP+TP without stage)
# Dense product 4*1*2*1=8; EP=2 borrows from FSDP*TP=8 ≥ 2.
run_config "noppc_fsdp4_tp2_ep2"  4 1 2 1 2  1 1  kimi_linear_436m_block_attn_res_n4

echo "" >> "$ORCH_LOG"
echo "[$(date)] all alignment configs done; running compare_pp_vs_fsdp reports" >> "$ORCH_LOG"

bash "$WORKSPACE_DIR/phase6_upstream_pr_prep/run_alignment_reports.sh" >> "$ORCH_LOG" 2>&1 || true

# Also run extract_collectives on every tier_c trace
echo "" >> "$ORCH_LOG"
echo "[$(date)] extracting NCCL collectives from Tier C traces" >> "$ORCH_LOG"
for d in "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_"*/tier_c_trace; do
    [[ -d "$d" ]] || continue
    /venv/main/bin/python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/extract_collectives.py" "$d" >> "$ORCH_LOG" 2>&1 || true
done

# Then chain into Tier B + Tier A trace recording
echo "" >> "$ORCH_LOG"
echo "[$(date)] starting Tier B + Tier A trace recording" >> "$ORCH_LOG"
bash "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/run_tier_b_a_traces.sh" >> "$ORCH_LOG" 2>&1 || true

# Re-extract collectives now that tier_b and tier_a traces are present
echo "" >> "$ORCH_LOG"
echo "[$(date)] extracting NCCL collectives from Tier B/A traces" >> "$ORCH_LOG"
for d in "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_"*/tier_{b,a}_trace; do
    [[ -d "$d" ]] || continue
    /venv/main/bin/python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/extract_collectives.py" "$d" >> "$ORCH_LOG" 2>&1 || true
done

echo "" >> "$ORCH_LOG"
echo "[$(date)] building phase 7 pattern catalog" >> "$ORCH_LOG"
/venv/main/bin/python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/build_pattern_catalog.py" >> "$ORCH_LOG" 2>&1 || true

echo "" >> "$ORCH_LOG"
echo "[$(date)] alignment matrix + trace tiers + catalog complete; ready for v10" >> "$ORCH_LOG"
echo "[$(date)] To launch v10 multimodal pretrain: bash phase6_upstream_pr_prep/run_v10_pretrain.sh" >> "$ORCH_LOG"
