#!/usr/bin/env bash
# Tier B + Tier A trace recording for the configs that actually work
# (B0 FSDP=8 already done in earlier orchestrator pass; A2 PP=4 V=2
# tier_b/tier_a were interrupted).

set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"
ORCH_LOG="$WORKSPACE_DIR/phase7/orchestrator_tiers_working.log"

run_tier() {
    local cfg="$1"; local fsdp="$2"; local pp="$3"; local tp="$4"; local cp="$5"; local ep="$6"
    local v="$7"; local local_bs="$8"; local flavor="$9"
    local tier="${10}"; local steps="${11}"; local gbs="${12}"
    local out="$WORKSPACE_DIR/phase5/runs/8gpu_${cfg}_${tier}"

    if [[ -f "$out/train.log" ]] && grep -qE "step:[ ]+${steps}[ ]+" "$out/train.log" 2>/dev/null; then
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
    if pgrep -f "phase5.train_mm" >/dev/null 2>&1; then
        pkill -TERM -f "phase5.train_mm" 2>/dev/null || true; sleep 20
        pkill -KILL -f "phase5.train_mm" 2>/dev/null || true; sleep 5
    fi
}

# A2 Tier B (GBS=120, 50 steps), Tier A (GBS=384, 100 steps)
run_tier "a2_fsdp2_pp4"  2 4 1 1 1   2 5  kimi_linear_436m_block_attn_res_n4 tier_b 50  120
run_tier "a2_fsdp2_pp4"  2 4 1 1 1   2 8  kimi_linear_436m_block_attn_res_n4 tier_a 100 384

# Extract collectives from all newly captured traces
for d in "$WORKSPACE_DIR/phase5/runs/8gpu_a2_"*/tier_{a,b}_trace; do
    [[ -d "$d" ]] || continue
    /venv/main/bin/python "$WORKSPACE_DIR/phase7/extract_collectives.py" "$d" >> "$ORCH_LOG" 2>&1 || true
done

# Rebuild the phase 7 catalog with the fresh tier B/A data
/venv/main/bin/python "$WORKSPACE_DIR/phase7/build_pattern_catalog.py" >> "$ORCH_LOG" 2>&1 || true

echo "" >> "$ORCH_LOG"
echo "[$(date)] A2 tier B + A traces complete; ready for v10" >> "$ORCH_LOG"
