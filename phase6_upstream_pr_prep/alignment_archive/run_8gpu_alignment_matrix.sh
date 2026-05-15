#!/usr/bin/env bash
# Phase 6 8-GPU alignment matrix orchestrator.
#
# Runs each 3D config back-to-back, generating Tier C trace as a slice
# of each alignment run. Output goes to phase5_vlm_multimodal_sft/runs/<config_id>/.
#
# After all configs PASS alignment vs B0, downstream Tier B / Tier A
# trace recording is launched separately (run_8gpu_trace_tiers.sh).

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6_upstream_pr_prep/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"

if [[ ! -d "$PHASE4_CKPT" ]]; then
    echo "ERROR: missing $PHASE4_CKPT" >&2
    exit 1
fi

ALIGN_STEPS="${ALIGN_STEPS:-500}"
GBS_ALIGN="${GBS_ALIGN:-12}"
SEED=42

ORCH_LOG="${WORKSPACE_DIR}/phase6_upstream_pr_prep/orchestrator_8gpu_alignment.log"
exec >>"$ORCH_LOG" 2>&1

echo "==============================================================="
echo "[$(date)] phase 6 8-GPU alignment matrix START"
echo "  ALIGN_STEPS=$ALIGN_STEPS  GBS=$GBS_ALIGN  SEED=$SEED"
echo "==============================================================="

# Stage 0: GPU sanity
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
    echo "[warn] phase5_vlm_multimodal_sft.train_mm running; killing"
    pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 30
    pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 10
fi

run_config() {
    local cfg="$1"; local fsdp="$2"; local pp="$3"; local tp="$4"; local cp="$5"; local ep="$6"
    local v="$7"; local local_bs="$8"; local flavor="$9"
    local out="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_${cfg}_seed${SEED}"
    echo ""
    echo "==============================================================="
    echo "[$(date)] config=$cfg mesh=FSDP${fsdp}xPP${pp}xTP${tp}xCP${cp}xEP${ep} V=$v LBS=$local_bs"
    echo "  OUT_DIR=$out"
    echo "==============================================================="
    OUT_DIR="$out" \
    FSDP="$fsdp" PP="$pp" TP="$tp" CP="$cp" EP="$ep" \
    V="$v" \
    STEPS="$ALIGN_STEPS" LOCAL_BS="$local_bs" GLOBAL_BS="$GBS_ALIGN" SEQ_LEN=260 \
    FLAVOR="$flavor" \
    STUDENT_CKPT="$PHASE4_CKPT" \
    SEED="$SEED" DETERMINISTIC=1 COMPILE=1 \
    TRACE_TIER="tier_c" TRACE_STEPS=50 \
    bash "$LAUNCHER" || {
        echo "[ERROR] config=$cfg failed; continuing to next"
        return 0
    }
    echo "[$(date)] config=$cfg done"
    sleep 30
}

# B0 anchor: pure FSDP=8
run_config "b0_fsdp8"           8 1 1 1 1   1 1  kimi_linear_436m_block_attn_res_n4

# A2: FSDP=2 × PP=4 (drop-in)
run_config "a2_fsdp2_pp4"       2 4 1 1 1   2 1  kimi_linear_436m_block_attn_res_n4

# A3: FSDP=2 × PP=2 × TP=2 (requires TP plan in parallelize.py)
run_config "a3_fsdp2_pp2_tp2"   2 2 2 1 1   2 1  kimi_linear_436m_block_attn_res_n4

# A6: FSDP=2 × PP=2 × EP=2 (n4 flavor already has MoE; first_k_dense_replace=1)
run_config "a6_fsdp2_pp2_ep2"   2 2 1 1 2   2 1  kimi_linear_436m_block_attn_res_n4

# CP=2: blocked on fla-core ring-recurrence support for chunk_kda.
# parallelize.py raises NotImplementedError when CP > 1 because KDA
# layers' fla-core kernel cannot operate on seq-sharded inputs. See
# parallelize.py:cp_enabled branch for the full rationale. This entry
# stays in the orchestrator as documentation; running it produces the
# expected NotImplementedError + a graceful failure marker.
# run_config "cp_fsdp2_pp2_cp2"   2 2 1 2 1   2 1  kimi_linear_436m_block_attn_res_n4
echo "[$(date)] [skipped] CP=2 alignment — out of scope: see parallelize.py CP branch"

# Phase 7 control #4: TP=2 × PP=2 × EP=2 (FSDP=1)
run_config "p7c4_tp2_pp2_ep2"   1 2 2 1 2   2 1  kimi_linear_436m_block_attn_res_n4

# Phase 7 control #5: FSDP=2 × TP=2 × EP=2 (PP=1)
run_config "p7c5_fsdp2_tp2_ep2" 2 1 2 1 2   1 2  kimi_linear_436m_block_attn_res_n4

echo ""
echo "==============================================================="
echo "[$(date)] phase 6 8-GPU alignment matrix DONE"
echo "==============================================================="
