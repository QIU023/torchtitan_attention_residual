#!/usr/bin/env bash
# Local AttnRes inference bench — kill stage 0, run bench suite, restart stage 0.
#
# Bench targets (controlled by SCOPE env var):
#   SCOPE=quick      — Qwen3 TP=1 only (~15 min, GQA backbone, cleanest AttnRes-only test)
#   SCOPE=standard   — Qwen3 TP=1 + Kimi TP=8 shard (~55 min, paper-figure data)
#   SCOPE=full       — Qwen3 TP=1 + Kimi TP=1 + Kimi TP=8 (~85 min, complete)
#
# Workflow:
#   1. Kill stage 0 training
#   2. Verify GPUs idle
#   3. Generate dummy HF ckpts (CPU/GPU mix)
#   4. Run bench suite per SCOPE
#   5. Restart stage 0 fresh (loses current C progress, but C only has step 1 anyway)

set -e

WORKSPACE=/workspace/torchtitan_attention_residual
SCOPE="${SCOPE:-standard}"
BENCH_OUT="$WORKSPACE/phase11/bench_results_local_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BENCH_OUT"

echo "==========================================="
echo "[$(date)] AttnRes local bench — SCOPE=$SCOPE"
echo "Results -> $BENCH_OUT"
echo "==========================================="

# 1. Kill stage 0
echo "[1] kill stage 0"
pkill -9 -f "torchtitan.train" 2>/dev/null || true
sleep 10
if pgrep -f "torchtitan.train" | grep -v "/bin/bash" >/dev/null 2>&1; then
    echo "ERROR: torchtitan.train procs still running"; exit 1
fi

# 2. GPU sanity
echo "[2] GPU mem after kill:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8

# 3. Generate dummy HF ckpts
echo "[3a] Qwen3 dummy (~30s)"
QWEN3_CKPT="$WORKSPACE/phase11/hf_qwen3_attn_res_dummy"
python3 "$WORKSPACE/phase11/dump_qwen3_attn_res_smoke.py" --out "$QWEN3_CKPT"

KIMI_CKPT="$WORKSPACE/phase11/hf_aligned_dummy"
if [[ "$SCOPE" != "quick" ]]; then
    echo "[3b] Kimi aligned dummy (~30s)"
    python3 "$WORKSPACE/phase11/dump_aligned_smoke.py" --out "$KIMI_CKPT"
fi

# 4. Bench
echo "[4] Bench Qwen3 TP=1 (~15 min, 3 modes)"
python3 "$WORKSPACE/phase11/bench_attn_res.py" \
    --model "$QWEN3_CKPT" \
    --tp 1 \
    --prefill 1024 --decode 256 \
    --out "$BENCH_OUT/qwen3_tp1.json"

if [[ "$SCOPE" == "full" ]]; then
    echo "[4b] Bench Kimi TP=1 (~20 min, 3 modes)"
    python3 "$WORKSPACE/phase11/bench_attn_res.py" \
        --model "$KIMI_CKPT" --tp 1 \
        --out "$BENCH_OUT/kimi_tp1.json"
fi

if [[ "$SCOPE" != "quick" ]]; then
    echo "[4c] Bench Kimi TP=8 (~40 min, 4 modes incl shard)"
    python3 "$WORKSPACE/phase11/bench_attn_res.py" \
        --model "$KIMI_CKPT" --tp 8 \
        --out "$BENCH_OUT/kimi_tp8.json"
fi

# 5. Restart stage 0
echo "[5] Restart stage 0 (lr=1.5e-3 warmup=150)"
cd "$WORKSPACE"
rm -rf "$WORKSPACE/phase4/runs/lm_447m_fp8_paperalign_C/" 2>/dev/null
LOCAL_BS=4 GLOBAL_BS=384 LR=1.5e-3 WARMUP=150 STEPS=12750 \
    OUT_DIR="$WORKSPACE/phase4/runs/lm_447m_fp8_paperalign_C" \
    nohup bash phase4/launch_redo_paperalign_10B.sh \
    > "$WORKSPACE/phase4/fp8_paperalign_C.log" 2>&1 &
STAGE0_PID=$!
echo "stage 0 restarted pid=$STAGE0_PID"

echo
echo "==========================================="
echo "[$(date)] DONE — bench at $BENCH_OUT"
echo "Stage 0 pid=$STAGE0_PID, log: phase4/fp8_paperalign_C.log"
echo "==========================================="
