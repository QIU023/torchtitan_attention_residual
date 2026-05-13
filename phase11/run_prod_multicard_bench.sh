#!/usr/bin/env bash
# 8x 5090 production multi-card AttnRes inference bench.
#
# Validates the AttnRes SGLang overlay on prod-grade model layouts:
#   1. Kimi Linear 48B-layout (paper) with downscaled MoE expert count
#      → ~7-28B params, KDA + MLA + MoE hybrid → exercises full overlay path
#   2. Qwen3 14B + AttnRes overlay
#      → GQA backbone (no MLA NaN) → cleanest AttnRes-on-vanilla-attention test
#
# Both run at TP=8, 4 modes (vanilla / naive / two-phase / shard), with the
# fp32 MLA fallback env vars active for Kimi (Blackwell SM 12.0 unblock).
#
# Outputs JSON to phase11/bench_results_prod_<timestamp>/ for paper figures.
#
# Workflow:
#   1. Stop stage 0 training (saves first)
#   2. Generate dummy HF ckpts (CPU/GPU mix; ~5-15 min depending on model size)
#   3. Bench TP=8 4-mode sweep (per model: ~30-50 min)
#   4. Restart stage 0 (auto-resumes from latest ckpt)
#
# Env knobs:
#   KIMI_EXPERTS={16,32,64,128,256}  default 32 (~7B, light)
#                                    64 = ~14B (50% fill of 8x5090)
#                                    128 = ~28B (80% fill — near paper density)
#   QWEN3_SIZE={qwen3_7b,qwen3_14b,qwen3_32b}  default qwen3_14b
#   SKIP_KIMI=1  skip Kimi bench (only Qwen3)
#   SKIP_QWEN3=1 skip Qwen3 bench (only Kimi)
#   BENCH_TP=8   TP size for bench (default 8)
#   BENCH_CTX=4096  prompt length for bench (default 4096)

set -e

WORKSPACE=/workspace/torchtitan_attention_residual
KIMI_EXPERTS="${KIMI_EXPERTS:-32}"
QWEN3_SIZE="${QWEN3_SIZE:-qwen3_14b}"
BENCH_TP="${BENCH_TP:-8}"
BENCH_CTX="${BENCH_CTX:-4096}"
SKIP_KIMI="${SKIP_KIMI:-0}"
SKIP_QWEN3="${SKIP_QWEN3:-0}"
STAGE0_RESUME="${STAGE0_RESUME:-1}"  # auto-restart stage 0 after bench

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BENCH_OUT="$WORKSPACE/phase11/bench_results_prod_$TIMESTAMP"
mkdir -p "$BENCH_OUT"

echo "============================================================"
echo "[$(date)] 8-card TP=$BENCH_TP AttnRes prod bench"
echo "  Kimi 48B-layout: experts=$KIMI_EXPERTS (skip=$SKIP_KIMI)"
echo "  Qwen3 size:      $QWEN3_SIZE (skip=$SKIP_QWEN3)"
echo "  Output dir:      $BENCH_OUT"
echo "============================================================"

# [1] Kill stage 0
echo "[1] Stop stage 0 training"
pkill -9 -f "torchtitan.train" 2>/dev/null || true
sleep 10
if pgrep -f "torchtitan.train" | grep -v "/bin/bash" >/dev/null 2>&1; then
    echo "ERROR: torchtitan procs still running"; exit 1
fi
echo "    GPU mem after kill:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8

# [2] Dummy ckpt generation
KIMI_CKPT="$WORKSPACE/phase11/hf_kimi_48b_e${KIMI_EXPERTS}_dummy"
QWEN3_CKPT="$WORKSPACE/phase11/hf_${QWEN3_SIZE}_attn_res_dummy"

if [[ "$SKIP_KIMI" == "0" && ! -d "$KIMI_CKPT" ]]; then
    echo "[2a] Dump Kimi 48B-layout (experts=$KIMI_EXPERTS) — may take 5-15 min"
    python3 "$WORKSPACE/phase11/dump_kimi_48b_attn_res_dummy.py" \
        --num-experts "$KIMI_EXPERTS" --out "$KIMI_CKPT"
fi

if [[ "$SKIP_QWEN3" == "0" && ! -d "$QWEN3_CKPT" ]]; then
    echo "[2b] Dump Qwen3 ($QWEN3_SIZE) AttnRes — may take 3-10 min"
    python3 "$WORKSPACE/phase11/dump_qwen3_big_attn_res_dummy.py" \
        --size "$QWEN3_SIZE" --out "$QWEN3_CKPT"
fi

# [3] Bench
if [[ "$SKIP_KIMI" == "0" ]]; then
    echo "[3a] Bench Kimi 48B-layout TP=$BENCH_TP (~30-50 min, 4 modes)"
    # Kimi needs fp32 MLA fallback on Blackwell (RTX 5090 SM 12.0)
    ATTNRES_MLA_FP32_FALLBACK=1 ATTNRES_FP32_NORM=1 ATTNRES_INPUT_CLAMP=32 \
    python3 "$WORKSPACE/phase11/bench_attn_res.py" \
        --model "$KIMI_CKPT" \
        --tp "$BENCH_TP" \
        --prefill "$BENCH_CTX" --decode 256 \
        --out "$BENCH_OUT/kimi_48b_e${KIMI_EXPERTS}_tp${BENCH_TP}.json" \
        || echo "WARN: Kimi bench failed (continuing)"
fi

if [[ "$SKIP_QWEN3" == "0" ]]; then
    echo "[3b] Bench Qwen3 ($QWEN3_SIZE) TP=$BENCH_TP (~30-50 min, 4 modes)"
    # Qwen3 uses GQA, no MLA → no fallback needed; cleanest AttnRes signal
    python3 "$WORKSPACE/phase11/bench_attn_res.py" \
        --model "$QWEN3_CKPT" \
        --tp "$BENCH_TP" \
        --prefill "$BENCH_CTX" --decode 256 \
        --out "$BENCH_OUT/${QWEN3_SIZE}_tp${BENCH_TP}.json" \
        || echo "WARN: Qwen3 bench failed (continuing)"
fi

# [4] Resume stage 0 if requested (auto-resumes from latest ckpt in dump_folder)
if [[ "$STAGE0_RESUME" == "1" ]]; then
    echo "[4] Restart stage 0 (auto-resumes from latest ckpt)"
    cd "$WORKSPACE"
    LOCAL_BS=4 GLOBAL_BS=384 LR=1.5e-3 WARMUP=150 STEPS=12750 SAVE_FREQ=100 \
        OUT_DIR="$WORKSPACE/phase4/runs/lm_447m_fp8_paperalign_C" \
        nohup bash phase4/launch_redo_paperalign_10B.sh \
        > "$WORKSPACE/phase4/fp8_paperalign_C.log" 2>&1 &
    STAGE0_PID=$!
    echo "    stage 0 restarted pid=$STAGE0_PID (SAVE_FREQ=100)"
fi

echo
echo "============================================================"
echo "[$(date)] DONE — bench at $BENCH_OUT"
echo "============================================================"
ls -la "$BENCH_OUT"
