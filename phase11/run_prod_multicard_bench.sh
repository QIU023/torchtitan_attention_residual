#!/usr/bin/env bash
# 8x 5090 multi-card AttnRes overlay correctness + perf bench.
#
# Goal: validate AttnRes SGLang overlay at production multi-card scale
# by running the 4-mode comparison (vanilla / naive / two-phase / shard)
# on two prod-grade dummy ckpts:
#
#   * Kimi 48B-layout (paper L=27 d=2304 N=9, ~14B params at num_experts=64)
#     — exercises full overlay path: KDA + MLA + MoE + AttnRes
#     — needs fp32 MLA fallback on Blackwell (SM 12.0 RTX 5090)
#
#   * Qwen3-14B + AttnRes overlay (GQA backbone, no MLA)
#     — cleanest AttnRes-only signal (no MLA NaN risk)
#
# What the 4 modes test:
#   vanilla   — AttnRes bypassed (baseline)
#   naive     — single-pass aggregator (correctness reference)
#   two-phase — Phase 1 batched + Phase 2 fused Triton kernel
#   shard     — two-phase + seq-dim TP shard (AR bytes -58%)
#
# Pass criterion: vanilla/naive/two-phase/shard outputs within noise band;
# two-phase decode tps > naive; shard mode at TP=8 has lower AR bytes.
#
# Workflow:
#   1. Stop stage 0
#   2. Dump both ckpts (skipped if dirs already exist)
#   3. Bench Kimi TP=8 4-mode (~30-50 min)
#   4. Bench Qwen3 TP=8 4-mode (~20-40 min)
#   5. Restart stage 0 (auto-resume from latest ckpt + SAVE_FREQ=100)

set -e

WORKSPACE=/workspace/torchtitan_attention_residual
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BENCH_OUT="$WORKSPACE/phase11/bench_results_prod_$TIMESTAMP"
mkdir -p "$BENCH_OUT"

# Single fixed prod-grade size per model (no size sweep — this is a
# correctness/perf comparison across 4 modes, not a scaling study).
KIMI_CKPT="$WORKSPACE/phase11/hf_kimi_48b_e32_dummy"
QWEN3_CKPT="$WORKSPACE/phase11/hf_qwen3_14b_attn_res_dummy"

echo "============================================================"
echo "[$(date)] 8-card TP=8 AttnRes overlay bench"
echo "  Output: $BENCH_OUT"
echo "============================================================"

# [1] Kill stage 0
echo "[1] Stop stage 0"
pkill -9 -f "torchtitan.train" 2>/dev/null || true
sleep 10
pgrep -f "torchtitan.train" | grep -v "/bin/bash" >/dev/null && { echo "ERROR: stage 0 still alive"; exit 1; } || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -2

# [2] Dump dummies
if [[ ! -d "$KIMI_CKPT" ]]; then
    echo "[2a] Dump Kimi 48B-layout (num_experts=32, ~7B params, ~14 GB safetensors)"
    python3 "$WORKSPACE/phase11/dump_kimi_48b_attn_res_dummy.py" \
        --num-experts 32 --out "$KIMI_CKPT"
fi
if [[ ! -d "$QWEN3_CKPT" ]]; then
    echo "[2b] Dump Qwen3-14B + AttnRes (~14B params, ~28 GB safetensors)"
    python3 "$WORKSPACE/phase11/dump_qwen3_big_attn_res_dummy.py" \
        --size qwen3_14b --out "$QWEN3_CKPT"
fi

# [3] Bench Kimi TP=8 — ctx sweep 4K + 16K (perf scaling)
for CTX in 4096 16384; do
    echo "[3.$CTX] Kimi TP=8 4-mode bench @ ctx=$CTX (fp32 MLA fallback on)"
    ATTNRES_MLA_FP32_FALLBACK=1 ATTNRES_FP32_NORM=1 ATTNRES_INPUT_CLAMP=32 \
    python3 "$WORKSPACE/phase11/bench_attn_res.py" \
        --model "$KIMI_CKPT" --tp 8 \
        --prefill "$CTX" --decode 256 \
        --disable-cuda-graph \
        --out "$BENCH_OUT/kimi_48b_e64_tp8_ctx${CTX}.json" \
        || echo "WARN: Kimi bench ctx=$CTX failed"
done

# [3-mem] Memory probe at TP=8 shard=0 vs shard=1
echo "[3-mem] Kimi TP=8 memory probe (shard=0 vs shard=1)"
ATTNRES_MLA_FP32_FALLBACK=1 \
python3 "$WORKSPACE/phase11/probe_memory.py" \
    --model "$KIMI_CKPT" --tp 8 --prefill 16384 \
    --out "$BENCH_OUT/kimi_48b_e64_tp8_memprobe.json" \
    || echo "WARN: memprobe failed (probe_memory.py optional)"

# [4] Bench Qwen3 TP=8 — ctx sweep 4K + 16K
for CTX in 4096 16384; do
    echo "[4.$CTX] Qwen3-14B TP=8 4-mode bench @ ctx=$CTX (GQA, no MLA)"
    python3 "$WORKSPACE/phase11/bench_attn_res.py" \
        --model "$QWEN3_CKPT" --tp 8 \
        --prefill "$CTX" --decode 256 \
        --disable-cuda-graph \
        --out "$BENCH_OUT/qwen3_14b_tp8_ctx${CTX}.json" \
        || echo "WARN: Qwen3 bench ctx=$CTX failed"
done

# [5] Restart stage 0 (auto-resume from latest ckpt)
echo "[5] Restart stage 0 (SAVE_FREQ=100, auto-resume)"
cd "$WORKSPACE"
LOCAL_BS=4 GLOBAL_BS=384 LR=1.5e-3 WARMUP=150 STEPS=12750 SAVE_FREQ=100 \
    OUT_DIR="$WORKSPACE/phase4/runs/lm_447m_fp8_paperalign_C" \
    nohup bash phase4/launch_redo_paperalign_10B.sh \
    > "$WORKSPACE/phase4/fp8_paperalign_C.log" 2>&1 &
echo "    stage 0 restarted pid=$!"

echo
echo "============================================================"
echo "[$(date)] DONE — results at $BENCH_OUT"
echo "============================================================"
ls -la "$BENCH_OUT"
