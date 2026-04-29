#!/usr/bin/env bash
# Phase 6 A1 — real-multimodal cache-adapter alignment orchestrator.
#
# Sequence:
#   1. Wait for currently-running Arm 1 (FSDP, no-seed) to log past step 6000
#      so its ckpt is safely persisted (caption-quality story deliverable).
#   2. SIGTERM the Arm 1 training process so GPUs free up.
#   3. Launch Arm 1' = FSDP=4 baseline, seed=42, from Phase 4 step-8000,
#      2000 steps, GLOBAL_BS=12 (matched to Arm 2). Produces the FSDP
#      reference loss curve.
#   4. Launch Arm 2 = PP=4 V=2 + Interleaved1F1B + cache adapter, seed=42,
#      from Phase 4 step-8000, 2000 steps, GLOBAL_BS=12. Produces the
#      PP+adapter loss curve.
#   5. Run compare_pp_vs_fsdp.py → alignment report at
#      phase6/alignment_report_arm2_real_mm.txt.
#
# Both Arm 1' and Arm 2 share: same init (Phase 4 step-8000), same projector
# init (seed=42), same data shuffle (seed=42), same LR/optimizer, same
# global_batch_size. Only difference is parallelism strategy. Pass criterion:
# max |loss_arm2[step] - loss_arm1prime[step]| ≤ 0.13 nats over matched steps.

set -euo pipefail

WORKSPACE_DIR=/root/torchtitan_attention_residual
ARM1_DIR="${WORKSPACE_DIR}/phase5/runs/arm1_fsdp"
PHASE4_CKPT="${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"

ARM1P_DIR="${WORKSPACE_DIR}/phase5/runs/arm1prime_fsdp_seed42_from_p4_8k"
ARM2_DIR="${WORKSPACE_DIR}/phase5/runs/arm2_pp4v2_adapter_seed42_from_p4_8k"

STEPS="${STEPS:-2000}"
SEED="${SEED:-42}"
GLOBAL_BS="${GLOBAL_BS:-12}"

LOG=/root/phase6_alignment_orchestrator.log
exec >>"$LOG" 2>&1

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 alignment orchestrator START"
echo "  STEPS=$STEPS SEED=$SEED GLOBAL_BS=$GLOBAL_BS"
echo "==============================================================="

# === Stage 1: wait for Arm 1 to pass step 6000 (ckpt safely on disk) ===
echo "[$(date)] [stage 1] waiting for Arm 1 to log past step 6000..."
while true; do
    if grep -qE "step: 60(1[0-9]|2[0-9])" "$ARM1_DIR/train.log" 2>/dev/null; then
        echo "[$(date)] [stage 1] Arm 1 logged past step 6000"
        break
    fi
    if [[ -d "${ARM1_DIR}/checkpoint/step-6000" ]]; then
        size=$(du -sb "${ARM1_DIR}/checkpoint/step-6000" 2>/dev/null | awk '{print $1}')
        if [[ -n "$size" && "$size" -gt 14000000000 ]]; then
            # Belt-and-suspenders: ckpt dir is full size, give one more minute
            echo "[$(date)] [stage 1] step-6000 ckpt complete ($((size/1024/1024))MB)"
            sleep 60
            break
        fi
    fi
    sleep 60
done

# === Stage 2: SIGTERM Arm 1, wait for clean exit ===
echo "[$(date)] [stage 2] sending SIGTERM to phase5.train_mm processes"
pkill -TERM -f "phase5.train_mm" 2>/dev/null || true

for i in $(seq 1 24); do
    sleep 5
    if ! pgrep -f "phase5.train_mm" >/dev/null 2>&1; then
        echo "[$(date)] [stage 2] training processes exited cleanly after ${i}*5s"
        break
    fi
done
if pgrep -f "phase5.train_mm" >/dev/null 2>&1; then
    echo "[$(date)] [stage 2] forcing SIGKILL (clean exit timed out)"
    pkill -KILL -f "phase5.train_mm" 2>/dev/null || true
    sleep 15
fi

# Wait for GPU memory to actually free
echo "[$(date)] [stage 2] GPU memory before stage 3:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
sleep 30

# === Stage 3: launch Arm 1' (FSDP, seed=42, from Phase 4 step-8000) ===
echo "[$(date)] [stage 3] launching Arm 1' FSDP baseline"
mkdir -p "$ARM1P_DIR"
cd "$WORKSPACE_DIR"
source /venv/main/bin/activate

# GLOBAL_BS=12 with FSDP=4 → LOCAL_BS=3
LOCAL_BS_FSDP=$((GLOBAL_BS / 4))

PYTHONPATH="$WORKSPACE_DIR:$WORKSPACE_DIR/torchtitan${PYTHONPATH:+:$PYTHONPATH}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5.train_mm \
    --mm.json /root/hf_cache/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --mm.images /root/hf_cache/LLaVA-Pretrain \
    --mm.vision-model google/siglip-base-patch16-224 \
    --mm.tokenizer NousResearch/Meta-Llama-3.1-8B \
    --mm.cache-dir /root/hf_cache \
    --mm.proj-lr-mult 50.0 \
    --mm.global-seq-len 258 \
    --module kimi_linear --config kimi_linear_436m_block_attn_res_n4 \
    --hf_assets_path "$WORKSPACE_DIR/torchtitan/assets/hf/Llama-3.1-8B" \
    --training.steps "$STEPS" \
    --training.local_batch_size "$LOCAL_BS_FSDP" \
    --training.global_batch_size "$GLOBAL_BS" \
    --training.seq_len 260 \
    --optimizer.lr 1e-5 \
    --lr_scheduler.warmup_steps 50 \
    --lr_scheduler.total_steps "$STEPS" \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --checkpoint.enable \
    --checkpoint.initial_load_path "$PHASE4_CKPT" \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval 999999 \
    --checkpoint.keep_latest_k 1 \
    --debug.seed "$SEED" \
    --debug.deterministic \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 1 \
    --dump_folder "$ARM1P_DIR" \
    --compile.enable \
    >"$ARM1P_DIR/train.log" 2>&1

echo "[$(date)] [stage 3] Arm 1' done"
sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# === Stage 4: launch Arm 2 (PP=4 V=2 + cache adapter, seed=42, from Phase 4 step-8000) ===
echo "[$(date)] [stage 4] launching Arm 2 PP+adapter"
mkdir -p "$ARM2_DIR"
cd "$WORKSPACE_DIR"
INIT=weak_ckpt INIT_CKPT="$PHASE4_CKPT" \
NGPU=4 PP=4 V=2 LOCAL_BS=1 GLOBAL_BS="$GLOBAL_BS" STEPS="$STEPS" \
ADAPTER=1 SEED="$SEED" DETERMINISTIC=1 \
COMPILE=1 \
OUT_DIR="$ARM2_DIR" \
bash phase5/launch_pp_adapter.sh

echo "[$(date)] [stage 4] Arm 2 done"

# === Stage 5: alignment report ===
echo "[$(date)] [stage 5] running compare_pp_vs_fsdp.py"
cd "$WORKSPACE_DIR"
python phase5/compare_pp_vs_fsdp.py \
    --pp "$ARM2_DIR/tb" \
    --fsdp "$ARM1P_DIR/tb" \
    > "$WORKSPACE_DIR/phase6/alignment_report_arm2_real_mm.txt" 2>&1 || true

echo "[$(date)] [stage 5] alignment report:"
cat "$WORKSPACE_DIR/phase6/alignment_report_arm2_real_mm.txt"

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 alignment orchestrator COMPLETE"
echo "==============================================================="
