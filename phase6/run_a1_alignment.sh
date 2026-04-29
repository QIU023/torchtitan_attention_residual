#!/usr/bin/env bash
# Phase 6 A1 — real-multimodal cache-adapter alignment orchestrator.
#
# Sequence:
#   1. Launch Arm 1' = FSDP=4 baseline, seed=42, from Phase 4 step-8000,
#      4000 steps, GLOBAL_BS=12. Doubles as caption-quality story curve
#      (covers 2k alignment window + 2k convergence-evidence tail).
#   2. Launch Arm 2 = PP=4 V=2 + Interleaved1F1B + cache adapter, seed=42,
#      from Phase 4 step-8000, 2000 steps, GLOBAL_BS=12. Produces the
#      PP+adapter loss curve.
#   3. Run compare_pp_vs_fsdp.py against Arm 1''s first 2000 steps →
#      alignment report at phase6/alignment_report_arm2_real_mm.txt.
#
# Both Arm 1' and Arm 2 share: same init (Phase 4 step-8000), same projector
# init (seed=42), same data shuffle (seed=42), same LR/optimizer, same
# global_batch_size, same warmup_steps. Only difference is parallelism
# strategy. Pass criterion:
#   max |loss_arm2[step] - loss_arm1prime[step]| ≤ 0.13 nats
# over matched steps {1, 100, 500, 1000, 2000}.
#
# History: an earlier no-seed Arm 1 (GLOBAL_BS=32) was killed at step ~2800
# because (a) its GLOBAL_BS=32 did not match Arm 2's required GLOBAL_BS=12
# so it could not double as the alignment baseline, (b) without
# --debug.seed it was not reproducible. The current Arm 1' replaces it
# with a single matched-config seeded run that serves both purposes.

set -euo pipefail

WORKSPACE_DIR=/root/torchtitan_attention_residual
ARM1_DIR="${WORKSPACE_DIR}/phase5/runs/arm1_fsdp"
PHASE4_CKPT="${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"

ARM1P_DIR="${WORKSPACE_DIR}/phase5/runs/arm1prime_fsdp_seed42_from_p4_8k"
ARM2_DIR="${WORKSPACE_DIR}/phase5/runs/arm2_pp4v2_adapter_seed42_from_p4_8k"

STEPS_ARM1P="${STEPS_ARM1P:-4000}"   # FSDP baseline doubles as caption-quality curve
STEPS_ARM2="${STEPS_ARM2:-2000}"     # PP+adapter alignment window
SEED="${SEED:-42}"
GLOBAL_BS="${GLOBAL_BS:-32}"         # PP=4 V=2 needs num_microbatches >= 8;
                                     # GLOBAL_BS=32 / LOCAL_BS_PP=2 = 16 microbatches (healthy headroom).
                                     # Matches original Arm 1 GLOBAL_BS so caption-loss tier is comparable.
LOCAL_BS_FSDP=$((GLOBAL_BS / 4))     # FSDP=4 → 8 per rank
LOCAL_BS_PP="${LOCAL_BS_PP:-2}"      # num_microbatches = GLOBAL_BS / LOCAL_BS_PP = 16 (>= V*PP=8 ✓)

LOG=/root/phase6_alignment_orchestrator.log
exec >>"$LOG" 2>&1

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 alignment orchestrator START"
echo "  STEPS_ARM1P=$STEPS_ARM1P STEPS_ARM2=$STEPS_ARM2 SEED=$SEED"
echo "  GLOBAL_BS=$GLOBAL_BS LOCAL_BS_FSDP=$LOCAL_BS_FSDP LOCAL_BS_PP=$LOCAL_BS_PP"
echo "==============================================================="

# === Stage 0: GPU sanity check (current Arm 1 was killed before this script ran) ===
echo "[$(date)] [stage 0] GPU memory check:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
if pgrep -f "phase5.train_mm" >/dev/null 2>&1; then
    echo "[$(date)] [stage 0] WARNING: phase5.train_mm still running, killing"
    pkill -TERM -f "phase5.train_mm" 2>/dev/null || true
    sleep 30
    pkill -KILL -f "phase5.train_mm" 2>/dev/null || true
    sleep 10
fi

# === Stage 1: launch Arm 1' (FSDP, seed=42, from Phase 4 step-8000) ===
echo "[$(date)] [stage 1] launching Arm 1' FSDP baseline (STEPS=$STEPS_ARM1P, GLOBAL_BS=$GLOBAL_BS, LOCAL_BS=$LOCAL_BS_FSDP, seed=$SEED)"
mkdir -p "$ARM1P_DIR"
cd "$WORKSPACE_DIR"
source /venv/main/bin/activate

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
    --training.steps "$STEPS_ARM1P" \
    --training.local_batch_size "$LOCAL_BS_FSDP" \
    --training.global_batch_size "$GLOBAL_BS" \
    --training.seq_len 260 \
    --optimizer.lr 1e-5 \
    --lr_scheduler.warmup_steps 10 \
    --lr_scheduler.total_steps "$STEPS_ARM1P" \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --checkpoint.enable \
    --checkpoint.initial_load_path "$PHASE4_CKPT" \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval 1000 \
    --checkpoint.keep_latest_k 2 \
    --debug.seed "$SEED" \
    --debug.deterministic \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 1 \
    --dump_folder "$ARM1P_DIR" \
    --compile.enable \
    >"$ARM1P_DIR/train.log" 2>&1

echo "[$(date)] [stage 1] Arm 1' done"
sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# === Stage 2: launch Arm 2 (PP=4 V=2 + cache adapter, seed=42, from Phase 4 step-8000) ===
echo "[$(date)] [stage 2] launching Arm 2 PP+adapter (STEPS=$STEPS_ARM2, GLOBAL_BS=$GLOBAL_BS, LOCAL_BS_PP=$LOCAL_BS_PP, seed=$SEED)"
mkdir -p "$ARM2_DIR"
cd "$WORKSPACE_DIR"
INIT=weak_ckpt INIT_CKPT="$PHASE4_CKPT" \
NGPU=4 PP=4 V=2 LOCAL_BS="$LOCAL_BS_PP" GLOBAL_BS="$GLOBAL_BS" STEPS="$STEPS_ARM2" \
ADAPTER=1 SEED="$SEED" DETERMINISTIC=1 \
COMPILE=1 \
OUT_DIR="$ARM2_DIR" \
bash phase5/launch_pp_adapter.sh

echo "[$(date)] [stage 2] Arm 2 done"

# === Stage 3: alignment report ===
echo "[$(date)] [stage 3] running compare_pp_vs_fsdp.py"
cd "$WORKSPACE_DIR"
python phase5/compare_pp_vs_fsdp.py \
    --pp "$ARM2_DIR/tb" \
    --fsdp "$ARM1P_DIR/tb" \
    > "$WORKSPACE_DIR/phase6/alignment_report_arm2_real_mm.txt" 2>&1 || true

echo "[$(date)] [stage 3] alignment report:"
cat "$WORKSPACE_DIR/phase6/alignment_report_arm2_real_mm.txt"

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 alignment orchestrator COMPLETE"
echo "==============================================================="
