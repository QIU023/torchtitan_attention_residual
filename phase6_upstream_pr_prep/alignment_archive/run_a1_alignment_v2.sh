#!/usr/bin/env bash
# Phase 6 A1 v2 — alignment at the documented GLOBAL_BS=12 config.
#
# After v1 hit two issues at GLOBAL_BS=32:
#   1. LOCAL_BS=2 caused an early NCCL P2P timeout (likely image-cache
#      cold start + per-rank time skew during compile).
#   2. LOCAL_BS=1 ran cleanly for 10 smoke steps but the PP grad_norm
#      reading was ~600× the FSDP grad_norm — confirmed against the
#      prior C4-only alignment to be a *reporting artifact* (PP
#      scheduler reports unscaled grads, FSDP reports clip-scaled).
#      Loss values trained fine.
#
# v2 strategy: drop GLOBAL_BS=32 entirely for the alignment pair. Use
# the documented GLOBAL_BS=12 (LOCAL_BS=1 PP, LOCAL_BS=3 FSDP) for
# both Arm 1'-align and Arm 2. The v1 GBS=32 Arm 1' run is preserved
# as the caption-quality story curve; it is NOT part of the alignment
# pair.
#
# Sequence:
#   stage 1: Arm 1'-align = FSDP=4, seed=42, from Phase 4 step-8000,
#            2000 steps, GLOBAL_BS=12 LOCAL_BS=3.
#   stage 2: Arm 2 = PP=4 V=2 + Interleaved1F1B + cache adapter,
#            same seed/init, 2000 steps, GLOBAL_BS=12 LOCAL_BS=1.
#   stage 3: compare_pp_vs_fsdp.py → phase6_upstream_pr_prep/alignment_report_arm2_real_mm.txt.
#
# Pass criterion: max |loss_arm2[step] - loss_arm1align[step]| ≤ 0.13 nats
# over matched steps {1, 100, 500, 1000, 2000}.

set -euo pipefail

WORKSPACE_DIR=/root/torchtitan_attention_residual
PHASE4_CKPT="${WORKSPACE_DIR}/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"

ARM1A_DIR="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/arm1align_fsdp_gbs12_seed42"
ARM2_DIR="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/arm2_pp4v2_adapter_gbs12_seed42"

STEPS="${STEPS:-2000}"
SEED="${SEED:-42}"
GLOBAL_BS=12        # documented Arm 2 config — proven to work for PP=4 V=2

LOG=/root/phase6_alignment_orchestrator.log
exec >>"$LOG" 2>&1

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 v2 alignment orchestrator START"
echo "  STEPS=$STEPS SEED=$SEED GLOBAL_BS=$GLOBAL_BS"
echo "==============================================================="

# Stage 0: GPU sanity
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
    echo "[$(date)] WARNING: phase5_vlm_multimodal_sft.train_mm still running, killing"
    pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 30
    pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 10
fi

# === Stage 1: Arm 1'-align (FSDP=4, GBS=12) ===
echo "[$(date)] [stage 1] launching Arm 1'-align (FSDP=4, GLOBAL_BS=12, LOCAL_BS=3, seed=$SEED)"
mkdir -p "$ARM1A_DIR"
cd "$WORKSPACE_DIR"
source /venv/main/bin/activate

PYTHONPATH="$WORKSPACE_DIR:$WORKSPACE_DIR/torchtitan${PYTHONPATH:+:$PYTHONPATH}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5_vlm_multimodal_sft.train_mm \
    --mm.json /root/hf_cache/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --mm.images /root/hf_cache/LLaVA-Pretrain \
    --mm.vision-model google/siglip-base-patch16-224 \
    --mm.tokenizer NousResearch/Meta-Llama-3.1-8B \
    --mm.cache-dir /root/hf_cache \
    --mm.proj-lr-mult 50.0 \
    --mm.global-seq-len 258 \
    --module attention_residual --config kimi_linear_436m_block_attn_res_n4 \
    --hf_assets_path "$WORKSPACE_DIR/torchtitan/assets/hf/Llama-3.1-8B" \
    --training.steps "$STEPS" \
    --training.local_batch_size 3 \
    --training.global_batch_size 12 \
    --training.seq_len 260 \
    --optimizer.lr 1e-5 \
    --lr_scheduler.warmup_steps 10 \
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
    --checkpoint.keep_latest_k 2 \
    --debug.seed "$SEED" \
    --debug.deterministic \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 1 \
    --dump_folder "$ARM1A_DIR" \
    --compile.enable \
    >"$ARM1A_DIR/train.log" 2>&1

echo "[$(date)] [stage 1] Arm 1'-align done"
sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# === Stage 2: Arm 2 (PP=4 V=2 + cache adapter, GBS=12) ===
echo "[$(date)] [stage 2] launching Arm 2 PP+adapter (GBS=12, LOCAL_BS=1, seed=$SEED)"
mkdir -p "$ARM2_DIR"
cd "$WORKSPACE_DIR"
INIT=weak_ckpt INIT_CKPT="$PHASE4_CKPT" \
NGPU=4 PP=4 V=2 LOCAL_BS=1 GLOBAL_BS=12 STEPS="$STEPS" \
ADAPTER=1 SEED="$SEED" DETERMINISTIC=1 \
COMPILE=1 \
OUT_DIR="$ARM2_DIR" \
bash phase5_vlm_multimodal_sft/launch_pp_adapter.sh

echo "[$(date)] [stage 2] Arm 2 done"

# === Stage 3: alignment report ===
echo "[$(date)] [stage 3] running compare_pp_vs_fsdp.py"
cd "$WORKSPACE_DIR"
python phase5_vlm_multimodal_sft/compare_pp_vs_fsdp.py \
    --pp "$ARM2_DIR/tb" \
    --fsdp "$ARM1A_DIR/tb" \
    > "$WORKSPACE_DIR/phase6_upstream_pr_prep/alignment_report_arm2_real_mm.txt" 2>&1 || true

echo "[$(date)] [stage 3] alignment report:"
cat "$WORKSPACE_DIR/phase6_upstream_pr_prep/alignment_report_arm2_real_mm.txt"

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 v2 alignment orchestrator COMPLETE"
echo "==============================================================="
