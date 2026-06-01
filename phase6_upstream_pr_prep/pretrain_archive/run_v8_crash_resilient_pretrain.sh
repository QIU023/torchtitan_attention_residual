#!/usr/bin/env bash
# Crash-resilient continued multimodal pretrain (commit 57a4b47 + later).
#
# The fla-core KDA Triton "device-side assert triggered" hits roughly
# every 1500-2700 steps regardless of seed/BS. Earlier runs (v1-v7)
# manually relaunched after each crash with model_only init, paying a
# ~50-100 step projector re-alignment cost each time.
#
# After commit 57a4b47, the multimodal trainer registers
# ``mm_projector`` (projector + AdamW state) with the checkpointer's
# ``self.states``. So a same-dump_folder auto-resume now restores
# projector + optimizer + LR scheduler + dataloader state — full
# DCP resume.
#
# This orchestrator is the autonomous loop:
#   - First iter: launch with --checkpoint.initial_load_path pointing
#     at the best prior ckpt (v7/step-800, loss 2.79) + model_only.
#     Trainer saves its first ckpt at step 200, including mm_projector.
#   - Subsequent iters: detect at least one ckpt in dump_folder, drop
#     initial_load_path, let torchtitan auto-resume from the latest
#     in-dir ckpt. Full state preserved.
#   - Stop when train.log shows "Training completed" or budget exhaust.
#
# Designed to run for ~10-12 hours. Safe to interrupt with SIGTERM
# at the orchestrator level.

set -uo pipefail

DUMP_FOLDER=/root/torchtitan_attention_residual/phase5_vlm_multimodal_sft/runs/v8_pretrain_resilient_from_v7_step800
INIT_CKPT=/root/torchtitan_attention_residual/phase5_vlm_multimodal_sft/runs/v7_pretrain_bs120_from_v6_step1200_BEST/checkpoint/step-800
WORKSPACE_DIR=/root/torchtitan_attention_residual

mkdir -p "$DUMP_FOLDER"
LOG=/root/v8_pretrain_orchestrator.log
exec >>"$LOG" 2>&1

echo ""
echo "==============================================================="
echo "[$(date)] v8 crash-resilient pretrain orchestrator START"
echo "  init ckpt: $INIT_CKPT"
echo "  dump folder: $DUMP_FOLDER"
echo "==============================================================="

ITER=0
MAX_ITER=20  # safety cap on relaunch loops

while [ $ITER -lt $MAX_ITER ]; do
    ITER=$((ITER + 1))
    LATEST_CKPT_COUNT=0
    if [ -d "$DUMP_FOLDER/checkpoint" ]; then
        LATEST_CKPT_COUNT=$(ls -d "$DUMP_FOLDER/checkpoint"/step-* 2>/dev/null | wc -l)
    fi

    INITIAL_ARGS=""
    if [ "$LATEST_CKPT_COUNT" -eq 0 ]; then
        echo "[$(date)] [iter $ITER] no in-dir ckpt yet — initial-load model-only from $INIT_CKPT"
        INITIAL_ARGS="--checkpoint.initial_load_path $INIT_CKPT --checkpoint.initial_load_model_only"
    else
        echo "[$(date)] [iter $ITER] $LATEST_CKPT_COUNT in-dir ckpt(s) found — auto-resume full state"
    fi

    # Each torchrun is a foreground call; orchestrator sleeps until it
    # exits (clean or crash).
    cd "$WORKSPACE_DIR" && source /venv/main/bin/activate
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
        --training.steps 10000 \
        --training.local_batch_size 30 \
        --training.global_batch_size 120 \
        --training.seq_len 260 \
        --optimizer.lr 1e-5 \
        --lr_scheduler.warmup_steps 100 \
        --lr_scheduler.total_steps 10000 \
        --lr_scheduler.decay_ratio 0.0 \
        --parallelism.pipeline_parallel_degree 1 \
        --parallelism.data_parallel_shard_degree 4 \
        --parallelism.data_parallel_replicate_degree 1 \
        --parallelism.tensor_parallel_degree 1 \
        --checkpoint.enable \
        --checkpoint.async_mode async \
        $INITIAL_ARGS \
        --checkpoint.interval 200 \
        --checkpoint.keep_latest_k 2 \
        --debug.seed $((50 + ITER)) \
        --metrics.save_tb_folder tb \
        --metrics.log_freq 50 \
        --dump_folder "$DUMP_FOLDER" \
        --compile.enable \
        >>"$DUMP_FOLDER/train.log" 2>&1
    EXIT=$?

    echo "[$(date)] [iter $ITER] torchrun exit code $EXIT"

    # Did training complete cleanly?
    if grep -q "Training completed" "$DUMP_FOLDER/train.log" 2>/dev/null; then
        TAIL_COMPLETE=$(grep -c "Training completed" "$DUMP_FOLDER/train.log")
        if [ "$TAIL_COMPLETE" -gt 0 ]; then
            echo "[$(date)] [iter $ITER] Training completed found in log — exiting"
            break
        fi
    fi

    # If exit was zero, training finished without "completed" — odd, but stop.
    if [ "$EXIT" -eq 0 ]; then
        echo "[$(date)] [iter $ITER] exit 0 without 'Training completed' — assuming done"
        break
    fi

    echo "[$(date)] [iter $ITER] worker died with exit $EXIT — sleeping 30s before relaunch"
    pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 30
done

echo ""
echo "==============================================================="
echo "[$(date)] v8 crash-resilient pretrain orchestrator COMPLETE (iter=$ITER)"
echo "==============================================================="
