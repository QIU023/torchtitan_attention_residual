#!/usr/bin/env bash
# Master orchestrator for the 18-hour GPU window.
#
# Sequence:
#   stage 0: wait for the in-flight A6 partial to finish.
#   stage 1: A4 async DCP smoke (50 steps with --checkpoint.async_mode=async).
#   stage 2: A5 mid-save resume smoke (50 steps, SIGTERM at step 30, restart).
#   stage 3: real multimodal pretrain at high BS, init from arm1prime
#            step-4000 (loss 3.03), LOCAL_BS=16 GBS=64, target ~10K steps
#            in ~14h on 4×5090.
#
# Each stage waits for the prior to release GPUs before launching.

set -euo pipefail

WORKSPACE_DIR=/root/torchtitan_attention_residual
PHASE4_CKPT="${WORKSPACE_DIR}/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"
ARM1P_CKPT="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/arm1prime_fsdp_seed42_from_p4_8k/checkpoint/step-4000"

LOG=/root/phase6_closure_orchestrator.log
exec >>"$LOG" 2>&1

echo ""
echo "==============================================================="
echo "[$(date)] phase 6 closure + pretrain orchestrator START"
echo "==============================================================="

# ------------------------------------------------------------------
# Helper: wait for GPUs to be free.
# ------------------------------------------------------------------
wait_gpus_free() {
    local timeout_sec="${1:-300}"
    local elapsed=0
    while [ $elapsed -lt $timeout_sec ]; do
        if ! pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
            sleep 10
            local mem
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
            if [ "$mem" -lt 200 ]; then
                echo "[$(date)] GPUs free (mem=${mem}MiB)"
                return 0
            fi
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "[$(date)] WARNING: GPU wait timed out at ${timeout_sec}s; proceeding anyway"
    return 1
}

# ------------------------------------------------------------------
# Stage 0: wait for A6 partial.
# ------------------------------------------------------------------
echo "[$(date)] [stage 0] waiting for A6 partial to finish..."
A6_LOG="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/a6_fsdp2_pp2_gbs12_seed42/train.log"
while true; do
    if grep -qE "Training completed|FAILED" "$A6_LOG" 2>/dev/null; then
        echo "[$(date)] [stage 0] A6 partial finished"
        break
    fi
    sleep 30
done
wait_gpus_free 300 || true

# ------------------------------------------------------------------
# Stage 1: A4 async DCP smoke.
# ------------------------------------------------------------------
A4_DIR="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/a4_async_dcp_smoke"
echo "[$(date)] [stage 1] A4 async DCP smoke launching"
mkdir -p "$A4_DIR"
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
    --training.steps 50 \
    --training.local_batch_size 3 \
    --training.global_batch_size 12 \
    --training.seq_len 260 \
    --optimizer.lr 1e-5 \
    --lr_scheduler.warmup_steps 5 \
    --lr_scheduler.total_steps 50 \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --checkpoint.enable \
    --checkpoint.async_mode async \
    --checkpoint.initial_load_path "$PHASE4_CKPT" \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval 25 \
    --checkpoint.keep_latest_k 2 \
    --debug.seed 42 \
    --debug.deterministic \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 5 \
    --dump_folder "$A4_DIR" \
    --compile.enable \
    >"$A4_DIR/train.log" 2>&1 || echo "[$(date)] A4 returned non-zero"

echo "[$(date)] [stage 1] A4 done"
wait_gpus_free 300 || true

# ------------------------------------------------------------------
# Stage 2: A5 mid-save resume smoke.
# ------------------------------------------------------------------
A5_DIR="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/a5_mid_save_resume_smoke"
echo "[$(date)] [stage 2] A5 mid-save resume smoke launching"
mkdir -p "$A5_DIR"

# Phase 2a: run to step 30, SIGTERM during/after the step-25 save trigger.
echo "[$(date)] [stage 2a] start phase2a (run + SIGTERM during save window)"
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
    --training.steps 100 \
    --training.local_batch_size 3 \
    --training.global_batch_size 12 \
    --training.seq_len 260 \
    --optimizer.lr 1e-5 \
    --lr_scheduler.warmup_steps 5 \
    --lr_scheduler.total_steps 100 \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --checkpoint.enable \
    --checkpoint.async_mode async \
    --checkpoint.initial_load_path "$PHASE4_CKPT" \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval 25 \
    --checkpoint.keep_latest_k 3 \
    --debug.seed 42 \
    --debug.deterministic \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 5 \
    --dump_folder "$A5_DIR" \
    --compile.enable \
    >"$A5_DIR/train_phase2a.log" 2>&1 &
A5_PID=$!

# Wait for step-25 save to start, then SIGTERM.
while ! grep -q "step:.* (25\|26\|27\|28\|29\|30)" "$A5_DIR/train_phase2a.log" 2>/dev/null; do
    sleep 5
    if ! kill -0 "$A5_PID" 2>/dev/null; then
        echo "[$(date)] [stage 2a] worker died early"
        break
    fi
done
echo "[$(date)] [stage 2a] sending SIGTERM to torchrun pid=$A5_PID"
kill -TERM "$A5_PID" 2>/dev/null || true
wait "$A5_PID" 2>/dev/null || true
sleep 10
pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
sleep 10

# Phase 2b: relaunch from the auto-saved ckpt.
echo "[$(date)] [stage 2b] start phase2b (resume from saved ckpt)"
ls "$A5_DIR/checkpoint/" || true

# torchtitan will auto-resume from the latest ckpt in dump_folder/checkpoint
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
    --training.steps 50 \
    --training.local_batch_size 3 \
    --training.global_batch_size 12 \
    --training.seq_len 260 \
    --optimizer.lr 1e-5 \
    --lr_scheduler.warmup_steps 5 \
    --lr_scheduler.total_steps 100 \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --checkpoint.enable \
    --checkpoint.async_mode async \
    --checkpoint.interval 999999 \
    --checkpoint.keep_latest_k 3 \
    --debug.seed 42 \
    --debug.deterministic \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 5 \
    --dump_folder "$A5_DIR" \
    --compile.enable \
    >"$A5_DIR/train_phase2b.log" 2>&1 || echo "[$(date)] A5 phase2b returned non-zero"

echo "[$(date)] [stage 2] A5 done"
wait_gpus_free 300 || true

# ------------------------------------------------------------------
# Stage 3: real multimodal pretrain (overnight). Continue from
# arm1prime step-4000 (loss 3.03 caption-quality init point).
# LOCAL_BS=16 GBS=64 — 2× the prior caption-story config.
# ------------------------------------------------------------------
PT_DIR="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/overnight_mm_pretrain_bs64_seed42"
echo "[$(date)] [stage 3] overnight pretrain launching"
mkdir -p "$PT_DIR"

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
    --training.local_batch_size 16 \
    --training.global_batch_size 64 \
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
    --checkpoint.initial_load_path "$ARM1P_CKPT" \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval 1000 \
    --checkpoint.keep_latest_k 3 \
    --debug.seed 42 \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 50 \
    --dump_folder "$PT_DIR" \
    --compile.enable \
    >"$PT_DIR/train.log" 2>&1 || echo "[$(date)] pretrain returned non-zero"

echo ""
echo "==============================================================="
echo "[$(date)] phase 6 closure + pretrain orchestrator COMPLETE"
echo "==============================================================="
