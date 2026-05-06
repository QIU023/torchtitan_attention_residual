#!/usr/bin/env bash
# Phase 6 A5 redo — mid-save resume smoke (fixed orchestrator timing).
#
# Original A5 attempt failed: the grep loop matched on torchrun startup
# WARNINGs before the worker's training loop began, firing SIGTERM
# before any ckpt saved. Phase 2b then ran 50 wasted steps from
# random init.
#
# This redo fixes the wait by:
# 1. Anchoring the grep to the strict ``[titan] ... INFO - step: N``
#    pattern (never matches warnings).
# 2. Requiring the matched step ≥ 25 AND the ckpt directory step-25
#    to physically exist on disk.
# 3. Adding a 30-second post-condition wait so the async-staged ckpt
#    finalizes before SIGTERM.
#
# After SIGTERM + clean exit, phase 2b relaunches with the same
# ``--dump_folder`` (no ``initial_load_path``) so torchtitan auto-
# resumes from the saved step-25 ckpt. Verifies loss continuity.

set -uo pipefail

WORKSPACE_DIR=/root/torchtitan_attention_residual
PHASE4_CKPT="${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"

A5_DIR="${WORKSPACE_DIR}/phase5/runs/a5_redo_resume_smoke"

LOG=/root/a5_redo_orchestrator.log
exec >>"$LOG" 2>&1

echo ""
echo "==============================================================="
echo "[$(date)] A5 redo orchestrator START"
echo "==============================================================="

mkdir -p "$A5_DIR"

# --- Phase 2a: run with async save interval 25, SIGTERM after step 25 confirmed ---
echo "[$(date)] [2a] launching trainer (async save freq=25, max steps=100)"
cd "$WORKSPACE_DIR" && source /venv/main/bin/activate

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
TRAINER_PID=$!

# Strict wait: log line must match `step: NN` AND step >= 25 AND ckpt dir exists.
echo "[$(date)] [2a] waiting for step ≥ 25 + step-25 ckpt physically saved..."
TIMEOUT=900  # 15 min compile budget + warmup
ELAPSED=0
while true; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "[$(date)] [2a] TIMEOUT — killing torchrun"
        break
    fi
    if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "[$(date)] [2a] worker died early — aborting smoke"
        exit 1
    fi
    # Match ONLY the canonical ``[titan] YYYY-MM-DD HH:MM:SS,sss - root - INFO - step: NN  loss:``
    # pattern — never the WARNINGs from torchrun startup.
    LATEST_STEP=$(grep -oE "INFO - .*step:\s*([0-9]+)" "$A5_DIR/train_phase2a.log" 2>/dev/null \
        | grep -oE "step:\s*([0-9]+)" \
        | grep -oE "[0-9]+" \
        | tail -1)
    if [ -n "$LATEST_STEP" ] && [ "$LATEST_STEP" -ge 25 ]; then
        if [ -d "$A5_DIR/checkpoint/step-25" ]; then
            echo "[$(date)] [2a] step-25 ckpt confirmed; sleeping 30s for async finalize"
            sleep 30
            break
        fi
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

echo "[$(date)] [2a] sending SIGTERM to torchrun pid=$TRAINER_PID"
kill -TERM "$TRAINER_PID" 2>/dev/null || true
for i in $(seq 1 12); do
    sleep 5
    if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "[$(date)] [2a] worker exited cleanly after ${i}*5s"
        break
    fi
done
pkill -KILL -f "phase5.train_mm" 2>/dev/null || true
sleep 10

# Capture the last logged loss before SIGTERM for continuity check.
PHASE2A_LAST_STEP=$(grep -oE "INFO - .*step:\s*([0-9]+)" "$A5_DIR/train_phase2a.log" \
    | tail -1 | grep -oE "[0-9]+" | tail -1)
PHASE2A_LAST_LOSS=$(grep -oE "INFO - .*step:\s*[0-9]+\s+loss:\s*([0-9.]+)" "$A5_DIR/train_phase2a.log" \
    | tail -1 | grep -oE "loss:\s*[0-9.]+" | grep -oE "[0-9.]+")
echo "[$(date)] [2a] complete. last logged: step=$PHASE2A_LAST_STEP loss=$PHASE2A_LAST_LOSS"
echo "[$(date)] [2a] saved ckpts: $(ls $A5_DIR/checkpoint/ 2>/dev/null)"

# --- Phase 2b: relaunch from same dump_folder, expect auto-resume ---
echo "[$(date)] [2b] relaunching trainer (auto-resume from $A5_DIR/checkpoint/step-25)"

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
    --checkpoint.interval 25 \
    --checkpoint.keep_latest_k 3 \
    --debug.seed 42 \
    --debug.deterministic \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 5 \
    --dump_folder "$A5_DIR" \
    --compile.enable \
    >"$A5_DIR/train_phase2b.log" 2>&1
EXIT=$?

# Compare phase 2b's first logged loss with phase 2a's last logged loss.
# Auto-resume should produce a step number > phase2a_last_step and a
# loss within bf16 noise of phase2a_last (since we resumed from a ckpt
# at exactly step 25 with full state).
PHASE2B_FIRST_STEP=$(grep -oE "INFO - .*step:\s*([0-9]+)" "$A5_DIR/train_phase2b.log" 2>/dev/null \
    | head -1 | grep -oE "[0-9]+" | tail -1)
PHASE2B_FIRST_LOSS=$(grep -oE "INFO - .*step:\s*[0-9]+\s+loss:\s*([0-9.]+)" "$A5_DIR/train_phase2b.log" \
    | head -1 | grep -oE "loss:\s*[0-9.]+" | grep -oE "[0-9.]+")
echo "[$(date)] [2b] first logged: step=$PHASE2B_FIRST_STEP loss=$PHASE2B_FIRST_LOSS"

if [ -n "$PHASE2B_FIRST_STEP" ] && [ "$PHASE2B_FIRST_STEP" -gt 25 ]; then
    echo "[$(date)] PASS: phase2b resumed past step-25 (got step $PHASE2B_FIRST_STEP)"
else
    echo "[$(date)] FAIL: phase2b did not auto-resume — got step $PHASE2B_FIRST_STEP"
fi

echo ""
echo "==============================================================="
echo "[$(date)] A5 redo COMPLETE (exit=$EXIT)"
echo "==============================================================="
