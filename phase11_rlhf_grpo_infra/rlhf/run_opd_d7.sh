#!/usr/bin/env bash
# D-7: D-6 extended to 300 steps + eval cascade — the "does encoder match
# actually transfer capability" test.
#
# Background (REPORT_OPD_D5.md + D-6 result):
#   * D-4 (CLIP-336 teacher, 50 step caption):  9.3% GQA
#   * D-6 (Mantis SigLIP-384 teacher, 50 step caption): 10.7% GQA (+1.4pp)
#     → matched-encoder family helps but 50 steps not enough to cross
#       baseline 12.3%
#   * D-5 (CLIP teacher, 600 step VQA): 0.3-2.3% (BROKE the model)
#     → CLIP+long+small_student is unstable; we don't know if Mantis+long
#       is stable
#
# D-7 question: does Mantis-teacher OPD steadily improve as we scale steps,
# or does it also crash like D-5? If it crosses 12.3% at 200-300 steps,
# we have proper capability-lift evidence for PR14. If it plateaus or
# crashes, we know cross-VLM-encoder match is necessary but not sufficient
# for our 447M-vs-8B capacity gap.
set -uo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

D7_DIR="phase11_rlhf_grpo_infra/rlhf/outputs/opd_d7"
mkdir -p "$D7_DIR"

echo "==========================================================="
echo "D-7 START @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================================="
echo "[preflight] disk=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)G"

# ---- background disk watchdog ----
(
    while true; do
        sleep 90
        F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
        if (( F < 12 )); then
            echo "[watchdog] PANIC: disk ${F}G; killing OPD procs"
            pkill -9 -f run_grpo_llava_kimi.py 2>/dev/null
            pkill -9 -f dcp_to_hf_kimi_attn_res_vl.py 2>/dev/null
            pkill -9 -f gqa_eval.py 2>/dev/null
            touch "$D7_DIR/DISK_PANIC"
            exit 1
        fi
    done
) &
WD=$!
echo "[watchdog] PID=$WD"

cleanup() { kill -9 "$WD" 2>/dev/null; }

# ---- Phase 1: 300-step OPD ----
RUN_LOG="$D7_DIR/run_300step.log"
echo "[$(date '+%H:%M:%S')] Phase 1: 300-step OPD → $RUN_LOG"
echo "       teacher=Mantis-SigLIP-Llama3  task=caption  lr=1e-5  T=2.0  β=0.5  NEPS=2  ckpt every 50"
echo "       ETA ~4h"

export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export HF_HOME=/workspace/.hf_home
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DCP="${PWD}/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200"
HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
CKPT_DIR="${PWD}/phase11_rlhf_grpo_infra/rlhf/outputs/opd_d7_ckpts"
TEACHER_ID="TIGER-Lab/Mantis-8B-siglip-llama3"

/usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$DCP" \
    --hf-model-path "$HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --task opd \
    --teacher-model-id "$TEACHER_ID" \
    --tokenizer-path "$HF" \
    --num-steps 300 \
    --num-episodes-per-step 2 \
    --opd-beta 0.5 \
    --opd-temperature 2.0 \
    --opd-lr 1e-5 \
    --opd-weight-decay 0.01 \
    --opd-task-type caption \
    --opd-ckpt-interval 50 \
    --opd-ckpt-dir "$CKPT_DIR" \
    --kl-coef 0.0 \
    > "$RUN_LOG" 2>&1
RUN_EXIT=$?
echo "[$(date '+%H:%M:%S')] Phase 1: run exit=$RUN_EXIT"
ls -la "$CKPT_DIR" 2>&1 | head -10

if [[ ! -d "$CKPT_DIR" ]] || [[ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]]; then
    echo "FATAL: no ckpts; aborting eval"
    cleanup
    exit 1
fi

# ---- Phase 2: eval cascade with auto-cleanup ----
echo "[$(date '+%H:%M:%S')] Phase 2: eval cascade"
SUMMARY="$D7_DIR/eval_summary.md"
{
    echo "# D-7 OPD Eval Cascade (Mantis SigLIP teacher, caption task, 300 steps)"
    echo
    echo "Baseline (SFT step-5200): **12.3%** (37/300)"
    echo "Teacher LLaVA-NeXT-8B upper:  63.7%"
    echo "D-4 (CLIP teacher 50 step): 9.3%"
    echo "D-6 (Mantis 50 step):       10.67%"
    echo
    echo "| Ckpt | GQA acc | Notes |"
    echo "|---|---|---|"
} > "$SUMMARY"

SRC_HF="phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
for STEP_DIR in $(ls -d "$CKPT_DIR"/step-* 2>/dev/null | sort -V); do
    STEP=$(basename "$STEP_DIR" | sed 's/step-//')
    HF_OUT="phase11_rlhf_grpo_infra/hf/opd_d7_step${STEP}"
    EVAL_LOG="$D7_DIR/eval_step${STEP}.log"
    echo "[$(date '+%H:%M:%S')]   step-$STEP: convert + eval"

    torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
        --in "$STEP_DIR" --out "$HF_OUT" \
        --config kimi_linear_447m_aligned_block_attn_res_n4 \
        --vision-tower google/siglip-base-patch16-224 \
        --processor-source "$SRC_HF" --projector-from-hf "$SRC_HF" \
        > "$EVAL_LOG" 2>&1
    CONV=$?
    if (( CONV != 0 )); then
        echo "[$(date '+%H:%M:%S')]     CONVERT FAILED (exit $CONV)"
        echo "| step-$STEP | CONVERT_FAILED | exit=$CONV |" >> "$SUMMARY"
        continue
    fi

    HF_MODEL_PATH="$(pwd)/$HF_OUT" N_EVAL=300 \
        ATTNRES_MLA_FP32_FALLBACK=1 SGLANG_DISABLE_SHM_MM=1 \
        SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts" \
        /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/gqa_eval.py \
        >> "$EVAL_LOG" 2>&1

    ACC=$(grep "GQA testdev greedy accuracy:" "$EVAL_LOG" | tail -1 \
            | sed -n 's/.*= \(.*\) =====/\1/p')
    if [[ -n "$ACC" ]]; then
        echo "[$(date '+%H:%M:%S')]     step-$STEP: $ACC"
        echo "| step-$STEP | $ACC | eval_step${STEP}.log |" >> "$SUMMARY"
    else
        echo "[$(date '+%H:%M:%S')]     step-$STEP: eval FAILED"
        echo "| step-$STEP | EVAL_FAILED | eval_step${STEP}.log |" >> "$SUMMARY"
    fi

    # Cleanup DCP after eval done — only HF dirs retained
    rm -rf "$STEP_DIR"
    F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
    echo "[$(date '+%H:%M:%S')]     disk after cleanup: ${F}G"
done

echo "[$(date '+%H:%M:%S')] Phase 2 DONE. Summary:"
cat "$SUMMARY"

cleanup
echo "[$(date '+%H:%M:%S')] D-7 DONE"
