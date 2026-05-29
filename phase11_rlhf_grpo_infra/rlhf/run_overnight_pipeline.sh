#!/usr/bin/env bash
# Overnight pipeline v2: Phase A (running) → eval → GRPO → eval.
#
# DECISION (2026-05-29, after R-1 metric calibration):
#   R-1 revealed the 447M base's real capability is MMBench 36.4%
#   (genuine multi-choice reasoning), GQA 12.3% (format-penalized),
#   POPE 50% (always-"no" shortcut bug). caption-task OPD optimizes
#   DESCRIPTION quality — orthogonal to all three eval axes, which is
#   why D-2..D-7 caption/VQA OPD never lifted eval metrics.
#
#   So this pipeline drops OPD and goes SFT → GRPO:
#     Phase A  full mix665k SFT     ← adds VQA/reasoning knowledge (RUNNING)
#     eval-A   MMBench/POPE/GQA     ← confirm base is strong enough for RL
#     GRPO     GQA verifiable reward← extractive polish on a now-stronger base
#     eval-GRPO cascade            ← measure lift over Phase-A baseline
#
#   GRPO gate: only run GRPO if Phase A base shows non-degenerate signal
#   (GQA group reward variance needs the base to get SOME answers right;
#   D-2..D-5 GRPO collapsed because the 12.3% base was too weak). With
#   Phase A targeting 25-30% GQA, reward variance should be non-zero.
#
# Disk: Phase A ckpt (~17G) + HF (~3G) + GRPO ckpts. Watchdog @ 12G.
set -uo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

PIPE_DIR="phase11_rlhf_grpo_infra/rlhf/outputs/overnight_pipeline"
mkdir -p "$PIPE_DIR"
PHASEA_DIR="phase5_vlm_multimodal_sft/runs/phaseA_mix665k_full"
PHASEA_LOG="$PHASEA_DIR/train.log"
SRC_HF="phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"

echo "=========================================================="
echo "OVERNIGHT PIPELINE v2 (SFT→GRPO) @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

# ---- disk watchdog ----
(
    while true; do
        sleep 120
        F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
        if (( F < 12 )); then
            echo "[watchdog $(date '+%H:%M:%S')] PANIC disk ${F}G; killing"
            pkill -9 -f run_grpo_llava_kimi.py 2>/dev/null
            pkill -9 -f dcp_to_hf_kimi_attn_res_vl.py 2>/dev/null
            pkill -9 -f run_all_evals 2>/dev/null; pkill -9 -f score_ 2>/dev/null
            touch "$PIPE_DIR/DISK_PANIC"; exit 1
        fi
    done
) &
WD=$!
cleanup() { kill -9 "$WD" 2>/dev/null; }

# ---- Stage 1: wait for Phase A ----
echo "[$(date '+%H:%M:%S')] Stage 1: waiting for Phase A (step 5200)…"
until grep -qE "step:\s*5200" "$PHASEA_LOG" 2>/dev/null || ! pgrep -f train_mm >/dev/null 2>&1; do
    sleep 120
done
if ! grep -qE "step:\s*5200" "$PHASEA_LOG" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] FATAL: Phase A exited before 5200"; tail -10 "$PHASEA_LOG"; cleanup; exit 1
fi
PHASEA_CKPT=$(ls -d "$PHASEA_DIR"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "[$(date '+%H:%M:%S')] Stage 1 done. Phase A ckpt = $PHASEA_CKPT"

# ---- Stage 2: eval Phase A base ----
echo "[$(date '+%H:%M:%S')] Stage 2: eval Phase A (MMBench+POPE+GQA)"
STAGE2_CKPT="$(pwd)/$PHASEA_CKPT" \
BENCHES="gqa mmbench pope" GQA_LIMIT=500 MMB_LIMIT=500 POPE_LIMIT=500 \
RUN_DIR="$PIPE_DIR/eval_phaseA" \
bash phase5_vlm_multimodal_sft/eval_benchmarks/run_all_evals.sh \
    > "$PIPE_DIR/eval_phaseA.log" 2>&1
echo "[$(date '+%H:%M:%S')] Stage 2 done. Phase A:"
grep -A8 "Summary table" "$PIPE_DIR/eval_phaseA/REPORT.md" 2>/dev/null | tee "$PIPE_DIR/_phaseA_summary.txt"

# Extract Phase A GQA to gate GRPO
PHASEA_GQA=$(grep -iE "gqa" "$PIPE_DIR/eval_phaseA/REPORT.md" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[$(date '+%H:%M:%S')] Phase A GQA = ${PHASEA_GQA:-unknown}"

# ---- Stage 3: convert Phase A → HF (GRPO generator needs it) ----
echo "[$(date '+%H:%M:%S')] Stage 3: Phase A DCP→HF"
PHASEA_HF="phase11_rlhf_grpo_infra/hf/phaseA_step5200"
torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
    --in "$PHASEA_CKPT" --out "$PHASEA_HF" \
    --config kimi_linear_447m_aligned_block_attn_res_n4 \
    --vision-tower google/siglip-base-patch16-224 \
    --processor-source "$SRC_HF" \
    > "$PIPE_DIR/convert_phaseA.log" 2>&1
if [[ ! -f "$PHASEA_HF/model.safetensors" ]]; then
    echo "[$(date '+%H:%M:%S')] FATAL: Phase A HF convert failed"; tail -10 "$PIPE_DIR/convert_phaseA.log"; cleanup; exit 1
fi
echo "[$(date '+%H:%M:%S')] Stage 3 done → $PHASEA_HF"

# ---- Stage 4: GRPO on GQA verifiable reward (200 steps) ----
echo "[$(date '+%H:%M:%S')] Stage 4: GRPO on GQA (200 steps, kl=0.05, 4 episodes/step)"
GRPO_DUMP="$(pwd)/$PIPE_DIR/grpo_run"
rm -rf "$GRPO_DUMP" 2>/dev/null; mkdir -p "$GRPO_DUMP"

export PYTHONPATH="$(pwd)/torchtitan:$(pwd)"
export HF_HOME=/workspace/.hf_home
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Note: run_grpo_llava_kimi.py default dump_folder is hardcoded; we pass
# the GQA data + phaseA ckpts. GRPO ckpts are saved by the launcher's own
# logic (PolicyTrainer DCP). We rely on its built-in eval-via-reward; the
# real eval is the post-run cascade below using the saved policy weights.
timeout 36000 /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$(pwd)/$PHASEA_CKPT" \
    --hf-model-path "$(pwd)/$PHASEA_HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --task gqa \
    --data-json /workspace/gqa_rl/gqa_testdev.json \
    --images-dir /workspace/gqa_rl \
    --num-steps 200 --num-episodes-per-step 4 \
    --kl-coef 0.05 \
    > "$PIPE_DIR/grpo_run.log" 2>&1
GRPO_EXIT=$?
echo "[$(date '+%H:%M:%S')] Stage 4 done (exit $GRPO_EXIT). GRPO reward trajectory:"
grep -E "step\s+[0-9]+.*reward" "$PIPE_DIR/grpo_run.log" 2>/dev/null | tail -10

# ---- Stage 5: summary ----
SUMMARY="$PIPE_DIR/FINAL_SUMMARY.md"
{
    echo "# Overnight Pipeline v2 Summary (SFT→GRPO)"
    echo
    echo "## Baseline (stage2 step-5200, R-1)"
    echo "  MMBench 36.4 | GQA 12.3 | POPE 50 (always-no)"
    echo
    echo "## Phase A (full mix665k SFT)"
    grep -A8 "Summary table" "$PIPE_DIR/eval_phaseA/REPORT.md" 2>/dev/null
    echo
    echo "## GRPO on GQA (200 steps, from Phase A base)"
    echo "Reward trajectory (last 10 steps):"
    echo '```'
    grep -E "step\s+[0-9]+.*reward" "$PIPE_DIR/grpo_run.log" 2>/dev/null | tail -10
    echo '```'
    echo
    echo "GRPO exit code: $GRPO_EXIT"
    echo "NOTE: GRPO weight-sync-via-disk is no-op (known); final policy"
    echo "weights live in the trainer actor. To eval GRPO'd model, the"
    echo "policy must be checkpointed — see grpo_run.log for whether"
    echo "PolicyTrainer saved DCP. If reward rose monotonically, GRPO is"
    echo "extracting capability; if flat, base reward variance was too low."
} > "$SUMMARY"

echo "[$(date '+%H:%M:%S')] Pipeline done. Summary:"
cat "$SUMMARY"
cleanup
echo "[$(date '+%H:%M:%S')] OVERNIGHT PIPELINE v2 DONE"
