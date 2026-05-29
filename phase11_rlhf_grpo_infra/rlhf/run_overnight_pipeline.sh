#!/usr/bin/env bash
# Overnight pipeline: Phase A (already running) → eval → OPD → eval cascade.
#
# Sequencing rationale (additive-before-extractive):
#   Phase A  SFT on full mix665k        ← adds VQA/reasoning knowledge (RUNNING)
#   eval-A   MMBench/POPE/GQA           ← new base capability snapshot
#   OPD      Mantis-SigLIP teacher      ← additive distillation on stronger base
#   eval-OPD cascade across ckpts       ← measure lift over Phase-A baseline
#   (GRPO deferred — extractive polish, needs OPD result + reward design)
#
# Teacher choice: Mantis-8B-siglip-llama3 (SigLIP-so400m-384) — matched
# encoder family with the student's SigLIP-base-224. D-2..D-7 established
# encoder match is necessary (if not sufficient); the open question this
# run answers is whether a STRONGER Phase-A base lets OPD finally lift.
#
# Disk: Phase A ckpt (~17G) + phaseA HF (~3G) + Mantis (~16G) + OPD ckpts
#   (4×6G, deleted after eval). Watchdog kills all at <12G.
set -uo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

PIPE_DIR="phase11_rlhf_grpo_infra/rlhf/outputs/overnight_pipeline"
mkdir -p "$PIPE_DIR"
PHASEA_DIR="phase5_vlm_multimodal_sft/runs/phaseA_mix665k_full"
PHASEA_LOG="$PHASEA_DIR/train.log"
SRC_HF="phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"   # projector donor for OPD

echo "=========================================================="
echo "OVERNIGHT PIPELINE @ $(date '+%Y-%m-%d %H:%M:%S')"
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
            pkill -9 -f run_all_evals 2>/dev/null
            pkill -9 -f score_ 2>/dev/null
            touch "$PIPE_DIR/DISK_PANIC"; exit 1
        fi
    done
) &
WD=$!
cleanup() { kill -9 "$WD" 2>/dev/null; }

# ---- Stage 1: wait for Phase A to finish ----
echo "[$(date '+%H:%M:%S')] Stage 1: waiting for Phase A (step 5200)…"
until grep -qE "step:\s*5200" "$PHASEA_LOG" 2>/dev/null || ! pgrep -f train_mm >/dev/null 2>&1; do
    sleep 120
done
if ! grep -qE "step:\s*5200" "$PHASEA_LOG" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] FATAL: Phase A exited before step 5200"
    tail -10 "$PHASEA_LOG"; cleanup; exit 1
fi
# Resolve final ckpt
PHASEA_CKPT=$(ls -d "$PHASEA_DIR"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "[$(date '+%H:%M:%S')] Stage 1 done. Phase A ckpt = $PHASEA_CKPT"

# ---- Stage 2: eval Phase A base (DCP direct, no HF convert) ----
echo "[$(date '+%H:%M:%S')] Stage 2: eval Phase A base (MMBench+POPE+GQA)"
STAGE2_CKPT="$(pwd)/$PHASEA_CKPT" \
BENCHES="gqa mmbench pope" GQA_LIMIT=500 MMB_LIMIT=500 POPE_LIMIT=500 \
RUN_DIR="$PIPE_DIR/eval_phaseA" \
bash phase5_vlm_multimodal_sft/eval_benchmarks/run_all_evals.sh \
    > "$PIPE_DIR/eval_phaseA.log" 2>&1
echo "[$(date '+%H:%M:%S')] Stage 2 done. Phase A eval:"
grep -A6 "Summary table" "$PIPE_DIR/eval_phaseA/REPORT.md" 2>/dev/null || echo "  (see eval_phaseA.log)"

# ---- Stage 3: convert Phase A → HF (for OPD SGLang generator) ----
echo "[$(date '+%H:%M:%S')] Stage 3: Phase A DCP→HF"
PHASEA_HF="phase11_rlhf_grpo_infra/hf/phaseA_step5200"
torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
    --in "$PHASEA_CKPT" --out "$PHASEA_HF" \
    --config kimi_linear_447m_aligned_block_attn_res_n4 \
    --vision-tower google/siglip-base-patch16-224 \
    --processor-source "$SRC_HF" \
    > "$PIPE_DIR/convert_phaseA.log" 2>&1
if [[ ! -f "$PHASEA_HF/model.safetensors" ]]; then
    echo "[$(date '+%H:%M:%S')] FATAL: Phase A HF convert failed"
    tail -10 "$PIPE_DIR/convert_phaseA.log"; cleanup; exit 1
fi
echo "[$(date '+%H:%M:%S')] Stage 3 done → $PHASEA_HF"

# ---- Stage 4: download Mantis teacher ----
echo "[$(date '+%H:%M:%S')] Stage 4: download Mantis teacher"
/usr/bin/python3 -c "
import os; os.environ['HF_HOME']='/workspace/.hf_home'
from huggingface_hub import snapshot_download
snapshot_download('TIGER-Lab/Mantis-8B-siglip-llama3',
    allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt'])
print('Mantis ready')
" > "$PIPE_DIR/mantis_dl.log" 2>&1
echo "[$(date '+%H:%M:%S')] Stage 4 done"

# ---- Stage 5: OPD from Phase-A base, Mantis teacher, 100 steps ----
echo "[$(date '+%H:%M:%S')] Stage 5: OPD (Mantis teacher, caption, lr=1e-5, T=2.0, 100 step, ckpt 25)"
OPD_CKPT_DIR="$(pwd)/$PIPE_DIR/opd_ckpts"
rm -rf "$OPD_CKPT_DIR" 2>/dev/null

export PYTHONPATH="$(pwd)/torchtitan:$(pwd)"
export HF_HOME=/workspace/.hf_home
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1 TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$(pwd)/$PHASEA_CKPT" \
    --hf-model-path "$(pwd)/$PHASEA_HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --task opd \
    --teacher-model-id "TIGER-Lab/Mantis-8B-siglip-llama3" \
    --tokenizer-path "$(pwd)/$PHASEA_HF" \
    --num-steps 100 --num-episodes-per-step 2 \
    --opd-beta 0.5 --opd-temperature 2.0 \
    --opd-lr 1e-5 --opd-weight-decay 0.01 \
    --opd-task-type caption \
    --opd-ckpt-interval 25 --opd-ckpt-dir "$OPD_CKPT_DIR" \
    --kl-coef 0.0 \
    > "$PIPE_DIR/opd_run.log" 2>&1
echo "[$(date '+%H:%M:%S')] Stage 5 done. OPD ckpts:"
ls -d "$OPD_CKPT_DIR"/step-* 2>/dev/null

# ---- Stage 6: eval cascade on OPD ckpts ----
echo "[$(date '+%H:%M:%S')] Stage 6: OPD eval cascade"
SUMMARY="$PIPE_DIR/FINAL_SUMMARY.md"
{
    echo "# Overnight Pipeline Summary"
    echo
    echo "Baseline (stage2 step-5200): MMBench 36.4 / GQA 12.3 / POPE 50(always-no)"
    echo
    echo "## Phase A (full mix665k SFT)"
    grep -A6 "Summary table" "$PIPE_DIR/eval_phaseA/REPORT.md" 2>/dev/null
    echo
    echo "## OPD on Phase-A base (Mantis teacher)"
    echo "| Ckpt | GQA | MMBench | POPE-f1 |"
    echo "|---|---|---|---|"
} > "$SUMMARY"

for STEP_DIR in $(ls -d "$OPD_CKPT_DIR"/step-* 2>/dev/null | sort -t- -k2 -n); do
    STEP=$(basename "$STEP_DIR" | sed 's/step-//')
    echo "[$(date '+%H:%M:%S')]   OPD eval step-$STEP"
    STAGE2_CKPT="$STEP_DIR" \
    BENCHES="gqa mmbench pope" GQA_LIMIT=500 MMB_LIMIT=500 POPE_LIMIT=500 \
    RUN_DIR="$PIPE_DIR/eval_opd_step${STEP}" \
    bash phase5_vlm_multimodal_sft/eval_benchmarks/run_all_evals.sh \
        > "$PIPE_DIR/eval_opd_step${STEP}.log" 2>&1
    GQA=$(grep -oE "gqa.*\| [0-9.]+ \|" "$PIPE_DIR/eval_opd_step${STEP}/REPORT.md" 2>/dev/null | grep -oE "[0-9.]+" | head -1)
    MMB=$(grep "mmbench" "$PIPE_DIR/eval_opd_step${STEP}/REPORT.md" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "| step-$STEP | ${GQA:-?} | ${MMB:-?} | see log |" >> "$SUMMARY"
    rm -rf "$STEP_DIR"   # free disk after eval
done

echo "[$(date '+%H:%M:%S')] Stage 6 done. Summary:"
cat "$SUMMARY"
cleanup
echo "[$(date '+%H:%M:%S')] OVERNIGHT PIPELINE DONE"
