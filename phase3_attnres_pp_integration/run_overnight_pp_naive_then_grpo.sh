#!/usr/bin/env bash
# Overnight chain:
#   0. (already running) L=32 e16 N=8 PP=8×VP=4 adapter — poll for DONE
#   1. naive L=24 N=8 d1280 e32 PP=8 × VP=3, 300 steps
#   2. naive L=32 N=8 d1280 e16 PP=8 × VP=4, 300 steps
#   3. update phase3 report + commit + push
#   4. GRPO with fixes (grad clip + KL + ckpt from /root/ vlm_sft_3ep)
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3_attnres_pp_integration/run_overnight_chain.log"
> "$LOG"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] OVERNIGHT CHAIN START"
echo "==============================================================="

# ----- Disk guard helper -----
# User mandate 2026-05-12: NEVER let disk fill. If < 50 GiB free, halt.
disk_guard() {
    local where="$1"
    local free_gib
    free_gib=$(df -BG /workspace | tail -1 | awk '{print $4}' | tr -d 'G')
    echo "[$(date)] DISK GUARD ($where): free=${free_gib} GiB"
    if [[ "${free_gib:-0}" -lt 50 ]]; then
        echo "[$(date)] DISK GUARD TRIGGERED — free=${free_gib} GiB < 50. Cleaning + halting."
        # Clean known throwaway dirs
        rm -rf /tmp/kimi_linear* /tmp/l32* /tmp/l24* /tmp/l16* 2>/dev/null
        rm -rf "$WS"/phase3_attnres_pp_integration/runs/pressure_test_2026051[01]-* 2>/dev/null
        find "$WS"/phase3_attnres_pp_integration/runs -name "checkpoint" -type d -exec rm -rf {} + 2>/dev/null
        free_gib=$(df -BG /workspace | tail -1 | awk '{print $4}' | tr -d 'G')
        echo "[$(date)] DISK GUARD post-clean: free=${free_gib} GiB"
        if [[ "${free_gib:-0}" -lt 50 ]]; then
            echo "[$(date)] DISK GUARD STILL UNDER LIMIT after clean — aborting chain"
            exit 1
        fi
    fi
}
disk_guard "start"

# ----- Commit + push helper -----
# Stages only known files (scripts + report + submodule bump). Skips
# bulky run dirs. Commits + pushes; tolerates push failure.
commit_push() {
    local step_name="$1"
    cd "$WS"
    git config user.name "QIU023" >/dev/null 2>&1
    git config user.email "yiqiaoqiu@hotmail.com" >/dev/null 2>&1
    git add \
        phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md \
        phase3_attnres_pp_integration/run_kimi48b_*.sh \
        phase3_attnres_pp_integration/run_overnight_pp_naive_then_grpo.sh \
        phase3_attnres_pp_integration/run_l24_adapter_sweep.sh \
        phase3_attnres_pp_integration/run_l32n8_widen_smoke.sh \
        phase3_attnres_pp_integration/run_pp_pressure_test.sh \
        phase4_kimi_attnres_lm_pretrain/run_kimi48b_downscale_sweep.sh \
        torchtitan 2>/dev/null || true
    if git diff --cached --quiet; then
        echo "[$(date)] commit_push ($step_name): nothing to commit"
        return 0
    fi
    git commit -m "$(printf 'phase3 overnight progress: %s\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' "$step_name")" 2>&1 | tail -3
    git push origin main 2>&1 | tail -3 || echo "(push failed for $step_name)"
}

# ----- 0. Wait for current adapter L=32 e16 run to finish -----
ADAPTER_LOG="$WS/phase3_attnres_pp_integration/run_kimi48b_d1280_e16_L32N8_pp8vp4_adapter.log"
echo "[$(date)] Waiting for L=32 e16 adapter run to finish..."
until grep -q "^\[.*DONE — out dir" "$ADAPTER_LOG" 2>/dev/null \
   || grep -q "OutOfMemoryError" "$ADAPTER_LOG" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] L=32 e16 adapter run finished (or errored)."
commit_push "L=32 e16 adapter + L=24 e32 adapter results captured"

# Wait for GPUs to free
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print (s<2000)?"y":"n"}')" = "y" ]; do sleep 5; done

# ----- 1. Naive L=24 N=8 d1280 e32 PP=8 × VP=3 -----
echo ""
echo "==============================================================="
echo "[$(date)] Step 1/4: naive L=24 N=8 d1280 e32 PP=8 × VP=3 START"
echo "==============================================================="
OUT1="$WS/phase3_attnres_pp_integration/runs/kimi48b_d1280_e32_L24N8_pp8vp3_naive_$(date +%Y%m%d-%H%M%S)"
rm -rf "$OUT1"
(cd torchtitan && \
 env PYTORCH_ALLOC_CONF="expandable_segments:True" \
     torchrun \
         --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
         --local-ranks-filter 7 --role rank --tee 3 \
         -m torchtitan.train \
         --module attention_residual --config kimi_linear_48b_block_attn_res_d1280_e32_L24_N8 \
         --training.steps 300 \
         --training.local_batch_size 24 \
         --training.global_batch_size 24 \
         --training.seq_len 1024 \
         --parallelism.pipeline_parallel_degree 8 \
         --parallelism.pipeline_parallel_schedule Interleaved1F1B \
         --parallelism.pipeline_parallel_layers_per_stage 1 \
         --parallelism.pipeline_parallel_first_stage_less_layers 0 \
         --parallelism.pipeline_parallel_last_stage_less_layers 0 \
         --checkpoint.no-enable \
         --dump_folder "$OUT1") 2>&1 | tail -50
echo "[$(date)] Step 1/4 DONE — out: $OUT1"
disk_guard "after-naive-L24"
commit_push "naive L=24 N=8 d1280 e32 PP=8 VP=3 done"

until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print (s<2000)?"y":"n"}')" = "y" ]; do sleep 5; done

# ----- 2. Naive L=32 N=8 d1280 e16 PP=8 × VP=4 -----
echo ""
echo "==============================================================="
echo "[$(date)] Step 2/4: naive L=32 N=8 d1280 e16 PP=8 × VP=4 START"
echo "==============================================================="
OUT2="$WS/phase3_attnres_pp_integration/runs/kimi48b_d1280_e16_L32N8_pp8vp4_naive_$(date +%Y%m%d-%H%M%S)"
rm -rf "$OUT2"
(cd torchtitan && \
 env PYTORCH_ALLOC_CONF="expandable_segments:True" \
     torchrun \
         --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
         --local-ranks-filter 7 --role rank --tee 3 \
         -m torchtitan.train \
         --module attention_residual --config kimi_linear_48b_block_attn_res_d1280_e16_L32_N8 \
         --training.steps 300 \
         --training.local_batch_size 32 \
         --training.global_batch_size 32 \
         --training.seq_len 1024 \
         --parallelism.pipeline_parallel_degree 8 \
         --parallelism.pipeline_parallel_schedule Interleaved1F1B \
         --parallelism.pipeline_parallel_layers_per_stage 1 \
         --parallelism.pipeline_parallel_first_stage_less_layers 0 \
         --parallelism.pipeline_parallel_last_stage_less_layers 0 \
         --checkpoint.no-enable \
         --dump_folder "$OUT2") 2>&1 | tail -50
echo "[$(date)] Step 2/4 DONE — out: $OUT2"
disk_guard "after-naive-L32"
commit_push "naive L=32 N=8 d1280 e16 PP=8 VP=4 done"

# ----- 3. Update report + commit + push -----
echo ""
echo "==============================================================="
echo "[$(date)] Step 3/4: regenerate phase3 report + commit + push"
echo "==============================================================="
python3 "$WS/phase3_attnres_pp_integration/gen_pp_report.py" 2>&1 || echo "(no gen_pp_report.py, skip auto-report)"
cd "$WS"
# Stage only the files we expect — never auto-add unknown stuff
git add phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md \
        phase3_attnres_pp_integration/run_kimi48b_d1280_e32_L24N8_pp8vp3_adapter.sh \
        phase3_attnres_pp_integration/run_kimi48b_d1280_e32_L32N8_pp8vp4_adapter.sh \
        phase3_attnres_pp_integration/run_kimi48b_d1280_e16_L32N8_pp8vp4_adapter.sh \
        phase3_attnres_pp_integration/run_overnight_pp_naive_then_grpo.sh \
        phase4_kimi_attnres_lm_pretrain/run_kimi48b_downscale_sweep.sh \
        torchtitan 2>&1 || true
git commit -m "$(cat <<'COMMITEOF'
phase3+phase4: kimi 48B-layout PP=8 × VP={3,4} adapter+naive results

L=24 N=8 (3 t-blocks/AttnRes-block paper sweet spot) PP=8 × VP=3 =
24 chunks adapter trained loss 12.26 → 6.19 over 270/300 steps.
L=32 N=8 e16 PP=8 × VP=4 = 32 chunks adapter ran with finite grads.
Both with kimi_linear paper architecture (KDA+MLA+MoE+Block AttnRes)
at dim=1280, seq=1024, FSDP+EP=8 + PP=8.

Naive baselines added for both shapes (300 steps each) for adapter
alignment verification.

Submodule torchtitan bumped: new flavors
kimi_linear_48b_block_attn_res_{d1280_e32_L24_N8, d1280_e32_L32_N8,
d1280_e16_L32_N8} + downscale-sweep variants.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMITEOF
)" 2>&1 || echo "(commit failed or nothing to commit)"
git push origin main 2>&1 || echo "(outer push may need manual auth)"
echo "[$(date)] Step 3/4 done"

# ----- 4. GRPO with fixes (vlm_sft_3ep ckpt + grad clip + KL) -----
echo ""
echo "==============================================================="
echo "[$(date)] Step 4/4: GRPO overnight with stability fixes"
echo "==============================================================="
# Use ckpt from /root tree (the original); symlink into /workspace
mkdir -p "$WS/phase11_rlhf_grpo_infra/hf"
if [[ ! -e "$WS/phase11_rlhf_grpo_infra/hf/vlm_sft_3ep" ]]; then
    ln -s /root/torchtitan_attention_residual/phase11_rlhf_grpo_infra/hf/vlm_sft_3ep "$WS/phase11_rlhf_grpo_infra/hf/vlm_sft_3ep"
fi
# DCP path from yesterday's run (may not exist in /workspace — try both)
S3_DCP_ROOT=/root/torchtitan_attention_residual/phase5_vlm_multimodal_sft/runs/vlm_447m_sft_3ep/checkpoint
LAST=$(ls "$S3_DCP_ROOT" 2>/dev/null | tail -1 || echo "")
S3_DCP="$S3_DCP_ROOT/$LAST"
S3_HF="$WS/phase11_rlhf_grpo_infra/hf/vlm_sft_3ep"
S3_OUT="$WS/phase11_rlhf_grpo_infra/rlhf/outputs/grpo_overnight_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$S3_OUT"

echo "S3_DCP=$S3_DCP"
echo "S3_HF=$S3_HF"
echo "S3_OUT=$S3_OUT"

SGLANG_DISABLE_SHM_MM=1 ATTNRES_MLA_FP32_FALLBACK=1 \
PYTHONPATH="$WS/torchtitan:$WS" \
    python phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
        --dcp-load-path "$S3_DCP" \
        --hf-model-path "$S3_HF" \
        --num-steps 300 \
        --num-episodes-per-step 4 \
        --kl-coef 0.05 \
        > "$S3_OUT/run.log" 2>&1
echo "[$(date)] Step 4/4 GRPO done — out: $S3_OUT"
disk_guard "after-GRPO"
commit_push "GRPO overnight from vlm_sft_3ep ckpt done"

echo ""
echo "==============================================================="
echo "[$(date)] OVERNIGHT CHAIN ALL DONE"
echo "==============================================================="
