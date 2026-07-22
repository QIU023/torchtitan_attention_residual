#!/usr/bin/env bash
# Overnight PP pressure re-run on the CURRENT fork (for the RFC's numerics
# evidence -- reproducible on the pinned commit, unlike the 2026-05-12 report
# whose torch-2.9-era runs no longer reproduce on torch 2.12 + current fla).
#
# Phase A: 48B-layout carrier (d1280 e16 L32 N8) PP=8 x VP=4, naive vs adapter.
#          seq_len=512 (seq_len=1024 OOMs on 5090 31GiB on the current stack;
#          the 05-12 run fit at 1024 on torch 2.9 -- memory grew since).
# Phase B: 175m L16 sweep (pp8vp2, pp4vp2, pp4vp4) naive vs adapter, 1000 steps.
# Phase C: pp4vp2 naive #2 -- naive-vs-naive nondeterminism band, to contextualize
#          the adapter |dLoss| (05-12 band was 0.06-0.13).
set -uo pipefail

WS=/workspace/torchtitan_attention_residual
P3="$WS/phase3_attnres_pp_integration"
TT="$WS/torchtitan"
RESULTS="$P3/OVERNIGHT_RESULTS_2026-07-22.md"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SHA=$(git -C "$TT" rev-parse HEAD)
{
  echo "# Overnight PP pressure re-run -- 2026-07-22"
  echo ""
  echo "torchtitan commit: \`$SHA\`"
  echo "torch: $(/venv/main/bin/python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
  echo ""
  echo "| phase | run | steps | final loss | note |"
  echo "|---|---|---|---|---|"
} > "$RESULTS"

# extract last "loss: X" from a titan rank-7 log
last_loss() { sed -E 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "loss: *[0-9.]+" | tail -1 | grep -oE "[0-9.]+"; }
last_step() { sed -E 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "step: *[0-9]+" | tail -1 | grep -oE "[0-9]+"; }

run_48b_pp8vp4() {
  local mode="$1" cacheflag="$2" out log
  out="$P3/runs/pp8vp4_48b_seq512_${mode}_$(date +%H%M%S)"; rm -rf "$out"
  log="$P3/overnight_pp8vp4_${mode}.log"; > "$log"
  echo "[$(date)] START 48B pp8vp4 $mode" | tee -a "$RESULTS.progress"
  ( cd "$TT" && env $cacheflag ATTNRES_DBG=0 \
      torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
        --local-ranks-filter 7 --role rank --tee 3 -m torchtitan.train \
        --module kimi_k3 --config kimi_linear_48b_block_attn_res_d1280_e16_L32_N8 \
        --training.steps 300 --training.local_batch_size 32 --training.global_batch_size 32 \
        --training.seq_len 512 \
        --parallelism.pipeline_parallel_degree 8 \
        --parallelism.pipeline_parallel_schedule Interleaved1F1B \
        --parallelism.pipeline_parallel_layers_per_stage 1 \
        --parallelism.pipeline_parallel_first_stage_less_layers 0 \
        --parallelism.pipeline_parallel_last_stage_less_layers 0 \
        --checkpoint.no-enable --dump_folder "$out" ) >>"$log" 2>&1
  echo "| A | 48B pp8vp4 $mode (seq512) | $(last_step "$log") | $(last_loss "$log") | |" >> "$RESULTS"
}

echo "===== PHASE A: 48B-carrier PP8xVP4 (seq512, 300 steps) ====="
run_48b_pp8vp4 naive   "-u TORCHTITAN_ATTNRES_CACHE"
run_48b_pp8vp4 adapter "TORCHTITAN_ATTNRES_CACHE=1"

echo "===== PHASE B: 175m L16 sweep (1000 steps) ====="
STEPS=1000 NGPU=8 SWEEP_OUT_ROOT="$P3/runs/pressure_test_20260722" \
  bash "$P3/run_pp_pressure_test.sh" >>"$P3/overnight_l16_sweep.log" 2>&1 || true
if [ -f "$P3/runs/pressure_test_20260722/SUMMARY.md" ]; then
  echo "" >> "$RESULTS"; echo "### L16 sweep SUMMARY (1000 steps)" >> "$RESULTS"
  cat "$P3/runs/pressure_test_20260722/SUMMARY.md" >> "$RESULTS"
fi

echo "===== PHASE C: pp4vp2 naive #2 (nondeterminism band) ====="
STEPS=1000 NGPU=8 RUN_ADAPTER=0 SWEEP="175m_attn_res_L16_n8:4:2:8:16" \
  SWEEP_OUT_ROOT="$P3/runs/band_pp4vp2_naive2_20260722" \
  bash "$P3/run_pp_pressure_test.sh" >>"$P3/overnight_band.log" 2>&1 || true
if [ -f "$P3/runs/band_pp4vp2_naive2_20260722/SUMMARY.md" ]; then
  echo "" >> "$RESULTS"; echo "### pp4vp2 naive #2 (for naive-vs-naive band)" >> "$RESULTS"
  cat "$P3/runs/band_pp4vp2_naive2_20260722/SUMMARY.md" >> "$RESULTS"
fi

echo "OVERNIGHT_PP_DONE" >> "$RESULTS"
echo "[$(date)] ALL DONE" | tee -a "$RESULTS.progress"
