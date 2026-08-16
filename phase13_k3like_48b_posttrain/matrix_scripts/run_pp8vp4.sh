#!/usr/bin/env bash
# The pp8 x vp4 pair: multimodal and LoRA. Neither the 13-cell block nor maxdeg can
# host it -- the debug twin's 13 layers cannot make 32 virtual stages -- so it takes
# its own script and its own flavors, whose 30 text layers plus the multimodal
# wrapper's two children give exactly 8 x 4.
#
# The topology is spelled out rather than taken from a cell name: the flavor is called
# pp8vp4 and that is a name, not a configuration. Running it through the pp8 cell
# would silently measure plain pp8, which is the mistake the first version of the DEP
# prefetch experiment made.
set -uo pipefail
: "${TITAN:?}"; : "${OUT:?}"
STEPS=${STEPS:-10}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate
# BUBBLE=1 replaces PREFETCH here: they are alternatives (the installer refuses both),
# and this is the pair where the bubble runtime has evidence -- 8/8 planned encodes in
# idle intervals with the loss identical to the mechanism off, on both flavors. The
# prefetch keeps its coverage through the mm arms' knobs.
#
# 24 micro-batches, not 8: with mb == pp every micro-batch falls in the report's upfront
# prefix, so there is nothing left to place and the cell reports 0/0 while looking green.
#
# The dump folder is cleared per cell. A run killed mid-step leaves a partial checkpoint
# that the next one resumes from, and it fails with "Missing key in checkpoint
# state_dict: optimizer.state...step" -- which reads as a defect in the cell.
for pair in "mm:kimi_k3_debugmodel_report_arch_pp8vp4" "lora:kimi_k3_debugmodel_report_arch_pp8vp4_lora"; do
  label=${pair%%:*}; flavor=${pair#*:}
  rm -rf "$OUT/$label"
  KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1 KIMI_VIT_BUBBLE=1 KIMI_VIT_BUBBLE_COST_RATIO=0.493 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 3600 torchrun \
    --nproc_per_node=8 --master_port=29751 -m torchtitan.train \
    --module kimi_k3 --config "$flavor" --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps "$STEPS" --training.seq_len 256 \
    --training.global-batch-size 24 --training.local-batch-size 24 \
    --parallelism.pipeline_parallel_degree 8 \
    --parallelism.pipeline_parallel_layers_per_stage 1 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --dump-folder "$OUT/$label" > "$OUT/$label.log" 2>&1
  n=$(grep -oE "loss: +[0-9.]+" "$OUT/$label.log" | wc -l)
  occ=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$label.log" | grep -oE 'DEP bubble runtime: [0-9]+/[0-9]+ planned' | tail -1)
  echo "  pp8vp4 $label: $n loss lines  ${occ:-no bubble report}"
  # Occupancy is the point of the cell, so a green loss count with 0/N placed is a
  # regression this must not pass over in silence.
  case "$occ" in *" 0/"*) echo "    WARNING: nothing was placed in a bubble" ;; esac
done
