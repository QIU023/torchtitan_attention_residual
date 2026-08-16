#!/usr/bin/env bash
# Two cells that put the vision tower on TWO pipeline stages
# (KIMI_VIT_DEP_STAGES=2), which nothing in the 54-cell gate exercises.
#
# What this does and does not claim. The tech report's sec 5.2.3 describes DEP as
# splitting ViT and text into separate stages and balancing the vision forward and
# backward passes across PP stages, and gives NO stage count -- K3 goes further and
# schedules the ViT passes into 1F1B pipeline bubbles instead. So a stage count
# above one is OUR generalization, not a K3 configuration, and these cells validate
# our implementation rather than fidelity to the report.
#
# Only pp4 and pp8. Vision stages come OUT of the text budget, so two of them need
# pp_degree >= 4, and the microbatch count has to be at least the stage count --
# both constraints cost a run each when they were learned.
#
# mm_full only: the text flavor has no tower (DEP there is invalid, not inert), and
# mm_lora exercises the same adapter path with fewer trainable params, so it would
# add runtime without adding coverage of the thing under test.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${TITAN:?}"; : "${OUT:?}"
FLAVOR=${FLAVOR:-kimi_k3_debugmodel_report_arch}
STEPS=${STEPS:-10}

# Invoked directly rather than through run_cells.sh: that script resolves a cell name
# against run13_flav.sh, and pp4/pp8 live in run_maxdeg.sh. It printed
# "pp4: no such cell in run13_flav.sh" and continued, so these two cells silently never
# ran while the gate reported them as part of its count -- the exact failure mode the
# gate exists to prevent. Degrees and batch sizes copied from run_maxdeg.sh.
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

for pair in "pp4:4" "pp8:8"; do
  cell=${pair%%:*}; deg=${pair#*:}
  gpus=$(seq -s, 0 $((deg - 1)))
  rm -rf "$OUT/$cell"
  KIMI_VIT_DEP_STAGES=2 KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1 \
  CUDA_VISIBLE_DEVICES=$gpus timeout 3600 torchrun \
    --nproc_per_node="$deg" --master_port=$((29820 + deg)) -m torchtitan.train \
    --module kimi_k3 --config "$FLAVOR" --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps "$STEPS" \
    --training.global-batch-size 8 --training.local-batch-size "$deg" \
    --parallelism.data_parallel_shard_degree 1 \
    --parallelism.pipeline_parallel_degree "$deg" \
    --dump-folder "$OUT/$cell" > "$OUT/$cell.log" 2>&1
  n=$(grep -oE "loss: +[0-9.]+" "$OUT/$cell.log" | wc -l)
  roles=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$cell.log" | grep -oE "DEP vision stage wiring: .{0,50}" | tail -1)
  echo "  dep2 $cell: $n loss lines  ${roles:-no DEP wiring line}"
done
