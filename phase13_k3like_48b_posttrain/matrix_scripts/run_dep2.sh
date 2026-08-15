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

for cell in pp4 pp8; do
  KIMI_VIT_DEP_STAGES=2 KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1 \
    TITAN=$TITAN OUT=$OUT FLAVOR=$FLAVOR STEPS=$STEPS \
    bash "$HERE/run_cells.sh" "$cell"
done
