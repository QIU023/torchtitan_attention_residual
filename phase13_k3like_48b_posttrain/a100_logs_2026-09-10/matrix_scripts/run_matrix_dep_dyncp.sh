#!/usr/bin/env bash
# Full 18-cell matrix, twice: multimodal and text-only, eager, DEP and dynamic CP ON.
#
# Every earlier 18-cell table was multimodal only, and the three TP+CP cells in it were
# run with KIMI_VIT_DYNAMIC_CP=0 because of the open vision-TP x dynamic-CP gap. This run
# leaves both knobs on, so those cells are expected to fail here -- that is the recorded
# defect showing, not a new regression. Reading the table needs that distinction up front.
#
# The text arm carries the same knobs for symmetry, but they are inert there: no vision
# tower means no vision stage to place and no patch dimension to shard. Its value is
# separating "our 5D parallelism is sound" from anything the vision path contributes.
#
# Flavors: multimodal kimi_k3_debugmodel_report_arch (13 layers) matches the current
# reference table. Text kimi_k3_mini_block_attn_res (21 layers, 4 heads, 8 experts) is
# the smallest text flavor that can express every cell: pp8 needs 8+ layers, tp4/cp4 need
# 4+ heads, ep8 needs 8 experts.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/workspace/mx_dep}
STEPS=${STEPS:-10}
mkdir -p "$OUT"

export KIMI_VIT_DEP=1
export KIMI_VIT_DYNAMIC_CP=1

echo "=== DEP=1 DYNAMIC_CP=1 eager, steps=$STEPS, out=$OUT ==="
df -h /workspace | tail -1

for arm in mm text; do
  case $arm in
    mm)   FLAVOR=kimi_k3_debugmodel_report_arch ;;
    text) FLAVOR=kimi_k3_mini_block_attn_res ;;
  esac
  echo
  echo "########## $arm : $FLAVOR ##########"
  FLAVOR=$FLAVOR OUT=$OUT/${arm}_13 STEPS=$STEPS bash "$HERE/run13_flav.sh"
  FLAVOR=$FLAVOR OUT=$OUT/${arm}_max STEPS=$STEPS bash "$HERE/run_maxdeg.sh"
  df -h /workspace | tail -1
done
echo "=== ALL DONE ==="
