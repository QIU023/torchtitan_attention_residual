#!/usr/bin/env bash
# The AttnRes adapter's DELTA mode: a repeatable regression run, plus the one
# geometry it has never been able to run.
#
# What is and is not new here. The delta path is NOT untested -- phase3 carries
# adapter/naive twins at 4 and 8 GPUs and at 48B pp8xvp4 and pp8xvp3, phase4 launches
# with it on by default, and PP_STATUS_2026-08-14 recorded twelve delta cells all
# BITWISE equal to naive, the densest being ep2_fsdp2_tp2_pp2 on the LoRA flavor
# (LoRA x EP x TP x PP x DEP x dynamic CP, cache on, equal to cache off).
#
# What is true is narrower and still matters: the 58-cell gate never sets
# TORCHTITAN_ATTNRES_CACHE, so "58/58" is not a regression statement about delta mode.
# Those twelve cells were a separate, hand-driven run. This script makes it a command,
# so a change to layout.py or the grad bridge can be checked the same way twice.
#
# The four cells and their arguments come from run13_flav.sh through run_cells.sh --
# never retyped. Six 8-GPU runs were once lost to a hand-written --local-batch-size.
#
# Genuinely new: the MULTICOMMIT geometry. A stage whose layer span crosses more than
# one AttnRes block boundary was refused outright by the layout, which raised a
# NotImplementedError pointing at _RecvBlockGradsFromConsumers -- a class deleted in
# 89868bde5 when the custom grad P2P became the rank-local capture and augment hooks.
# So this configuration has never run, and it is the one A2 opened up.
#
# The judge is cache-off vs cache-on, not a healthy loss curve: delta mode is
# numerically neutral by design, so a green cell says nothing about whether it engaged.
# Every cell therefore runs twice and the log must also carry the wrap line.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${TITAN:?}"; : "${OUT:?}"
STEPS=${STEPS:-10}
FLAVOR=${FLAVOR:-kimi_k3_debugmodel_report_arch}
# The gate's multimodal knobs. The adapter has DEP-specific wiring
# (_install_vision_stage_wiring, _dep_vision_share_index), so delta x DEP is real code
# rather than a formality -- and a harness that dropped these knobs is what once made
# the same tree report 12.04691 here and 12.07827 in the gate.
export KIMI_VIT_DEP=${KIMI_VIT_DEP:-1} KIMI_VIT_DYNAMIC_CP=${KIMI_VIT_DYNAMIC_CP:-1}
mkdir -p "$OUT"

CELLS=${CELLS:-"pp2 fsdp2_tp2_pp2 tp2_pp2_cp2 ep2_fsdp2_tp2_pp2"}

losses() { sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "loss: +[0-9.]+" | tr -s ' '; }
wrapline() {
  sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null \
    | grep -oE "cross-stage cache adapter wrapped [0-9]+ stage" | tail -1
}
# Both are canaries the adapter now raises on or logs. A slot cleared at step end
# outside an exception path means an mb-end assertion did not run.
anomalies() {
  sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null \
    | grep -oE "capture-count mismatch|cleared [0-9]+ captured-grad slot" | sort -u | tr '\n' ' '
}

echo "### delta-mode regression: $CELLS (flavor=$FLAVOR, steps=$STEPS) ###"
for mode in off on; do
  cache=0; [ "$mode" = on ] && cache=1
  TORCHTITAN_ATTNRES_CACHE=$cache TITAN=$TITAN OUT="$OUT/cache_$mode" STEPS=$STEPS \
    FLAVOR=$FLAVOR bash "$HERE/run_cells.sh" $CELLS > "$OUT/cache_$mode.driver.log" 2>&1
done

for name in $CELLS; do
  on="$OUT/cache_on/$name.log"; off="$OUT/cache_off/$name.log"
  n=$(losses "$on" | wc -l)
  w=$(wrapline "$on")
  if [ -z "$w" ]; then
    verdict="DELTA MODE NEVER ENGAGED -- proves nothing"
  elif [ "$(losses "$off")" = "$(losses "$on")" ]; then
    verdict="loss IDENTICAL to cache off"
  else
    verdict="LOSS DIVERGED from cache off"
  fi
  a=$(anomalies "$on")
  printf "  delta %-20s %2d loss lines  %-38s %s%s\n" \
    "$name" "$n" "${w:-no wrap line}" "$verdict" "${a:+  ANOMALY: $a}"
done

# ----- the multicommit geometry, which run13_flav.sh has no cell for ----------
# 32 split children (30 text layers + the multimodal wrapper's two) over
# layers_per_stage 8 gives 4 stages = pp2 x vp2, and attn_res_block_size 1 makes every
# one of those eight layers a block boundary: eight commits a stage.
echo "### multicommit: pp2 x vp2, 8 commits per stage ###"
cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate
MC_FLAVOR=${MC_FLAVOR:-kimi_k3_debugmodel_report_arch_pp8vp4}
for mode in off on; do
  cache=0; [ "$mode" = on ] && cache=1
  name="multicommit_$mode"
  rm -rf "$OUT/$name"
  TORCHTITAN_ATTNRES_CACHE=$cache CUDA_VISIBLE_DEVICES=0,1 timeout 3600 torchrun \
    --nproc_per_node=2 --master_port=29926 -m torchtitan.train \
    --module kimi_k3 --config "$MC_FLAVOR" --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps "$STEPS" --training.seq_len 256 \
    --training.global-batch-size 8 --training.local-batch-size 8 \
    --parallelism.pipeline_parallel_degree 2 \
    --parallelism.pipeline_parallel_layers_per_stage 8 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1
  rm -rf "$OUT/$name/checkpoint"
done
n=$(losses "$OUT/multicommit_on.log" | wc -l)
w=$(wrapline "$OUT/multicommit_on.log")
if [ -z "$w" ]; then
  verdict="DELTA MODE NEVER ENGAGED -- proves nothing"
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/multicommit_on.log" \
    | grep -oiE "(supports only Interleaved1F1B|NotImplementedError|falling back to passthrough).{0,60}" | head -1
elif [ "$(losses "$OUT/multicommit_off.log")" = "$(losses "$OUT/multicommit_on.log")" ]; then
  verdict="loss IDENTICAL to cache off"
else
  verdict="LOSS DIVERGED from cache off"
fi
a=$(anomalies "$OUT/multicommit_on.log")
printf "  %-26s %2d loss lines  %-38s %s%s\n" \
  "multicommit" "$n" "${w:-no wrap line}" "$verdict" "${a:+  ANOMALY: $a}"
