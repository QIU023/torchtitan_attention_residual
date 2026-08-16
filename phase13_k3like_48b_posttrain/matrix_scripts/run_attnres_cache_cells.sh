#!/usr/bin/env bash
# The AttnRes adapter's DELTA mode: the cells that actually enter it.
#
# The first version of this script did not, and that is the whole lesson. It reused the
# 13-cell matrix's PP cells through run_cells.sh, which is the right instinct -- never
# retype a cell's arguments -- but those cells run the DEFAULT schedule, and delta mode
# is gated on `isinstance(pp_schedule, Interleaved1F1B)`. Every cell logged "supports
# only Interleaved1F1B; running without the adapter", both halves of the A/B ran naive
# passthrough, and the losses matched perfectly while measuring nothing. The wrap-line
# assertion is what caught it; without that check the run reads as a clean pass.
#
# Two further constraints have to be satisfied together, and no existing matrix cell
# satisfies either:
#
# * `BlockLayoutTables` needs `n_layers % num_stages == 0` under the default layer map.
#   The multimodal pp8vp4 flavor has 30 layers, which is not divisible by 4, so pp2 x vp2
#   raised ValueError before reaching any of the code under test.
# * the model must actually BE AttnRes. `report_arch_pp8vp4` has `num_blocks=None`, so
#   the adapter passes through no matter what the schedule is -- checked after the fact,
#   which is why the first version failed twice for two unrelated reasons.
#
# And no existing flavor can express multi-commit at all. The k3mini parent is 21 layers
# over K3's block size 12, so a span wider than 12 layers leaves fewer than two stages;
# the 48B L32_N8 carrier fits the arithmetic but OOMs on two ranks with 16 layers each.
# Hence kimi_k3_mini_attnres_multicommit: 16 layers in 8 blocks of 2, k3mini in every
# other respect. That flavor exists for this geometry, the same way report_arch_pp8vp4
# exists so pp8xvp4 is expressible at all.
#
#   single  pp2 x vp4 (layers_per_stage 2)  -> 8 stages, ONE commit per stage
#   multi   pp2 x vp2 (layers_per_stage 4)  -> 4 stages, TWO commits per stage
#
# first/last_stage_less_layers 0 is required, not cosmetic: the defaults make the split
# uneven and the adapter's contiguous-layout check then refuses with "layer 24 sits on
# stage 2, but the contiguous layout this adapter assumes".
#
# `multi` is the geometry the layout used to refuse outright, pointing at a class deleted
# in 89868bde5. It has never run, in any form, so it is the only evidence that removing
# that restriction was right.
#
# Judged against the cache-off twin, because a green loss curve says nothing about
# whether delta mode engaged. But NOT by bit equality, and that distinction cost a
# round: delta mode is not bit-identical to naive and is not meant to be. A mid-stage
# rebuild of the block stack is "the only reorder point" (attn_res_model.py), so the
# floating-point summation order differs and the curves separate from step 2 on --
# step 1 agrees to the digit, which is what says the forward path is the same.
# Measured here: max |dLoss| over six steps is 0.0198 at one commit per stage and
# 0.0083 at two, against phase3's recorded |dLoss| <= 0.011 at PP8xVP4. So the bar is
# a tolerance, and the sharper check is COMPARATIVE -- multi-commit must not be worse
# than single-commit, since that is the claim removing the multi-commit restriction
# rests on.
set -uo pipefail
: "${TITAN:?}"; : "${OUT:?}"
STEPS=${STEPS:-6}
FLAVOR=${FLAVOR:-kimi_k3_mini_attnres_multicommit}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

losses() { sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "loss: +[0-9.]+" | tr -s ' '; }

# cell:layers_per_stage:expected_commits_per_stage
for spec in "single:2:1" "multi:4:2"; do
  IFS=: read -r cell lps commits <<<"$spec"
  for mode in off on; do
    cache=0; [ "$mode" = on ] && cache=1
    name="${cell}_${mode}"
    # Cleared per run: a killed one leaves a partial checkpoint the next resumes from,
    # failing with "Missing key in checkpoint state_dict: optimizer.state...step".
    rm -rf "$OUT/$name"
    TORCHTITAN_ATTNRES_CACHE=$cache CUDA_VISIBLE_DEVICES=0,1 timeout 3600 torchrun \
      --nproc_per_node=2 --master_port=29940 -m torchtitan.train \
      --module kimi_k3 --config "$FLAVOR" --debug.seed 42 --debug.deterministic \
      --metrics.log_freq 1 --training.steps "$STEPS" --training.seq_len 256 \
      --training.global-batch-size 8 --training.local-batch-size 8 \
      --parallelism.data_parallel_shard_degree 1 \
      --parallelism.pipeline_parallel_degree 2 \
      --parallelism.pipeline_parallel_layers_per_stage "$lps" \
      --parallelism.pipeline_parallel_first_stage_less_layers 0 \
      --parallelism.pipeline_parallel_last_stage_less_layers 0 \
      --parallelism.pipeline_parallel_schedule Interleaved1F1B \
      --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1
    rm -rf "$OUT/$name/checkpoint"
  done

  on="$OUT/${cell}_on.log"
  n=$(losses "$on" | wc -l)
  w=$(sed 's/\x1b\[[0-9;]*m//g' "$on" | grep -oE "cross-stage cache adapter wrapped [0-9]+ stage" | tail -1)
  if [ -z "$w" ]; then
    verdict="DELTA MODE NEVER ENGAGED -- proves nothing"
    sed 's/\x1b\[[0-9;]*m//g' "$on" | grep -oiE \
      "(supports only Interleaved1F1B|NotImplementedError|ValueError|falling back to passthrough).{0,70}" | head -1
  else
    # Tolerance, not equality, and step 1 separately: step 1 must agree to the printed
    # digits (same forward), later steps only within TOL.
    verdict=$(python3 - "$OUT/${cell}_off.log" "$on" <<'PY'
import re, sys
TOL = 0.03
def losses(p):
    return [float(x) for x in re.findall(r"loss: +([0-9.]+)",
            re.sub(r"\x1b\[[0-9;]*m", "", open(p).read()))]
off, on = losses(sys.argv[1]), losses(sys.argv[2])
if not off or len(off) != len(on):
    print(f"UNCOMPARABLE ({len(off)} vs {len(on)} loss lines)")
elif off[0] != on[0]:
    print(f"FORWARD DIVERGED at step 1: {off[0]} vs {on[0]}")
else:
    d = max(abs(a - b) for a, b in zip(off, on))
    print(f"max |dLoss| {d:.5f} {'within' if d <= TOL else 'OVER'} tol {TOL}")
PY
)
  fi
  # capture-count mismatch now raises; a step-end slot sweep outside an exception path
  # means an mb-end assertion did not run. Either in a log is a real signal.
  a=$(sed 's/\x1b\[[0-9;]*m//g' "$on" \
    | grep -oE "capture-count mismatch|cleared [0-9]+ captured-grad slot" | sort -u | tr '\n' ' ')
  printf "  %-8s (%s commit/stage) %2d loss lines  %-38s %s%s\n" \
    "$cell" "$commits" "$n" "${w:-no wrap line}" "$verdict" "${a:+  ANOMALY: $a}"
done
