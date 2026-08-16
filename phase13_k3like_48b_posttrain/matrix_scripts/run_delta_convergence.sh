#!/usr/bin/env bash
# Does the AttnRes delta path converge to the same place as the naive one?
#
# The open question behind the adapter's whole value proposition. Delta mode is a
# TRANSPORT optimisation -- it ships only newly committed blocks instead of the full
# stack -- and it buys phase3's -11.4% peak memory and +2.7% tps. What it costs is a
# small arithmetic difference: the mid-stage rebuild of the block stack is "the only
# reorder point" (attn_res_model.py), so the floating-point summation order differs from
# the naive path and the curves separate after step 1.
#
# Six steps put that at max |dLoss| 0.0198 (matrix_scripts/run_attnres_cache_cells.sh).
# On a loss of about 7.7 that is 0.26%, which is inside bf16 noise for a single step and
# says nothing about where a real run ends up. Two trajectories can differ by noise at
# every step and still track each other, or slowly diverge -- and only the second would
# make the memory saving a bad trade. Nothing in this repo has ever run long enough to
# tell those apart: every delta-vs-naive comparison on record is tens of steps.
#
# So: the same configuration, one run each, long enough for the curves to either rejoin
# or separate. The judge is not a tolerance at any single step, it is the SHAPE:
#
#   * relative gap at the end vs at the start -- growing means divergence, flat or
#     shrinking means the difference is noise being averaged away;
#   * the gap against the run-to-run spread of the SAME configuration, which is the
#     floor any comparison has to clear. Two naive runs with different seeds bound how
#     much of the delta-vs-naive gap is attributable to the transport at all.
#
# That third arm is the part it would be easy to leave out and be fooled by. Without it
# a 0.5% end gap looks like evidence, when the same config reseeded might show 0.5% too.
set -uo pipefail
: "${TITAN:?}"; : "${OUT:?}"
STEPS=${STEPS:-3000}
FLAVOR=${FLAVOR:-kimi_k3_mini_attnres_multicommit}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

# arm:cache:seed
#   naive       the transport this is measured against
#   delta       the same run with delta mode on
#   naive_seed7 the same transport, different seed -- the run-to-run floor
ARMS=("naive:0:42" "delta:1:42" "naive_seed7:0:7")

for spec in "${ARMS[@]}"; do
  IFS=: read -r arm cache seed <<<"$spec"
  rm -rf "$OUT/$arm"
  # log_freq 10: 3000 steps at freq 1 is a 300k-line log per rank and the shape is what
  # matters, not every step.
  TORCHTITAN_ATTNRES_CACHE=$cache CUDA_VISIBLE_DEVICES=0,1 timeout 86400 torchrun \
    --nproc_per_node=2 --master_port=29990 -m torchtitan.train \
    --module kimi_k3 --config "$FLAVOR" --debug.seed "$seed" --debug.deterministic \
    --metrics.log_freq 10 --training.steps "$STEPS" --training.seq_len 256 \
    --training.global-batch-size 8 --training.local-batch-size 8 \
    --parallelism.data_parallel_shard_degree 1 \
    --parallelism.pipeline_parallel_degree 2 \
    --parallelism.pipeline_parallel_layers_per_stage 4 \
    --parallelism.pipeline_parallel_first_stage_less_layers 0 \
    --parallelism.pipeline_parallel_last_stage_less_layers 0 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --dump-folder "$OUT/$arm" > "$OUT/$arm.log" 2>&1
  rc=$?
  w=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$arm.log" | grep -c "cache adapter wrapped")
  echo "  $arm: rc=$rc, wrap lines=$w, $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$arm.log" | grep -cE 'step: +[0-9]+ +loss') loss lines"
  # A delta arm with no wrap line measured naive twice, which is the failure mode that
  # already wasted one round of this work.
  [ "$arm" = delta ] && [ "$w" -eq 0 ] && echo "    ABORT: delta arm never engaged delta mode" && exit 1
  rm -rf "$OUT/$arm/checkpoint"
done

python3 - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])

def curve(arm):
    text = re.sub(r"\x1b\[[0-9;]*m", "", (out / f"{arm}.log").read_text())
    pairs = re.findall(r"step: +(\d+) +loss: +([0-9.]+)", text)
    seen = {}
    for step, loss in pairs:          # ranks duplicate each step; keep one
        seen.setdefault(int(step), float(loss))
    return [seen[k] for k in sorted(seen)]

naive, delta, reseed = curve("naive"), curve("delta"), curve("naive_seed7")
n = min(len(naive), len(delta), len(reseed))
if n < 10:
    print(f"too few points to judge a shape ({n})")
    raise SystemExit(1)
naive, delta, reseed = naive[:n], delta[:n], reseed[:n]

def gap(a, b, lo, hi):
    seg = [abs(x - y) / max(abs(x), 1e-9) for x, y in zip(a[lo:hi], b[lo:hi])]
    return sum(seg) / len(seg)

head, tail = slice(0, max(2, n // 10)), slice(n - max(2, n // 10), n)
print(f"\npoints={n}  final loss: naive {naive[-1]:.5f}  delta {delta[-1]:.5f}  reseed {reseed[-1]:.5f}")
for label, other in (("delta vs naive", delta), ("reseed vs naive", reseed)):
    h = gap(naive, other, head.start, head.stop)
    t = gap(naive, other, tail.start, tail.stop)
    print(f"  {label:16s} mean rel gap: first 10% {h:.5f} -> last 10% {t:.5f}"
          f"  ({'growing' if t > h * 1.5 else 'flat or shrinking'})")
print(
    "\nThe delta-vs-naive gap only means something if it exceeds the reseed gap: that\n"
    "second row is the same transport with a different seed, i.e. how far apart two runs\n"
    "of the SAME configuration land. A delta gap at or below it is not attributable to\n"
    "the transport."
)
PY
