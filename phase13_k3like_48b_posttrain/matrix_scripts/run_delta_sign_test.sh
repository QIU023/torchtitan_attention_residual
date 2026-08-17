#!/usr/bin/env bash
# Is delta mode's difference from naive a BIAS, or does it average out?
#
# run_delta_convergence.sh established that the difference is smaller than the
# run-to-run spread of one configuration, and that neither gap grows over 3000 steps.
# That answers "does it diverge" and leaves a sharper question open, because the two
# quantities being compared are not the same KIND of thing:
#
#   * the delta-vs-naive difference is DETERMINISTIC. Same seed, same result to the last
#     digit -- measured, two delta runs at six steps agree bitwise -- because it is a
#     floating-point summation order difference (the mid-stage block-stack rebuild, and
#     the explicit `grad + captured` where autograd would have accumulated), not a
#     sampling of anything;
#   * the reseed difference is genuine randomness.
#
# So "smaller than the reseed spread" bounds the IMPACT without ruling out a systematic
# bias. And in the single pair that was run, delta finished worse (1.78017 against
# 1.77437). One pair cannot tell a coincidence from a sign.
#
# This runs the pair at several seeds and looks at the SIGN. If delta is worse at every
# seed, the reorder is a bias and the magnitude question becomes "how much bias is the
# memory saving worth". If the sign is mixed, the difference averages out and the earlier
# conclusion stands unqualified.
#
# Shorter runs than the convergence test on purpose: this asks about the sign of a
# consistent effect across seeds, not about long-horizon shape, and three pairs at 3000
# steps would take most of a day on two GPUs.
set -uo pipefail
: "${TITAN:?}"; : "${OUT:?}"
STEPS=${STEPS:-1500}
SEEDS=${SEEDS:-"42 7 1234"}
FLAVOR=${FLAVOR:-kimi_k3_mini_attnres_multicommit}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

for seed in $SEEDS; do
  for mode in off on; do
    cache=0; [ "$mode" = on ] && cache=1
    name="s${seed}_${mode}"
    rm -rf "$OUT/$name"
    TORCHTITAN_ATTNRES_CACHE=$cache CUDA_VISIBLE_DEVICES=0,1 timeout 86400 torchrun \
      --nproc_per_node=2 --master_port=29992 -m torchtitan.train \
      --module kimi_k3 --config "$FLAVOR" --debug.seed "$seed" --debug.deterministic \
      --metrics.log_freq 10 --training.steps "$STEPS" --training.seq_len 256 \
      --training.global-batch-size 8 --training.local-batch-size 8 \
      --parallelism.data_parallel_shard_degree 1 \
      --parallelism.pipeline_parallel_degree 2 \
      --parallelism.pipeline_parallel_layers_per_stage 4 \
      --parallelism.pipeline_parallel_first_stage_less_layers 0 \
      --parallelism.pipeline_parallel_last_stage_less_layers 0 \
      --parallelism.pipeline_parallel_schedule Interleaved1F1B \
      --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1
    w=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" | grep -c "cache adapter wrapped")
    if [ "$mode" = on ] && [ "$w" -eq 0 ]; then
      echo "  ABORT seed $seed: delta arm never engaged delta mode"
      exit 1
    fi
    rm -rf "$OUT/$name/checkpoint"
  done
  echo "  seed $seed done"
done

python3 - "$OUT" "$SEEDS" <<'PY'
import re, sys, pathlib, statistics
out, seeds = pathlib.Path(sys.argv[1]), sys.argv[2].split()

def curve(name):
    text = re.sub(r"\x1b\[[0-9;]*m", "", (out / f"{name}.log").read_text())
    seen = {}
    for step, loss in re.findall(r"step: +(\d+) +loss: +([0-9.]+)", text):
        seen.setdefault(int(step), float(loss))
    return [seen[k] for k in sorted(seen)]

print(f"\nseed    naive      delta      delta-naive   tail mean (last 10%)")
signs, tails = [], []
for seed in seeds:
    off, on = curve(f"s{seed}_off"), curve(f"s{seed}_on")
    n = min(len(off), len(on))
    if n < 10:
        print(f"{seed:6s} too few points ({n})")
        continue
    off, on = off[:n], on[:n]
    k = max(2, n // 10)
    # Tail mean rather than the final point: one step is a sample of a noisy quantity,
    # and the question is whether delta sits consistently above or below.
    t_off = statistics.fmean(off[-k:])
    t_on = statistics.fmean(on[-k:])
    d = t_on - t_off
    signs.append(1 if d > 0 else -1)
    tails.append(d / max(abs(t_off), 1e-9))
    print(f"{seed:6s} {off[-1]:.5f}  {on[-1]:.5f}   {on[-1] - off[-1]:+.5f}     "
          f"{t_off:.5f} vs {t_on:.5f}  ({d:+.5f}, {d / max(abs(t_off), 1e-9) * 100:+.2f}%)")

if len(signs) >= 2:
    same = abs(sum(signs)) == len(signs)
    print(
        f"\nsign of (delta - naive) across {len(signs)} seeds: "
        f"{'CONSISTENT -- ' + ('delta worse' if signs[0] > 0 else 'delta better') if same else 'MIXED'}"
    )
    print(f"mean relative tail difference: {statistics.fmean(tails) * 100:+.3f}%")
    if same:
        print(
            "A consistent sign means the summation-order difference is a BIAS, not\n"
            "something that averages away. The magnitude above is then the real cost to\n"
            "weigh against the -11.4% peak memory, and it should be re-checked at a\n"
            "production shape before the gate flips."
        )
    else:
        print(
            "A mixed sign means the difference does not favour either transport, so the\n"
            "convergence result stands without the bias caveat."
        )
PY
