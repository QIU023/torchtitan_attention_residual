#!/bin/bash
# Per-parameter TP check with EP ON -- a different code path from everything
# verified so far.
#
# The moe_sharding fix landed in the `else` branch of `if enable_ep`. With EP on,
# in_grad_placements takes the OTHER branch (dense_sequence_parallel_placement)
# and the routed experts live on the ep mesh instead of being Replicate on tp, so
# none of the EP-off verification transfers.
#
# dp and ep are held fixed and only tp varies. EP is carved out of the
# data-parallel axes, so dp_shard >= ep: dp2 x ep2 gives tp1 on 4 GPUs and tp2 on
# 8. tp4 would need 16.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/tp_ep_perparam}
SEED=$OUT/seed
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=${FLAVOR:-kimi_linear_k3mini_diag_4l_moe_depth}
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1"

rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 1800 torchrun --nproc_per_node=1 --master_port=47000 \
  -m torchtitan.train $BASE --training.steps 1 --training.global-batch-size 1 \
  --training.local-batch-size 1 --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint --dump-folder "$SEED" >/dev/null 2>&1
SEED_PATH=$(find "$SEED" -maxdepth 3 -type d -name "step-0" | head -1)
[ -z "${SEED_PATH:-}" ] && { echo "ABORT: no seed"; exit 1; }
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SEED_PATH \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=47100
run() {
  local name="$1" ngpu="$2" tp="$3" ep="$4"
  PORT=$((PORT+1)); rm -rf "$OUT/run_$name" "$OUT/$name.json"*
  echo "=== $name (${ngpu} GPU: dp_shard4 ep=$ep tp=$tp) ==="
  GRADCHK_DUMP="$OUT/$name.json" timeout 2400 torchrun --nproc_per_node=$ngpu \
    --master_port=$PORT ../phase13_k3like_48b_posttrain/tp_trainer_grad_probe.py \
    $BASE $LOAD --training.steps 3 --training.global-batch-size 8 \
    --training.local-batch-size 2 --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree $tp --parallelism.expert_parallel_degree $ep \
    --dump-folder "$OUT/run_$name" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' \
    | grep -oE "step: +1 +loss: +[0-9.]+ +grad_norm: +[0-9.]+|[A-Za-z_.]*Error: .*" \
    | sort -u | head -2
}

run ep2_tp1 4 1 2
run ep2_tp2 8 2 2

python3 - "$OUT" <<'PY'
import glob, json, os, statistics, sys
o = sys.argv[1]
def merge(n):
    d = {}
    for f in sorted(glob.glob(os.path.join(o, n + ".json.r*"))):
        d.update(json.load(open(f)))
    return d or None
b, t = merge("ep2_tp1"), merge("ep2_tp2")
if not (b and t):
    print(f"MISSING dumps: base={bool(b)} tp2={bool(t)}"); raise SystemExit
rows = [(k, b[k] / t[k]) for k in b if b[k] > 1e-9 and t.get(k, 0) > 1e-9]
rows.sort(key=lambda r: -abs(r[1] - 1))
print(f"\nEP2 + dp_shard4 held fixed, tp1 -> tp2. {len(rows)} comparable params. 8 worst:")
for k, r in rows[:8]:
    print(f"  {k:<56}{r:>10.5f}")
print(f"\n  max |ratio-1| = {max(abs(r[1]-1) for r in rows):.5f}")
print(f"  med |ratio-1| = {statistics.median(abs(r[1]-1) for r in rows):.5f}")
exp = [r for r in rows if "expert" in r[0] or "w1_" in r[0] or "w2_" in r[0] or "w3_" in r[0]]
print(f"\n  routed/shared expert params: {len(exp)}")
for k, r in exp[:6]:
    print(f"    {k:<54}{r:>10.5f}")
PY
