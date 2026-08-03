#!/bin/bash
# Per-parameter TP verification in combination with PP and CP, plus tp8.
#
# Everything verified so far was TP alone. A 4D-mesh run was checked once, but
# only at grad_norm, which is exactly the aggregate that hid the AttnRes defect.
# Each leg holds the OTHER parallelism fixed and varies only tp, so tp remains
# the single variable.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/tp_combo}
SEED=$OUT/seed
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=${FLAVOR:-kimi_k3_mini_diag_4l_moe_depth}
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1"

rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 1800 torchrun --nproc_per_node=1 --master_port=48000 \
  -m torchtitan.train $BASE --training.steps 1 --training.global-batch-size 1 \
  --training.local-batch-size 1 --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint --dump-folder "$SEED" >/dev/null 2>&1
SEED_PATH=$(find "$SEED" -maxdepth 3 -type d -name "step-0" | head -1)
[ -z "${SEED_PATH:-}" ] && { echo "ABORT: no seed"; exit 1; }
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SEED_PATH \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=48100
run() {
  local name="$1" ngpu="$2" lbs="$3"; shift 3
  PORT=$((PORT+1)); rm -rf "$OUT/run_$name" "$OUT/$name.json"*
  GRADCHK_DUMP="$OUT/$name.json" timeout 2400 torchrun --nproc_per_node=$ngpu \
    --master_port=$PORT ../phase13_k3like_48b_posttrain/tp_trainer_grad_probe.py \
    $BASE $LOAD --training.steps 3 --training.global-batch-size 8 \
    --training.local-batch-size $lbs --parallelism.data_parallel_shard_degree 1 \
    "$@" --dump-folder "$OUT/run_$name" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' \
    | grep -oE "step: +1 +loss: +[0-9.]+ +grad_norm: +[0-9.]+|[A-Za-z_.]*Error: .*" \
    | sort -u | head -1
}
cmp() {
  python3 - "$OUT" "$1" "$2" "$3" <<'PY'
import glob, json, os, statistics, sys
o, base, leg, label = sys.argv[1:5]
def merge(n):
    d = {}
    for f in sorted(glob.glob(os.path.join(o, n + ".json.r*"))):
        d.update(json.load(open(f)))
    return d or None
b, t = merge(base), merge(leg)
if not (b and t):
    print(f"{label}: MISSING (base={bool(b)} leg={bool(t)})"); raise SystemExit
rows = [(k, b[k]/t[k]) for k in b if b[k] > 1e-9 and t.get(k, 0) > 1e-9]
rows.sort(key=lambda r: -abs(r[1]-1))
print(f"{label:<22} n={len(rows):3d}  max |r-1|={max(abs(r[1]-1) for r in rows):.5f}"
      f"  med={statistics.median(abs(r[1]-1) for r in rows):.5f}"
      f"   worst: {rows[0][0].split('.')[-2]}.{rows[0][0].split('.')[-1]} {rows[0][1]:.4f}")
PY
}

echo "### gap 1: tp8 (8-head flavor; k3mini has 4 heads so tp8 is structurally impossible there) ###"
run h8_tp1 1 2
run h8_tp2 2 2 --parallelism.tensor_parallel_degree 2
run h8_tp4 4 2 --parallelism.tensor_parallel_degree 4
run h8_tp8 8 2 --parallelism.tensor_parallel_degree 8
cmp h8_tp1 h8_tp2 "8head: tp1->tp2"
cmp h8_tp1 h8_tp4 "8head: tp1->tp4"
cmp h8_tp1 h8_tp8 "8head: tp1->tp8"
