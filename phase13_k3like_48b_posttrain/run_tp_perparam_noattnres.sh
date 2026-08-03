#!/bin/bash
# Per-parameter gradient attribution across tp degrees, with AttnRes DISABLED.
#
# The AttnRes finding (TP_GRAD_FINDING_2026-07-29) localized the ONE thing that
# grows with tp degree. It does not explain the flat ~5-10% carried by nearly
# every other parameter -- the no-AttnRes leg lands at 1.0468, not 1.0. This
# script asks the remaining question directly: with AttnRes gone, does anything
# left still move with tp, or is the residual genuinely flat?
#
# Same instrument as before: one model, one seed checkpoint, vary only tp.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/tp_perparam_noattnres}
SEED=$OUT/seed
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=${FLAVOR:-kimi_k3_mini_diag_1l_mla_noattnres}
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1"

echo "########## seed checkpoint ($FLAVOR) ##########"
rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 1800 torchrun --nproc_per_node=1 --master_port=35000 \
  -m torchtitan.train $BASE --training.steps 1 --training.global-batch-size 1 \
  --training.local-batch-size 1 --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint --dump-folder "$SEED" 2>&1 | tail -2
SEED_PATH=$(find "$SEED" -maxdepth 3 -type d -name "step-0" | head -1)
echo "seed: ${SEED_PATH:-MISSING}"
[ -z "${SEED_PATH:-}" ] && { echo "ABORT: no seed"; exit 1; }
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SEED_PATH \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=35100
run() {
  local name="$1" ngpu="$2" dp="$3" tp="$4"
  PORT=$((PORT+1))
  echo "=== $name (${ngpu} GPU: dp=$dp tp=$tp) ==="
  rm -rf "$OUT/run_$name"
  GRADCHK_DUMP="$OUT/$name.json" timeout 2400 torchrun --nproc_per_node=$ngpu \
    --master_port=$PORT ../phase13_k3like_48b_posttrain/tp_trainer_grad_probe.py \
    $BASE $LOAD --training.steps 3 --training.global-batch-size $((dp*2)) --training.local-batch-size 2 \
    --parallelism.data_parallel_shard_degree $dp \
    --parallelism.tensor_parallel_degree $tp --dump-folder "$OUT/run_$name" 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +1 +loss|GRADCHK|Traceback|Error" | head -5
}

run dp2      2 2 1
run dp2_tp2  4 2 2
run dp2_tp4  8 2 4

echo "########## per-parameter ratios (dp2 / dp2xTP) ##########"
python - "$OUT" <<'PY'
import json, sys, os
o = sys.argv[1]
def load(n):
    p = os.path.join(o, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None
base, t2, t4 = load("dp2"), load("dp2_tp2"), load("dp2_tp4")
if not base: print("no baseline dump"); raise SystemExit
rows = []
for k, v in base.items():
    r2 = (v / t2[k]) if t2 and t2.get(k) else float("nan")
    r4 = (v / t4[k]) if t4 and t4.get(k) else float("nan")
    rows.append((k, v, r2, r4))
rows.sort(key=lambda r: -abs((r[3] if r[3] == r[3] else r[2]) - 1.0))
print(f"{'parameter':<52}{'dp2':>10}{'/tp2':>9}{'/tp4':>9}  grows?")
for k, v, r2, r4 in rows:
    grow = "YES" if (r2 == r2 and r4 == r4 and abs(r4-1) > abs(r2-1) + 0.05) else ""
    print(f"{k:<52}{v:>10.4f}{r2:>9.4f}{r4:>9.4f}  {grow}")
PY
