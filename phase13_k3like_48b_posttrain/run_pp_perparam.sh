#!/bin/bash
# Per-parameter gradient attribution across PP degrees, incl. VP.
#
# PP has only ever been validated at the loss / grad_norm level. Those are
# aggregates: a wrong gradient on one parameter can hide inside a global norm,
# and the AttnRes cross-stage adapter is exactly the kind of hand-routed
# backward path that would show up per-parameter first. This applies the same
# instrument that isolated the TP defect -- one model, one seed checkpoint,
# vary only the parallelism.
#
# dp_shard is held at 2 on every leg, so PP is the only variable. It cannot be
# dropped to 1: without FSDP's mixed-precision cast the KDA params stay fp32 and
# fla's kernel asks for 108,160 B of dynamic shared memory against this GPU's
# 101,376 B limit, at any seq_len or batch size. That also caps this study at
# pp4 on 8 GPUs (pp8 would need dp_shard=1).
#
# Every leg is arranged to accumulate exactly 4 partial sums of batch 1, so the
# bf16 accumulation structure is identical and only the parallelism differs:
#   pp1  dp2 local1 global8 -> 4 accumulation steps x 1 microbatch
#   pp2  dp2 local2 global8 -> 2 accumulation steps x 2 microbatches
#   pp4  dp2 local4 global8 -> 1 accumulation step  x 4 microbatches
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/pp_perparam}
SEED=$OUT/seed
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=${FLAVOR:-kimi_k3_mini_block_attn_res}
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1"

echo "########## seed checkpoint ($FLAVOR) ##########"
rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 1800 torchrun --nproc_per_node=1 --master_port=38000 \
  -m torchtitan.train $BASE --training.steps 1 --training.global-batch-size 1 \
  --training.local-batch-size 1 --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint --dump-folder "$SEED" 2>&1 | tail -1
SEED_PATH=$(find "$SEED" -maxdepth 3 -type d -name "step-0" | head -1)
[ -z "${SEED_PATH:-}" ] && { echo "ABORT: no seed"; exit 1; }
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SEED_PATH \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=38100
# KDA's fla kernel wants 108,160 B of dynamic shared memory at batch 8 on this
# GPU (limit 101,376 B), so the no-PP baseline cannot run the local batch in one
# forward. It runs 8 gradient-accumulation steps of batch 1 instead, which is
# exactly the microbatch structure the PP legs see (8 microbatches of batch 1).
run() {
  local name="$1" ngpu="$2" lbs="$3"; shift 3
  PORT=$((PORT+1))
  rm -rf "$OUT/run_$name" "$OUT/$name.json"*
  echo "=== $name (${ngpu} GPU) ==="
  GRADCHK_DUMP="$OUT/$name.json" timeout 2400 torchrun --nproc_per_node=$ngpu \
    --master_port=$PORT ../phase13_k3like_48b_posttrain/tp_trainer_grad_probe.py \
    $BASE $LOAD --training.steps 3 --training.global-batch-size 8 \
    --training.local-batch-size $lbs --parallelism.data_parallel_shard_degree 2 \
    "$@" --dump-folder "$OUT/run_$name" 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*m//g' \
    | grep -E "step: +1 +loss|Traceback|RuntimeError|ValueError|AssertionError" | head -3
}

run pp1  2 1
run pp2  4 2 --parallelism.pipeline_parallel_degree 2
run pp4  8 4 --parallelism.pipeline_parallel_degree 4
run pp2_vp2 4 2 --parallelism.pipeline_parallel_degree 2 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_layers_per_stage 6
run pp4_vp2 8 4 --parallelism.pipeline_parallel_degree 4 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_layers_per_stage 3

echo; echo "########## per-parameter ratios (pp1 / leg) ##########"
python3 - "$OUT" <<'PY'
import glob, json, os, sys
o = sys.argv[1]
def merge(name):
    d = {}
    for f in sorted(glob.glob(os.path.join(o, name + ".json.r*"))):
        d.update(json.load(open(f)))
    return d or None
base = merge("pp1")
if not base:
    print("no baseline dump"); raise SystemExit
print(f"baseline pp1: {len(base)} parameters with gradients")
for leg in ("pp2", "pp4", "pp2_vp2", "pp4_vp2"):
    d = merge(leg)
    if not d:
        print(f"\n{leg:<10} NO DUMP (leg failed)"); continue
    common = [k for k in base if k in d and base[k] > 1e-8 and d[k] > 1e-8]
    missing = [k for k in base if k not in d and base[k] > 1e-8]
    rows = sorted(((k, base[k] / d[k]) for k in common), key=lambda r: -abs(r[1] - 1))
    print(f"\n{leg:<10} {len(d)} params, {len(common)} comparable, "
          f"max|ratio-1| = {abs(rows[0][1]-1):.5f}")
    if missing:
        print(f"           MISSING from leg: {len(missing)} -> {missing[:3]}")
    for k, r in rows[:5]:
        print(f"     {k:<56}{r:>10.5f}")
PY
