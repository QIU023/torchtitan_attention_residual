#!/usr/bin/env bash
# Does the gate_up_combine hook change a STOCK model's arithmetic?
#
# torchtitan's own rule for a refactor is identical loss AND grad_norm with
# --debug.seed 42 --debug.deterministic. deepseek_v3_debugmodel is the check that
# matters here because it is upstream's, uses GroupedExperts unmodified, and has
# nothing of ours in its path -- an identical run on OUR model would only show that
# our subclass is consistent with itself.
#
# Five significant digits on stdout is not enough to call two runs identical (the
# repo's own guidance), so this reads loss and grad_norm out of the TensorBoard
# event files instead, at full float precision.
#
#   TITAN=/path/to/tree bash run_stock_identity.sh
#
# Run it twice: once on a tree with the hook and once with the hook reverted
# (git stash), then diff the two dumps. Same tree twice is a control, not evidence.
set -uo pipefail

TITAN=${TITAN:?set TITAN to the tree under test}
OUT=${OUT:-/workspace/pr27_identity}
TAG=${TAG:?set TAG to "hook" or "base"}
STEPS=${STEPS:-10}

mkdir -p "$OUT/$TAG"
cd "$TITAN"
export PYTHONPATH=$TITAN
source /venv/main/bin/activate

torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  -m torchtitan.train \
  --module deepseek_v3 --config deepseek_v3_debugmodel \
  --training.steps "$STEPS" --training.seq_len 512 \
  --debug.seed 42 --debug.deterministic \
  --metrics.log_freq 1 --metrics.enable_tensorboard \
  --dump-folder "$OUT/$TAG" \
  > "$OUT/$TAG.log" 2>&1
echo "rc=$? -> $OUT/$TAG.log"

python - "$OUT/$TAG" <<'PY'
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root = sys.argv[1]
paths = sorted(glob.glob(f"{root}/**/events.out.tfevents.*", recursive=True))
if not paths:
    raise SystemExit(f"no event files under {root} -- did the run reach step 1?")
acc = EventAccumulator(paths[-1]); acc.Reload()
tags = [t for t in acc.Tags()["scalars"] if "loss" in t or "grad_norm" in t]
for t in sorted(tags):
    for e in acc.Scalars(t):
        print(f"{t}\tstep={e.step}\t{e.value!r}")
PY
