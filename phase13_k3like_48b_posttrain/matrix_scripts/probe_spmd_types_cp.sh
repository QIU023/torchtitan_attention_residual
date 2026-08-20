#!/usr/bin/env bash
# Does our CP survive spmd_types, and does KCP fail differently from Ulysses?
#
#   TITAN=/workspace/torchtitan_attention_residual/torchtitan bash probe_spmd_types_cp.sh
#
# Upstream now requires spmd_backend=spmd_types for CP (validate_cp_backend, PR-4218)
# while our gate pins partial_dtensor. Two separate questions hide behind that, and
# guessing at either has been expensive before:
#
#   1. does CP-on-spmd_types work at all for this model?
#   2. is KCP specifically the thing that breaks?  KCP passes recurrence state rank to
#      rank inside what would be a local_map region -- a sequential dependency the
#      placement vocabulary cannot express -- whereas Ulysses is a plain S(1)->S(2)
#      redistribute. If only the KCP flavor fails, that separates "our CP is imperative"
#      from "KCP is inexpressible".
#
# 2x2 so the answer is read off a table instead of inferred: {kcp, ulysses} x
# {partial_dtensor, spmd_types}. Two steps each -- this asks whether it RUNS, not
# whether it converges.
set -uo pipefail

TITAN=${TITAN:?set TITAN}
OUT=${OUT:-/workspace/mx_spmd_probe}
STEPS=${STEPS:-2}
mkdir -p "$OUT"
cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

for flavor in kimi_k3_mini_kcp kimi_k3_mini_kda_ulysses; do
  for backend in partial_dtensor spmd_types; do
    tag="${flavor#kimi_k3_mini_}_${backend}"
    echo "--- $tag"
    CUDA_VISIBLE_DEVICES=0,1 timeout 1200 torchrun --nproc_per_node=2 \
      --master_port=57311 -m torchtitan.train --module kimi_k3 --config "$flavor" \
      --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
      --training.steps "$STEPS" --training.global-batch-size 8 --training.seq-len 4096 \
      --parallelism.data_parallel_shard_degree 1 \
      --parallelism.context_parallel_degree 2 \
      --parallelism.spmd_backend "$backend" \
      --training.disable-cuda-graphs \
      --dump-folder "$OUT/$tag" > "$OUT/$tag.log" 2>&1
    rc=$?
    rm -rf "$OUT/$tag/checkpoint"
    steps=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag.log" \
      | grep -oE "step: +[0-9]+" | awk '{print $2}' | sort -un | wc -l)
    if [ "$steps" -eq "$STEPS" ]; then
      echo "    OK ($steps/$STEPS steps)"
    else
      echo "    FAIL rc=$rc ($steps/$STEPS steps)"
      # The first error is the informative one; later ones are usually its wreckage.
      sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag.log" \
        | grep -oE "[A-Za-z_]*(Error|Exception): .{0,140}" | head -2 | sed 's/^/      /'
    fi
  done
done

echo
echo "=== table ==="
printf "%-28s %s\n" "cell" "result"
for f in "$OUT"/*.log; do
  t=$(basename "$f" .log)
  n=$(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -oE "step: +[0-9]+" | awk '{print $2}' | sort -un | wc -l)
  printf "%-28s %s\n" "$t" "$([ "$n" -eq "$STEPS" ] && echo ok || echo "fail: $(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -oE '[A-Za-z_]*(Error|Exception): .{0,80}' | head -1)")"
done
