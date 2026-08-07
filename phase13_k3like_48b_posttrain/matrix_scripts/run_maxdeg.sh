#!/usr/bin/env bash
# Max-degree cells on the twin flavor, WITHOUT touching it.
#
# Expressibility is fixed by the twin's own config and was checked, not guessed:
#   num_attention_heads = 4, kda_num_heads = 4, num_experts = 8, 13 layers.
#   runnable        : ep8, pp4, pp8, tp4, cp4
#   not expressible : tp8 / cp8 / tp4xcp4 (only 4 heads to shard),
#                     pp8xvp4 (13 layers cannot host 32 virtual stages)
# A cell that would need the flavor changed is recorded as not expressible,
# never accommodated -- the twin's whole value is "their exact config".
set -uo pipefail
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/tmp/maxdeg}; STEPS=${STEPS:-3}; EXTRA=${EXTRA:-}
FLAVOR=${FLAVOR:-kimi_k3_debugmodel_pr_4025}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN; source /venv/main/bin/activate
BASE="--module kimi_k3 --config $FLAVOR --debug.seed 42 --debug.deterministic \
 --metrics.log_freq 1 --training.steps $STEPS --training.global-batch-size 8 $EXTRA"
D=--parallelism.data_parallel_shard_degree; T=--parallelism.tensor_parallel_degree
P=--parallelism.pipeline_parallel_degree;   C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree

rows() { sed -E 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*step: +([0-9]+) +loss: +([-0-9.]+).*/\1 \2/' | grep -vE ' -' \
  | sort -u -n -k1,1 | grep -c .; }

run() {
  local name="$1" n="$2"; shift 2
  local gpus; gpus=$(seq -s, 0 $((n-1)))
  for attempt in 1 2 3; do
    local port=$((50000 + RANDOM % 9000))
    rm -rf "$OUT/$name"
    CUDA_VISIBLE_DEVICES="$gpus" timeout 7200 torchrun --nproc_per_node="$n" \
      --master_port="$port" ${ENTRY:--m torchtitan.train} $BASE "$@" \
      --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1
    # See run13_flav.sh: the checkpoint is ~700 MB per cell and unread. Dropped
    # here rather than after the whole matrix, because the disk fills DURING a
    # matrix, not after it. tb/ is kept as the full-precision record.
    rm -rf "$OUT/$name/checkpoint"
    [ "$(rows "$OUT/$name.log")" -eq "$STEPS" ] && break
    grep -q EADDRINUSE "$OUT/$name.log" || break   # retry only the one transient mode we have observed
    sleep 15
  done
  local r; r=$(rows "$OUT/$name.log")
  if [ "$r" -eq "$STEPS" ]; then
    printf "%-16s %s\n" "$name" "$(sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" \
      | grep -E "step: +[0-9]+ +loss" | sed -E 's/.*loss: +([-0-9.]+).*/\1/' \
      | grep -vE '^-' | head -3 | tr '\n' ' ')..."
  else
    printf "%-16s FAIL (%d/%d) %s\n" "$name" "$r" "$STEPS" \
      "$(sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" | grep -oiE \
       "(RuntimeError|ValueError|AssertionError|InternalError|NotImplementedError): .{0,70}" | head -1)"
  fi
}

echo "### max-degree cells, flavor=$FLAVOR steps=$STEPS ###"
run ep8_fsdp8 8 $D 8 $E 8
run pp4       4 --training.local-batch-size 4 $D 1 $P 4
run pp8       8 --training.local-batch-size 8 $D 1 $P 8
run tp4       4 $D 1 $T 4
run cp4       4 $D 1 $C 4
echo "### skipped, not expressible on the twin config ###"
echo "tp8 / cp8 / tp4xcp4 : only 4 attention heads (MLA and KDA) to shard"
echo "pp8 x vp4           : 13 layers cannot host 32 virtual stages"
echo "### MAXDEG DONE ###"
