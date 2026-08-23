#!/usr/bin/env bash
# Parallelism matrix for the tree built on PR-4025 (branch k3_on_4025).
#
# Covers the axes that are migrated. TP cells are absent on purpose: TP is still
# behind NotImplementedError on that tree, and a cell that cannot express its
# topology is not a pass.
#
# WARM THE CACHE FIRST. Measured on this tree: the first run against a cold
# inductor cache gives a different loss from every run after it (12.40963 vs
# 12.38712, each reproducible). Without the warm-up the first cells differ from
# the rest and it reads as a parallelism axis changing numerics.
#
# SERIAL on purpose. 8 GPUs are one resource.
set -uo pipefail

TITAN=${TITAN:-/workspace/tt_4025/torchtitan}
OUT=${OUT:-/workspace/mx_4025_$(date +%m%d_%H%M)}
STEPS=${STEPS:-10}
mkdir -p "$OUT"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$OUT/inductor}

echo "=== 4025-tree matrix: TITAN=$TITAN OUT=$OUT steps=$STEPS ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2

cell() {  # name nproc flavor extra-flags
  local name=$1 np=$2 flavor=$3; shift 3
  local port=$((62000 + RANDOM % 2000))
  ( source /venv/main/bin/activate && cd "$TITAN" && PYTHONPATH="$TITAN" \
    timeout 1800 torchrun --nproc_per_node="$np" --master_port="$port" -m torchtitan.train \
    --module kimi_k3 --config "$flavor" --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps "$STEPS" "$@" \
    --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1 )
  local rc=$?
  local n; n=$(grep -oE "loss: +[0-9.]+" "$OUT/$name.log" | awk '{print $2}' | uniq | wc -l)
  printf "%-26s rc=%-3s steps=%-3s %s\n" "$name" "$rc" "$n" \
    "$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" | grep -oE 'loss: +[0-9.]+' | head -1)"
  [ "$rc" -ne 0 ] && sed 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" \
    | grep -oiE "(RuntimeError|ValueError|NotImplementedError|AssertionError): .{0,90}" | sort -u | head -1
  return 0
}

CPNB="--parallelism.context_parallel_load_balancer None"
BIG="--training.num-tokens-per-microbatch-per-dp-rank 2048"

echo; echo "########## warm-up (discarded) ##########"
cell warmup 2 kimi_k3_debugmodel_text --parallelism.data_parallel_shard_degree 2

for arm in text mm lora; do
  case $arm in
    text) F=kimi_k3_debugmodel_text ;;
    mm)   F=kimi_k3_debugmodel ;;
    # LoRA is not compared against the dense arms: it trains 36 parameters of
    # 786, so its losses live somewhere else entirely. Its cells are judged
    # against each other, per the pairing rule.
    lora) F=kimi_k3_debugmodel_lora ;;
  esac
  echo; echo "########## $arm : $F ##########"
  cell ${arm}_dp1            1 $F
  cell ${arm}_fsdp2          2 $F --parallelism.data_parallel_shard_degree 2
  cell ${arm}_cp2            2 $F --parallelism.context_parallel_degree 2 $CPNB
  cell ${arm}_cp4            4 $F --parallelism.context_parallel_degree 4 $CPNB $BIG
  cell ${arm}_pp2            2 $F --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 2
  cell ${arm}_pp4            4 $F --parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4
  cell ${arm}_ep2_fsdp2      2 $F --parallelism.expert_parallel_degree 2 --parallelism.data_parallel_shard_degree 2
  cell ${arm}_ep2_cp2        4 $F --parallelism.expert_parallel_degree 2 --parallelism.data_parallel_shard_degree 2 --parallelism.context_parallel_degree 2 $CPNB
  cell ${arm}_fsdp2_pp2_cp2  8 $F --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 2 --parallelism.context_parallel_degree 2 $CPNB
  cell ${arm}_ep8_fsdp8      8 $F --parallelism.expert_parallel_degree 8 --parallelism.data_parallel_shard_degree 8
done

echo; echo "=== accounting ==="
found=0; passed=0
while IFS= read -r log; do
  case "$log" in *warmup*) continue;; esac
  found=$((found+1))
  n=$(grep -oE "loss: +[0-9.]+" "$log" | awk '{print $2}' | uniq | wc -l)
  if [ "$n" -ge "$STEPS" ]; then passed=$((passed+1)); else
    echo "  FAIL $(basename "$log" .log): $(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -oE '[A-Za-z_]*Error: .{0,70}' | sort -u | head -1)"
  fi
done < <(find "$OUT" -maxdepth 1 -name '*.log' | sort)
echo "  cells with a log: $found; passed: $passed"
echo "=== DONE ==="
