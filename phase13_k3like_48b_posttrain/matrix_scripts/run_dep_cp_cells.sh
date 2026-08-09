#!/usr/bin/env bash
# The three matrix cells that carry BOTH pipeline and context parallelism, run with
# dynamic CP and DEP enabled together.
#
# Why three and not eighteen, and why combined rather than a separate axis:
#
# * DEP changes the stage split, so it is meaningful only at pp > 1; dynamic CP only
#   activates at cp > 1. Only cells with both can show the interaction, and the 18-cell
#   matrix has exactly three: fsdp2_pp2_cp2, tp2_pp2_cp2, ep2_fsdp2_pp2_cp2.
# * A dynamic-CP-OFF arm would add nothing. It is ON by default, so every existing cp cell
#   already ran it, and its numerical equivalence is established at fp32 with
#   max|delta| = 0.000e+00 in dedicated tests. The matrix runs bf16, whose resolution is
#   strictly worse -- rechecking a high-resolution result with a low-resolution instrument.
# * The matrix's job is regression detection, not first verification. Both features are
#   independently established (dynamic CP exact at four configurations; DEP bit-identical
#   at n_vit 1, 2 and 4). If the combined arm regresses, turning one off bisects it after
#   the fact, which is far cheaper than paying 4x on every run.
#
# DEP is at n_vit=1 deliberately: a tower split across stages refuses CP
# (_dep_reject_cp), because every share would have to recompute the dynamic-CP patch plan
# identically and a mismatch there hangs rather than raises.
#
# Each cell is run TWICE, with the features off and on, from the same seed. The features
# are numerically neutral by design, so the pass condition is bit-identical loss -- and a
# difference would mean one of them perturbs the arithmetic under that parallelism
# combination, which is exactly what this is here to catch.
set -uo pipefail

TITAN=${TITAN:-/workspace/torchtitan_attention_residual/torchtitan}
OUT=${OUT:-/workspace/mxdep}
STEPS=${STEPS:-10}
FLAVOR=${FLAVOR:-kimi_k3_debugmodel_report_arch}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

BASE="--module kimi_k3 --config $FLAVOR --debug.seed 42 --debug.deterministic \
 --metrics.log_freq 1 --training.steps $STEPS --training.global-batch-size 8 \
 --training.local-batch-size 2"
D=--parallelism.data_parallel_shard_degree
T=--parallelism.tensor_parallel_degree
P=--parallelism.pipeline_parallel_degree
C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree
ALL=0,1,2,3,4,5,6,7

run_cell() {
  local name="$1" mode="$2" port="$3"; shift 3
  local env_on=""
  [ "$mode" = on ] && env_on="KIMI_VIT_DEP=1 KIMI_VIT_SIDE_STREAM=1"
  rm -rf "$OUT/${name}_${mode}"
  CUDA_VISIBLE_DEVICES=$ALL env $env_on timeout 7200 torchrun \
    --nproc_per_node=8 --master_port="$port" -m torchtitan.train \
    $BASE "$@" --dump-folder "$OUT/${name}_${mode}" \
    > "$OUT/${name}_${mode}.log" 2>&1
  local ex=$?
  printf "%-22s %-4s exit=%s wiring=%s loss1=%s loss%s=%s err=%s\n" \
    "$name" "$mode" "$ex" \
    "$(grep -ac 'vision stage wiring' "$OUT/${name}_${mode}.log")" \
    "$(grep -a 'step:  1 ' "$OUT/${name}_${mode}.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -v 'loss: -' | grep -oE 'loss: *[0-9.]+' | head -1 | tr -d ' ')" \
    "$STEPS" \
    "$(grep -a "step: *$STEPS " "$OUT/${name}_${mode}.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -v 'loss: -' | grep -oE 'loss: *[0-9.]+' | head -1 | tr -d ' ')" \
    "$(grep -aoE '(RuntimeError|ValueError|AssertionError|OutOfMemory\w*|NotImplementedError): .{0,60}' "$OUT/${name}_${mode}.log" | head -1)" \
    >> "$OUT/results.txt"
  rm -rf "$OUT/${name}_${mode}/checkpoint"
}

: > "$OUT/results.txt"
for mode in off on; do
  run_cell fsdp2_pp2_cp2       $mode 49201 $D 2 $P 2 $C 2
  run_cell tp2_pp2_cp2         $mode 49202 $D 1 $T 2 $P 2 $C 2
  run_cell ep2_fsdp2_pp2_cp2   $mode 49203 $D 2 $E 2 $P 2 $C 2
done
echo "### DEPCP DONE ###" >> "$OUT/results.txt"
