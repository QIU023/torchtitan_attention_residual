#!/usr/bin/env bash
# Re-run every CP cell in the gate (21 = 7 per arm x 3 arms) and diff against a baseline.
#
#   TITAN=/workspace/tt_merge OUT=/workspace/mx_cp bash run_cp_cells.sh
#
# Exists because the CP cells span TWO matrix files with different call shapes:
# six live in run13_flav.sh as `launch <name> <gpus> <port> ...` and cp4 lives in
# run_maxdeg.sh as `run <name> <ngpus> ...`. run_cells.sh only greps the first and
# prints "no such cell" then CONTINUES -- which is exactly how two cells once did
# nothing while the tally still said 58. Here an unresolvable name is fatal.
#
# Cell arguments are never retyped: they are read out of the matrix files.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TITAN=${TITAN:?set TITAN to the tree under test}
OUT=${OUT:-/workspace/mx_cp}
STEPS=${STEPS:-10}
BASELINE=${BASELINE:-$HERE/../gate_logs/gate_58_2026-08-19_merged_percell.txt}
mkdir -p "$OUT"

CP_CELLS_13="cp2 fsdp2_tp2_cp2 tp2_pp2_cp2 fsdp2_pp2_cp2 ep2_fsdp2_tp2_cp2 ep2_fsdp2_pp2_cp2"
CP_CELLS_MAX="cp4"
# Narrow the sweep without retyping any cell's arguments or losing the drift assertions
# below -- a targeted re-check has to run the SAME thing the full sweep ran, or it is
# not comparable to it.
ARMS=${ARMS:-"text mm_full mm_lora"}
CELLS=${CELLS:-"$CP_CELLS_13 $CP_CELLS_MAX"}

# The per-arm knobs and extras MUST match run_postmerge_gate.sh or the numbers are not
# comparable to the baseline -- the same tree once gave 12.04691 under one set and
# 12.07827 under the other, and the gap was briefly read as an adapter defect. Rather
# than trust that this copy stays in sync, assert the gate still says what we assume.
GATE=$HERE/run_postmerge_gate.sh
# The gate's settings, and the default here. Overridable for a targeted sweep on a
# different backend (SPMD_BACKEND=spmd_types), but the drift assertion below still
# checks the gate against its OWN value -- an override must not disable the check that
# this script and the gate have not diverged.
GATE_BACKEND=${SPMD_BACKEND:-partial_dtensor}
GATE_EXTRA="--parallelism.spmd_backend $GATE_BACKEND --training.disable-cuda-graphs"
GATE_EXTRA_EXPECTED="--parallelism.spmd_backend partial_dtensor --training.disable-cuda-graphs"
MM_KNOBS="KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1 KIMI_VIT_PREFETCH=1"
for expect in "$GATE_EXTRA_EXPECTED" "$MM_KNOBS" "--training.seq-len 4096"; do
  grep -qF -- "$expect" "$GATE" || {
    echo "FATAL: run_postmerge_gate.sh no longer contains: $expect" >&2
    echo "       This script's settings have drifted from the gate; fix before running." >&2
    exit 1
  }
done

arm_flavor() {
  case $1 in
    text)    echo kimi_k3_mini_block_attn_res ;;
    mm_full) echo kimi_k3_debugmodel_report_arch ;;
    mm_lora) echo kimi_k3_debugmodel_report_arch_lora ;;
  esac
}
# KNOBS overrides the per-arm default, for probes that need to vary one knob (e.g.
# running a cell with DEP off to see whether DEP is what makes it nondeterministic).
# Set it to the empty string with KNOBS="" -- unset and empty mean different things.
arm_knobs() {
  if [ "${KNOBS+set}" = set ]; then echo "$KNOBS"; return; fi
  case $1 in text) echo "" ;; *) echo "$MM_KNOBS" ;; esac
}
arm_extra() {
  case $1 in
    text) echo "--training.seq-len 4096 $GATE_EXTRA" ;;
    *)    echo "$GATE_EXTRA" ;;
  esac
}

cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

D=--parallelism.data_parallel_shard_degree
T=--parallelism.tensor_parallel_degree
P=--parallelism.pipeline_parallel_degree
C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree
PPB="--training.local-batch-size 2"
ALL=0,1,2,3,4,5,6,7

# Resolve a cell name to "<gpus>|<args>", from whichever matrix file defines it.
resolve_cell() {
  local name="$1" line rest gpus n
  line=$(grep -E "^launch +${name} " "$HERE/run13_flav.sh" | head -1)
  if [ -n "$line" ]; then
    rest=$(sed -E "s/^launch +${name} +//" <<<"$line" | sed 's/ *&$//')
    gpus=$(awk '{print $1}' <<<"$rest")
    echo "$gpus|$(cut -d' ' -f3- <<<"$rest")"
    return 0
  fi
  line=$(grep -E "^run +${name} " "$HERE/run_maxdeg.sh" | head -1)
  if [ -n "$line" ]; then
    rest=$(sed -E "s/^run +${name} +//" <<<"$line")
    n=$(awk '{print $1}' <<<"$rest")
    echo "$(seq -s, 0 $((n - 1)))|$(cut -d' ' -f2- <<<"$rest")"
    return 0
  fi
  return 1
}

total=0; ran=0; port_base=52000
for arm in $ARMS; do
  FLAVOR=$(arm_flavor "$arm")
  EXTRA=$(arm_extra "$arm")
  CELL_KNOBS=$(arm_knobs "$arm")
  BASE="--module kimi_k3 --config $FLAVOR --debug.seed 42 --debug.deterministic \
 --metrics.log_freq 1 --training.steps $STEPS --training.global-batch-size 8 $EXTRA"
  echo
  echo "########## $arm : $FLAVOR ##########"
  for name in $CELLS; do
    total=$((total + 1))
    spec=$(resolve_cell "$name") || {
      echo "FATAL: cell '$name' is defined in neither run13_flav.sh nor run_maxdeg.sh" >&2
      exit 1
    }
    gpus=${spec%%|*}; cellargs=${spec#*|}
    eval "gpus=\"$gpus\"; cellargs=\"$cellargs\""
    n=$(awk -F, '{print NF}' <<<"$gpus")
    tag="${arm}_${name}"
    echo "--- $tag  gpus=$gpus  args: $cellargs"
    for attempt in 1 2 3; do
      port=$((port_base + total * 7 + 300 * (attempt - 1)))
      rm -rf "$OUT/$tag"
      CUDA_VISIBLE_DEVICES="$gpus" env $CELL_KNOBS timeout 7200 torchrun \
        --nproc_per_node="$n" --master_port="$port" ${ENTRY:--m torchtitan.train} \
        $BASE $cellargs --dump-folder "$OUT/$tag" > "$OUT/$tag.log" 2>&1
      rm -rf "$OUT/$tag/checkpoint"
      grep -q EADDRINUSE "$OUT/$tag.log" || break
      echo "  ($tag: port $port in use, retrying)"
    done
    # Every rank logs the same step, so a raw match count is rank_count x steps --
    # that reported FAIL (20/10), (40/10), (80/10) on cells that ran perfectly.
    # Dedup on the step NUMBER rather than with uniq: uniq collapses ADJACENT
    # duplicates, so two consecutive steps agreeing to five digits would silently
    # cost a step.
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag.log")
    nsteps=$(grep -oE "step: +[0-9]+" <<<"$clean" | awk '{print $2}' | sort -un | wc -l)
    got=$(grep -oE "step: +[0-9]+ +loss: +[0-9.]+" <<<"$clean" \
      | awk '{print $2, $4}' | sort -un -k1,1 | awk '{printf "%s ", $2}')
    if [ "$nsteps" -eq "$STEPS" ]; then
      ran=$((ran + 1))
      echo "  ok  $got"
    else
      echo "  FAIL ($nsteps/$STEPS steps): $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag.log" \
        | grep -oE '[A-Za-z_]*Error: .{0,90}' | sort -u | head -1)"
    fi
  done
done

echo
echo "=== ran $ran of $total CP cells ==="
echo "=== diff vs baseline: $BASELINE ==="
python3 "$HERE/cmp_cp_cells.py" "$OUT" "$BASELINE"
