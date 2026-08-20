#!/usr/bin/env bash
# Post-merge gate: three arms, serial, on the merged tree.
#
# Runs after the EP x TP fix in common/moe_sharding.py (three layouts that describe what
# the EP path PRODUCES were keyed on enable_sp instead of enable_ep; see
# MERGE_GATE_EP_TP_2026-08-12.md).
#
# Arms, and what each one is FOR:
#   text       kimi_k3_mini_block_attn_res        the MoE lives in the language model, so
#                                                 this is the arm the fixed bug lives in,
#                                                 with no vision path to confound it
#   mm_full    kimi_k3_debugmodel_report_arch     the real target: K3 trains MoonViT-V2
#                                                 jointly, so multimodal is not a variant
#   mm_lora    ..._report_arch_lora               LoRA x DEP, 18/18 pre-merge
#
# Text LoRA is deliberately NOT run: it would only tell us whether a LoRA problem is
# multimodal-specific, and there is no such problem open.
#
# READ THE BASELINE NOTE BEFORE JUDGING THE TABLES: the merge shifts six cells in the 5th
# significant digit (#4099 moved the valid-token-count reduction onto the device, changing
# summation order). "Byte-identical to the pre-merge table" is NOT the pass criterion here.
# The criteria are: every expressible cell trains 10/10, and the two EP x TP cells that the
# merge broke are back.
#
# SERIAL on purpose. 8 GPUs are one resource; overlapping runs on 2 of them is how a
# previous session's evidence got polluted.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TITAN=${TITAN:-/workspace/tt_merge}
OUT=${OUT:-/workspace/mx_postmerge}
STEPS=${STEPS:-10}
mkdir -p "$OUT"

export TITAN

# Topology is set PER ARM, not exported globally. DEP on a text flavor is INVALID, not
# inert: the finding-50 guard fires with "this rank owns 1 vision stage(s) by stage index
# but 0 were wired" on every PP cell. That was established once already and a global export
# repeated it. Finding 32 made the config field the source of truth, so "off" here means
# simply not setting the retired env name.
arm_knobs() {
  case $1 in
    text) echo "" ;;
    # KIMI_VIT_PREFETCH=1 joins them: the vision encode for micro-batch m+1 is issued
    # during m's text compute -- the one path implementing report 5.2.3's concurrency,
    # which had otherwise run exactly once, on two cells.
    #
    # This comment claimed until 2026-08-20 that prefetch was "proven numerically inert"
    # and cost nothing to carry. It did cost something: with DEP on it made
    # mm_full/tp2_pp2_cp2 unreproducible (seven runs, seven distinct 10-step traces),
    # because the encode recorded its autograd graph on the vision side stream and
    # several micro-batches then accumulated into the tower unordered. Fixed the same
    # day -- the side stream is now skipped while a graph is being recorded -- so the
    # cell is ordinary evidence again and prefetch really does carry for free here.
    # The old claim's four pp8xvp4 pairs compared too few steps to see any of it.
    # Full account: NONDETERMINISM_tp2_pp2_cp2_2026-08-20.md.
    *)    echo "KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1 KIMI_VIT_PREFETCH=1" ;;
  esac
}

echo "=== post-merge gate: TITAN=$TITAN OUT=$OUT steps=$STEPS ==="
df -h /workspace | tail -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Smoke one cell per arm first. The text arm's defaults bit once before (seq 8192 /
# float32 -> OOM on three cells), and finding that out 40 minutes in wastes the whole arm.
#
# NO --training.local-batch-size here: only the PP cells set it, so passing it made the
# smoke HEAVIER than any cell it gates and it OOMed on a configuration that runs fine.
# A gate that fails on something the real run does not do is worse than no gate.
# Per-arm overrides. The text flavor defaults to seq_len 8192 with dtype float32, which
# does not fit 16 GB per rank -- this is the trap that cost a whole text-arm launch once
# and it OOMs the smoke here too. seq_len 2048 cuts activations 4x; float32 is KEPT so the
# cells that did pass in the earlier text table stay comparable.
# GATE_EXTRA is appended to every arm. It exists because the 2026-08-19 upstream merge
# changed two defaults out from under us and both have to be pinned or all 58 cells fail
# identically, which reads as "our tree broke" rather than "a default moved":
#   --parallelism.spmd_backend partial_dtensor
#       spmd_types is the new default (#4085) and needs every parameter to be a DTensor on
#       the full SPMD mesh before fully_shard. That is the declarative conversion, not a
#       flag.
#   --training.disable-cuda-graphs
#       CUDA graph capture is new in the core trainer (#3559) and on by default; our vision
#       patch count varies per batch, so validation rejects it.
# Unset GATE_EXTRA once either of those stops being needed, rather than editing this.
GATE_EXTRA=${GATE_EXTRA-"--parallelism.spmd_backend partial_dtensor --training.disable-cuda-graphs"}

arm_extra() {
  case $1 in
    text) echo "--training.seq-len 4096 $GATE_EXTRA" ;;
    *)    echo "$GATE_EXTRA" ;;
  esac
}

for arm in text mm_full mm_lora; do
  case $arm in
    text)    FLAVOR=kimi_k3_mini_block_attn_res ;;
    mm_full) FLAVOR=kimi_k3_debugmodel_report_arch ;;
    mm_lora) FLAVOR=kimi_k3_debugmodel_report_arch_lora ;;
  esac
  echo
  echo "########## SMOKE $arm : $FLAVOR $(arm_extra $arm) ##########"
  ( source /venv/main/bin/activate && cd "$TITAN" && PYTHONPATH="$TITAN" \
      env $(arm_knobs "$arm") timeout 900 torchrun --nproc_per_node=2 \
      --master_port=60941 -m torchtitan.train --module kimi_k3 --config "$FLAVOR" \
      --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 2 \
      --training.global-batch-size 8 \
      --parallelism.data_parallel_shard_degree 2 $(arm_extra "$arm") \
      --dump-folder "$OUT/smoke_$arm" > "$OUT/smoke_$arm.log" 2>&1 )
  rc=$?
  echo "  smoke $arm rc=$rc"
  grep -oE "step: +[0-9]+ +loss: +[0-9.]+" "$OUT/smoke_$arm.log" | head -2
  if [ $rc -ne 0 ]; then
    echo "  SMOKE FAILED -- stopping before the matrices. Last error:"
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/smoke_$arm.log" \
      | grep -oiE "(RuntimeError|ValueError|AssertionError|OutOfMemoryError): .{0,120}" \
      | sort -u | head -3
    exit 1
  fi
done

for arm in text mm_full mm_lora; do
  case $arm in
    text)    FLAVOR=kimi_k3_mini_block_attn_res ;;
    mm_full) FLAVOR=kimi_k3_debugmodel_report_arch ;;
    mm_lora) FLAVOR=kimi_k3_debugmodel_report_arch_lora ;;
  esac
  echo
  echo "########## $arm : $FLAVOR ##########"
  env $(arm_knobs "$arm") EXTRA="$(arm_extra "$arm")" FLAVOR=$FLAVOR \
    OUT=$OUT/${arm}_13 STEPS=$STEPS bash "$HERE/run13_flav.sh"
  env $(arm_knobs "$arm") EXTRA="$(arm_extra "$arm")" FLAVOR=$FLAVOR \
    OUT=$OUT/${arm}_max STEPS=$STEPS bash "$HERE/run_maxdeg.sh"
  # Two extra cells on the fullest multimodal arm: the vision tower split across
  # two pipeline stages. Nothing in the 18 above sets KIMI_VIT_DEP_STAGES, so
  # multi-stage DEP had never been in a gate. 54 becomes 56.
  if [ "$arm" = mm_full ]; then
    env $(arm_knobs "$arm") EXTRA="$(arm_extra "$arm")" FLAVOR=$FLAVOR \
      OUT=$OUT/${arm}_dep2 STEPS=$STEPS bash "$HERE/run_dep2.sh"
    # The pp8xvp4 pair, on the two flavors that can express it. This is where the
    # prefetch is meaningfully exercised: eight micro-batches, so seven of them can be
    # served from an encode issued ahead. 56 becomes 58.
    OUT=$OUT/pp8vp4 STEPS=$STEPS bash "$HERE/run_pp8vp4.sh"
  fi
  echo "--- $arm 13-cell table ---"
  bash "$HERE/collect13.sh" "$OUT/${arm}_13"
  df -h /workspace | tail -1
done

# Count what actually produced a log, and say so against what was expected. This exists
# because run_dep2.sh spent several gate runs calling run_cells.sh with pp4/pp8, which
# resolves cell names against run13_flav.sh where those two do not live: it printed
# "no such cell" and continued, so two cells silently did nothing while every tally
# reported 58. A gate that cannot tell "passed" from "never ran" is not a gate.
EXPECTED=58
echo
echo "=== cell accounting ==="
found=0
passed=0
while IFS= read -r log; do
  found=$((found + 1))
  n=$(grep -oE "loss: +[0-9.]+" "$log" | awk '{print $2}' | uniq | wc -l)
  if [ "$n" -ge 10 ]; then
    passed=$((passed + 1))
  else
    echo "  FAIL $(echo "$log" | sed "s|$OUT/||"): $(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -oE '[A-Za-z_]*Error: .{0,70}' | sort -u | head -1)"
  fi
done < <(find "$OUT" -name '*.log' -not -name 'smoke_*' | sort)
echo "  logs found: $found of $EXPECTED expected; passed: $passed"
if [ "$found" -ne "$EXPECTED" ]; then
  echo "  WARNING: $((EXPECTED - found)) cell(s) produced no log at all -- they did not run."
fi

echo
echo "=== POST-MERGE GATE DONE ==="
