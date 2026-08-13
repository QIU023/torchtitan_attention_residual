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
    *)    echo "KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1" ;;
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
arm_extra() {
  case $1 in
    text) echo "--training.seq-len 4096" ;;
    *)    echo "" ;;
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
  echo "--- $arm 13-cell table ---"
  bash "$HERE/collect13.sh" "$OUT/${arm}_13"
  df -h /workspace | tail -1
done

echo
echo "=== POST-MERGE GATE DONE ==="
