#!/usr/bin/env bash
# The old 58-cell gate with the TP cells removed, on the 4025-based tree.
#
# Cell list and flags come from run13_flav.sh / run_maxdeg.sh unchanged; the six
# cells naming tensor parallel are dropped because TP is still behind
# NotImplementedError here. Nothing is invented: a new cell list would not be
# comparable to the gate this has to replace.
#
# Weights come from ONE seed checkpoint -- a single one for the whole matrix,
# not one per arm -- copied into every cell, the way verify_refactor.sh does it.
# Measured: the text flavor's 703 keys are a strict subset of the multimodal
# flavor's 773 (the 70 missing are the tower), and the LoRA flavor's 809 contain
# all 773 plus its 36 adapters, which start from their own init anyway. The
# loader is strict=False, so one seed built from the multimodal flavor serves
# all three arms and initialization stops being a variable across arms as well
# as within one. Both batch
# knobs are pinned -- global AND local -- because the local one decides what
# each rank sees per micro-batch, and leaving it free makes a change of DP
# degree look like a numerical difference.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
TITAN=${TITAN:-/workspace/tt_frozen2}
STEPS=${STEPS:-2}
OUT=${OUT:-/workspace/gate_notp_$(date +%m%d_%H%M%S)}; mkdir -p "$OUT"
export TORCHINDUCTOR_CACHE_DIR=$OUT/inductor
R=$OUT/results.txt; : > $R
BATCH="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"

seed_for(){ local cfg=$1 dir=$2
  rm -rf "$dir"
  ( source /venv/main/bin/activate && PYTHONPATH=$TITAN timeout 900 torchrun --nproc_per_node=1 \
    --master_port=$((49500+RANDOM%300)) -m torchtitan.train --module kimi_k3 --config "$cfg" \
    --debug.seed 42 --debug.deterministic --training.steps 1 $BATCH \
    --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint \
    --checkpoint.enable --dump-folder "$dir" > "$OUT/seed_$3.log" 2>&1 )
  echo "  seed[$3] rc=$? $(ls "$dir/checkpoint" 2>/dev/null | head -1)" >> $R; }

cell(){ local arm=$1 cfg=$2 seed=$3 name=$4 np=$5; shift 5
  local d="$OUT/${arm}_$name"; rm -rf "$d"; mkdir -p "$d"
  cp -r "$seed/checkpoint" "$d/checkpoint" 2>/dev/null
  ( source /venv/main/bin/activate && PYTHONPATH=$TITAN timeout 1800 torchrun \
    --nproc_per_node=$np --master_port=$((49800+RANDOM%1200)) -m torchtitan.train \
    --module kimi_k3 --config "$cfg" --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps $STEPS $BATCH --checkpoint.enable \
    --checkpoint.interval 100000 "$@" --dump-folder "$d" > "$OUT/${arm}_$name.log" 2>&1 )
  local rc=$?
  printf "%-10s %-20s rc=%s  %s\n" "$arm" "$name" "$rc" \
    "$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/${arm}_$name.log" | grep -oE 'step: +1 +loss: +[0-9.]+' | head -1)" >> $R
  [ $rc -ne 0 ] && sed 's/\x1b\[[0-9;]*m//g' "$OUT/${arm}_$name.log" \
    | grep -oiE "(RuntimeError|ValueError|AssertionError|NotImplementedError): .{0,70}" | sort -u | head -1 >> $R
  rm -rf "$d/checkpoint"; return 0; }

D=--parallelism.data_parallel_shard_degree
P=--parallelism.pipeline_parallel_degree
C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree
CPNB="--parallelism.context_parallel_load_balancer None"
MB2="--parallelism.num-pp-microbatches 2"
MB4="--parallelism.num-pp-microbatches 4"

S="$OUT/seed_all"; seed_for kimi_k3_debugmodel "$S" all

for arm in text mm lora; do
  case $arm in
    text) CFG=kimi_k3_debugmodel_text ;;
    mm)   CFG=kimi_k3_debugmodel ;;
    lora) CFG=kimi_k3_debugmodel_lora ;;
  esac
  echo "########## $arm ##########" >> $R
  cell $arm $CFG $S warm_discard      1 $D 1
  cell $arm $CFG $S dp1               1 $D 1
  cell $arm $CFG $S fsdp2             2 $D 2
  cell $arm $CFG $S pp2               2 $D 1 $P 2 $MB2
  cell $arm $CFG $S cp2               2 $D 1 $C 2 $CPNB
  cell $arm $CFG $S ep2_fsdp2         2 $D 2 $E 2
  cell $arm $CFG $S fsdp2_pp2_cp2     8 $D 2 $P 2 $MB2 $C 2 $CPNB
  cell $arm $CFG $S ep2_fsdp2_pp2_cp2 8 $D 2 $E 2 $P 2 $MB2 $C 2 $CPNB
  cell $arm $CFG $S ep8_fsdp8         8 $D 8 $E 8
  cell $arm $CFG $S pp4               4 $D 1 $P 4 $MB4
  cell $arm $CFG $S pp8               8 $D 1 $P 8 --parallelism.num-pp-microbatches 8
done
rm -rf "$S"
echo "GATE-NO-TP DONE" >> $R
