#!/bin/bash
# Matrix runner reporting steps 1 / 3 / 10.
# Usage: TITAN=<tree> CFG=<flavor> BATCH="<flags>" CELLS="<name|nproc|flags>..." mx3.sh <tag>
# Shared seed checkpoint cache, per-cell warm-up pass, seed-load assertion.
set -uo pipefail
TAG=$1
OUT=/workspace/mx3_${TAG}_$(date +%m%d_%H%M%S); mkdir -p "$OUT"
export TORCHINDUCTOR_CACHE_DIR=$OUT/inductor
R=$OUT/results.txt; : > $R
echo "tree=$TITAN cfg=$CFG batch=$BATCH" >> $R

# The seed checkpoint is a function of the model shape alone -- 5.8G per copy,
# 25s to build. Keying it on the tree would defeat before/after comparisons,
# which need BOTH trees to start from the same init, so the key is the flavor
# and the batch flags that shape it. A stale cache cannot be silent: a shape
# that no longer matches fails the DCP load. SEED_CACHE=0 forces a rebuild.
SEED_ROOT=${SEED_ROOT:-/workspace/.mx3_seeds}
SEED_KEY=$(printf '%s|%s' "$CFG" "$BATCH" | sha1sum | cut -c1-12)
S=$SEED_ROOT/${CFG}_${SEED_KEY}
if [ "${SEED_CACHE:-1}" = "0" ]; then rm -rf "$S"; fi
if [ -s "$S/.built" ]; then
  echo "seed cached=$S" >> $R
else
  T=$S.tmp.$$; rm -rf "$T"; mkdir -p "$T"
  ( source /venv/main/bin/activate && cd "$TITAN" && PYTHONPATH=$TITAN timeout 900 torchrun \
    --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train \
    --module kimi_k3 --config $CFG --debug.seed 42 --debug.deterministic --training.steps 1 $BATCH \
    --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint \
    --checkpoint.enable --dump-folder "$T" > "$OUT/seed.log" 2>&1 ); echo "seed rc=$?" >> $R
  if [ "$(find "$T/checkpoint" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
    printf 'cfg=%s\nbatch=%s\ntree=%s\nbuilt=%s\n' \
      "$CFG" "$BATCH" "$TITAN" "$(date -Iseconds)" > "$T/.built"
    mkdir -p "$SEED_ROOT"
    # A concurrent matrix may have won the race; its copy is equivalent.
    mv -T "$T" "$S" 2>/dev/null || rm -rf "$T"
  else
    rm -rf "$T"
  fi
  echo "seed built=$S" >> $R
fi
SEED_FILES=$(find "$S/checkpoint" -type f 2>/dev/null | wc -l)
echo "seed files=$SEED_FILES" >> $R

loss_at(){ grep -oE "step: *$2 .*loss: *[0-9.]+" "$1" | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+'; }

# A cell whose seed copy silently truncated trains from a fresh init and looks
# like an ordinary result. Copy, then count the files back before running.
stage_seed(){ local d=$1
  rm -rf "$d"; mkdir -p "$d"
  cp -r --reflink=auto "$S/checkpoint" "$d/checkpoint" 2>/dev/null
  local n; n=$(find "$d/checkpoint" -type f 2>/dev/null | wc -l)
  [ "$n" -eq "$SEED_FILES" ] && [ "$n" -gt 0 ]
}

cell(){ local nm=$1 np=$2; shift 2
  local avail; avail=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  if [ "${avail:-0}" -lt 20 ]; then
    printf "%-12s %-18s DISK-LOW(%sG)\n" "$nm" "ABORT" "$avail" >> $R; tail -1 $R; return
  fi
  for pass in warm measure; do
    local d="$OUT/${nm}_$pass"
    if ! stage_seed "$d"; then
      printf "%-12s %-18s SEED-COPY-FAIL(%s)\n" "$nm" "ABORT" "$pass" >> $R; tail -1 $R; return
    fi
    ( source /venv/main/bin/activate && cd "$TITAN" && PYTHONPATH=$TITAN timeout 2400 torchrun \
      --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) -m torchtitan.train \
      --module kimi_k3 --config $CFG --debug.seed 42 --debug.deterministic \
      --metrics.log_freq 1 --training.steps 10 $BATCH --checkpoint.enable \
      --checkpoint.interval 100000 "$@" --dump-folder "$d" > "$OUT/${nm}_$pass.log" 2>&1 )
    rm -rf "$d/checkpoint"
  done
  local L="$OUT/${nm}_measure.log" ok="seed-ok"
  grep -q "Loading the checkpoint from" "$L" || ok="ASSERT-SEED-FAIL"
  printf "%-12s %-18s s1=%-9s s3=%-9s s10=%-9s\n" "$nm" "$ok" \
    "$(loss_at "$L" 1)" "$(loss_at "$L" 3)" "$(loss_at "$L" 10)" >> $R; tail -1 $R; }

while IFS='|' read -r nm np flags; do
  [ -z "${nm// }" ] && continue
  cell "$nm" "$np" $flags
done <<< "$CELLS"
echo "DONE $OUT" >> $R
