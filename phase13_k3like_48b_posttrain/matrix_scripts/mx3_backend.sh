#!/bin/bash
# mx3.sh with the config (and an env prefix) chosen per cell, for the EP comm-backend
# matrix: each backend is its own config_registry flavor. Reports steps 1 / 3 / 10.
# Usage: TITAN=<tree> BATCH="<flags>" CELLS="<name|nproc|config|ENV=..|flags>..." mx3_backend.sh <tag>
# One seed checkpoint per invocation (SEED_CFG, default kimi_k3_debugmodel), per-cell
# warm-up pass, seed-load assertion, disk gate. mx3.sh itself is unchanged.
set -uo pipefail
TAG=$1
OUT=/workspace/mx3_${TAG}_$(date +%m%d_%H%M%S); mkdir -p "$OUT"
export TORCHINDUCTOR_CACHE_DIR=$OUT/inductor
R=$OUT/results.txt; : > $R
SEED_CFG=${SEED_CFG:-kimi_k3_debugmodel}
echo "tree=$TITAN seed_cfg=$SEED_CFG batch=$BATCH" >> $R
S=$OUT/seed
( source /venv/main/bin/activate && cd "$TITAN" && PYTHONPATH=$TITAN timeout 900 torchrun \
  --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train \
  --module kimi_k3 --config $SEED_CFG --debug.seed 42 --debug.deterministic --training.steps 1 $BATCH \
  --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint \
  --checkpoint.enable --dump-folder "$S" > "$OUT/seed.log" 2>&1 ); echo "seed rc=$?" >> $R
SEED_FILES=$(find "$S/checkpoint" -type f 2>/dev/null | wc -l)
echo "seed files=$SEED_FILES" >> $R

loss_at(){ grep -oE "step: *$2 .*loss: *[0-9.]+" "$1" | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+'; }

stage_seed(){ local d=$1
  rm -rf "$d"; mkdir -p "$d"
  cp -r "$S/checkpoint" "$d/checkpoint" 2>/dev/null
  local n; n=$(find "$d/checkpoint" -type f 2>/dev/null | wc -l)
  [ "$n" -eq "$SEED_FILES" ] && [ "$n" -gt 0 ]
}

cell(){ local nm=$1 np=$2 cfg=$3 envs=$4; shift 4
  local avail; avail=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  if [ "${avail:-0}" -lt 20 ]; then
    printf "%-22s %-18s DISK-LOW(%sG)\n" "$nm" "ABORT" "$avail" >> $R; tail -1 $R; return
  fi
  local rc=0
  for pass in warm measure; do
    local d="$OUT/${nm}_$pass"
    if ! stage_seed "$d"; then
      printf "%-22s %-18s SEED-COPY-FAIL(%s)\n" "$nm" "ABORT" "$pass" >> $R; tail -1 $R; return
    fi
    ( source /venv/main/bin/activate && cd "$TITAN" && env $envs PYTHONPATH=$TITAN timeout 2400 torchrun \
      --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) -m torchtitan.train \
      --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic \
      --metrics.log_freq 1 --training.steps 10 $BATCH --checkpoint.enable \
      --checkpoint.interval 100000 --dump-folder "$d" "$@" > "$OUT/${nm}_$pass.log" 2>&1 ); rc=$?
    rm -rf "$d/checkpoint"
  done
  local L="$OUT/${nm}_measure.log" ok="seed-ok"
  grep -q "Loading the checkpoint from" "$L" || ok="ASSERT-SEED-FAIL"
  printf "%-22s %-18s rc=%-3s s1=%-9s s3=%-9s s10=%-9s\n" "$nm" "$ok" "$rc" \
    "$(loss_at "$L" 1)" "$(loss_at "$L" 3)" "$(loss_at "$L" 10)" >> $R; tail -1 $R; }

while IFS='|' read -r nm np cfg envs flags; do
  [ -z "${nm// }" ] && continue
  cell "$nm" "$np" "$cfg" "${envs:-_=_}" $flags
done <<< "$CELLS"
echo "DONE $OUT" >> $R
