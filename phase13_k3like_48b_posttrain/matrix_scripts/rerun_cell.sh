#!/bin/bash
# Rerun one cell inside an existing matrix directory, against that matrix's own
# seed, so the result stays comparable with the cells around it.
# Usage: rerun_cell.sh <OUT_dir> <cell> <nproc> <flags...>
set -uo pipefail
OUT=$1; nm=$2; np=$3; shift 3
TITAN=$(grep -oP '(?<=^tree=)\S+' $OUT/results.txt)
CFG=$(grep -oP '(?<=cfg=)\S+' $OUT/results.txt)
BATCH=$(sed -n 1p $OUT/results.txt | sed 's/.*batch=//')
export TORCHINDUCTOR_CACHE_DIR=$OUT/inductor
S=$OUT/seed
SEED_FILES=$(find "$S/checkpoint" -type f | wc -l)
for pass in warm measure; do
  d="$OUT/${nm}_$pass"; rm -rf "$d"; mkdir -p "$d"
  cp -r "$S/checkpoint" "$d/checkpoint"
  [ "$(find "$d/checkpoint" -type f | wc -l)" -eq "$SEED_FILES" ] || { echo "SEED-COPY-FAIL"; exit 1; }
  ( source /venv/main/bin/activate && cd "$TITAN" && PYTHONPATH=$TITAN timeout 2400 torchrun \
    --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) -m torchtitan.train \
    --module kimi_k3 --config $CFG --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 10 $BATCH --checkpoint.enable \
    --checkpoint.interval 100000 "$@" --dump-folder "$d" > "$OUT/${nm}_$pass.log" 2>&1 )
  rm -rf "$d/checkpoint"
done
L="$OUT/${nm}_measure.log"; ok=seed-ok
grep -q "Loading the checkpoint from" "$L" || ok=ASSERT-SEED-FAIL
la(){ grep -oE "step: *$1 .*loss: *[0-9.]+" "$L"|head -1|grep -oE 'loss: *[0-9.]+'|grep -oE '[0-9.]+'; }
printf "%-12s %-18s s1=%-9s s3=%-9s s10=%-9s (rerun)\n" "$nm" "$ok" "$(la 1)" "$(la 3)" "$(la 10)" | tee -a $OUT/results.txt
