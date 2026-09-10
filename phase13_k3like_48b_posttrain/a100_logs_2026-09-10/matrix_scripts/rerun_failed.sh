#!/usr/bin/env bash
# Re-run any leg whose log is missing / short / shows EADDRINUSE, on a fresh
# port each attempt. Port collision with a previous leg's TIME_WAIT socket is
# the observed failure mode, so retrying on a new port is the actual fix, not
# a "retry until it passes" loop -- it stops as soon as the leg produces STEPS
# rows and reports honestly if it never does.
set -uo pipefail
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/tmp/twin10}; STEPS=${STEPS:-10}; EXTRA=${EXTRA:-}
cd "$TITAN"; export PYTHONPATH=$TITAN; source /venv/main/bin/activate
FLAVOR=kimi_k3_debugmodel_pr_4025
BASE="--module kimi_k3 --config $FLAVOR --debug.seed 42 --debug.deterministic \
 --metrics.log_freq 1 --training.steps $STEPS --training.global-batch-size 8 $EXTRA"
PPB="--training.local-batch-size 2"
D=--parallelism.data_parallel_shard_degree; T=--parallelism.tensor_parallel_degree
P=--parallelism.pipeline_parallel_degree;   C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree
declare -A ARGS=(
 [dp1]="$D 1"                                  [fsdp2]="$D 2"
 [pp2]="$PPB $D 1 $P 2"                        [cp2]="$D 1 $C 2"
 [tp2]="$D 1 $T 2"                             [ep2_fsdp2]="$D 2 $E 2"
 [fsdp2_tp2_pp2]="$PPB $D 2 $T 2 $P 2"         [fsdp2_tp2_cp2]="$D 2 $T 2 $C 2"
 [tp2_pp2_cp2]="$PPB $D 1 $T 2 $P 2 $C 2"      [fsdp2_pp2_cp2]="$PPB $D 2 $P 2 $C 2"
 [ep2_fsdp2_tp2_pp2]="$PPB $D 2 $E 2 $T 2 $P 2" [ep2_fsdp2_tp2_cp2]="$D 2 $E 2 $T 2 $C 2"
 [ep2_fsdp2_pp2_cp2]="$PPB $D 2 $E 2 $P 2 $C 2")
declare -A NG=([dp1]=1 [fsdp2]=2 [pp2]=2 [cp2]=2 [tp2]=2 [ep2_fsdp2]=2)
rows() { sed -E 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*step: +([0-9]+) +loss: +([-0-9.]+).*/\1 \2/' | grep -vE ' -' | sort -u -n -k1,1 | grep -c .; }
for name in "${!ARGS[@]}"; do
  f="$OUT/$name.log"
  [ -f "$f" ] && [ "$(rows "$f")" -eq "$STEPS" ] && continue
  n=${NG[$name]:-8}; gpus=$(seq -s, 0 $((n-1)))
  for attempt in 1 2 3; do
    port=$((50000 + RANDOM % 9000))
    echo ">> $name attempt $attempt on port $port ($n GPUs)"
    rm -rf "$OUT/$name"
    CUDA_VISIBLE_DEVICES="$gpus" timeout 7200 torchrun --nproc_per_node="$n" \
      --master_port="$port" -m torchtitan.train $BASE ${ARGS[$name]} \
      --dump-folder "$OUT/$name" > "$f" 2>&1
    [ "$(rows "$f")" -eq "$STEPS" ] && { echo ">> $name OK"; break; }
    grep -q EADDRINUSE "$f" || { echo ">> $name failed for a reason other than EADDRINUSE; stopping retries"; break; }
    sleep 20
  done
done
echo "### RERUN DONE ###"
