#!/bin/bash
# AC table on the branch as filed (main b21f7d43e): shared warm cache, a fresh-cache noise floor, a vision probe.
set -uo pipefail; set -f
MAIN=/tmp/wt_acreuse_main BR=/tmp/wt_fix_acreuse; export CUDA_VISIBLE_DEVICES=${GPU:-2}
OUT=/workspace/.acreuse_gpu/b21_$(date +%m%d_%H%M%S); mkdir -p $OUT/cache $OUT/cache_fresh $OUT/cache_probe
R=$OUT/results.txt; echo "main=$(git -C $MAIN rev-parse --short HEAD) branch=$(git -C $BR rev-parse --short HEAD)" > $R
for t in $MAIN $BR; do sed -i 's/if capability not in {(10, 0), (10, 3)}:/if capability < (8, 0):  # LOCAL GUARD LIFT/' $t/torchtitan/models/kimi_k3/kda.py
  grep -q "LOCAL GUARD LIFT" $t/torchtitan/models/kimi_k3/kda.py || { echo "guard lift failed $t" >> $R; exit 1; }; done
B="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 512"
run(){ local nm=$1 tree=$2 steps=$3 cache=$4 entry=$5; shift 5
  ( source /workspace/venv_bfx9/bin/activate && cd $tree && TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$tree timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
    $entry --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
    --training.steps $steps $B --dump-folder $OUT/$nm "$@" > $OUT/$nm.log 2>&1 ); echo $?; }
row(){ local nm=$1 rc=$2 L=$OUT/$1.log
  echo "$nm rc=$rc peak_mem_GiB=$(grep -oE 'memory: *[0-9.]+GiB' $L | grep -oE '[0-9.]+' | sort -g | tail -1) tps_last5=$(grep -oE 'tps: *[0-9,]+' $L | tail -5 | grep -oE '[0-9,]+' | tr -d , | awk '{s+=$1;n++} END{if(n) printf "%d", s/n}')" >> $R
  echo "  loss=$(grep -oE 'step: *[0-9]+ .*loss: *[0-9.]+' $L | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+' | paste -sd,)" >> $R
  echo "  gnorm=$(grep -oE 'grad_norm: *[0-9.]+' $L | grep -oE '[0-9.]+' | paste -sd,)" >> $R; }
E="-m torchtitan.train"
CELLS=(
 "main_none|$MAIN|activation-checkpoint:none"
 "branch_none|$BR|activation-checkpoint:none"
 "branch_selective|$BR|activation-checkpoint:selective"
 "branch_region_attn|$BR|activation-checkpoint:region --activation-checkpoint.save-regions *attention.*"
 "branch_full|$BR|activation-checkpoint:full"
)
for c in "${CELLS[@]}"; do IFS='|' read -r nm tree a <<< "$c"; echo "warm $nm rc=$(run ${nm}_warm $tree 1 $OUT/cache "$E" $a)" >> $R; done
for c in "${CELLS[@]}"; do IFS='|' read -r nm tree a <<< "$c"; rc=$(run $nm $tree ${STEPS:-10} $OUT/cache "$E" $a); row $nm $rc; done
rc=$(run main_none_freshcache $MAIN ${STEPS:-10} $OUT/cache_fresh "$E" activation-checkpoint:none); row main_none_freshcache $rc
rc=$(run branch_none_visionprobe $BR ${STEPS:-10} $OUT/cache_probe "/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/vision_probe.py" activation-checkpoint:none)
echo "vision probe rc=$rc $(grep -h VISION_PROBE $OUT/branch_none_visionprobe.log | tail -1)" >> $R
for t in $MAIN $BR; do git -C $t checkout -- torchtitan/models/kimi_k3/kda.py; done
echo "B21 DONE $OUT" >> $R; cat $R
