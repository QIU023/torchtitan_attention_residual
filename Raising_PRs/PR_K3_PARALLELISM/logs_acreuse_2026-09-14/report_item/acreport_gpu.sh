#!/bin/bash
# Acceptance for the attention-residual recompute: main b21f7d43e vs the branch on one shared warm cache.
set -uo pipefail; set -f
MAIN=/tmp/wt_acreuse_main BR=/tmp/wt_fix_acreuse
OUT=/workspace/.acreuse_gpu/report_$(date +%m%d_%H%M%S); mkdir -p $OUT/cache
export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor TRITON_CACHE_DIR=$OUT/cache/triton
R=$OUT/results.txt; echo "main=$(git -C $MAIN rev-parse --short HEAD) branch=$(git -C $BR rev-parse --short HEAD)" > $R
for t in $MAIN $BR; do sed -i 's/if capability not in {(10, 0), (10, 3)}:/if capability < (8, 0):  # LOCAL GUARD LIFT/' $t/torchtitan/models/kimi_k3/kda.py
  grep -q "LOCAL GUARD LIFT" $t/torchtitan/models/kimi_k3/kda.py || { echo "guard lift failed $t" >> $R; exit 1; }; done
B="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 512"
run(){ local nm=$1 tree=$2 steps=$3 gpus=$4 mod=$5 cfg=$6; shift 6
  local np=$(( $(echo $gpus | tr -cd , | wc -c) + 1 ))
  ( source /workspace/venv_bfx9/bin/activate && cd $tree && CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=/workspace/pylib/attn_gym_main:$tree timeout 2400 \
    torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module $mod --config $cfg \
    --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps $steps --dump-folder $OUT/$nm "$@" > $OUT/$nm.log 2>&1 ); echo $?; }
row(){ local nm=$1 rc=$2 L=$OUT/$1.log
  echo "$nm rc=$rc peak_mem_GiB=$(grep -oE 'memory: *[0-9.]+GiB' $L | grep -oE '[0-9.]+' | sort -g | tail -1) tps_last5=$(grep -oE 'tps: *[0-9,]+' $L | tail -5 | grep -oE '[0-9,]+' | tr -d , | awk '{s+=$1;n++} END{if(n) printf "%d", s/n}') oom=$(grep -c OutOfMemoryError $L)" >> $R
  echo "  loss=$(grep -oE 'step: *[0-9]+ .*loss: *[0-9.]+' $L | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+' | paste -sd,)" >> $R
  echo "  gnorm=$(grep -oE 'grad_norm: *[0-9.]+' $L | grep -oE '[0-9.]+' | paste -sd,)" >> $R; }
K="kimi_k3 kimi_k3_debugmodel"; RB="torchtitan_recipes.tests.b200 kimi_k3_debugmodel_mm_fsdp2"
CELLS=(
 "dp1_main_none|$MAIN|10|2|$K|$B activation-checkpoint:none"
 "dp1_branch_none|$BR|10|2|$K|$B activation-checkpoint:none"
 "dp1_main_selective|$MAIN|10|2|$K|$B activation-checkpoint:selective"
 "dp1_branch_selective|$BR|10|2|$K|$B activation-checkpoint:selective"
 "dp1_branch_region_attn|$BR|10|2|$K|$B activation-checkpoint:region --activation-checkpoint.save-regions *attention.*"
 "dp1_branch_full|$BR|10|2|$K|$B activation-checkpoint:full"
 "mmfsdp2_main_none|$MAIN|3|2,3|$RB|activation-checkpoint:none"
 "mmfsdp2_branch_none|$BR|3|2,3|$RB|activation-checkpoint:none"
)
for c in "${CELLS[@]}"; do IFS='|' read -r nm tree steps gpus mc a <<< "$c"; read -r mod cfg <<< "$mc"; echo "warm $nm rc=$(run ${nm}_warm $tree 1 $gpus $mod $cfg $a)" >> $R; done
for c in "${CELLS[@]}"; do IFS='|' read -r nm tree steps gpus mc a <<< "$c"; read -r mod cfg <<< "$mc"; rc=$(run $nm $tree $steps $gpus $mod $cfg $a); row $nm $rc; done
for t in $MAIN $BR; do git -C $t checkout -- torchtitan/models/kimi_k3/kda.py; done
echo "REPORT GPU DONE $OUT" >> $R; cat $R
