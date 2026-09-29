#!/bin/bash
# #4656 TP/SP matrix after #4499's, 4 x H100: main f35966713 against #4656 5d469fdf3, 100 steps per cell,
# seed 42, deterministic. Each configuration's caches are warmed by a 1-step main run; both trees start
# from copies of it. Cells that fit in half the box run main and #4656 side by side.
M=~/mep; K=~/kit/m4656; O=$M/results/m4656; P=$O/progress.txt; mkdir -p $O
. $M/venv/bin/activate
A=$M/w/main_f359; B=$M/w/pr4656
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
for t in $A $B; do git -C $t apply $K/torch_compat_shim.patch || { note "shim failed on $t"; exit 1; }; done
note "main $(git -C $A rev-parse --short HEAD) pr4656 $(git -C $B rev-parse --short HEAD) torch $(python -c 'import torch; print(torch.__version__)')"
run() {  # <name> <tree> <gpus> <cache> <config> <steps> [args...]
  local name=$1 tree=$2 gpus=$3 cache=$4 cfg=$5 steps=$6; shift 6
  local D=$O/$name; rm -rf $D; mkdir -p $D; local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3000 torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module m4656_local --config $cfg --training.steps $steps --metrics.log-freq 1 \
    --debug.seed 42 --debug.deterministic --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out; note "$name rc=$rc"
}
side() {  # <name> <ngpu 1|2> <config> [args...]
  local name=$1 n=$2 cfg=$3; shift 3; local C=$O/cache_$name g1=0 g2=2
  [ $n = 2 ] && { g1=0,1; g2=2,3; }
  rm -rf $C; mkdir -p $C
  run warm_$name $A $g1 $C $cfg 1 "$@"
  rm -rf $C.a $C.b; cp -r $C $C.a; cp -r $C $C.b
  run main_$name $A $g1 $C.a $cfg 100 "$@" & run pr_$name $B $g2 $C.b $cfg 100 "$@" & wait
  rm -rf $C $C.a $C.b
}
quad() {  # <name> <config> <steps> [args...]
  local name=$1 cfg=$2 steps=$3; shift 3; local C=$O/cache_$name
  rm -rf $C; mkdir -p $C
  run warm_$name $A 0,1,2,3 $C $cfg 1 "$@"
  for x in "main:$A" "pr:$B"; do rm -rf $C.c; cp -r $C $C.c; run ${x%%:*}_$name ${x#*:} 0,1,2,3 $C.c $cfg $steps "$@"; done
  rm -rf $C $C.c
}
T1="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
T2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data-parallel-shard-degree 2"
TP="--parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2"
NOSP="--parallelism.no-enable-sequence-parallel"
# Table 1: dp1, 256 tokens per step
side t1_tp1 1 k3_adamw $T1
mkdir -p $O/cache_fresh; run pr_t1_tp1_fresh $B 0 $O/cache_fresh k3_adamw 100 $T1; rm -rf $O/cache_fresh
side t1_tp2ep2_sp 2 k3_adamw $T1 $TP
side t1_tp2ep2_nosp 2 k3_adamw $T1 $TP $NOSP
# Table 2: dp2, 512 tokens per step
side t2_dp2 2 k3_adamw $T2
side t2_dp2ep2 2 k3_adamw $T2 --parallelism.expert_parallel_degree 2
quad t2_dp2tp2ep2_sp k3_adamw 100 $T2 $TP
quad t2_dp2tp2ep2_nosp k3_adamw 100 $T2 $TP $NOSP
# Activation checkpointing off, so #4656's own recompute runs under TP and SP
side t1_tp2ep2_sp_noac 2 k3_adamw_noac $T1 $TP
side t1_tp2ep2_nosp_noac 2 k3_adamw_noac $T1 $TP $NOSP
quad t2_dp2tp2ep2_sp_noac k3_adamw_noac 100 $T2 $TP
quad t2_dp2tp2ep2_nosp_noac k3_adamw_noac 100 $T2 $TP $NOSP
# Type checking, 3 steps as in #4499: the b200 mm cell with SP on and off, and dp1 x tp2 x ep2;
# then the mm cell with type checking off for the step-by-step comparison
quad tc_mm_sp k3_mm 3
quad tc_mm_nosp k3_mm 3 $NOSP
side tc_t1_tp2ep2 2 k3_adamw_typecheck $T1 $TP
quad mm_notc_sp k3_mm_notc 100
for t in $A $B; do git -C $t checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; done
for t in $A $B; do echo "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done > $O/worktrees_after.txt
note "m4656 done"; echo done > $O/DONE
