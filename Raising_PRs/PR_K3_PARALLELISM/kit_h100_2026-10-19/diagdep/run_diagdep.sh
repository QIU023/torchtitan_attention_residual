#!/bin/bash
# DEP diagnosis on the 5060: the H100 numerics cell (pp4 x vpp4, seq 2048, 224 px, one image per sample, M16, cost ratio
# 0.116, deterministic, automatic dynamic shapes off) at the debug width (DEPW_DIM=256); DEP off, K2.5 and bubble, 2 steps
# each on a copy of one cache warmed by 2 steps of each mode. New tree 7529daeea on GPUs 0-3 (main pins), old tree
# 4e2ec143b on GPUs 4-7 (venv_1006's own spmd_types and torch_remat, which that tree pins).
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; G=$S/diagdep
source /workspace/venv_1006/bin/activate
export DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=256 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 \
  DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$G/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
run() {  # <tree> <name> <config> <cache> <gpus>
  local tree=$1 name=$2 cfg=$3 C=$4 gpus=$5 O=$G/$1_$2 pins=; rm -rf $O; mkdir -p $O
  [ $tree = new ] && pins=$S/main_pins_1016:
  ( cd $G/$tree && DEPN_STEPS=2 DUMP_DIR=$O/dump PYTHONPATH=$G/kit_$tree:$pins. TORCHINDUCTOR_CACHE_DIR=$C/ic \
      TRITON_CACHE_DIR=$C/tc CUDA_VISIBLE_DEVICES=$gpus timeout 2400 torchrun --nproc_per_node=4 --rdzv_backend c10d \
      --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_upd1 --config $cfg \
      --output-dir $O/out > $O/run.log 2>&1 )
  local rc=$?; rm -rf $O/out; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $O/run.log)
  note "$tree $name rc=$rc patched=$(echo "$clean" | grep -c 'optim_step patched') dumps=$(ls $O/dump 2>/dev/null | wc -l) $(echo "$clean" | grep -a -o 'DEP_UPD1_GRAD_NORM rank 0 [^ ]*') $(echo "$clean" | grep -a -o 'step: *[0-9]* *loss: *[0-9.-]*' | tr '\n' ' ') $(echo "$clean" | grep -a -o 'vision_dep: .*' | head -1 | cut -c1-110)"
}
lane() {  # <tree> <gpus>
  local tree=$1 gpus=$2 W=$G/cache_$1; mkdir -p $W/ic $W/tc
  for c in w_dep_off w_dep_k25 w_dep_bubble; do run $tree warm_$c $c $W $gpus; rm -rf $G/${tree}_warm_$c/dump; done
  for c in w_dep_off w_dep_k25 w_dep_bubble; do rm -rf $G/cc_$tree; cp -r $W $G/cc_$tree; run $tree $c $c $G/cc_$tree $gpus; done
  rm -rf $G/cc_$tree
}
note "new $(ls $G/new/torchtitan/models/kimi_k3/pipeline_parallel/vision_dep | wc -l) files, old $(ls $G/old/torchtitan/models/kimi_k3/pipeline_parallel/vision_dep 2>/dev/null | wc -l) files"
lane new 0,1,2,3 &
lane old 4,5,6,7 &
wait
python $G/analyze_diagdep.py $G > $G/analysis.txt 2>&1
note "DIAGDEP_DONE analysis rc=$?"
