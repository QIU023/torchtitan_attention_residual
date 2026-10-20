#!/bin/bash
# The old tree 4e2ec143b needs the 09-28 torch (68e0ae4, the 10-02/10-04 H100 build): rerun its lane with venv_0928 on
# GPUs 4-7 once the first, venv_1006 attempt of that lane has failed through; then redo the analysis after both lanes.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; G=$S/diagdep
until [ $(grep -c "^[0-9:]* old " $G/progress.txt) -ge 6 ]; do sleep 10; done
source /workspace/venv_0928/bin/activate
export DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=256 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 \
  DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$G/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
run() {  # <name> <config> <cache>
  local name=$1 cfg=$2 C=$3 O=$G/old_$1; rm -rf $O; mkdir -p $O
  ( cd $G/old && DEPN_STEPS=2 DUMP_DIR=$O/dump PYTHONPATH=$G/kit_old:. TORCHINDUCTOR_CACHE_DIR=$C/ic \
      TRITON_CACHE_DIR=$C/tc CUDA_VISIBLE_DEVICES=4,5,6,7 timeout 2400 torchrun --nproc_per_node=4 --rdzv_backend c10d \
      --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_upd1 --config $cfg \
      --output-dir $O/out > $O/run.log 2>&1 )
  local rc=$?; rm -rf $O/out; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $O/run.log)
  note "old0928 $name rc=$rc patched=$(echo "$clean" | grep -c 'optim_step patched') dumps=$(ls $O/dump 2>/dev/null | wc -l) $(echo "$clean" | grep -a -o 'DEP_UPD1_GRAD_NORM rank 0 [^ ]*') $(echo "$clean" | grep -a -o 'step: *[0-9]* *loss: *[0-9.-]*' | tr '\n' ' ') $(echo "$clean" | grep -a -o 'vision_dep: .*' | head -1 | cut -c1-110)"
}
note "old0928 lane: torch $(python -c 'import torch; print(torch.__version__)')"
W=$G/cache_old0928; rm -rf $W; mkdir -p $W/ic $W/tc
for c in w_dep_off w_dep_k25 w_dep_bubble; do run warm_$c $c $W; rm -rf $G/old_warm_$c/dump; done
for c in w_dep_off w_dep_k25 w_dep_bubble; do rm -rf $G/cc_old0928; cp -r $W $G/cc_old0928; run $c $c $G/cc_old0928; done
rm -rf $G/cc_old0928
until grep -q "DIAGDEP_DONE" $P; do sleep 10; done
/workspace/venv_1006/bin/python $G/analyze_diagdep.py $G > $G/analysis.txt 2>&1
note "OLD0928_DONE analysis rc=$?"
