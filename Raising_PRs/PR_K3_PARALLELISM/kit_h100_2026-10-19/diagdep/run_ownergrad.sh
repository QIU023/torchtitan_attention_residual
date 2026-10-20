#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; G=$S/diagdep
export DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=256 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 \
  DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=$G/own_new; rm -rf $O $G/ccown; mkdir -p $O; cp -r $G/cache_new $G/ccown
( cd $G/new && DEPN_STEPS=1 DUMP_DIR=$O/dump PYTHONPATH=$G/kit_new:$S/main_pins_1016:. TORCHINDUCTOR_CACHE_DIR=$G/ccown/ic \
    TRITON_CACHE_DIR=$G/ccown/tc CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 2400 /workspace/venv_1006/bin/torchrun --nproc_per_node=4 \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_ownergrad \
    --config w_dep_off --output-dir $O/out > $O/run.log 2>&1 )
echo "$(date +%H:%M:%S) own_new rc=$? $(grep -a -o 'DEP_OWNERGRAD.*' $O/run.log | head -1)" >> $G/progress_own.txt
rm -rf $O/out $G/ccown
echo OWN_DONE >> $G/progress_own.txt
