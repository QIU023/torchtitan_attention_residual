#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; G=$S/diagdep
export DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=256 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 \
  DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
one() {  # <name> <tree> <config> <gpus>
  local name=$1 tree=$2 cfg=$3 gpus=$4 O=$G/s20_$1; rm -rf $O $G/cc20_$1; mkdir -p $O; cp -r $G/cache_new $G/cc20_$1
  ( cd $G/$tree && DEPN_STEPS=20 PYTHONPATH=$G/kit_new:$S/main_pins_1016:. TORCHINDUCTOR_CACHE_DIR=$G/cc20_$1/ic \
      TRITON_CACHE_DIR=$G/cc20_$1/tc CUDA_VISIBLE_DEVICES=$gpus timeout 2400 /workspace/venv_1006/bin/torchrun --nproc_per_node=4 \
      --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_new_local \
      --config $cfg --output-dir $O/out > $O/run.log 2>&1 )
  echo "$(date +%H:%M:%S) $name rc=$?" >> $G/progress_20.txt; rm -rf $O/out $G/cc20_$1
}
one off new w_dep_off 0,1,2,3 &
one k25_gd new_gd w_dep_k25 4,5,6,7 &
wait
one k25 new w_dep_k25 0,1,2,3 &
one bubble_gd new_gd w_dep_bubble 4,5,6,7 &
wait
echo S20_DONE >> $G/progress_20.txt
