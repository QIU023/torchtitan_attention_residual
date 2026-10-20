#!/bin/bash
# Local patch check (never committed): the replica's parameters get grad_dtype = mixed_precision_reduce, like the stage-0
# tower under FSDP; K2.5 and bubble, 2 steps, compared with the unpatched DEP-off dumps.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; G=$S/diagdep
export DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=256 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 \
  DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
one() {  # <config> <gpus>
  local cfg=$1 gpus=$2 O=$G/gd_$1; rm -rf $O $G/ccgd_$1; mkdir -p $O; cp -r $G/cache_new $G/ccgd_$1
  ( cd $G/new_gd && DEPN_STEPS=2 DUMP_DIR=$O/dump PYTHONPATH=$G/kit_new:$S/main_pins_1016:. TORCHINDUCTOR_CACHE_DIR=$G/ccgd_$1/ic \
      TRITON_CACHE_DIR=$G/ccgd_$1/tc CUDA_VISIBLE_DEVICES=$gpus timeout 2400 /workspace/venv_1006/bin/torchrun --nproc_per_node=4 \
      --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_upd1 \
      --config $cfg --output-dir $O/out > $O/run.log 2>&1 )
  echo "$(date +%H:%M:%S) gd $cfg rc=$? $(sed 's/\x1b\[[0-9;]*m//g' $O/run.log | grep -a -o 'step: *[0-9]* *loss: *[0-9.]*' | grep -v ' -1' | tr '\n' ' ')" >> $G/progress_gd.txt
  rm -rf $O/out $G/ccgd_$1
}
one w_dep_k25 0,1,2,3 &
one w_dep_bubble 4,5,6,7 &
wait
echo GD_DONE >> $G/progress_gd.txt
