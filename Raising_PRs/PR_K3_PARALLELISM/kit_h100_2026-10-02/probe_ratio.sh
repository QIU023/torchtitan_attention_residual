#!/bin/bash
# One traced K2.5 step (step 10 of 12) at a (seq, image side) not in dep_new_calib.sh's grid, and its measured cost ratio
# (trace_ratio.py). Usage: probe_ratio.sh <seq> <res>. Same setup as dep_new_calib.sh; results next to its trace probes.
M=~/mep; KN=~/kit/kit_h100_2026-10-02; K30=~/kit/kit_h100_2026-09-30; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src; W=$M/w/dep
O=$M/results/dep_new_calib; P=$O/progress.txt
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
seq=$1; res=$2; name=tr_s${seq}_r$res; D=$O/$name; C=$O/cache_$name; rm -rf $D $C; mkdir -p $D $C
( cd $W && DEPR_SEQ=$seq DEPR_RES=$res DEPR_NMAX=1 DEPN_MBS=16 DEPN_STEPS=12 DEPN_PROFILE=1 PYTHONPATH=$KN:$KD:. \
  TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d \
  --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_new_local --config w_dep_k25 \
  --output-dir $D/out > $D/run.log 2>&1 )
rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint $C
n=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'encodes [0-9]* before' | head -1 | grep -o '[0-9]*')
t=$(find $D/out -name "rank0_trace.json*" | head -1)
if [ -n "$t" ] && [ -n "$n" ]; then PYTHONPATH=$K30 $V/bin/python $KN/trace_ratio.py $(dirname $t) $n > $D/ratio.txt 2>&1; fi
note "$name rc=$rc $(grep -a 'cost ratio' $D/ratio.txt 2>/dev/null)"
echo done > $D/DONE
