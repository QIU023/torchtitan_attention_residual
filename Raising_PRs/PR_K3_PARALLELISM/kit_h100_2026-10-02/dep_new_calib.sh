#!/bin/bash
# DEP on the rebased tree (dep_review1 a93cd48ea on main db050eb3f), step 1 of the 10-02 H100 plan: waits for the MoonEP
# recheck to finish, then (one GPU job at a time)
#   smoke: K2.5 and bubble, 3 steps each, seq 2048, 448 px, M16 (the port of the local recipes must run end to end);
#   microbench: one GPU, dim 6144, the base tower, seq 2048 and 4096 (microbench_new.py; wall clock, local compile on);
#   trace ratios: one traced K2.5 step (step 10 of 12) at seq 2048 and 4096 x 224 and 1008 px, cap 1, M16, and the
#   planner's cost ratio from the trace's kernel time (trace_ratio.py), which decides the levels.
# Recipes: dep_new_local.py (kit only). Results in ~/mep/results/dep_new_calib.
M=~/mep; KN=~/kit/kit_h100_2026-10-02; K30=~/kit/kit_h100_2026-09-30; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src; W=$M/w/dep
O=$M/results/dep_new_calib; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $M/results/moonep_rb2/DONE ]; do sleep 20; done
note "dep $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
run() {  # <name> <config> <cache>; DEPN_* and DEPR_* from the caller's environment
  local name=$1 cfg=$2 cache=$3; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$KN:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_new_local --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
C=$O/cache_smoke; rm -rf $C; mkdir -p $C
for c in w_dep_k25 w_dep_bubble; do DEPR_SEQ=2048 DEPR_RES=448 DEPR_NMAX=1 DEPN_MBS=16 DEPN_STEPS=3 run smoke_$c $c $C; done
rm -rf $C
cd $W
for seq in 2048 4096; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$KN:. timeout 1800 $V/bin/python $KN/microbench_new.py --dim 6144 --seq $seq \
    --stages 16 > $O/microbench_base_d6144_s$seq.txt 2>&1
  note "microbench seq $seq rc=$? $(grep -a 'text stage forward' $O/microbench_base_d6144_s$seq.txt)"
done
$V/bin/python $KN/choose_levels_k3range.py /root/dep_data/t2i1024_k4/MANIFEST.json $O/microbench_base_d6144_s*.txt \
  > $O/levels_microbench.md 2>&1
for seq in 2048 4096; do
  for res in 224 1024; do
    name=tr_s${seq}_r$res; C=$O/cache_$name; rm -rf $C; mkdir -p $C
    DEPR_SEQ=$seq DEPR_RES=$res DEPR_NMAX=1 DEPN_MBS=16 DEPN_STEPS=12 DEPN_PROFILE=1 run $name w_dep_k25 $C
    n=$(sed 's/\x1b\[[0-9;]*m//g' $O/$name/run.log | grep -a -o 'encodes [0-9]* before' | head -1 | grep -o '[0-9]*')
    t=$(find $O/$name/out -name "rank0_trace.json*" | head -1)
    if [ -n "$t" ] && [ -n "$n" ]; then
      PYTHONPATH=$K30 $V/bin/python $KN/trace_ratio.py $(dirname $t) $n > $O/$name/ratio.txt 2>&1
    fi
    note "$name $(grep -a 'cost ratio' $O/$name/ratio.txt 2>/dev/null)"
    rm -rf $C
  done
done
note "calib done"
echo done > $O/DONE
