#!/bin/bash
# 10-02 (the user: check the DEP refactor and measure the hiding rate with the new code), on the H100:
# 1) numerics: one cell (seq 2048, 224 px, one image per sample, M16, bubble at cost ratio 0.116), deterministic, automatic
#    dynamic shapes off, 20 steps, the measured tree a93cd48ea (~/mep/w/dep) and the refactor b2a57dff7
#    (~/mep/w/dep_new), each on a copy of one warm cache; loss and grad norm compared on every step (cmp_steps.py).
# 2) hiding rate on b2a57dff7 with DEP's encodes and backwards annotated (probe_annotate_vision_dep_pkg.patch in
#    ~/mep/w/dep_probe, never committed): per level K2.5 traced first gives the measured cost ratio, then bubble traced
#    at that ratio gives where the vision work ran (ana_dep_place.py). One GPU job at a time.
M=~/mep; KN=~/kit/kit_h100_2026-10-02; K30=~/kit/kit_h100_2026-09-30; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src
OLD=$M/w/dep; NEW=$M/w/dep_new; PRB=$M/w/dep_probe; O=$M/results/dep_refactor_check; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
for t in $OLD $NEW $PRB; do note "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done
run() {  # <tree> <name> <config> <cache> <steps>; DEPN_*, DEPR_* and DEPV_* from the environment
  local tree=$1 name=$2 cfg=$3 cache=$4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && DEPN_STEPS=$5 PYTHONPATH=$KN:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_new_local --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
export DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1
C0=$O/cache_num; rm -rf $C0; mkdir -p $C0
run $OLD nwarm w_dep_bubble $C0 1
for x in old:$OLD new:$NEW; do
  rm -rf $O/cc; cp -r $C0 $O/cc; run ${x#*:} num_${x%%:*} w_dep_bubble $O/cc 20; rm -rf $O/num_${x%%:*}/out
done
rm -rf $O/cc $C0
$V/bin/python $KN/cmp_steps.py $O/num_old/run.log $O/num_new/run.log > $O/num_cmp.txt 2>&1
note "numerics old vs new: $(tail -1 $O/num_cmp.txt)"
unset DEPN_DET DEPN_STATIC DEPV_COST_RATIO
for lvl in c012:2048:224 c022:2048:1024 c030:1536:1024; do
  IFS=: read -r name seq res <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_RES=$res DEPR_NMAX=1 DEPN_MBS=16 DEPN_PROFILE=1
  C=$O/cache_$name; rm -rf $C; mkdir -p $C
  run $PRB ${name}_k25 w_dep_k25 $C 12
  t=$(find $O/${name}_k25/out -name "rank0_trace.json*" | head -1)
  [ -n "$t" ] && PYTHONPATH=$K30 $V/bin/python $KN/ana_dep_place.py $(dirname $t) > $O/${name}_k25/place.txt 2>&1
  r=$(grep -a -o 'cost ratio (annotated kernel time): [0-9.]*' $O/${name}_k25/place.txt | grep -o '[0-9.]*$')
  note "$name k25: $(grep -a 'cost ratio' $O/${name}_k25/place.txt); $(tail -1 $O/${name}_k25/place.txt)"
  if [ -n "$r" ]; then
    DEPV_COST_RATIO=$r run $PRB ${name}_bubble w_dep_bubble $C 12
    t=$(find $O/${name}_bubble/out -name "rank0_trace.json*" | head -1)
    [ -n "$t" ] && PYTHONPATH=$K30 $V/bin/python $KN/ana_dep_place.py $(dirname $t) > $O/${name}_bubble/place.txt 2>&1
    note "$name bubble at $r: $(tail -1 $O/${name}_bubble/place.txt)"
  fi
  rm -rf $C
done
unset DEPN_PROFILE
note "check done"
echo done > $O/DONE
