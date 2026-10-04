#!/bin/bash
# DEP on dep_review1 = 8518f473f (Figure 11 layout) on the 4 x H100 box, after the PR 4656 kit finishes; one GPU job
# at a time. Layout and data as the 10-02 box (~/mep, ~/kit, DEPR_DATA rebuilt by kit_h100_2026-09-29/dep prep).
# 1) the body's numerics table: seq 2048, 224 px, one image per sample, M16, cost ratio 0.116, deterministic, automatic
#    dynamic shapes off, 100 steps; DEP off twice (the floor), K2.5 and bubble, each on a copy of one cache warmed by
#    one step of each mode; cmp_steps.py against the first DEP off run.
# 2) the body's covered share per level: K2.5 traced gives the measured cost ratio, then bubble traced at that ratio,
#    with DEP's encodes and backwards annotated (probe_annotate_vision_dep_pkg.patch, never committed);
#    ana_dep_place.py reports how much of the vision work ran inside pipeline bubbles.
M=~/mep; KN=~/kit/kit_h100_2026-10-02; K30=~/kit/kit_h100_2026-09-30; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src
O=$M/results/dep_fig11; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $M/results/m4656_list/DONE ]; do sleep 30; done
SHA=8518f473f; git -C $M/tt fetch -q origin dep_review1
NUM=$M/w/dep_fig11; TRC=$M/w/dep_fig11_probe
[ -d $NUM ] || git -C $M/tt worktree add -q --detach $NUM $SHA
[ -d $TRC ] || { git -C $M/tt worktree add -q --detach $TRC $SHA && git -C $TRC apply $KN/probe_annotate_vision_dep_pkg.patch; }
for t in $NUM $TRC; do note "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done
run() {  # <tree> <name> <config> <cache> <steps>
  local tree=$1 name=$2 cfg=$3 cache=$4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && DEPN_STEPS=$5 PYTHONPATH=$KN:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_new_local --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
export DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1
C0=$O/cache_num; rm -rf $C0; mkdir -p $C0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run $NUM warm_$c $c $C0 1; rm -rf $O/warm_$c/out; done
for x in off_a:w_dep_off off_b:w_dep_off k25:w_dep_k25 bubble:w_dep_bubble; do
  rm -rf $O/cc; cp -r $C0 $O/cc; run $NUM num_${x%%:*} ${x#*:} $O/cc 100; rm -rf $O/num_${x%%:*}/out
  [ ${x%%:*} = off_a ] || { $V/bin/python $KN/cmp_steps.py $O/num_off_a/run.log $O/num_${x%%:*}/run.log > $O/cmp_off_${x%%:*}.txt 2>&1
    note "off_a vs ${x%%:*}: $(tail -1 $O/cmp_off_${x%%:*}.txt)"; }
done
rm -rf $O/cc $C0
unset DEPN_DET DEPN_STATIC DEPV_COST_RATIO
for lvl in c012:2048:224 c022:2048:1024 c030:1536:1024; do
  IFS=: read -r name seq res <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_RES=$res DEPR_NMAX=1 DEPN_MBS=16 DEPN_PROFILE=1
  C=$O/cache_$name; rm -rf $C; mkdir -p $C
  run $TRC ${name}_k25 w_dep_k25 $C 12
  t=$(find $O/${name}_k25/out -name "rank0_trace.json*" | head -1)
  [ -n "$t" ] && PYTHONPATH=$K30 $V/bin/python $KN/ana_dep_place.py $(dirname $t) > $O/${name}_k25/place.txt 2>&1
  r=$(grep -a -o 'cost ratio (annotated kernel time): [0-9.]*' $O/${name}_k25/place.txt | grep -o '[0-9.]*$')
  note "$name k25: $(grep -a 'cost ratio' $O/${name}_k25/place.txt); $(tail -1 $O/${name}_k25/place.txt)"
  if [ -n "$r" ]; then
    DEPV_COST_RATIO=$r run $TRC ${name}_bubble w_dep_bubble $C 12
    t=$(find $O/${name}_bubble/out -name "rank0_trace.json*" | head -1)
    [ -n "$t" ] && PYTHONPATH=$K30 $V/bin/python $KN/ana_dep_place.py $(dirname $t) > $O/${name}_bubble/place.txt 2>&1
    note "$name bubble at $r: $(tail -1 $O/${name}_bubble/place.txt)"
  fi
  rm -rf $C
done
unset DEPN_PROFILE
note "all done"; echo done > $O/DONE
