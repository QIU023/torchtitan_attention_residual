#!/bin/bash
# Where DEP's vision work runs, measured: waits for run_dep_new.sh to finish, then in a separate worktree
# (~/mep/w/dep_probe at the same head plus probe_annotate_vision_dep.patch, never committed) one traced step (step 10 of
# 12) of K2.5 and bubble at each campaign level, analysed by ana_dep_place.py. DEP off once per level gives the baseline
# compute before the schedule, so the per-encode time no longer counts it (trace_ratio.py did).
M=~/mep; KN=~/kit/kit_h100_2026-10-02; K30=~/kit/kit_h100_2026-09-30; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src
W=$M/w/dep_probe; O=$M/results/dep_new_place; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $M/results/dep_new_k3range/DONE ]; do sleep 30; done
HEAD=$(git -C $M/w/dep rev-parse HEAD)
if [ ! -d $W ]; then git -C $M/tt worktree add -q --detach $W $HEAD; fi
git -C $W checkout -q --detach $HEAD && git -C $W apply $KN/probe_annotate_vision_dep.patch
note "probe tree $(git -C $W rev-parse --short HEAD) + annotations, dirty=$(git -C $W status --short | wc -l)"
run() {  # <name> <config> <cache>
  local name=$1 cfg=$2 cache=$3; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && DEPN_STEPS=12 DEPN_PROFILE=1 PYTHONPATH=$KN:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_new_local --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
for lvl in c012:2048:224:0.116 c022:2048:1024:0.220 c030:1536:1024:0.276; do
  IFS=: read -r name seq res ratio <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_RES=$res DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=$ratio
  C=$O/cache_$name; rm -rf $C; mkdir -p $C
  for c in w_dep_off w_dep_k25 w_dep_bubble; do run ${name}_$c $c $C; done
  for c in w_dep_k25 w_dep_bubble; do
    t=$(find $O/${name}_$c/out -name "rank0_trace.json*" | head -1)
    [ -n "$t" ] && PYTHONPATH=$K30 $V/bin/python $KN/ana_dep_place.py $(dirname $t) > $O/${name}_$c/place.txt 2>&1
    note "$name $c: $(tail -1 $O/${name}_$c/place.txt)"
  done
  rm -rf $C
done
git -C $W checkout -q -- .
note "probe done"
echo done > $O/DONE
