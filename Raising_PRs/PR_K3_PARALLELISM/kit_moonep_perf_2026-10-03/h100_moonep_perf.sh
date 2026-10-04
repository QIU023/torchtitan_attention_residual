#!/bin/bash
# MoonEP performance round (review branch moonep_review1 = 0b698e3c2), real MoonEP 33327eb on a 4 x H100 NVSwitch box,
# queued behind the DEP run of 10-04 (waits for ~/mep/results/dep_fig11/DONE).
# Layout as the 09-30 box: venv ~/mep/venv_src (torch + MoonEP + titan deps), fork clone ~/mep/tt, this kit in
# ~/kit/kit_moonep_perf_2026-10-03. One GPU job at a time; progress in $O/progress.txt.
#   1. GPU unit tests on the new head c5 (test_moonep.py 9 cases: 7 on 2 GPUs, 2 on 4; test_moonep_ops.py; the
#      shared-stream test).
#   2. Bitwise dumps, real kernels: old vs c1, c3 vs c4 and c4 vs c5 must be bitwise equal; old vs c2 reports the gaps and
#      each tree's error against an fp32 dense reference.
#   3. Microbench at K3's per-expert width (latent 3584, hidden 3072), S 4096, K 8, E 32 (8 per rank), 2 layers:
#      the expert op alone for old/c1/c2/c4 under uniform and skewed routing; the whole MoE layer for old and for
#      c4 with the shared-experts stream off and on.
#   4. End-to-end Kimi K3 debug cells (FSDP 4 x EP 4, seq 512), 20 steps, deterministic, one warm cache per tree:
#      old standard (twice), old MoonEP, old MoonEP FullAC; c5 MoonEP, FullAC, shared stream, standard + stream; then
#      c5 standard and MoonEP with expert FSDP over two ranks (dp_shard 4 x EP 2) and with HSDP (2 x 2 x EP 2).
M=~/mep; V=$M/venv_src; K=~/kit/kit_moonep_perf_2026-10-03
O=$M/results/moonep_perf; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $M/results/dep_fig11/DONE ]; do sleep 30; done
. $V/bin/activate
declare -A REV=( [old]=5e4596dc7 [c1]=a7864a5a3 [c2]=06d8c338d [c3]=493ce26e0 [c4]=1db6fbcd1 [c5]=0b698e3c2 )
git -C $M/tt fetch -q origin moonep_review1 k3_moonep_seam
for t in old c1 c2 c3 c4 c5; do
  W=$M/w/mep_$t; [ -d $W ] || git -C $M/tt worktree add -q --detach $W ${REV[$t]}
  note "$t $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
done
note "multicast $(python3 -c "import ctypes;c=ctypes.CDLL('libcuda.so.1');c.cuInit(0);v=ctypes.c_int();c.cuDeviceGetAttribute(ctypes.byref(v),132,0);print(v.value)")"

( cd $M/w/mep_c5 && timeout 2400 python -m pytest tests/unit_tests/gpu/test_moonep.py tests/unit_tests/cpu/test_moonep_ops.py \
    tests/unit_tests/gpu/test_moe_shared_experts_stream.py -q -rA > $O/gpu_tests.log 2>&1 )
note "gpu tests rc=$? $(tail -1 $O/gpu_tests.log)"

bitwise() {  # <old tree> <new tree> <tag>
  local R=$O/bitwise_$3; rm -rf $R; mkdir -p $R
  for sc in "2 1 1 hot none" "2 2 2 hot none" "2 2 2 hot selective" "2 2 2 hot full" "2 2 2 hot region" "4 2 2 hot2 none"; do
    set -- $sc; local n=$1 name="n$1_l$2_m$3_$4_$5"
    for side in a b; do
      local T=$M/w/mep_$OLDT; [ $side = b ] && T=$M/w/mep_$NEWT
      PYTHONPATH=$T CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) timeout 900 torchrun --nproc_per_node=$n \
        --master_port=$((29600 + RANDOM % 400)) $K/bitwise_dump.py --out $R/$side/$name --layers $2 --mbs $3 \
        --routing $4 --ac $5 > $R/$side.$name.log 2>&1
      note "bitwise $TAGB $name $side rc=$?"
    done
    python $K/bitwise_cmp.py $R/a/$name $R/b/$name > $R/cmp.$name.txt 2>&1
    python $K/ref_err.py $R/a/$name $R/b/$name > $R/ref.$name.txt 2>&1
    note "bitwise $TAGB $name: $(tail -1 $R/cmp.$name.txt)"
  done
}
OLDT=old NEWT=c1 TAGB=old_c1 bitwise old c1 old_c1
OLDT=c3 NEWT=c4 TAGB=c3_c4 bitwise c3 c4 c3_c4
OLDT=old NEWT=c2 TAGB=old_c2 bitwise old c2 old_c2
OLDT=c4 NEWT=c5 TAGB=c4_c5 bitwise c4 c5 c4_c5

mb() {  # <tree> <tag> <args...>
  local t=$1 tag=$2; shift 2
  ( cd $M/w/mep_$t && PYTHONPATH=. timeout 1800 torchrun --nproc_per_node=4 --master_port=$((29600 + RANDOM % 400)) \
      $K/microbench.py --out $O/mb_${tag}.json "$@" > $O/mb_${tag}.log 2>&1 )
  note "mb $tag rc=$? $(python3 -c "import json;d=json.load(open('$O/mb_${tag}.json'));print('ms',[round(x,2) for x in d['median_ms_per_rank']],'GiB',[round(x,2) for x in d['peak_gib_per_rank']])" 2>/dev/null)"
}
for routing in uniform skew; do
  for t in old c1 c2 c5; do mb $t experts_${routing}_$t --mode experts --routing $routing; done
done
mb old moe_old --mode moe
mb c5 moe_c5 --mode moe
mb c5 moe_c5_stream --mode moe --stream

cell() {  # <tree> <name> <config> <cache> <steps>
  local T=$M/w/mep_${1%x} name=$2 cfg=$3 cache=$4 D=$O/$2; rm -rf $D; mkdir -p $D
  ( cd $T && PROBE_STEPS=$5 PROBE_DET=1 PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    NGPU=4 LOG_RANK=0,1,2,3 MODULE=moonep_probe_1003 CONFIG=$cfg timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$name rc=$rc"
}
for t in old c5 c5x; do
  cells="standard moonep moonep_full"; [ $t = c5 ] && cells="moonep moonep_full moonep_shared_stream standard_shared_stream"
  [ $t = c5x ] && cells="standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp"
  C0=$O/cache_$t; rm -rf $C0; mkdir -p $C0
  for c in $cells; do cell $t warm_${t}_$c ${c}_cell $C0 1; done
  for c in $cells; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell $t num_${t}_$c ${c}_cell $O/cache_c 20; done
  [ $t = old ] && { rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell old num_old_standard_b standard_cell $O/cache_c 20; }
  rm -rf $O/cache_c $C0
done
note "all done"
echo done > $O/DONE
