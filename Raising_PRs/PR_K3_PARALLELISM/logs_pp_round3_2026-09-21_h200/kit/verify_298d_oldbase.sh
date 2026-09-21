#!/bin/bash
# The fix 298d9589e on the OLD base (42ad4f3c5 on 27951c2f6, whose seed checkpoint and reference exist; the new base's
# #4808 changes the checkpoint layout): apply the commit's diff there, re-apply the matrix hacks, rerun three body cells.
set -eu
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3; LOG=/workspace/results/pp_r3_run.log; K=/workspace/kit
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0 GN_FP32=1
cd $TT
git checkout -q -- .
git checkout -q 42ad4f3c5
git diff 6153cfc00 298d9589e -- torchtitan | git apply --3way 2>>$LOG || { echo "V298O-APPLY-Error" >> $LOG; exit 1; }
echo "tree at 42ad4f3c5 + 298d9589e diff: $(git status --short | wc -l) files changed: $(git status --short | awk '{print $2}' | tr '\n' ' ')" >> $LOG
python3 $K/probe_apply_h200.py $TT >> $LOG 2>&1
python3 -m py_compile torchtitan/trainer.py torchtitan/training_engine.py torchtitan/distributed/pipeline_parallel.py torchtitan/models/kimi_k3/pipeline_parallel/__init__.py torchtitan/models/kimi_k3/config_registry.py
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 100"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"; P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"; P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4"
cell() {  # cell <name> <gpus> <nproc> <port> <config> <flags...>
  local nm=$1 gpus=$2 np=$3 port=$4 cfg=$5; shift 5
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $OUT/seed_c4_1024/checkpoint $d/
  cp -r $OUT/ind_dp1_ns_seqsum $OUT/ind_$nm; cp -r $OUT/tri_dp1 $OUT/tri_$nm
  set +e
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
      torchrun --nproc_per_node=$np --master_port=$port $COMMON --config $cfg "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; set -e; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $OUT/$nm.log) $(date -u +%T)" >> $LOG
}
echo "== VO: 298d9589e on the old base, pp2 x vp2 naive / cached and pp4 x vp4 cached $(date -u +%T)" >> $LOG
cell v298o_vp2n 0,1 2 47301 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $IL &
cell v298o_vp2c 2,3 2 47302 kimi_k3_debugmodel_c4 $B4 $P2 $IL &
wait
cell v298o_pp4vp4c 0,1,2,3 4 47303 kimi_k3_debugmodel_c4_16stages $B4 $P4 $IL
export TABLE_STEPS="1 10 50 100"
{ for pair in "vp2n v298o_vp2n" "vp2c v298o_vp2c" "pp4vp4c v298o_pp4vp4c"; do set -- $pair; python3 $K/tables.py $OUT $1 $2 | tail -n 1 | awk -F"|" -v a=$1 -v b=$2 '{print b" vs "a":", $(NF-2), $(NF-1)}'; done; } >> $LOG 2>&1
echo "V298O-DONE $(date -u +%T)" >> $LOG
