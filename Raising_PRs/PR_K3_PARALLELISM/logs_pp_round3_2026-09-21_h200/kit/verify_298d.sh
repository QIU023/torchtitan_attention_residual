#!/bin/bash
# Verify 298d9589e (the applied split comes back from pipeline_llm) leaves the numerics untouched: the tree moves to that
# commit with only the matrix hacks re-applied, then pp2 x vp2 naive / cached and pp4 x vp4 cached run 100 steps on the
# first lineage caches and are compared with the body's cells (expected bitwise).
set -eu
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3; LOG=/workspace/results/pp_r3_run.log; K=/workspace/kit
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0 GN_FP32=1
cd $TT
git remote -v | head -n 1 >> $LOG
git diff > /workspace/results/hacks_before_298d.patch
git fetch -q origin pp_review4
git checkout -q -- .
git checkout -q 298d9589e
echo "tree at $(git rev-parse --short HEAD), clean=$(git status --short | wc -l)" >> $LOG
python3 $K/probe_apply_h200.py $TT >> $LOG 2>&1
python3 -m py_compile torchtitan/trainer.py torchtitan/training_engine.py torchtitan/distributed/utils.py torchtitan/models/kimi_k3/config_registry.py torchtitan/models/kimi_k3/kda.py torchtitan/models/common/moe.py
echo "hacks re-applied: $(git status --short | wc -l) files dirty" >> $LOG
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
echo "== V: 298d9589e, the body's pp2 x vp2 naive / cached and pp4 x vp4 cached $(date -u +%T)" >> $LOG
cell v298_vp2n 0,1 2 47201 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $IL &
cell v298_vp2c 2,3 2 47202 kimi_k3_debugmodel_c4 $B4 $P2 $IL &
wait
cell v298_pp4vp4c 0,1,2,3 4 47203 kimi_k3_debugmodel_c4_16stages $B4 $P4 $IL
export TABLE_STEPS="1 10 50 100"
{ for pair in "vp2n v298_vp2n" "vp2c v298_vp2c" "pp4vp4c v298_pp4vp4c"; do set -- $pair; python3 $K/tables.py $OUT $1 $2 | tail -n 1 | awk -F"|" -v a=$1 -v b=$2 '{print b" vs "a":", $(NF-2), $(NF-1)}'; done; } >> $LOG 2>&1
echo "V298-DONE $(date -u +%T)" >> $LOG
