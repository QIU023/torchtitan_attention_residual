#!/bin/bash
# Is the dp2 x pp2 departure the quantile-balancing hook's cross-rank histogram sum? Same cells without the hook.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; TT=/tmp/wt_pp4312; O=$S/ppmx
for i in $(seq 1 120); do grep -q "OVERNIGHT-DONE\|SANITY FAILED" $S/overnight.log && break; sleep 5; done
source /workspace/venv_bfx9/bin/activate; export PYTHONPATH=$TT GN_FP32=1
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
cell() { local nm=$1 gpus=$2 np=$3 csrc=$4; shift 4; local d=$O/$nm; rm -rf $d $O/ind_$nm $O/tri_$nm; mkdir -p $d; cp -r $O/seed_c4_2048/checkpoint $d/; cp -r $O/ind_$csrc $O/ind_$nm; cp -r $O/tri_$csrc $O/tri_$nm 2>/dev/null
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$O/ind_$nm TRITON_CACHE_DIR=$O/tri_$nm torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $COMMON --config kimi_k3_debugmodel_c4_noqb --training.steps 100 "$@" --dump-folder $d > $O/$nm.log 2>&1 ); local rc=$?; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $O/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $O/$nm.log)"; }
( export NOSYNC_GA=1; cell d2_dp2_ns_noqb 0,1 2 d2_dp2_ns $B8 --parallelism.data_parallel_shard_degree 2 )
cell d2_pp2_noqb 4,5,6,7 4 d2_dp2_ns_noqb $B8 --parallelism.data_parallel_shard_degree 2 $P &
cell d2_dp2_noqb 0,1 2 d2_dp2_ns_noqb $B8 --parallelism.data_parallel_shard_degree 2 &
wait
echo "# noqb: 2048 tokens per step, dp2, no quantile-balancing hook"; python3 $S/tables.py $O d2_dp2_ns_noqb d2_dp2_noqb d2_pp2_noqb
echo NOQB-DONE
