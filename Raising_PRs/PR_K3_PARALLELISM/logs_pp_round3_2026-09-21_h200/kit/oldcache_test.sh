#!/bin/bash
# The equalized reference on a copy of the FIRST reference's finished cache (ind_dp1_ns_seqsum): separates the
# equalization hack (a code change) from a second cold compile (a cache change). GPU 3, 100 steps.
set -u
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3; nm=dp1_ns_oldcache
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0 GN_FP32=1 NOSYNC_GA=1
d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $OUT/seed_c4_1024/checkpoint $d/
cp -r $OUT/ind_dp1_ns_seqsum $OUT/ind_$nm; cp -r $OUT/tri_dp1_ns $OUT/tri_$nm 2>/dev/null
( cd $TT && CUDA_VISIBLE_DEVICES=3 TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm torchrun --nproc_per_node=1 --master_port=47001 \
    -m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --config kimi_k3_debugmodel_c4 --training.steps 100 \
    --training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1 --dump-folder $d > $OUT/$nm.log 2>&1 )
rc=$?; rm -rf $d/checkpoint
echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $OUT/$nm.log) $(date -u +%T)" >> /workspace/results/pp_r3_run.log
