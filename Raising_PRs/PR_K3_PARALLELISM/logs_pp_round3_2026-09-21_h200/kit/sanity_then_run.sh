set -u
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=/workspace/tt_pp NCCL_NVLS_ENABLE=0 GN_FP32=1
O=/workspace/results/sanity2; rm -rf $O; mkdir -p $O; cd /workspace/tt_pp
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.steps 2 --parallelism.data_parallel_shard_degree 1"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
run() { nm=$1 gpus=$2 np=$3 cfg=$4; shift 4; CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$O/ind_$nm TRITON_CACHE_DIR=$O/tri_$nm timeout 900 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $COMMON --config $cfg "$@" --dump-folder $O/$nm > $O/$nm.log 2>&1; echo "$nm rc=$? steps=$(grep -a -c "step: " $O/$nm.log) $(grep -a -m1 -E "Error|Traceback" $O/$nm.log | cut -c1-120)"; }
echo "== sanity $(date -u +%T)"
run dp1 0 1 kimi_k3_debugmodel_c4 &
( export FP32_PROBE=1; run f32 1 1 kimi_k3_debugmodel_c4 --training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32 ) &
run pp2vp4 2,3 2 kimi_k3_debugmodel_c4_8stages --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4 $IL &
wait
run pp4vp4 0,1,2,3 4 kimi_k3_debugmodel_c4_16stages --parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL
echo "== sanity done $(date -u +%T)"
if [ "$(grep -c "rc=0" $O/../sanity2_run.log 2>/dev/null)" -ge 4 ]; then echo "launching the matrix"; bash /workspace/kit/run_pp_h200.sh > /workspace/results/pp_r3_run.log 2>&1; echo "matrix rc=$?"; else echo "SANITY FAILED, matrix not launched"; fi
