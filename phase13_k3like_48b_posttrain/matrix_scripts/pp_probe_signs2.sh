#!/bin/bash
# Sign census, second set: dp1 against two bug-free re-orderings with no pipeline in them, an FSDP dp2
# cell and a dp1 cell with 512-token micro-batches (a different accumulation order). dp1 on two
# fresh caches came out bitwise identical, so the control has to re-order the sums some other way.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
MS=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
T=/tmp/wt_ppprobe8; SEED=/workspace/.mx3_seeds_main30/kimi_k3_debugmodel_49ffe72a4588/seed_ckpt
OUT=/workspace/ppprobe8b_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
run(){ local nm=$1 np=$2 cache=$3; shift 3
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  ( source /venv/main/bin/activate && cd $T && PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$cache TRITON_CACHE_DIR=$OUT/tri_$cache \
    GRAD_TENSOR_DUMP=$DUMP/$nm timeout 2400 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 10 $B --checkpoint.enable --checkpoint.interval 100000 "$@" \
    --dump-folder $d > $OUT/$nm.log 2>&1 ); rm -rf $d/checkpoint
  printf "%-10s " $nm; sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -oE "step: +[0-9]+ +loss: +[0-9.]+ +grad_norm: +[0-9.]+" | awk '{printf "%s:%s/%s ", $2, $4, $6}'; echo
}
run dp1_C 1 C $D
run dp2 2 F --parallelism.data_parallel_shard_degree 2
run dp1_mb512 1 G $D --training.num-tokens-per-microbatch-per-dp-rank 512
source /venv/main/bin/activate
python $MS/pp_step10_census.py "dp1 vs dp2 (FSDP re-partition, no pipeline)" $DUMP/dp1_C $DUMP/dp2
python $MS/pp_step10_census.py "dp1 vs dp1 with 512-token micro-batches (accumulation order)" $DUMP/dp1_C $DUMP/dp1_mb512
rm -rf $DUMP $OUT/ind_* $OUT/tri_*
echo "PPPROBE8B DONE $OUT"
