#!/bin/bash
# cp8 reads 12.54963 at step 1 against 12.53972 (cp2) and 12.53932 (cp4). Two checks on the same
# tree and seed: upstream's generic Ulysses kernel at cp8 (does it read the same step 1), and the
# step-1 per-parameter gradients of the packed kernel at cp8 against dp1, cp2 and the generic
# kernel at cp8 (does any parameter group stand out).
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
MS=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
T=/tmp/wt_cprun5; MX=$MS/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel_mm
D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=$T CFG=kimi_k3_debugmodel_cp2_generic BATCH="$B" CELLS="cp8_generic|8|$D 1 $C 8" $MX cp5_cp8_generic
OUT=/workspace/gradprobe_cp8; rm -rf $OUT; mkdir -p $OUT
SEED=$(ls -d /workspace/.mx3_seeds_main/kimi_k3_debugmodel_mm_*/seed_ckpt | head -1)
source /venv/main/bin/activate
cd $T && python $SP/grad_dump_hack_full.py $T && python $SP/grad_hash_hack.py $T
run(){ local nm=$1 cfg=$2 cache=$3
  local d=$OUT/$nm; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  TORCHINDUCTOR_CACHE_DIR=$cache GRAD_DUMP=$OUT/$nm PYTHONPATH=$T timeout 1800 torchrun --nproc_per_node=8 \
    --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module kimi_k3 --config $cfg \
    --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 1 $B \
    --checkpoint.enable --checkpoint.interval 100000 $D 1 $C 8 --dump-folder $d > $OUT/$nm.log 2>&1
  echo "$nm rc=$? $(grep -oE 'step: *1 .*loss: *[0-9.]+' $OUT/$nm.log | grep -oE 'loss: *[0-9.]+')"
  rm -rf $d/checkpoint; }
run cp8 kimi_k3_debugmodel_cp2 "$(ls -d /workspace/mx3_cp5_cp8_[0-9]*/inductor | head -1)"
run cp8_generic kimi_k3_debugmodel_cp2_generic "$(ls -d /workspace/mx3_cp5_cp8_generic_*/inductor | head -1)"
cd $T && git checkout -- torchtitan/trainer.py
P5=/workspace/gradprobe_cp5
for pair in "$P5/dp1 $OUT/cp8" "$P5/cp2 $OUT/cp8" "$OUT/cp8 $OUT/cp8_generic" "$P5/dp1 $OUT/cp8_generic"; do
  set -- $pair; printf "%s vs %s: " $(basename $1) $(basename $2); python $MS/cmp_grad_dumps.py $1.rank0.step1.txt $2.rank0.step1.txt 2>&1 | head -3 | tr '\n' ' ' | cut -c1-260; echo
done
echo "CP8 CHECK DONE"
