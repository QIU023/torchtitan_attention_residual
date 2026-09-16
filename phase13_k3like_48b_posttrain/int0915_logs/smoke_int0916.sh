#!/bin/bash
# 3-step GPU smokes of the 09-16 integration tree (no seed checkpoint: rc and loss only).
set -uo pipefail
TITAN=/tmp/wt_int0916
OUT=/workspace/.smoke_int0916/$(date +%m%d_%H%M%S); mkdir -p $OUT
R=$OUT/results.txt; echo "tree=$(git -C $TITAN rev-parse --short HEAD)" > $R
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
K=kimi_k3; RB=torchtitan_recipes.tests.b200
run(){ local nm=$1 gpus=$2 mod=$3 cfg=$4; shift 4
  local np=$(( $(echo $gpus | tr -cd , | wc -c) + 1 )) d=$OUT/$nm; mkdir -p $d
  ( source /workspace/venv_bfx9/bin/activate && cd $TITAN && \
    CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$d/triton TORCHINDUCTOR_CACHE_DIR=$d/inductor TORCHINDUCTOR_COMPILE_THREADS=4 \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$TITAN timeout 1800 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module $mod --config $cfg --metrics.log_freq 1 --training.steps 3 --dump-folder $d "$@" > $d/log.txt 2>&1 )
  local rc=$?
  s(){ sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -oE "step: *$1 .*loss: *[0-9.]+ .*grad_norm: *[0-9.]+" | head -1 | grep -oE "loss: *[0-9.]+|grad_norm: *[0-9.]+" | grep -oE "[0-9.]+" | paste -sd/; }
  printf "%-16s rc=%-3s s1=%-18s s2=%-18s s3=%-18s\n" $nm $rc "$(s 1)" "$(s 2)" "$(s 3)" >> $R
  [ $rc -ne 0 ] && grep -m3 -E "Error|error:" $d/log.txt | grep -v lspci | cut -c1-220 | sed 's/^/    /' >> $R
  rm -rf $d/triton $d/inductor
}
D="--parallelism.data_parallel_shard_degree"
# wave 1: single-GPU flavors and dp2 cells
run dp1 0 $K kimi_k3_debugmodel $B $D 1 &
run dp1_compile 1 $K kimi_k3_debugmodel $B $D 1 --compile.enable --compile.components model &
run mtp 2 $K kimi_k3_debugmodel_mtp $B $D 1 &
run lora 3 $K kimi_k3_debugmodel_lora $B $D 1 &
run muon_dp2 4,5 $K kimi_k3_debugmodel_muon $B $D 2 &
run dp2_ep2 6,7 $K kimi_k3_debugmodel $B $D 2 --parallelism.expert_parallel_degree 2 &
wait
# wave 2
run region_ac 0 $K kimi_k3_debugmodel $B $D 1 activation-checkpoint:region --activation-checkpoint.save-regions "*attention.*" &
run mx_qat 1 $K kimi_k3_debugmodel_mx_qat $B $D 1 &
run qlora_mxfp4 2 $K kimi_k3_debugmodel_qlora_mxfp4 $B $D 1 &
run pp2vp2 3,4 $RB kimi_k3_debugmodel_pp2_vp2 &
run cp2_allgather 5,6 $RB kimi_k3_debugmodel_mm_allgather_kv_cp2 &
wait
# wave 3: the b200 TP cell (fsdp2 x tp2 x ep2) with SP on and off, Ulysses CP
run mm_tp_ep 0,1,2,3 $RB kimi_k3_debugmodel_mm &
run mm_tp_ep_nosp 4,5,6,7 $RB kimi_k3_debugmodel_mm --parallelism.no-enable-sequence-parallel &
wait
run cp2_ulysses 0,1 $RB kimi_k3_debugmodel_mm_ulysses_cp2 &
wait
run pp8vp4 0,1,2,3,4,5,6,7 $RB kimi_k3_debugmodel_pp8_vp4
run pp8vp4_vit_dep 0,1,2,3,4,5,6,7 $RB kimi_k3_debugmodel_pp8_vp4_vit_dep
echo "SMOKE DONE $OUT" >> $R; cat $R
