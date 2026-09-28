#!/bin/bash
# 5060 smokes for the PR heads that never ran on a GPU, with settings already on record:
#  A. 1 GPU, the body recipe (debug model, 2048 tokens/step in 512-token micro-batches, 10 steps, seed 42,
#     deterministic), main vs #4656 under none/selective/full on one warm cache; #4881 on its own cache.
#  B. the B200 composition cell (8 GPUs, 10 steps) on #4656 and PR A.
#  C. the PR A table's PP layout (93 layers, block 12, 3 per stage, dim 2048, seq 2048, M16, full AC, pp8):
#     #4656, PR A, #4765 cpu_offload=all, #4764 planned; one warm cache, 6 steps each.
#  D. the b1 balance layout (same, no AC, dim 1024, seq 512): #4764 off, balance alone (tcp), planned + balance.
# torch 2.15.0.dev20260906 needs the local compat shim on the current-main trees; applied here, reverted at the end.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/s5060; V=/workspace/venv_bfx9/bin; SHIM=$S/lbplan/pr5_torch_compat_shim.patch
KL=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_pp_lowerbound_2026-09-26
declare -A T=([main]=$S/wt_s_main [l4656]=$S/wt_s_l4656 [zinit]=$S/wt_s_zinit [pra]=$S/wt_ppopt [o4765]=/tmp/wt_ppoff [b4764]=/tmp/wt_ppbal)
for t in main l4656 pra o4765 b4764; do git -C ${T[$t]} apply $SHIM || { echo "shim failed on $t"; exit 1; }; done
one() {  # <name> <tree> <gpu> <cache> <steps> <ac>
  local D=$R/$1; rm -rf $D; mkdir -p $D
  ( cd ${T[$2]} && CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$4/ic TRITON_CACHE_DIR=$4/tc \
    timeout 1800 $V/torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --training.steps $5 --metrics.log_freq 1 --training.num-tokens-per-train-step 2048 \
    --training.num-tokens-per-microbatch-per-dp-rank 512 --dump-folder $D/out activation-checkpoint:$6 > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) $1 $(cat $D/rc)" >> $R/progress.txt
}
# A
C0=$R/cache_a; mkdir -p $C0
for t in main l4656; do for ac in none selective full; do one warm_${t}_$ac $t 0 $C0 1 $ac; done; done
i=0
for t in main l4656; do for ac in none selective full; do
  rm -rf $R/cache_a_${t}_$ac; cp -r $C0 $R/cache_a_${t}_$ac
  one id_${t}_$ac $t $i $R/cache_a_${t}_$ac 10 $ac & i=$((i + 1))
done; done
mkdir -p $R/cache_z; one zinit_none zinit 6 $R/cache_z 10 none &
wait
# B
for t in l4656 pra; do
  D=$R/compose_$t; rm -rf $D; mkdir -p $D/cache
  ( cd ${T[$t]} && PYTHONPATH=/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
    timeout 1800 $V/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module torchtitan_recipes.tests.b200 --config kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4 \
    --training.steps 10 --dump-folder $D/out > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) compose_$t $(cat $D/rc)" >> $R/progress.txt; rm -rf $D/cache
done
# C and D through the lower-bound kit
pp() {  # <name> <tree> <cache> <steps> [VAR=val ...]
  local name=$1 t=$2 cache=$3 n=$4; shift 4
  env "$@" LB_ROOT=$R bash $KL/run_lb.sh $name ${T[$t]} $cache $n
  echo "$(date +%H:%M:%S) $name $(grep -o 'rc=[0-9]*' $R/$name/train.log | tail -1)" >> $R/progress.txt
}
export PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_LPS=3
C1=$R/cache_c; mkdir -p $C1
for spec in "c_b4656:l4656" "c_pra:pra" "c_o4765all:o4765:PPMEM_CPU_OFFLOAD=all" "c_b4764plan:b4764:PPMEM_CPU_OFFLOAD=planned"; do
  IFS=: read -r name t e <<< "$spec"; pp w$name $t $C1 1 PPMEM_DIM=2048 PPMEM_SEQ=2048 $e
done
for spec in "c_b4656:l4656" "c_pra:pra" "c_o4765all:o4765:PPMEM_CPU_OFFLOAD=all" "c_b4764plan:b4764:PPMEM_CPU_OFFLOAD=planned"; do
  IFS=: read -r name t e <<< "$spec"; rm -rf $R/cache_$name; cp -r $C1 $R/cache_$name
  pp $name $t $R/cache_$name 6 PPMEM_DIM=2048 PPMEM_SEQ=2048 $e
done
C2=$R/cache_d; mkdir -p $C2
for spec in "d_off:b4764" "d_bal:b4764:PPMEM_BALANCE=1" "d_planbal:b4764:PPMEM_CPU_OFFLOAD=planned,PPMEM_BALANCE=1"; do
  IFS=: read -r name t e <<< "$spec"; pp w$name $t $C2 1 PPMEM_DIM=1024 PPMEM_SEQ=512 PPMEM_AC=none ${e//,/ }
done
for spec in "d_off:b4764" "d_bal:b4764:PPMEM_BALANCE=1" "d_planbal:b4764:PPMEM_CPU_OFFLOAD=planned,PPMEM_BALANCE=1"; do
  IFS=: read -r name t e <<< "$spec"; rm -rf $R/cache_$name; cp -r $C2 $R/cache_$name
  pp $name $t $R/cache_$name 6 PPMEM_DIM=1024 PPMEM_SEQ=512 PPMEM_AC=none ${e//,/ }
done
for t in main l4656 pra o4765 b4764; do git -C ${T[$t]} checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; done
for t in main l4656 zinit pra o4765 b4764; do echo "$t $(git -C ${T[$t]} status --short | wc -l)"; done > $R/worktrees_after.txt
rm -rf $R/cache_*
echo done > $R/ALL_DONE
