#!/bin/bash
# PR A on 4 x H100: the debug model widened to dim 6144, pp4 x vp2, 16 micro-batches x 2048 tokens, FullAC,
# AdamW, seed 42.
# New torch (the 09-28 nightly built from source; since pytorch#196463 it allocates receive buffers just in
# time): PR A as committed fails there, so a 2-step run records how; main 5dc97a3e7 against PR A with its
# receive-buffer overrides removed and its send waits kept (pra_send_only_probe.patch), 10 steps for
# per-rank peaks and bitwise loss, and a timing run traced at step 8.
# Old torch (cu126 09-06 nightly, receive buffers preallocated per micro-batch, with the compat shims):
# dev 1777ad806 against its parent c87a5e102 (logbook only), the #4765 / #4764 smokes, and main against
# PR A as committed (PR A's claim on the torch it was written for). Then the two-GPU nvlink_intra round trip.
M=~/mep; K=~/kit/pra; export LB_ROOT=$M/results/pra; P=$LB_ROOT/progress.txt
NEW=$M/venv_src; OLD=$M/venv
mkdir -p $LB_ROOT
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
export PPMEM_DIM=${PPMEM_DIM:-6144} PPMEM_SEQ=2048
for t in devp:c87a5e102 pra_send:85eefa54b; do
  [ -d $M/w/${t%%:*} ] || git -C $M/tt worktree add -q --detach $M/w/${t%%:*} ${t#*:}
done
git -C $M/w/pra_send diff --quiet && git -C $M/w/pra_send apply $K/pra_send_only_probe.patch
for t in main_5dc9 pra pra_send dev devp o4765 b4764; do
  echo "$t $(git -C $M/w/$t rev-parse --short HEAD) dirty=$(git -C $M/w/$t status --short | wc -l)"
done > $LB_ROOT/trees.txt
note "new torch $($NEW/bin/python -c 'import torch; print(torch.__version__)'), old $($OLD/bin/python -c 'import torch; print(torch.__version__)'), dim $PPMEM_DIM"
export VENV=$NEW
bash $K/run_lb4.sh fit_main $M/w/main_5dc9 $LB_ROOT/cache_fit 2
if ! grep -q "rc=0" $LB_ROOT/fit_main/train.log; then
  if grep -qi "out of memory" $LB_ROOT/fit_main/train.log; then
    export PPMEM_DIM=5120; note "dim 6144 does not fit on main, widening to 5120 instead"
    rm -rf $LB_ROOT/cache_fit; bash $K/run_lb4.sh fit_main_5120 $M/w/main_5dc9 $LB_ROOT/cache_fit 2
    grep -q "rc=0" $LB_ROOT/fit_main_5120/train.log || { note "fit check failed at 5120"; exit 3; }
  else
    note "fit check failed"; exit 3
  fi
fi
note "fit ok at dim $PPMEM_DIM"
LB_LOG_RANK=0,1,2,3 bash $K/run_lb4.sh pra_committed_new_torch $M/w/pra $LB_ROOT/cache_fit 2
note "PR A as committed on the new torch: $(tail -1 $LB_ROOT/pra_committed_new_torch/train.log)"
rm -rf $LB_ROOT/cache_fit
bash $K/campaign4.sh s1 10 8 "main=$M/w/main_5dc9" "pra_send=$M/w/pra_send"; rm -rf $LB_ROOT/cache0 $LB_ROOT/cache_s1_*; note "s1 done"
export VENV=$OLD
for t in devp dev o4765 b4764; do git -C $M/w/$t apply ~/kit/m4656/torch_compat_shim.patch || note "shim failed on $t"; done
for t in main_5dc9 pra; do git -C $M/w/$t apply $K/shim_5dc97a3e7_pp.patch || note "shim failed on $t"; done
bash $K/campaign4.sh s2 10 8 "devp=$M/w/devp" "dev=$M/w/dev"; rm -rf $LB_ROOT/cache0 $LB_ROOT/cache_s2_*; note "s2 done"
C=$LB_ROOT/cache_smoke; rm -rf $C; mkdir -p $C
bash $K/run_lb4.sh smoke_b4764_none $M/w/b4764 $C 6
PPMEM_CPU_OFFLOAD=all bash $K/run_lb4.sh smoke_o4765_all $M/w/o4765 $C 6
PPMEM_CPU_OFFLOAD=planned bash $K/run_lb4.sh smoke_b4764_planned $M/w/b4764 $C 6
PPMEM_BALANCE=1 PPMEM_REMOTE_PROTOCOL=nvlink_intra bash $K/run_lb4.sh smoke_b4764_balance $M/w/b4764 $C 6
PPMEM_CPU_OFFLOAD=planned PPMEM_BALANCE=1 PPMEM_REMOTE_PROTOCOL=nvlink_intra \
  bash $K/run_lb4.sh smoke_b4764_planned_balance $M/w/b4764 $C 6
for s in smoke_b4764_none smoke_o4765_all smoke_b4764_planned smoke_b4764_balance smoke_b4764_planned_balance; do
  note "$s $(tail -1 $LB_ROOT/$s/train.log)"
done
rm -rf $C
bash $K/campaign4.sh s3 10 8 "main_old=$M/w/main_5dc9" "pra_old=$M/w/pra"; rm -rf $LB_ROOT/cache0 $LB_ROOT/cache_s3_*; note "s3 done"
for t in main_5dc9 pra devp dev o4765 b4764; do
  git -C $M/w/$t checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
done
for proto in nvlink_intra tcp; do
  ( cd $M/w/b4764 && PROTO=$proto CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. timeout 600 $NEW/bin/torchrun --nproc_per_node=2 \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 $K/nvlink_intra_test.py > $LB_ROOT/nvlink_$proto.log 2>&1 )
  note "nvlink round trip $proto rc=$?"
done
for step in 3 10; do
  $NEW/bin/python $K/tab_lb.py $LB_ROOT $step s1_mem_main s1_mem_pra_send > $LB_ROOT/tables_s1_step$step.txt 2>&1
  $NEW/bin/python $K/tab_lb.py $LB_ROOT $step s2_mem_devp s2_mem_dev > $LB_ROOT/tables_s2_step$step.txt 2>&1
  $NEW/bin/python $K/tab_lb.py $LB_ROOT $step s3_mem_main_old s3_mem_pra_old > $LB_ROOT/tables_s3_step$step.txt 2>&1
done
$NEW/bin/python $K/balance_table.py $LB_ROOT 5 smoke_b4764_none smoke_o4765_all smoke_b4764_planned smoke_b4764_balance \
  smoke_b4764_planned_balance > $LB_ROOT/tables_smoke.txt 2>&1
for c in s1_time_main s1_time_pra_send s2_time_devp s2_time_dev s3_time_main_old s3_time_pra_old; do
  t=$(find $LB_ROOT/$c/dump -name "rank0_trace.json*" 2>/dev/null | head -1)
  [ -n "$t" ] && $NEW/bin/python $K/analyze_trace.py $(dirname $t) > $LB_ROOT/$c/analysis.txt 2>&1
done
for t in main_5dc9 pra pra_send devp dev o4765 b4764; do
  echo "$t $(git -C $M/w/$t rev-parse --short HEAD) dirty=$(git -C $M/w/$t status --short | wc -l)"
done > $LB_ROOT/trees_after.txt
note "pra done"; echo done > $LB_ROOT/DONE
