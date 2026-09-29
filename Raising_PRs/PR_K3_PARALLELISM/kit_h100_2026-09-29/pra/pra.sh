#!/bin/bash
# PR A on 4 x H100: the debug model widened to dim 6144, pp4 x vp2, 16 micro-batches x 2048 tokens, FullAC,
# AdamW, seed 42. Fit check on main, then main 5dc97a3e7 against PR A 85eefa54b (10 steps: per-rank peaks,
# bitwise; a timing run traced at step 8), dev 1777ad806 against its parent c87a5e102 (logbook only), the
# #4765 / #4764 smokes on the same layout (6 steps), and the two-GPU nvlink_intra round trip.
M=~/mep; K=~/kit/pra; export LB_ROOT=$M/results/pra VENV=${VENV:-$M/venv_src}; P=$LB_ROOT/progress.txt
mkdir -p $LB_ROOT
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
export PPMEM_DIM=${PPMEM_DIM:-6144} PPMEM_SEQ=2048
[ -d $M/w/devp ] || git -C $M/tt worktree add -q --detach $M/w/devp c87a5e102
for t in main_5dc9 pra dev devp o4765 b4764; do
  echo "$t $(git -C $M/w/$t rev-parse --short HEAD) dirty=$(git -C $M/w/$t status --short | wc -l)"
done > $LB_ROOT/trees.txt
note "torch $($VENV/bin/python -c 'import torch; print(torch.__version__)') dim $PPMEM_DIM"
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
rm -rf $LB_ROOT/cache_fit; note "fit ok at dim $PPMEM_DIM"
bash $K/campaign4.sh s1 10 8 "main=$M/w/main_5dc9" "pra=$M/w/pra"; rm -rf $LB_ROOT/cache0 $LB_ROOT/cache_s1_*; note "s1 done"
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
for proto in nvlink_intra tcp; do
  ( cd $M/w/b4764 && PROTO=$proto CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. timeout 600 $VENV/bin/torchrun --nproc_per_node=2 \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 $K/nvlink_intra_test.py > $LB_ROOT/nvlink_$proto.log 2>&1 )
  note "nvlink round trip $proto rc=$?"
done
for step in 3 10; do
  $VENV/bin/python $K/tab_lb.py $LB_ROOT $step s1_mem_main s1_mem_pra > $LB_ROOT/tables_s1_step$step.txt 2>&1
  $VENV/bin/python $K/tab_lb.py $LB_ROOT $step s2_mem_devp s2_mem_dev > $LB_ROOT/tables_s2_step$step.txt 2>&1
done
$VENV/bin/python $K/balance_table.py $LB_ROOT 5 smoke_b4764_none smoke_o4765_all smoke_b4764_planned smoke_b4764_balance \
  smoke_b4764_planned_balance > $LB_ROOT/tables_smoke.txt 2>&1
for c in s1_time_main s1_time_pra s2_time_devp s2_time_dev; do
  t=$(find $LB_ROOT/$c/dump -name "rank0_trace.json*" 2>/dev/null | head -1)
  [ -n "$t" ] && $VENV/bin/python $K/analyze_trace.py $(dirname $t) > $LB_ROOT/$c/analysis.txt 2>&1
done
note "pra done"; echo done > $LB_ROOT/DONE
