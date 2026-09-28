#!/bin/bash
# G3: the PP cache stack with the PR A table's layout (93 layers in blocks of 12, 3 layers per stage
# = 32 stages, dim 2048, seq 2048, 16 micro-batches, full AC), pp4 x vp8 on 4 x H100:
# main, #4656 (list), #4780 (list + recompute), PR A (list + PR A), all three (e8d0a4aec).
# Then #4765 cpu_offload=all smoked on the same layout, and the composition cell without FSDP.
K=~/k927/kit; R=~/k927/res/pra; V=~/venv_pp/bin
mkdir -p $R
git -C ~/tt fetch -q origin
[ -d ~/w/stack ] || git -C ~/tt worktree add -q --detach ~/w/stack e8d0a4aec
echo "stack $(git -C ~/w/stack rev-parse --short HEAD)" > $R/layout.txt
export LB_ROOT=$R PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_SEQ=2048 PPMEM_LPS=3 PPMEM_DIM=2048
echo "93 layers, block 12, 3 layers per stage, dim 2048, seq 2048, 16 micro-batches, full AC, pp4" >> $R/layout.txt
# fit check on main (the heaviest tree); the name is what queue.sh looks for
bash $K/run_lb4.sh fit_b4656 ~/w/main $R/cache_fit 2
grep -q "rc=0" $R/fit_b4656/train.log || { echo "fit check failed" >> $R/layout.txt; exit 3; }
bash $K/campaign4.sh s7 10 8 "main=$HOME/w/main" "b4656=$HOME/w/l4656" "c4780=$HOME/w/c4780" "pra=$HOME/w/pra" "stack=$HOME/w/stack"
PPMEM_CPU_OFFLOAD=all bash $K/run_lb4.sh smoke_o4765all ~/w/o4765 $R/cache_smoke 4
for t in l4656 c4780 pra; do
  D=$R/compose_$t; rm -rf $D; mkdir -p $D/cache $D/tmp
  ( cd ~/w/$t && TMPDIR=$D/tmp PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
    timeout 2400 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module compose4 --config compose4 --training.steps 10 --dump-folder $D/out > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; rm -rf $D/cache
done
rm -rf $R/cache*
echo done > $R/G3_DONE
