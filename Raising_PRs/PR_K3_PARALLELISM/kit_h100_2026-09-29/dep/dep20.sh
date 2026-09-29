#!/bin/bash
# After dep4.sh: the numerics cells again at 20 steps (the debug set's standard columns 1 / 10 / 20),
# DEP off twice, K2.5, bubble, each on a copy of the numerics warm cache that dep4.sh used (kept aside
# as ~/mep/dep_cache0_keep before dep4.sh removed it). No dumps: steps 1 and 2 were dumped by dep4.sh.
M=~/mep; K=~/kit/dep; O=$M/results/dep; P=$O/progress.txt
. $M/venv_src/bin/activate; W=$M/w/dep
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $O/DONE ]; do sleep 20; done
note "dep20: 20-step numerics on the kept warm cache"
for x in off_a20:dep_off off_b20:dep_off k25_20:dep_k25 bubble20:dep_bubble; do
  n=${x%%:*}; c=${x#*:}; D=$O/$n; rm -rf $D $O/cache_c; mkdir -p $D; cp -r $M/dep_cache0_keep $O/cache_c
  ( cd $W && PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$O/cache_c/ic TRITON_CACHE_DIR=$O/cache_c/tc DEPN_MEM_OUT=$D/mem \
    timeout 2400 torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep4 --config $c --training.steps 20 --metrics.log-freq 1 \
    --debug.seed 42 --debug.deterministic --dump-folder $D/out > $D/run.log 2>&1 )
  note "$n rc=$?"; rm -rf $D/out
done
rm -rf $O/cache_c
python $K/tab_dep20.py $O > $O/tables_20.md 2>&1
note "dep20 done"; echo done > $O/DONE20
