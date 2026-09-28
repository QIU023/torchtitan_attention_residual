#!/bin/bash
# G5: MoonEP (draft #4751, ~/w/moonep = f556ab4fd on main 6c2dadbb3 of 09-18) in venv_k3 (torch 0921, MoonEP public release 33327eb).
K=~/k927/kit; R=~/k927/res/moonep; W=~/w/moonep; V=~/venv_k3/bin
mkdir -p $R
( cd $W && git apply --check ~/kda_guard_lift.patch && git apply ~/kda_guard_lift.patch ) > $R/guard.txt 2>&1; echo "guard rc=$?" >> $R/guard.txt
( cd $W && timeout 1800 $V/python -m pytest tests/unit_tests/gpu/test_kimi_k3_moon_ep.py -q > $R/ondevice.log 2>&1 ); echo "rc=$?" >> $R/ondevice.log
( cd $W && timeout 900 $V/python -m pytest tests/unit_tests/cpu -q -k "moon_ep or ep_token_dispatcher_capacity" > $R/cpu.log 2>&1 ); echo "rc=$?" >> $R/cpu.log
run() {  # <name> <module> <config> <cache> <steps> [extra...]
  local name=$1 module=$2 cfg=$3 cache=$4 steps=$5; shift 5
  local D=$R/$name; rm -rf $D; mkdir -p $D/tmp
  ( cd $W && MEP_SEQ=${MEP_SEQ:-512} TMPDIR=$D/tmp PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module $module --config $cfg --training.steps $steps --metrics.log_freq 1 --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) $name $(cat $D/rc)" >> $R/progress.txt
}
SEED="--debug.seed 42 --debug.deterministic"
# the committed h100 cell, unseeded
run cell torchtitan_recipes.tests.h100 kimi_k3_moonep_fsdp4_ep4 $R/cache_cell 10
# numerics: standard dispatcher against MoonEP on one warm cache, and each rerun
C0=$R/cache0; mkdir -p $C0
run warm_std moonep_ctrl std_fsdp4_ep4 $C0 1 $SEED
run warm_mep moonep_ctrl moonep_fsdp4_ep4 $C0 1 $SEED
for c in std mep std2 mep2; do rm -rf $R/cache_$c; cp -r $C0 $R/cache_$c; done
run std moonep_ctrl std_fsdp4_ep4 $R/cache_std 10 $SEED
run mep moonep_ctrl moonep_fsdp4_ep4 $R/cache_mep 10 $SEED
run std2 moonep_ctrl std_fsdp4_ep4 $R/cache_std2 10 $SEED
run mep2 moonep_ctrl moonep_fsdp4_ep4 $R/cache_mep2 10 $SEED
# DeepEP v2, NVLink only, if it built into venv_k3
if $V/python -c "from deep_ep import ElasticBuffer" > /dev/null 2>&1; then
  DL=$($V/python -c "import os, nvidia.nvshmem as a, nvidia.nccl as b; print(os.path.join(a.__path__[0], 'lib') + ':' + os.path.join(b.__path__[0], 'lib'))")
  export EP_DISABLE_GIN=1 EP_REUSE_NCCL_COMM=0 NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_DISABLE_MNNVL=1 NCCL_NVLS_ENABLE=0 LD_LIBRARY_PATH=$DL:${LD_LIBRARY_PATH:-}
  export MEP_SEQ=512
  D0=$R/cache_d0; mkdir -p $D0
  run warm_dep_std moonep_ctrl std_fsdp4_ep4 $D0 1 $SEED
  run warm_dep moonep_ctrl deepep_fsdp4_ep4 $D0 1 $SEED
  for c in dstd dep dep2; do rm -rf $R/cache_$c; cp -r $D0 $R/cache_$c; done
  run dstd moonep_ctrl std_fsdp4_ep4 $R/cache_dstd 10 $SEED
  run dep moonep_ctrl deepep_fsdp4_ep4 $R/cache_dep 10 $SEED
  run dep2 moonep_ctrl deepep_fsdp4_ep4 $R/cache_dep2 10 $SEED
else
  echo "deep_ep not importable in venv_k3; DeepEP cells skipped" > $R/deepep_skipped.txt
fi
( cd $W && git checkout -- . ) ; git -C $W status --short > $R/worktree_after.txt
rm -rf $R/cache*
echo done > $R/G5_DONE
