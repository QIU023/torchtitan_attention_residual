#!/bin/bash
# G4: #4764 smoke only (~/w/b4764 = 71e8bfaf2): the RemoteBackend round trip over tcp and nvlink_intra, then
# 4 steps each of no balance, balance alone over nvlink_intra, and planned offload with balance, on the
# 5060 b1 layout (93 layers in blocks of 12, 3 per stage, no AC, dim 1024, seq 512) at pp4.
K=~/k927/kit; R=~/k927/res/balance; V=~/venv_pp/bin
mkdir -p $R
for proto in tcp nvlink_intra; do
  ( cd ~/w/b4764 && PROTO=$proto PYTHONPATH=$K:. timeout 600 $V/torchrun --nproc_per_node=2 --rdzv_backend c10d \
    --rdzv_endpoint=localhost:0 $K/nvlink_intra_test.py > $R/roundtrip_$proto.log 2>&1 ); echo "rc=$?" >> $R/roundtrip_$proto.log
done
export LB_ROOT=$R PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_SEQ=512 PPMEM_LPS=3 PPMEM_DIM=1024 PPMEM_AC=none
bash $K/run_lb4.sh smoke_off ~/w/b4764 $R/cache_off 4
PPMEM_BALANCE=1 PPMEM_REMOTE_PROTOCOL=nvlink_intra bash $K/run_lb4.sh smoke_bal ~/w/b4764 $R/cache_bal 4
PPMEM_CPU_OFFLOAD=planned PPMEM_BALANCE=1 PPMEM_REMOTE_PROTOCOL=nvlink_intra bash $K/run_lb4.sh smoke_plan ~/w/b4764 $R/cache_plan 4
rm -rf $R/cache*
echo done > $R/G4_DONE
