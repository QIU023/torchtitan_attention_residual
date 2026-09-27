#!/bin/bash
# The mixed-image vit_dep cell under fsdp_reshard_after_forward=always, on DEP as it now stands.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/dep_mixed; W=/tmp/wt_depnew
cd $W && git apply $S/lbplan/pr5_torch_compat_shim.patch
D=$R/always; rm -rf $D; mkdir -p $D/cache
PYTHONPATH=$R:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
timeout 1200 /workspace/venv_bfx9/bin/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  --role rank --tee 3 -m torchtitan.train --module dep_mixed --config dep_dp2_mixed \
  --parallelism.fsdp-reshard-after-forward always --training.steps 4 --dump-folder $D/out > $D/run.log 2>&1
echo "rc=$?" > $D/rc
rm -rf $D/cache
git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
echo done > $R/ALWAYS_DONE
