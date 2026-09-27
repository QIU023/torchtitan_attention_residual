#!/bin/bash
# dp2 x pp4 vit_dep with even DP ranks text-only: A with the dummy encode at the placement, B without.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/dep_mixed; W=/tmp/wt_depnew
cd $W && git apply $S/lbplan/pr5_torch_compat_shim.patch
run() {  # <name>
  local D=$R/$1; rm -rf $D; mkdir -p $D/cache
  PYTHONPATH=$R:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
  timeout 1200 /workspace/venv_bfx9/bin/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 -m torchtitan.train --module dep_mixed --config dep_dp2_mixed \
    --training.steps 4 --dump-folder $D/out > $D/run.log 2>&1
  echo "rc=$?" > $D/rc
  rm -rf $D/cache
}
run with_dummy
python3 - <<'PY'
p = "/tmp/wt_depnew/torchtitan/models/kimi_k3/pipeline_parallel/vision_dep.py"
s = open(p).read()
old = "            return self._owner.dummy_vision_inputs(self._device)\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "            return None\n"))
PY
run without_dummy
git -C $W checkout -- torchtitan/models/kimi_k3/pipeline_parallel/vision_dep.py torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
git -C $W status --short > $R/worktree_after.txt
echo done > $R/ALL_DONE
