# Switch /workspace/tt_moonep to the rebased head from the bundle and reapply the three local patches.
set -e; source /workspace/kit/h200/env.sh; cd /workspace/tt_moonep
git checkout -q -- . && git fetch -q /workspace/kit/h200/moonep_rb_full.bundle HEAD && git checkout -q FETCH_HEAD
git apply /workspace/kit/0002-kimi_k3-moonep-debug-flavor-a706a881d.patch && git apply /workspace/kit/h200/0003-kimi_k3-c4-debug-flavors.patch && git apply /workspace/kit/kda_sm120_guard_lift.patch
echo "tree now: $(git rev-parse --short HEAD) $(git log --oneline -1 | cut -c11-70) | dirty: $(git status --porcelain | tr "\n" ";")"
python -c "from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_moonep, kimi_k3_debugmodel_moonep_c4; print(\"flavors ok\")" 2>&1 | grep -v "jax profiler" | tail -1
