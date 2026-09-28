#!/bin/bash
# 09-27 H100 takeover: worktrees for every tree under test, and venv_pp with a torch nightly new enough for main (no compat shim).
set -x
export PATH=$HOME/.local/bin:$PATH
git -C ~/tt remote -v | head -2
git -C ~/tt fetch -q origin
git -C ~/tt fetch -q https://github.com/pytorch/torchtitan.git main:refs/remotes/upstream/main
mkdir -p ~/w
for spec in main:f35966713 dep:31f372593 l4656:aa6d9fedc c4780:f14d681f4 pra:439bd2088 o4765:61734f376 b4764:71e8bfaf2 moonep:f556ab4fd; do
  name=${spec%%:*}; rev=${spec#*:}
  [ -d ~/w/$name ] || git -C ~/tt worktree add -q --detach ~/w/$name $rev
  echo "$name $(git -C ~/w/$name rev-parse --short HEAD) dirty=$(git -C ~/w/$name status --short | wc -l)"
done
[ -d ~/venv_pp ] || uv venv -q --python 3.12 ~/venv_pp
source ~/venv_pp/bin/activate
uv pip install -q --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
python -c "import torch; print('torch', torch.__version__)"
cd ~/w/main && uv pip install -q -r .ci/docker/requirements.txt
uv pip install -q pytest pytest-subtests expecttest pyflakes mooncake-transfer-engine==0.3.13.post1
uv pip install -q -r ~/w/main/.ci/docker/requirements-vlm.txt 2>/dev/null || true
python -c "import torch; print('torch after deps', torch.__version__)"
P=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")
echo "unshard_lookahead in schedules: $(grep -c unshard_lookahead $P/distributed/pipelining/schedules.py)"
python - <<'PY'
import torch.distributed as dist
cfg = getattr(dist, "config", None)
print("dist.config has pipeline_per_edge_p2p:", hasattr(cfg, "pipeline_per_edge_p2p") if cfg is not None else "no dist.config")
PY
uv pip list 2>/dev/null | grep -iE "cutlass|attn|spmd|remat|mooncake|^torch "
echo SETUP_DONE
