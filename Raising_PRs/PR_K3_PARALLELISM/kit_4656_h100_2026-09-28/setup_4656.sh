#!/bin/bash
# New 1 x H100 box (09-28): the fork, worktrees for main and #4656, and ~/venv_pp as on the 09-27 box.
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
[ -d ~/tt ] || git clone -q https://github.com/QIU023/torchtitan.git ~/tt
git -C ~/tt fetch -q origin main attnres_review1
git -C ~/tt fetch -q https://github.com/pytorch/torchtitan.git main:refs/remotes/upstream/main
mkdir -p ~/w
for spec in main:f35966713 c4780:f14d681f4; do
  name=${spec%%:*}; rev=${spec#*:}
  [ -d ~/w/$name ] || git -C ~/tt worktree add -q --detach ~/w/$name $rev
  echo "$name $(git -C ~/w/$name rev-parse --short HEAD) dirty=$(git -C ~/w/$name status --short | wc -l)"
done
[ -d ~/venv_pp ] || uv venv -q --python 3.12 ~/venv_pp
source ~/venv_pp/bin/activate
uv pip install -q --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
cd ~/w/main && uv pip install -q -r .ci/docker/requirements.txt
uv pip install -q -r .ci/docker/requirements-vlm.txt 2>/dev/null || true
uv pip install -q pytest pytest-subtests expecttest
python -c "import torch; print('torch after deps', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
uv pip list 2>/dev/null | grep -iE "cutlass|attn|spmd|remat|^torch |torchvision"
echo SETUP_DONE
