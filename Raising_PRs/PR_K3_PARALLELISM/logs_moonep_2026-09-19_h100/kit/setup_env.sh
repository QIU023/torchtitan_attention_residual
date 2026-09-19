set -x
cd /workspace
export PATH=/root/miniconda3/bin:$PATH
conda create -y -n k3 python=3.12 > /workspace/conda_create.log 2>&1; echo "CONDA_RC=$?"
source /root/miniconda3/bin/activate k3
pip install --no-cache-dir "torch==2.15.0.dev20260907+cu126" "torchvision" --index-url https://download.pytorch.org/whl/nightly/cu126 > /workspace/pip_torch.log 2>&1; echo "TORCH_RC=$?"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
git clone -q --branch k3_moonep_seam https://github.com/QIU023/torchtitan.git /workspace/tt_moonep && git -C /workspace/tt_moonep log --oneline -1; echo "TT_RC=$?"
git clone -q https://github.com/MoonshotAI/MoonEP.git /workspace/moonep && git -C /workspace/moonep checkout -q 2bd860b && git -C /workspace/moonep log --oneline -1; echo "MOONEP_RC=$?"
git clone -q https://github.com/QIU023/attention-gym.git /workspace/attn_gym && git -C /workspace/attn_gym checkout -q b19162e && git -C /workspace/attn_gym log --oneline -1; echo "AG_RC=$?"
echo "SETUP_STAGE1_DONE"
