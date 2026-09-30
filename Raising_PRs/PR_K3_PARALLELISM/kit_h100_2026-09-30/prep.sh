#!/bin/bash
# 09-30 box: everything that does not need the source-built torch. The titan fork with one worktree per head
# (DEP = k3_pp_mm d27839459, MoonEP = moonep_review1 ab191a771), and the DEP ratio dataset rebuilt from
# jackyhate/text-to-image-2M's data_1024_10K tar; its shards must hash as in the 09-29 MANIFEST.json.
set -x
M=~/mep; mkdir -p $M/w ~/dep_data; cd $M
[ -d tt ] || git clone -q https://github.com/QIU023/torchtitan.git tt
cd tt
git remote add upstream https://github.com/pytorch/torchtitan.git 2>/dev/null
git fetch -q upstream main
git fetch -q origin k3_pp_mm moonep_review1
for x in dep:d27839459 moonep:ab191a771; do
  n=${x%%:*}; c=${x#*:}
  [ -d $M/w/$n ] || git worktree add -q --detach $M/w/$n $c
  echo "worktree $n $(git -C $M/w/$n rev-parse --short HEAD)"
done
cd $M
[ -d venv_data ] || uv venv venv_data --python 3.12
. venv_data/bin/activate
uv pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("jackyhate/text-to-image-2M", repo_type="dataset",
                      allow_patterns=["data_1024_10K/*"], local_dir="/root/mep/t2i_src")
print("downloaded to", p)
PY
ls -la $M/t2i_src/data_1024_10K/
SRC=$(ls $M/t2i_src/data_1024_10K/*.tar | head -1)
python ~/kit/overnight/dep_ratio/build_dataset.py $SRC ~/dep_data/t2i1024_k4
python - <<'PY'
import json
new = json.load(open("/root/dep_data/t2i1024_k4/MANIFEST.json"))
ref = json.load(open("/root/kit/kit_h100_2026-09-30/MANIFEST_09-29.json"))
same = new["shards"] == ref["shards"]
print("MANIFEST identical to 09-29:", same, {k: v for k, v in new.items() if k != "shards"})
PY
echo PREP_DONE
