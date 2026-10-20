#!/bin/bash
# DEP ratio dataset on the 10-19 box: the first data_1024_10K tar of jackyhate/text-to-image-2M, rebuilt into
# /workspace/dep_data/t2i1024_k4; its shards must hash as in the 09-29 MANIFEST.
exec > /workspace/prep.log 2>&1
set -x
export UV_CACHE_DIR=/workspace/.uvcache
uv venv --python 3.12 /workspace/venv_data && . /workspace/venv_data/bin/activate && uv pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("jackyhate/text-to-image-2M", "data_1024_10K/data_000000.tar", repo_type="dataset", local_dir="/workspace/t2i_src")
print("downloaded", p)
PY
python /workspace/kit_dep/build_dataset.py /workspace/t2i_src/data_1024_10K/data_000000.tar /workspace/dep_data/t2i1024_k4
python - <<'PY'
import json
new = json.load(open("/workspace/dep_data/t2i1024_k4/MANIFEST.json"))
ref = json.load(open("/workspace/kit_dep/MANIFEST_09-29.json"))
print("MANIFEST identical to 09-29:", new["shards"] == ref["shards"], {k: v for k, v in new.items() if k != "shards"})
PY
df -h /workspace | tail -1
echo PREP_DONE
