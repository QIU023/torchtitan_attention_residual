"""Logbook probe: the full Kimi-K3 flavor on meta against the official 2.8T index (497,220 keys, packed MXFP4 experts)."""
import json, sys, time
import torch
from torchtitan.config import ParallelismConfig
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

view = sys.argv[1]
t0 = time.time()
config = build_model_config("Kimi-K3", parallelism=ParallelismConfig(enable_sequence_parallel=False))
with torch.device("meta"):
    model = config.build()
sd = model.state_dict()
print(f"model built on meta in {time.time() - t0:.0f}s: {len(sd)} state keys, {sum(v.numel() for v in sd.values()) / 1e12:.3f} T elements", flush=True)
adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
policy = adapter.mxfp4_policy(view)
qat = adapter.qat_weight_fqns(policy)
print(f"policy: {len(policy.weight_fqns)} packed HF weights; QAT selects {len(qat)} titan parameters", flush=True)
hf = adapter.to_hf(sd)
official = set(json.load(open(f"{view}/model.safetensors.index.json"))["weight_map"])
virtual = {k[: -len("_packed")] if k.endswith(".weight_packed") else k for k in official if not k.endswith(".weight_scale")}
packed = {k[: -len("_packed")] for k in official if k.endswith(".weight_packed")}
print(f"official: {len(official)} keys -> {len(virtual)} logical weights ({len(packed)} packed)")
missing, extra = virtual - set(hf), set(hf) - virtual
print(f"to_hf: {len(hf)} keys; official logical weights not produced: {len(missing)}; produced but not in the index: {len(extra)}")
for k in sorted(missing)[:8]: print("  missing", k)
for k in sorted(extra)[:8]: print("  extra  ", k)
back = adapter.from_hf(hf)
print(f"from_hf round trip: {len(back)} keys; equals the model's keys: {set(back) == set(sd)}")
print(f"packed weights QAT does not cover: {len(packed - set(policy.weight_fqns))}; QAT targets not packed: {len(set(policy.weight_fqns) - packed)}")
