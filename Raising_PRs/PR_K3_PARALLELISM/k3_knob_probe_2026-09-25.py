"""Kimi K3's pipeline entry with layers_per_stage and no split: what reaches pipeline_llm."""
import sys
from types import SimpleNamespace
import torch.nn as nn
from torchtitan.config import ParallelismConfig
import torchtitan.models.kimi_k3.pipeline_parallel as kpp

class Captured(Exception):
    pass

def capture(model, **kwargs):
    raise Captured(kwargs["parallelism"])

kpp.pipeline_llm = capture
model = nn.Module()
for name in ("vision_encoder", "output_res_proj", "output_res_norm"):
    setattr(model, name, nn.Linear(2, 2))
model.pipeline_first_stage_module_fqns = ("vision_encoder",)
model.pipeline_last_stage_module_fqns = ("output_res_proj", "output_res_norm")
user = ParallelismConfig(pipeline_parallel_degree=2, pipeline_parallel_schedule="Interleaved1F1B", pipeline_parallel_layers_per_stage=5)
try:
    kpp.pipeline_kimi_k3(model, parallelism=user, parallel_dims=SimpleNamespace(pp=2), model_config=SimpleNamespace(layers=[None] * 17))
except Captured as c:
    p = c.args[0]
    split = p.pipeline_parallel_module_fqns_per_model_part
    print(f"[{sys.argv[1]}] reached pipeline_llm: {len(split)} stages, copy layers_per_stage={p.pipeline_parallel_layers_per_stage}, user layers_per_stage={user.pipeline_parallel_layers_per_stage}; first {split[0][:2]} last {split[-1][-2:]}")
except ValueError as e:
    print(f"[{sys.argv[1]}] ValueError: {e}")
