# DEP 在 4312 的 7814d1f8b 上：设了 layers_per_stage 时，vit_dep 推导的切分写进 copy.copy，knob 保留，调用方配置不变；旧的 dataclasses.replace 会触发 ParallelismConfig 的检查。
# 运行：cd <dep_review1 的 checkout> && PYTHONPATH=<attn_gym>:. python dep_knob_probe_2026-09-26.py
import dataclasses
from types import SimpleNamespace

from torchtitan.config import ParallelismConfig
from torchtitan.models.kimi_k3.pipeline_parallel import _with_vit_dep_split
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep import vit_dep_split

model = SimpleNamespace(
    tok_embeddings=object(), layers=object(), norm=object(), lm_head=object(),
    output_res_proj=object(), output_res_norm=object(), vision_encoder=object(),
    pipeline_last_stage_module_fqns=("output_res_proj", "output_res_norm"),
)
# core accepts knob 2 at 29 layers on pp4 (16 stages); 3 and 5 give stage counts pp4 does not divide
for knob in (2, 3, 5):
    cfg = ParallelismConfig(
        pipeline_parallel_degree=4,
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_layers_per_stage=knob,
    )
    kwargs = {
        "parallelism": cfg,
        "parallel_dims": SimpleNamespace(pp=4),
        "model_config": SimpleNamespace(layers=list(range(29))),
    }
    try:
        derived = _with_vit_dep_split(model, kwargs)
    except ValueError as e:
        print(f"knob {knob}: core refuses the layout: {str(e)[:90]}")
        continue
    split = derived.pipeline_parallel_module_fqns_per_model_part
    print(f"knob {knob}: {len(split)} stages, first {split[0]}, copy knob "
          f"{derived.pipeline_parallel_layers_per_stage}, caller split "
          f"{cfg.pipeline_parallel_module_fqns_per_model_part}, caller knob "
          f"{cfg.pipeline_parallel_layers_per_stage}, same object {derived is cfg}")
    try:
        dataclasses.replace(cfg, pipeline_parallel_module_fqns_per_model_part=split)
        print("  dataclasses.replace: accepted")
    except ValueError as e:
        print(f"  dataclasses.replace (the pre-rebase code) raises: {str(e)[:100]}")
