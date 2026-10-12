"""Build every recipe module's plain debug model with LoRATransform (all targets) on meta, under SP on and off."""
import importlib, inspect, sys, traceback
import torch
from torchtitan.config import ParallelismConfig
from torchtitan.config.transform import LoRATransform, apply_transforms
from torchtitan.models.common.lora import _LoRAMixin

mods = ["llama3", "deepseek_v3", "deepseek_v4", "qwen3", "qwen3_5", "qwen3_6", "qwen3_8", "gpt_oss", "kimi_k2_7", "kimi_k3", "muse_glimmer"]
for name in mods:
    m = importlib.import_module(f"torchtitan_recipes.tests.models.{name}")
    fns = [f for f, obj in vars(m).items() if callable(obj) and f.endswith("debugmodel") and inspect.isfunction(obj)]
    for fn in fns[:2]:
        for sp in (False, True):
            try:
                kwargs = {}
                params = inspect.signature(getattr(m, fn)).parameters
                if "parallelism" in params:
                    kwargs["parallelism"] = ParallelismConfig(enable_sequence_parallel=sp)
                config = getattr(m, fn)(**kwargs)
                config = apply_transforms(config, [LoRATransform(rank=8, alpha=16.0)])
                with torch.device("meta"):
                    model = config.model.build()
                n = sum(isinstance(x, _LoRAMixin) for x in model.modules())
                print(f"OK   {name}.{fn} sp={sp} lora_modules={n}")
            except Exception as e:
                print(f"FAIL {name}.{fn} sp={sp}: {type(e).__name__}: {str(e)[:200]}")
            if "parallelism" not in params:
                break
