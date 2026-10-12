"""For each debug model, the single linears whose weight placement is not dense S(0) / S(1): what main's assert rejects."""
import collections, importlib, inspect, re
import torch, spmd_types as spmd
from torchtitan.config import ParallelismConfig
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear

ok = (dense_param_placement(tp=spmd.S(0)), dense_param_placement(tp=spmd.S(1)))
for name, fn in [("deepseek_v3", "deepseek_v3_debugmodel"), ("deepseek_v4", "deepseek_v4_debugmodel"), ("gpt_oss", "gpt_oss_debugmodel"),
                 ("qwen3_5", "qwen35_debugmodel"), ("kimi_k2_7", "kimi_k2_5_debugmodel"), ("kimi_k3", "kimi_k3_debugmodel")]:
    m = importlib.import_module(f"torchtitan_recipes.tests.models.{name}")
    f = getattr(m, fn)
    kw = {"parallelism": ParallelismConfig(enable_sequence_parallel=True)} if "parallelism" in inspect.signature(f).parameters else {}
    with torch.device("meta"):
        model = f(**kw).model.build()
    seen = collections.Counter()
    for fqn, mod in model.named_modules():
        if type(mod) is Linear or (isinstance(mod, Linear) and getattr(mod, "num_linears", 1) == 1):
            sc = getattr(mod, "_sharding_config", None)
            w = sc.state_shardings.get("weight") if sc else None
            if w is not None and w not in ok and getattr(mod, "num_linears", 1) == 1:
                key = re.sub(r"\.\d+\.", ".N.", fqn)
                seen[(key, str(w.local_type))] += 1
    print(f"== {name}")
    for (k, t), n in sorted(seen.items()):
        print(f"   {n:3d} x {k:55s} {t}")
