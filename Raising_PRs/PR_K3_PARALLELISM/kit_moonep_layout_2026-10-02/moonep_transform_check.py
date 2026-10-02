"""MoonEP transform on the DeepSeek V3 debug config (Kimi K3 cannot import on this box):
the K3-only refusal, the routed-experts conversion, and the built module's class."""
import copy
import sys

sys.path.append(sys.argv[1])
import torch

import types

_stub = types.ModuleType("torchtitan.models.kimi_k3.model")


class _KimiK3Model:
    class Config:
        pass


_stub.KimiK3Model = _KimiK3Model
sys.modules["torchtitan.models.kimi_k3.model"] = _stub

from torchtitan.config.transform import apply_transforms, TokenDispatcherTransform
from torchtitan.config.transform.apply import transform_model_config_
from torchtitan.config.transform.base import ModelConfigTransformContext
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher

try:
    from torchtitan.distributed.moonep.experts import MoonEPRoutedExperts
except ImportError:
    from torchtitan.models.common.moe import MoonEPRoutedExperts
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel

T = TokenDispatcherTransform(dispatcher=MoonEPTokenDispatcher, routed_experts=MoonEPRoutedExperts)
config = deepseek_v3_debugmodel()
config.parallelism.expert_parallel_degree = 2
try:
    apply_transforms(config, [T])
    print("refusal: NOT REFUSED")
except ValueError as e:
    print("refusal:", str(e).split(": ", 1)[-1][:160])
ctx = ModelConfigTransformContext(training=config.training, parallelism=config.parallelism)
model = transform_model_config_(copy.deepcopy(config.model), [T], context=ctx)
experts = [re for _, re, _, _ in model.traverse(RoutedExperts.Config)]
print("routed experts:", len(experts),
      "all MoonEP experts:", all(isinstance(e, MoonEPRoutedExperts.Config) for e in experts),
      "all MoonEP dispatchers:", all(isinstance(e.token_dispatcher, MoonEPTokenDispatcher.Config) for e in experts),
      "tokens per rank:", sorted({e.token_dispatcher.num_max_tokens_per_rank for e in experts}))
with torch.device("meta"):
    module = experts[0].build()
print("built:", type(module).__module__ + "." + type(module).__name__,
      "| moonep lib imported:", "moonep" in sys.modules)
