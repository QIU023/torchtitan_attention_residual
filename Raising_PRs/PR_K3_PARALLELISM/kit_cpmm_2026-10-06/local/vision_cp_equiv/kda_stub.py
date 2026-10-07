"""Windows-only: stub the attn-gym modules that need CuTeDSL so torchtitan.models imports on CPU."""
import sys
import types
from unittest import mock


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return mock.MagicMock(name=f"{self.__name__}.{name}")


for name in (
    "attn_gym.linear.short_conv",
    "attn_gym.linear.short_conv.cute",
    "attn_gym.linear.kda.impl.fused",
    "attn_gym.linear.kda.impl.cudnn",
):
    m = _Stub(name)
    m.__path__ = []
    sys.modules[name] = m

import torch.distributed.pipelining as _pp

for _n in ("analyze_pipeline_activation_liveness", "PipelineStageInfo", "PipelineActivationLiveness", "PipelineActivationLivenessAnalysis"):
    if not hasattr(_pp, _n):
        setattr(_pp, _n, mock.MagicMock(name=_n))
