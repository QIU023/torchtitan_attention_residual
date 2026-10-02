# Local-only harness for the Windows box: kimi_k3's model needs CuTeDSL (absent here), and
# torch 2.13 lacks the nightly step(arg_mbs=..., kwarg_mbs=...) interface titan uses.
import os
import sys
import types

_root = os.environ.get("DEP_ROOT")
if _root:
    for _name, _rel in (
        ("torchtitan.models.kimi_k3", "torchtitan/models/kimi_k3"),
        *( [("torchtitan.models.kimi_k3.pipeline_parallel", "torchtitan/models/kimi_k3/pipeline_parallel")] if os.environ.get("DEP_STUB_PP", "1") == "1" else [] ),
    ):
        _mod = types.ModuleType(_name)
        _mod.__path__ = [os.path.join(_root, _rel)]
        sys.modules[_name] = _mod

    def _shim():
        from torch.distributed.pipelining import schedules as S

        original = S.PipelineScheduleMulti.step

        def step(self, *args, arg_mbs=None, kwarg_mbs=None, target_mbs=None, losses=None,
                 return_outputs=True, loss_kwargs=None, target=None, **kwargs):
            if arg_mbs is None and kwarg_mbs is None and target_mbs is None:
                return original(self, *args, target=target, losses=losses,
                                return_outputs=return_outputs, loss_kwargs=loss_kwargs, **kwargs)
            for stage in self._stages:
                stage.has_backward = self._has_backward
            for stage in self._stages:
                stage.clear_runtime_states()
            self._step_microbatches(arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs,
                                    loss_kwargs=loss_kwargs)

        S.PipelineScheduleMulti.step = step

    try:
        _shim()
    except Exception as e:  # noqa
        print("shim failed", e)
