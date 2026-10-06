"""H100 probe shim, logbook kit only: torch 68e0ae4's MixedPrecisionPolicy predates param_dtype_override_fn, which titan
main (#5068) always passes. Accept it when None, the K3 cells' value, and refuse anything else."""

import sys

import torch.distributed.fsdp as _fsdp

_Policy = _fsdp.MixedPrecisionPolicy
if "param_dtype_override_fn" not in getattr(_Policy, "__dataclass_fields__", {}):

    def MixedPrecisionPolicy(*args, param_dtype_override_fn=None, **kwargs):
        if param_dtype_override_fn is not None:
            raise RuntimeError("mpp_shim: param_dtype_override_fn needs a newer torch")
        return _Policy(*args, **kwargs)

    _fsdp.MixedPrecisionPolicy = MixedPrecisionPolicy
    sys.stderr.write("mpp_shim: MixedPrecisionPolicy takes param_dtype_override_fn=None\n")
