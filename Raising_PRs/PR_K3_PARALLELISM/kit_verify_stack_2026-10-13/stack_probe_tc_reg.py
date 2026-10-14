"""Logbook probe: the type-checked MX QAT cells with Elfie's torchao Functions registered as local for SPMD type checking."""

import spmd_types as spmd

import stack_probe_1012 as base


def _register():
    from torchao.prototype.qat import mx

    for fn in (mx._MXFakeQuantize, mx._MXQATNativeGroupedMM, mx._MXQuantizedForwardFakeQuantizedBackward):
        spmd.register_local_autograd_function(fn)


def mm_mxqat_reg():
    _register()
    return base.mm_mxqat()


def mm_muon_mxqat_reg():
    _register()
    return base.mm_muon_mxqat()
