"""For the stage class K3 builds when DEP is on, print every method's resolution: the class
that defines it and a fingerprint of its code (bytecode, constants, names; docstrings dropped).

usage: python mro_fingerprint.py <tree> <old|new>
"""

import hashlib
import inspect
import json
import sys

sys.path.insert(0, sys.argv[1])
if sys.argv[2] == "old":
    from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.stage import (
        VisionDepPipelineStage as cls,
    )
else:
    from torchtitan.models.kimi_k3.pipeline_parallel import _VisionDepAttnResStage as cls


def code_key(code):
    consts = [code_key(c) if inspect.iscode(c) else repr(c) for c in code.co_consts]
    if consts and isinstance(code.co_consts[0], str):
        consts = consts[1:]  # docstring
    return hashlib.sha1(
        repr((code.co_code, consts, code.co_names, code.co_varnames, code.co_freevars)).encode()
    ).hexdigest()[:12]


out = {"mro": [k.__module__.split(".")[-1] + "." + k.__name__ for k in cls.__mro__ if k is not cls]}
for name in sorted(dir(cls)):
    attr = inspect.getattr_static(cls, name)
    owner = next(k for k in cls.__mro__ if name in k.__dict__)
    fn = getattr(attr, "__func__", attr)
    if inspect.isfunction(fn):
        out[name] = (owner.__name__ if owner is not cls else "<self>", code_key(fn.__code__))
print(json.dumps(out, sort_keys=True))
