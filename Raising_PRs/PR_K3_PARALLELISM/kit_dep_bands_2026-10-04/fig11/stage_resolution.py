"""Dump, for the stage class each tree installs when vision_dep is on, every attribute's resolved owner and source."""
import hashlib, inspect, json, sys
which = sys.argv[1]
if which == "old":
    from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.stage import VisionDepPipelineStage as C
else:
    from torchtitan.models.kimi_k3.pipeline_parallel import _VisionDepAttnResStage as C
out = {"mro": [k.__qualname__ for k in C.__mro__]}
for name in sorted(set(dir(C))):
    owner = next((k for k in C.__mro__ if name in vars(k)), None)
    attr = vars(owner)[name] if owner is not None else None
    fn = getattr(attr, "__func__", attr)
    try:
        src = inspect.getsource(fn)
    except (TypeError, OSError):
        src = repr(type(attr))
    out[name] = [owner.__qualname__ if owner else None, hashlib.sha1(src.encode()).hexdigest()[:12]]
json.dump(out, sys.stdout, indent=0)
