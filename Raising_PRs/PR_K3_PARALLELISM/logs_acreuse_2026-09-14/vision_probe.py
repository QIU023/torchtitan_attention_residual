# Counts the micro-batches that reach Kimi K3's vision tower, then runs torchtitan.train unchanged.
import atexit, runpy, sys
import torchtitan.models.kimi_k3.model as k3
_count = {"microbatches": 0, "with_images": 0, "images": 0}
_orig = k3.KimiK3Model._prepare_multimodal_embeds
def _wrapped(self, tokens, **kwargs):
    _count["microbatches"] += 1
    grid = kwargs.get("grid_thw")
    if kwargs.get("pixel_values") is not None:
        _count["with_images"] += 1
        _count["images"] += 0 if grid is None else int(grid.shape[0])
    return _orig(self, tokens, **kwargs)
k3.KimiK3Model._prepare_multimodal_embeds = _wrapped
atexit.register(lambda: print(f"VISION_PROBE {_count}", flush=True))
sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
