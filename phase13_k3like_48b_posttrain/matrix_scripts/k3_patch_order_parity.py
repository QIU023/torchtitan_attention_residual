"""One synthetic image through the released K3 processor and through the tree's
collator path (process_image + vision_to_patches raster); compare tensors."""
import importlib.util, json, sys
import numpy as np, torch
from PIL import Image

REL = "/root/models/kimi-k3-debug-nt-rel"
sys.path.insert(0, "/tmp/wt_int0916")
from torchtitan.hf_datasets.multimodal.utils.image import (
    process_image, resize_to_navit_patch_grid, vision_to_patches)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

# load the checkpoint's processor files as a package so the relative imports resolve
import transformers
from transformers import AutoImageProcessor
cfg = json.load(open(f"{REL}/preprocessor_config.json"))["media_proc_cfg"]
try:
    proc = AutoImageProcessor.from_pretrained(REL, trust_remote_code=True)
    how = "AutoImageProcessor(" + type(proc).__name__ + ")"
except Exception as e:
    print("AutoImageProcessor failed:", type(e).__name__, str(e)[:300]); raise
media = sys.modules[type(proc).__module__.rsplit(".", 1)[0] + ".media_utils"] if (type(proc).__module__.rsplit(".", 1)[0] + ".media_utils") in sys.modules else None
print("processor:", how, "| media_utils module:", media is not None)

rng = np.random.default_rng(0)
for (W, H) in [(300, 200), (97, 131), (640, 480)]:
    img = Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    # released path: navit_resize_image -> normalize -> navit_patchify
    out = proc.preprocess([{"type": "image", "image": img}], return_tensors="pt")
    theirs, thw = out["pixel_values"], out["grid_thws"]
    # tree path
    mine = process_image(img, patch_size=14, merge_size=2, image_mean=(0.5,)*3, image_std=(0.5,)*3,
                         resize_fn=resize_to_navit_patch_grid, max_patches=65536, max_patches_per_side=512)
    patches, grid = vision_to_patches(mine, 14, 1, 2, patch_order="raster")
    a = theirs.reshape(theirs.shape[0], -1).float(); b = patches.float()
    print(f"{W}x{H}: released {tuple(theirs.shape)} thw={thw.tolist()} via {how}; tree {tuple(patches.shape)} thw={grid.tolist()}; "
          f"shape_eq={a.shape==b.shape} max_abs_diff={(a-b).abs().max().item() if a.shape==b.shape else 'n/a'} "
          f"bitwise={torch.equal(a,b) if a.shape==b.shape else False}")
    if a.shape == b.shape and not torch.equal(a, b):
        # is it a permutation of the sequence (block vs raster)?
        pb, _ = vision_to_patches(mine, 14, 1, 2, patch_order="block")
        print("   block order equal:", torch.equal(a, pb.float()), " max diff vs block:", (a-pb.float()).abs().max().item())
