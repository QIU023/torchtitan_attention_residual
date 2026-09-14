import os, sys
from torchtitan.models.kimi_k3 import model_registry


def bound(config):
    return len({
        (l.attention is not None, l.layer_id % l.attn_res_block_size == 0,
         min(-(-l.layer_id // l.attn_res_block_size), 2))
        for l in config.layers
    }) + 2


flavor = sys.argv[1]
c = model_registry(flavor).model
print(f"{sys.argv[2]}: layers={len(c.layers)} MLA={sorted(l.layer_id for l in c.layers if l.attention is not None)[-4:]} bound={bound(c)} measured={sys.argv[3]}")
