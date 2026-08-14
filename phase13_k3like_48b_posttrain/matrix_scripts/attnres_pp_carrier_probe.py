"""Does the upstream AttnRes carrier survive a pipeline stage boundary?

Their block threads `block_residual_TND` through its own signature, which is what makes
the carrier declarable where ours (a Python list) is not. But the MODEL's forward does not:

    def forward(self, tokens, ...) -> torch.Tensor:
        block_residual_TND = h_BLD.new_zeros(B * L, 0, D)   # local, rebuilt every call
        for layer in self.layers.values():
            h_BLD, block_residual_TND = layer(h_BLD, block_residual_TND, ...)

torchtitan's PP runs that same forward on each stage with a subset of layers, so stage 1
starts the carrier at N=0 and everything stage 0 committed is gone. Nothing raises: the
attention residual simply restarts per stage, and the loss keeps going down.

This measures it without needing PP. One process, one model, no distribution:

    whole   the model's own forward over all layers
    split   layers[:k] then layers[k:], with the carrier reset between them, which is
            what a two-stage pipeline would do

If the split output matches the whole output, the carrier is not load-bearing across the
cut and PP needs nothing. It does not match, and the gap is the size of the error a
pipelined run would silently absorb.

    python attnres_pp_carrier_probe.py
"""

import torch

from torchtitan.models.kimi_k3_up.config_registry import kimi_k3_up_mini_block_attn_res

torch.manual_seed(0)
cfg = kimi_k3_up_mini_block_attn_res()
model = cfg.model_spec.model.build().to("cuda").to(torch.bfloat16)
model.init_states()
model.eval()

B, L = 1, 128
D = cfg.model_spec.model.dim
torch.manual_seed(1234)
h_BLD = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)

layers = list(model.layers.values())
n = len(layers)
cut = n // 2


def run(layer_list, h, carrier):
    for layer in layer_list:
        h, carrier = layer(h, carrier, None, None)
    return h, carrier


with torch.no_grad():
    whole_h, whole_carrier = run(layers, h_BLD, h_BLD.new_zeros(B * L, 0, D))

    # Stage 0, then stage 1 restarting the carrier -- what PP produces today.
    s0_h, s0_carrier = run(layers[:cut], h_BLD, h_BLD.new_zeros(B * L, 0, D))
    split_h, split_carrier = run(layers[cut:], s0_h, s0_h.new_zeros(B * L, 0, D))

    # And the same cut WITH the carrier handed across, to confirm the cut itself
    # is innocent and it is the reset that costs.
    carried_h, carried_carrier = run(layers[cut:], s0_h, s0_carrier)

print(f"layers={n} cut after {cut}, D={D}")
print(f"carrier N: whole={whole_carrier.shape[1]}  split={split_carrier.shape[1]}  "
      f"carried={carried_carrier.shape[1]}")


def rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    return d / max(a.float().abs().max().item(), 1e-12)


print(f"whole vs split (carrier reset at the cut): {rel(whole_h, split_h):.3e}")
print(f"whole vs carried (carrier handed across):  {rel(whole_h, carried_h):.3e}")

# And the fix: KimiK3PipelineModel threads the carrier through the model
# signature, so two stages built from it must reproduce the unsplit model.
from torchtitan.models.kimi_k3_up.pp_model import KimiK3PipelineModel

pp = KimiK3PipelineModel(cfg.model_spec.model).to("cuda").to(torch.bfloat16)
pp.load_state_dict(model.state_dict())
pp.eval()
pp_layers = list(pp.layers.values())

# No copy.copy on an nn.Module here: it is a SHALLOW copy that shares _modules,
# so assigning .tok_embeddings = None on one "stage" silently rewires all of them.
# Swap pp.layers in place instead and restore it, which touches nothing else.
pp.tok_embeddings = None
all_layers = type(pp.layers)(list(pp.layers.items()))
first_half = type(pp.layers)(list(pp.layers.items())[:cut])
second_half = type(pp.layers)(list(pp.layers.items())[cut:])

with torch.no_grad():
    pp.layers = all_layers
    ref_full = pp.forward(h_BLD)

    # Stage 0: no lm_head, so the tail is skipped and the carrier comes out.
    pp.layers, saved_head = first_half, pp.lm_head
    pp.lm_head = None
    h0, c0 = pp.forward(h_BLD)

    # Stage 1: lm_head restored, so it runs the output residual and the tail.
    pp.layers, pp.lm_head = second_half, saved_head
    pp_out = pp.forward(h0, c0)
    pp.layers = all_layers

print(f"unsplit model vs two PP stages:            {rel(ref_full, pp_out):.3e}")
