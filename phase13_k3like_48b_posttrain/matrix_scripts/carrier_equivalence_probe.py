"""Is the tensor carrier bitwise equal to the list carrier?

Migrating the AttnRes carrier from a Python list of per-block tensors to one
threaded ``[T, N, D]`` tensor is a change of container, not of arithmetic. So the
gate is not "the matrix still trains" -- it is BITWISE equality, on a single GPU,
before any parallelism is involved. Anything less and the 54-cell matrix cannot
tell a container bug from the numerical drift it is allowed to have elsewhere.

Why the change is worth making: no ``sharding_config`` can reach a value the
model holds in a Python local. That is what blocked the declarative TP migration
-- `ffn_out` became a DTensor while `partial_block` and `plain_stream` stayed
plain, and the residual add died with "aten.add.Tensor got mixed". A tensor in
the forward signature can be declared; a list element cannot.

Run on one GPU, no distribution:

    python carrier_equivalence_probe.py
"""

import torch

from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_report_arch

torch.manual_seed(0)
cfg = kimi_k3_debugmodel_report_arch()
spec = cfg.model_spec.model
model = spec.build().to("cuda")
# Our multimodal wrapper predates the Module protocol's init_states, so seed the
# parameters directly. Values do not matter; both paths see the same ones.
torch.manual_seed(7)
for prm in model.parameters():
    torch.nn.init.normal_(prm, std=0.02)
model.eval()
model = model.language_model if hasattr(model, "language_model") else model

B, L = 1, 64
D = spec.kimi_config.hidden_size
torch.manual_seed(1234)
h = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
model = model.to(torch.bfloat16)

layers = list(model.layers.items())


def run_list_carrier():
    """The current path: list of blocks plus a separate partial."""
    blocks: list[torch.Tensor] = []
    partial = h
    for key, layer in layers:
        is_start = int(key) % model.layers_per_block == 0
        blocks, partial, _ = layer(blocks, partial, is_start, None)
    return blocks, partial


def run_tensor_carrier():
    """The target path, once the block takes and returns [T, N, D]."""
    if not hasattr(layers[0][1], "forward_tensor_carrier"):
        raise SystemExit(
            "block has no forward_tensor_carrier yet -- run this after the change"
        )
    carrier = h.new_zeros(B * L, 0, D)
    x = h
    for key, layer in layers:
        is_start = int(key) % model.layers_per_block == 0
        x, carrier = layer.forward_tensor_carrier(x, carrier, is_start)
    return carrier, x


with torch.no_grad():
    blocks_ref, partial_ref = run_list_carrier()
    carrier_got, x_got = run_tensor_carrier()

# The list holds [B, L, D] per block; the tensor holds them as columns of
# [B*L, N, D]. Same values, different container.
ref_carrier = (
    torch.stack([b.reshape(-1, D) for b in blocks_ref], dim=1)
    if blocks_ref
    else h.new_zeros(B * L, 0, D)
)

print(f"blocks committed: list={len(blocks_ref)}  tensor N={carrier_got.shape[1]}")
same_carrier = torch.equal(ref_carrier, carrier_got)
same_partial = torch.equal(partial_ref, x_got)
print(f"carrier bitwise equal: {same_carrier}")
print(f"partial/hidden bitwise equal: {same_partial}")
print("EQUIVALENCE PASS" if same_carrier and same_partial else "EQUIVALENCE FAIL")
