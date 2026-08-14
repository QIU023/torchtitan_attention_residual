"""Is the Ulysses MLA forward equal to the single-rank forward?

This is the CP correctness check. The obvious end-to-end version -- train dp1,
train cp2, compare losses -- does NOT work and was tried: our own Ulysses, the
one whose docstring says it was validated bit-exact against a single-rank
reference, shows a LARGER step-1 discrepancy (8.6e-3) under that comparison than
the port being tested (5.1e-4). dp1 and cp2 are not the same computation at the
trainer level: CP reorders the sequence for head-tail load balancing and the
batch composition differs. The end-to-end comparison judges nothing.

So compare the module directly. Every rank builds the same module from the same
seed, the full input is generated identically everywhere, and then:

  reference  the module's ORIGINAL forward on the full [B, T, D]
  ulysses    the CP forward on this rank's [B, T/cp, D] shard

Rank r's output must equal reference[:, r*T/cp : (r+1)*T/cp]. Nothing else in
the trainer participates, so a failure is the implementation.

Note the probe this replaces did not survive: `kda_ulysses_cp_probe` is
referenced in a model.py comment and exists nowhere in the tree, so the
bit-exactness claim in those docstrings cannot currently be re-run.

    torchrun --nproc_per_node=2 ulysses_mla_parity.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.models.kimi_k3_up import _mla_config
from torchtitan.models.kimi_k3_up.cp_ulysses import apply_ulysses_cp

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)
mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))

DIM, HEADS, B, T = 512, 4, 2, 64
assert T % world == 0 and HEADS % world == 0

# Same seed on every rank, so every rank builds identical weights and the same
# input. Without this the comparison measures initialisation, not CP.
torch.manual_seed(0)
cfg = _mla_config(
    dim=DIM,
    num_heads=HEADS,
    q_lora_rank=128,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
)
mla = cfg.build().to(device).to(torch.float32)
for p in mla.parameters():
    torch.nn.init.normal_(p, std=0.02)
torch.manual_seed(1234)
x_BTD = torch.randn(B, T, DIM, device=device, dtype=torch.float32)

with torch.no_grad():
    reference_BTD = mla.forward(x_BTD)

t_loc = T // world
x_local = x_BTD[:, rank * t_loc : (rank + 1) * t_loc].contiguous()

apply_ulysses_cp([mla], mesh)
with torch.no_grad():
    got_BLD = mla.forward(x_local)

want_BLD = reference_BTD[:, rank * t_loc : (rank + 1) * t_loc]
diff = (got_BLD - want_BLD).abs().max().item()
scale = want_BLD.abs().max().item()
print(
    f"[rank{rank}] max abs diff {diff:.3e}  relative {diff / max(scale, 1e-12):.3e}",
    flush=True,
)

ok = torch.tensor([1.0 if diff / max(scale, 1e-12) < 1e-5 else 0.0], device=device)
dist.all_reduce(ok)
if rank == 0:
    print("PARITY PASS" if ok.item() == world else "PARITY FAIL", flush=True)
dist.destroy_process_group()
