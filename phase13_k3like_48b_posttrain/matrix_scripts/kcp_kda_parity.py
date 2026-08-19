"""Is the KCP KDA forward equal to the single-rank forward?

Same judge as ``ulysses_mla_parity.py`` and for the same reason: comparing dp1
and cp2 training losses does not measure this. Every rank builds the same module
# NOTE (2026-08-19): this probe imports the VENDORED upstream K3 tree, which has been
# deleted -- torchtitan/models/kimi_k3_up/ existed to be diffed against, and the pr4025
# git remote does that without a copy that drifts. Two of the four probes that used it
# were already broken by the 2026-08-15 rollback that stripped our work out of the
# vendored tree, and nothing noticed.
#
# To run this again, restore the tree from history:
#     git -C torchtitan checkout 0cadf15e0 -- torchtitan/models/kimi_k3_up
# and re-add "kimi_k3_up" to the registry list in torchtitan/models/__init__.py.
# Prefer pinning to their CURRENT head instead, since the vendored copy was already
# three reuse commits behind by the time it was deleted:
#     git -C torchtitan show pr4025/agent/add-kimi-k3-reference-model:<path>

from the same seed, generates the same full input, and then

  reference  the module's ORIGINAL forward on the full [B, T, D]
  kcp        the CP forward on this rank's [B, T/cp, D] shard

Rank r's output must equal reference[:, r*T/cp : (r+1)*T/cp].

Two things this has to catch that the MLA probe did not have to:

* ``chunk_kda``'s signature does not name ``A_log`` or ``dt_bias`` -- they
  arrive through ``**kwargs``. A misspelled keyword is therefore silently
  dropped instead of raising, and the run still produces plausible numbers.
* the short convolution's halo. Without the exchange each rank's first W-1
  outputs are computed against zero padding, which is a small localised error
  that a loss curve absorbs. Ranks above 0 are where it shows, so a probe that
  only reports rank 0 would miss it -- both ranks are checked.
* the recurrence's incoming state. Rank 0 starts from zero either way, so rank 0
  passing proves nothing about the prefix scan; rank 1 is the load-bearing row.

    torchrun --nproc_per_node=2 kcp_kda_parity.py
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.models.kimi_k3_up import _kda_config
from torchtitan.models.kimi_k3_up.cp_kcp import apply_kcp

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)
mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))

# B is 1 because fla's causal_conv1d_cp asserts "CP requires [1, T, D]": the CP
# path is built around cu_seqlens packing, where the batch is one packed
# sequence. This is a constraint on KCP itself, not on the probe.
DIM, HEADS, HEAD_DIM, CONV_W, B, T = 512, 4, 128, 4, 1, 256
assert T % world == 0

torch.manual_seed(0)
kda = (
    _kda_config(dim=DIM, num_heads=HEADS, head_dim=HEAD_DIM, conv_kernel_size=CONV_W)
    .build()
    .to(device)
)
# The fla kernels are bf16/fp16 paths; keep the module in bf16 so the reference
# and the CP run take the same kernel rather than differing by dtype.
kda = kda.to(torch.bfloat16)
# Use the model's OWN initialization. Hand-rolling one and skipping the 1-D
# parameters left A_log and dt_bias on uninitialized memory, the reference blew
# up to ~1e25, and relative error against an exploding scale reported 6.5e-3 --
# under the threshold, so the probe printed PARITY PASS while measuring nothing.
kda.init_states()
torch.manual_seed(1234)
x_BTD = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16)

with torch.no_grad():
    reference_BTD = kda.forward(x_BTD)

# Guard the judge before trusting it: a relative error is meaningless if the
# reference is not finite and of a sane magnitude. This is what the first
# version of this probe lacked.
ref_max = reference_BTD.float().abs().max().item()
if not torch.isfinite(reference_BTD).all() or not (1e-6 < ref_max < 1e3):
    raise RuntimeError(
        f"reference output is unusable (finite={torch.isfinite(reference_BTD).all()}, "
        f"max|.|={ref_max:.3e}); the parity number below would be meaningless"
    )

t_loc = T // world
x_local = x_BTD[:, rank * t_loc : (rank + 1) * t_loc].contiguous()

apply_kcp([kda], mesh)
with torch.no_grad():
    got_BLD = kda.forward(x_local)

want_BLD = reference_BTD[:, rank * t_loc : (rank + 1) * t_loc]
diff = (got_BLD.float() - want_BLD.float()).abs().max().item()
scale = want_BLD.float().abs().max().item()
rel = diff / max(scale, 1e-12)
print(f"[rank{rank}] max abs diff {diff:.3e}  relative {rel:.3e}", flush=True)

# bf16 kernels, so the bar is bf16 noise rather than fp32 noise.
ok = torch.tensor([1.0 if rel < 5e-2 else 0.0], device=device)
dist.all_reduce(ok)
if rank == 0:
    print("PARITY PASS" if ok.item() == world else "PARITY FAIL", flush=True)
dist.destroy_process_group()
