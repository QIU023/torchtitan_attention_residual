"""Is KCP on OUR tree exact, and is it still exact with a batch axis?

Sibling of ``kcp_kda_parity.py``, which probes the vendored upstream tree's
``cp_kcp`` at B = 1. Two things this one has to answer that that one does not:

* our ``KimiDeltaAttention._forward_kcp`` is the path a CP run now takes by
  default (``kda_cp_mode`` defaults to "kcp"), so its exactness is no longer a
  property of one opt-in flavor;
* it loops over the batch. fla's ``causal_conv1d_cp`` asserts ``[1, T, D]``, so a
  batch cannot be handed to the CP path whole. Flattening it into one packed
  sequence would be wrong rather than merely awkward: ``build_cp_context`` cuts
  the GLOBAL packed sequence into contiguous rank-ordered pieces, while this rank
  holds piece ``r`` of EVERY sequence. The loop is the layout that matches, and
  it is also what the recurrence needs, since the delta-rule state must not carry
  from one sequence in a batch into the next.

The judge is the same and is the only one that measures this: comparing training
losses at dp1 and cp2 does not, because both are plausible. Every rank builds the
same module from the same seed, generates the same full ``[B, T, D]``, and then

  reference  the module's forward with no CP group, on the full sequence
  kcp        the CP forward on this rank's ``[B, T/cp, D]`` shard

Rank r's output must equal ``reference[:, r*T/cp : (r+1)*T/cp]`` for every row of
the batch. B > 1 is the point: a loop that leaked state across rows, or that
reused one row's context for another, shows up as a per-row disagreement, so the
per-row error is reported and not just the max.

Rank 0 proves less than rank 1 does. Its incoming recurrent state is zero either
way and its conv needs no halo, so a probe that only printed rank 0 would pass
with the prefix scan and the halo exchange both broken.

    torchrun --nproc_per_node=2 kcp_batch_parity.py
"""

import torch
import torch.distributed as dist


from torchtitan.models.kimi_k3.model import KimiDeltaAttention, KimiK3Config

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)

DIM, HEADS, HEAD_DIM, CONV_W, B, T = 512, 4, 128, 4, 2, 256
assert T % world == 0, "KCP needs the sequence to divide evenly across CP ranks"

torch.manual_seed(0)
# Through make_config, so the module under test is built the way the model builds
# it -- a hand-rolled Config would be a second definition able to drift.
flat = KimiK3Config(
    hidden_size=DIM,
    kda_num_heads=HEADS,
    kda_head_dim=HEAD_DIM,
    kda_short_conv_kernel_size=CONV_W,
    kda_use_full_rank_gate=True,
    kda_gate_lower_bound=-5.0,
    kda_cp_mode="kcp",
)
kda = (
    KimiDeltaAttention.make_config(flat, layer_idx=0)
    .build()
    .to(device)
    .to(torch.bfloat16)
)
assert kda.cp_mode == "kcp", kda.cp_mode
# The model's own init: hand-rolling one and skipping A_log / dt_bias left them
# on uninitialized memory, the reference blew up to ~1e25, and a relative error
# against an exploding scale read as a pass while measuring nothing.
# no_grad: init_states writes parameters in place, which autograd refuses on a
# leaf that requires grad outside the trainer's own init path.
with torch.no_grad():
    kda.init_states()

torch.manual_seed(1234)
x_BTD = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16)

with torch.no_grad():
    reference_BTD = kda.forward(x_BTD)

ref_max = reference_BTD.float().abs().max().item()
if not torch.isfinite(reference_BTD).all() or not (1e-6 < ref_max < 1e3):
    raise RuntimeError(
        f"reference output is unusable (finite="
        f"{bool(torch.isfinite(reference_BTD).all())}, max|.|={ref_max:.3e}); the "
        "parity number below would be meaningless"
    )

t_loc = T // world
x_local = x_BTD[:, rank * t_loc : (rank + 1) * t_loc].contiguous()

kda._cp_group = dist.group.WORLD
with torch.no_grad():
    got_BLD = kda.forward(x_local)

want_BLD = reference_BTD[:, rank * t_loc : (rank + 1) * t_loc]
per_row = []
for b in range(B):
    diff = (got_BLD[b].float() - want_BLD[b].float()).abs().max().item()
    scale = want_BLD[b].float().abs().max().item()
    per_row.append(diff / max(scale, 1e-12))
rel = max(per_row)
rows = " ".join(f"row{b}={r:.3e}" for b, r in enumerate(per_row))
print(f"[rank{rank}] relative {rel:.3e}  ({rows})", flush=True)

# bf16 kernels, so the bar is bf16 noise rather than fp32 noise.
ok = torch.tensor([1.0 if rel < 5e-2 else 0.0], device=device)
dist.all_reduce(ok)
if rank == 0:
    print("PARITY PASS" if ok.item() == world else "PARITY FAIL", flush=True)
dist.destroy_process_group()
