"""Packed MLA CP kernels against main's generic ones on real collectives: outputs and gradients."""
import torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention.flex_attention import create_block_mask
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention as AG, UlyssesCPFlexInnerAttention as UG
from torchtitan.models.kimi_k3.cp_mla import MLAKVAllGatherCPFlexInnerAttention as AP, MLAUlyssesCPFlexInnerAttention as UP

dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r)
mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
T, H, N, R, V, cp = 256, 4, 32, 16, 32, 2; Tl = T // cp; sl = slice(r * Tl, (r + 1) * Tl)
g = torch.Generator().manual_seed(0)
Q, KN, KR, VV, W = (torch.randn(*s, generator=g) for s in ((T, H, N + R), (T, H, N), (T, R), (T, H, V), (T, H, V)))
full = lambda b, h, qi, ki: qi >= 0
masks = {"ulysses": create_block_mask(full, None, None, T, T, device="cuda"),
         "allgather": create_block_mask(full, None, None, Tl, T, device="cuda")}

def run(kernel, mask, dt):
    q, kn, kr, v = (x[sl].cuda().to(dt).requires_grad_() for x in (Q, KN, KR, VV))
    k = torch.cat((kn, kr.unsqueeze(1).expand(-1, H, -1)), dim=-1)
    with set_current_spmd_mesh(mesh):
        out = kernel(q, k, v, attention_masks=mask)
    (out.float() * W[sl].cuda()).sum().backward()
    return {"out": out.detach().float(), "dq": q.grad.float(), "dk_nope": kn.grad.float(), "dk_rope": kr.grad.float(), "dv": v.grad.float()}

for dt in (torch.float32, torch.bfloat16):
    for name, gen, pk in (("ulysses", UG(UG.Config()), UP(UP.Config(rope_head_dim=R))),
                          ("allgather", AG(AG.Config()), AP(AP.Config(rope_head_dim=R)))):
        a = run(gen.cuda(), masks[name], dt); b = run(pk.cuda(), masks[name], dt)
        rel = {k: (a[k] - b[k]).abs().max().item() / max(a[k].abs().max().item(), 1e-30) for k in a}
        bitwise = {k: torch.equal(a[k], b[k]) for k in a}
        if r == 0:
            print(f"{name:9s} {str(dt):15s} " + "  ".join(f"{k}: {'bitwise' if bitwise[k] else f'{rel[k]:.1e}'}" for k in a), flush=True)
dist.destroy_process_group()
