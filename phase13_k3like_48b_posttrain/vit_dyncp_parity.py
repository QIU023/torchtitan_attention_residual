"""Does the patch-partitioned tower reproduce the replicated one?

Every rank encodes the whole image once (the reference), then encodes only its
row band with a plan and gathers the merged tokens. The two must agree, or the
partition is not a partition of the same computation.
"""
import torch, torch.distributed as dist
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.vision_encoder import make_cp_patch_plan

dist.init_process_group("nccl")
r, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
torch.manual_seed(0)

m = model_registry("debugmodel").model.build().to("cuda").to(torch.float32)
m.init_states(); m.eval()
enc = m.vision_encoder
kh, kw = enc.merge_kernel_size

t, h, w = 1, kh * 2 * world, kw * 2          # rows divide the kernel and the ranks
torch.manual_seed(1)
px = torch.randn(t * h * w, enc.patch_embed.weight.shape[-1], device="cuda")
grid = torch.tensor([[t, h, w]], dtype=torch.long, device="cuda")

with torch.no_grad():
    ref = enc(px, grid_thw=grid)             # replicated: the whole image

plan, ranges = make_cp_patch_plan((t, h, w), group=dist.group.WORLD, rank=r,
                                  merge_kernel_h=kh)
shard = torch.cat([px[lo:hi] for lo, hi in ranges], dim=0)
with torch.no_grad():
    mine = enc(shard, grid_thw=grid, cp_plan=plan)

parts = [torch.empty_like(mine) for _ in range(world)]
dist.all_gather(parts, mine.contiguous())
got = torch.cat(parts, dim=0)

ok = got.shape == ref.shape
d = (got.float() - ref.float()).abs() if ok else None
if r == 0:
    if not ok:
        print(f"SHAPE MISMATCH: partitioned {tuple(got.shape)} vs replicated {tuple(ref.shape)}")
    else:
        print(f"vit dynamic CP: max_abs={d.max().item():.3e} "
              f"rel={(d.max()/ref.float().abs().max()).item():.3e}")
dist.destroy_process_group()
