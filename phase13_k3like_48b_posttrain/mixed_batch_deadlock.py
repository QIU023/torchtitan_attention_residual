"""One rank with images, one without: does the step complete?

This is the hazard the vision TODO in parallelize.py describes. FSDP2 issues
the tower's all-gather from its pre-forward hook, so a rank that skips the
tower leaves its peers in that collective. The failure is a watchdog timeout,
not an exception, so the check is "did we get here", with a short timeout.
"""
import os, torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torchtitan.models.kimi_k3 import model_registry

dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
torch.manual_seed(0)
spec = model_registry("debugmodel")
model = spec.model.build().to("cuda").to(torch.bfloat16)
model.init_states()
fully_shard(model.vision_encoder)          # what makes skipping it unsafe
fully_shard(model)

T = 128
tokens = torch.randint(0, 1000, (T,), device="cuda")
positions = torch.arange(T, device="cuda")
kw = dict(positions=positions, attention_masks=model.get_attention_masks(positions))

if r == 0:                                  # rank 0 carries an image
    kh, kw_ = model.vision_encoder.merge_kernel_size
    w = model.vision_encoder.patch_embed.weight
    px = torch.zeros(kh * kw_, w.shape[-1], dtype=w.dtype, device='cuda')
    grid = torch.tensor([[1, kh, kw_]], dtype=torch.int32, device="cuda")
    IMG = 7
    n_vis = (kh // 2) * (kw_ // 2)
    tokens[:n_vis] = IMG
    out = model(tokens, pixel_values=px, grid_thw=grid,
                special_tokens={"image_id": IMG}, **kw)
else:                                       # rank 1 is text only
    out = model(tokens, **kw)
out.sum().backward()
dist.barrier()
print(f"rank{r} completed", flush=True)
dist.destroy_process_group()
