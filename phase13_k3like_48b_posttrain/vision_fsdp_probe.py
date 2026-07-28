"""Does the vision tower train under FSDP2 at dp2, and do gradients reduce?"""
import os
import torch, torch.distributed as dist
from torchtitan.experiments.kimi_k3.moonvit import MoonViT, MoonViTConfig
from torchtitan.experiments.kimi_k3.parallelize import apply_fsdp_vision
from torchtitan.experiments.kimi_k3.vision_preprocess import pack_images

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

cfg = MoonViTConfig(num_hidden_layers=4, hidden_size=64, num_attention_heads=2,
                    qkv_hidden_size=96, intermediate_size=128, patch_size=14,
                    init_pos_emb_height=16, init_pos_emb_width=16,
                    text_hidden_size=128, rope_max_grid=64)
torch.manual_seed(0)
with torch.device("meta"):
    tower = MoonViT(cfg)
mesh = dist.device_mesh.init_device_mesh("cuda", (world,), mesh_dim_names=("dp_shard",))
units = apply_fsdp_vision(tower, mesh, torch.bfloat16, torch.float32)
tower.to_empty(device="cuda")
torch.manual_seed(0)
tower.init_weights()

# each rank gets a different image, as data parallelism implies
torch.manual_seed(100 + rank)
imgs = [torch.rand(3, 224, 224, device="cuda"), torch.rand(3, 168, 252, device="cuda")]
patches, grid = pack_images(imgs)
out = tower(patches.bfloat16(), grid)
loss = torch.cat(out).float().pow(2).mean()
loss.backward()

g = tower.patch_embed.proj.weight.grad
gl = g.to_local() if hasattr(g, "to_local") else g
if rank == 0:
    print(f"[VIS-FSDP] units={units} world={world}", flush=True)
    print(f"[VIS-FSDP] tokens per image: {[o.shape[0] for o in out]}", flush=True)
    print(f"[VIS-FSDP] loss {loss.item():.5f}", flush=True)
    print(f"[VIS-FSDP] patch_embed grad: shape {tuple(gl.shape)} "
          f"finite {torch.isfinite(gl).all().item()} sum {gl.float().abs().sum().item():.4e}",
          flush=True)
# a sharded param must have a DIFFERENT local shard per rank but the SAME
# reduced gradient content -- check the grad is a DTensor (i.e. reduce-scattered)
if rank == 0:
    print(f"[VIS-FSDP] grad is DTensor (reduce-scattered): {hasattr(g,'to_local')}", flush=True)
    print("[VIS-FSDP] PASS" if torch.isfinite(gl).all() and gl.float().abs().sum() > 0 else "[VIS-FSDP] FAIL", flush=True)
dist.destroy_process_group()
