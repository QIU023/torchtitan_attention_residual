"""Scratch: model-level dynamic CP gradient check at CP2 (run with torchrun, 2 GPUs). Prints per-rank results."""
import os
from dataclasses import replace

import torch
import torch.distributed as dist

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.models.common.multimodal import build_vision_bank_indices
from torchtitan.models.kimi_k3 import build_model_config


def _compile_flex_without_autotune():
    from torch.nn.attention.flex_attention import flex_attention
    from torchtitan.models.common.attention import FlexInnerAttention

    FlexInnerAttention.inductor_configs["max_autotune"] = False
    FlexInnerAttention.inductor_configs["coordinate_descent_tuning"] = False
    FlexInnerAttention._compiled_flex_attn = torch.compile(flex_attention, options=FlexInnerAttention.inductor_configs)


def run(ctx, min_patches):
    parallelism = ParallelismConfig(context_parallel_degree=2)
    config = build_model_config("debugmodel", parallelism=parallelism, attn_backend="flex")
    config = replace(config, vision_encoder=replace(config.vision_encoder, dynamic_cp_min_patches=min_patches))
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = config.build()
        model.init_states()
    model.parallelize(
        parallelism_context=ctx,
        training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
        parallelism=parallelism,
        local_compile_regions=[],
        activation_remat=None,
        dump_folder="/tmp",
    )
    tower = model.vision_encoder
    grids = [[1, 16, 16], [1, 12, 20]]
    kh, kw = tower.merge_kernel_size
    counts = [(h // kh) * (w // kw) for _, h, w in grids]
    placeholder = 7
    tokens = torch.cat([torch.tensor([3, 4]), torch.full((counts[0],), placeholder), torch.tensor([5]), torch.full((counts[1],), placeholder), torch.tensor([6, 8, 9])]).cuda()
    tokens = torch.cat([tokens, torch.full(((-tokens.numel()) % 2,), 1, device="cuda")])
    indices = build_vision_bank_indices(tokens, placeholder_id=placeholder)
    gen = torch.Generator(device="cuda").manual_seed(1)
    pixels = torch.randn(sum(t * h * w for t, h, w in grids), tower.patch_embed.in_features, device="cuda", generator=gen)
    upstream = torch.randn(tokens.numel(), config.dim, device="cuda", generator=gen)
    r, half = dist.get_rank(), tokens.numel() // 2
    local = slice(r * half, (r + 1) * half)
    with ctx.activate_spmd():
        embeds = model._prepare_multimodal_embeds(
            tokens[local], pixel_values=pixels, grid_thw=torch.tensor(grids, device="cuda"),
            special_tokens={"image_id": placeholder}, vision_bank_indices_T=indices[local],
        )
    (embeds * upstream[local]).sum().backward()
    full = lambda t: t.full_tensor() if hasattr(t, "full_tensor") else t
    grads = {n: full(p.grad).detach().clone() for n, p in tower.named_parameters() if p.grad is not None}
    return grads


dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
_compile_flex_without_autotune()
ctx = ParallelismContext(dp_replicate=1, dp_shard=1, cp=2, tp=1, pp=1, ep=1, world_size=2, enable_sequence_parallel=False)
ctx.build_mesh()
ref = run(ctx, None)
split = run(ctx, 16)
worst = max(((split[k] - ref[k]).norm() / ref[k].norm().clamp_min(1e-30)).item() for k in ref)
print(f"rank {dist.get_rank()}: {len(ref)} tower grads, worst relative diff split vs ref {worst:.3e}", flush=True)
dist.destroy_process_group()
