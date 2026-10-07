"""Old vs new dynamic CP tower on 4 gloo ranks (CPU, fp32, dense attention in both): outputs and grads.

usage: python dcp_equiv.py <tree> <out_dir> <old|new>
"""
import os
import sys
import tempfile
from dataclasses import replace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CASES = {
    "one image over the whole CP group": [[1, 12, 12], [1, 16, 16]],
    "two images over two sub-groups": [[1, 16, 16], [1, 12, 12], [1, 20, 12]],
    "four images, one rank each": [[1, 16, 16], [1, 16, 14], [1, 12, 12], [1, 20, 12], [1, 18, 16]],
    "a video whose last rank holds only padding": [[1, 12, 12], [2, 10, 16]],
}


def _ids(runs):
    return torch.repeat_interleave(
        torch.tensor([i for i, _ in runs]), torch.tensor([n for _, n in runs])
    )


CALLS = []


def dense_mask(query_runs, key_runs, device):
    CALLS.append(len(key_runs))
    return _ids(query_runs)[:, None] == _ids(key_runs)[None, :]


def dense_forward(self, q_THK, k_THK, v_THV, *, attention_metadata, scale=None, **kw):
    q, k, v = (t.transpose(0, 1) for t in (q_THK, k_THK, v_THV))
    s = (q @ k.transpose(-1, -2)) * (scale if scale is not None else q.shape[-1] ** -0.5)
    s = s.masked_fill(~attention_metadata, float("-inf"))
    return (s.softmax(-1) @ v).transpose(0, 1)


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, x):
        ctx.group = group
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, x, group=group)
        return tuple(parts)

    @staticmethod
    def backward(ctx, *grads):
        stacked = torch.stack(grads)
        dist.all_reduce(stacked, group=ctx.group)
        return None, stacked[dist.get_rank(ctx.group)]


def gloo_all_gather(tensor, group=None):
    return _AllGather.apply(group, tensor)


def worker(rank, tree, out_dir, which, store):
    sys.path.insert(0, tree)
    import kda_stub  # noqa: F401
    import torchtitan.distributed.parallelism_context as pc
    pc.device_type = "cpu"
    from torchtitan.distributed import ParallelismContext
    from torchtitan.models.common.attention.attention import FlexInnerAttention
    from torchtitan.models.kimi_k3 import build_model_config

    FlexInnerAttention.forward = dense_forward
    import torch.distributed.nn.functional as dist_nn
    dist_nn.all_gather = gloo_all_gather
    if which == "old":
        from torchtitan.models.kimi_k3.vision_encoder import build_cp_subgroups, KimiK3VisionEncoder
        KimiK3VisionEncoder._split_mask = staticmethod(dense_mask)
    else:
        from torchtitan.models.kimi_k3 import vision_cp
        from torchtitan.models.kimi_k3.vision_cp import build_cp_subgroups, encoder
        encoder._split_mask = dense_mask

    dist.init_process_group("gloo", init_method=f"file:///{store}", rank=rank, world_size=4)
    ctx = ParallelismContext(dp_replicate=1, dp_shard=1, cp=4, tp=1, pp=1, ep=1, world_size=4,
                             enable_sequence_parallel=False)
    ctx.build_mesh()
    config = replace(build_model_config("debugmodel").vision_encoder, dynamic_cp_min_patches=200)
    torch.manual_seed(0)
    tower = config.build()
    tower.init_states()
    tower.set_cp_subgroups(build_cp_subgroups(ctx.get_mesh("cp")))
    kh, kw = tower.merge_kernel_size
    patch_dim = tower.patch_embed.in_features
    results = {}
    for name, grids in CASES.items():
        g = torch.Generator().manual_seed(1)
        pixels = torch.randn(sum(t * h * w for t, h, w in grids), patch_dim, generator=g)
        upstream = torch.randn(sum((h // kh) * (w // kw) for _, h, w in grids),
                               tower.projector.linear_2.out_features, generator=g)
        tower.zero_grad(set_to_none=True)
        with ctx.activate_spmd():
            out = tower(pixels, grid_thw=torch.tensor(grids))
        (out * upstream).sum().backward()
        results[name] = (out.detach().clone(), {n: p.grad.clone() for n, p in tower.named_parameters()})
    assert len(CALLS) == len(CASES), CALLS
    import torchtitan.models.kimi_k3 as k3
    results["_tree"] = k3.__file__
    torch.save(results, os.path.join(out_dir, f"{which}.{rank}.pt"))
    dist.destroy_process_group()


if __name__ == "__main__":
    tree, out_dir, which = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    store = os.path.join(tempfile.mkdtemp(), "store")
    mp.spawn(worker, args=(tree, out_dir, which, store), nprocs=4, join=True)
    print("done", which)
