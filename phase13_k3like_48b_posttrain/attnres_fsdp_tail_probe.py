"""Is final_attn_res_proj.weight actually unsharded when block_attn_res reads it?

block_attn_res takes the pseudo-query from ``proj.weight`` directly instead of
calling ``proj(...)``, so FSDP2's pre-forward hook never fires on that module,
and FSDP2 warns "1 of the 2 modules passed to fully_shard did not run forward
before backward". The all-gather is supposed to come from
``final_attn_res_norm``, which shares the same FSDP param group and IS called
(``K = norm(V)``) one line earlier in block_attn_res. This probe asserts that
mechanism instead of inferring it from a sane loss: inside the forward, the
weight must be the full [1, D] plain tensor on every rank.

Result (2026-07-27, k3mini, dp2, real apply_fsdp, param bf16 / reduce fp32):
PASS on both ranks, grad finite. rank1 reports grad-sum 0.0 because dim 0 of a
[1, D] param is size 1, so rank 1's local shard is empty -- expected, not a
missing gradient.

Caveat worth keeping: a HAND-ROLLED fully_shard of the same module groups
(instead of apply_fsdp) produced a nan grad on rank 0 under bf16 compute while
fp32 compute stayed finite. The real path is clean, so this is an artifact of
the approximation, but it means this probe must call apply_fsdp -- an ad-hoc
wrapping does not reproduce the trainer's behavior here.

Launch: PYTHONPATH=<titan> torchrun --nproc_per_node=2 attnres_fsdp_tail_probe.py
"""

import os

import torch
import torch.distributed as dist

from torchtitan.experiments.kimi_k3.attn_res_model import KimiK3AttnResModel
from torchtitan.experiments.kimi_k3.model_configs import (
    build_kimi_linear_config,
    resolve_num_blocks,
)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    cfg = build_kimi_linear_config("k3mini", vocab_size=256)
    nb = resolve_num_blocks("k3mini", "block_attn_res")
    torch.manual_seed(0)
    model = KimiK3AttnResModel(cfg, num_blocks=nb).cuda()
    model.init_weights()

    from torchtitan.experiments.kimi_k3.parallelize import apply_fsdp

    mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )
    # the REAL trainer wrapping, not a hand-rolled approximation
    apply_fsdp(
        model,
        mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
    )

    seen = {}

    def record(_mod, _args):
        w = model.final_attn_res_proj.weight
        seen["shape"] = tuple(w.shape)
        seen["is_dtensor"] = hasattr(w, "to_local")

    model.final_attn_res_norm.register_forward_pre_hook(record)

    tokens = torch.randint(0, cfg.vocab_size, (1, 128), device="cuda")
    torch.nn.functional.cross_entropy(
        model(tokens).float().view(-1, cfg.vocab_size), tokens.view(-1)
    ).backward()

    D = cfg.hidden_size
    ok = seen.get("shape") == (1, D) and not seen["is_dtensor"]
    # sharded on dim 0 across 2 ranks, [1, D] leaves rank 1 with [0, D] --
    # so an all-gather failure shows up as an empty first dim, not a subtle
    # numerical difference.
    print(
        f"[rank{rank}] weight seen inside forward: shape={seen.get('shape')} "
        f"dtensor={seen.get('is_dtensor')} -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    g = model.final_attn_res_proj.weight.grad
    print(
        f"[rank{rank}] grad after backward: "
        f"{None if g is None else (tuple(g.shape), g.abs().sum().item())}",
        flush=True,
    )
    dist.destroy_process_group()
    assert ok


if __name__ == "__main__":
    main()
