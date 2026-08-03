"""KCP integrated into KimiDeltaAttention: does the module still compute KDA?

kda_kcp_probe.py validated fla's cp_context at the OP level and
conv_halo_probe.py validated the short-conv halo. This checks the assembled
module: KimiDeltaAttention with kda_cp_mode="kcp", sequence-sharded across
ranks, against the same module on one rank with the whole sequence.

Ulysses runs as a second arm. It is not a control -- both should match the
reference -- but it says whether a discrepancy is KCP-specific or a problem with
the module's CP plumbing generally.

The control that gives this power is the third arm: sharded with the halo and the
prefix scan DISABLED, which must be visibly wrong.

Launch: PYTHONPATH=<titan> torchrun --nproc_per_node=<cp> kda_kcp_module_probe.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from torchtitan.experiments.kimi_k3.model import (
    KimiDeltaAttention,
    KimiK3Config,
)


def config(cp_mode: str) -> KimiK3Config:
    return KimiK3Config(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        mla_use_nope=True,
        kda_num_heads=4,
        kda_head_dim=64,
        kda_short_conv_kernel_size=4,
        kda_gate_lower_bound=-5.0,
        kda_use_full_rank_gate=True,
        kda_cp_mode=cp_mode,
        kda_layers=[1],
        full_attn_layers=[2],
        num_experts=None,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=2,
        moe_layer_freq=1,
        num_expert_group=1,
        topk_group=1,
        rms_norm_eps=1e-5,
        hidden_act="silu",
    )


def build(cp_mode: str, group=None):
    torch.manual_seed(0)
    mod = KimiDeltaAttention(config(cp_mode), layer_idx=0).cuda().to(torch.bfloat16)
    mod.train()
    for p in mod.parameters():
        if p.dim() > 0:
            torch.nn.init.normal_(p, std=0.05)
        dist.broadcast(p.data, src=0)
    if group is not None:
        mod._cp_group = group
    return mod


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    assert T % world == 0
    D = 256

    g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn(1, T, D, device="cuda", dtype=torch.bfloat16, generator=g)
    part = T // world
    sl = slice(rank * part, (rank + 1) * part)

    # Reference: the whole sequence, no CP.
    ref_mod = build("ulysses", group=None)
    with torch.no_grad():
        ref = ref_mod(x)

    results = {}
    for mode in ("kcp", "ulysses"):
        mod = build(mode, group=dist.group.WORLD)
        with torch.no_grad():
            local = x[:, sl].contiguous() if mode == "kcp" else x[:, sl].contiguous()
            out = mod(local)
        buf = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(buf, out.contiguous())
        gathered = torch.cat(buf, dim=1)
        if mode == "ulysses":
            # Ulysses returns this rank's seq slice too (seq stays sharded on
            # the way out), so the gather reassembles the sequence the same way.
            pass
        results[mode] = gathered

    # CONTROL: sharded, no halo and no cp_context -- each rank in isolation.
    ctrl_mod = build("ulysses", group=None)
    with torch.no_grad():
        ctrl_local = ctrl_mod(x[:, sl].contiguous())
    buf = [torch.empty_like(ctrl_local) for _ in range(world)]
    dist.all_gather(buf, ctrl_local.contiguous())
    control = torch.cat(buf, dim=1)

    if rank == 0:

        def rel(a):
            return ((a.float() - ref.float()).norm() / ref.float().norm()).item()

        print(f"[KCP-MOD] cp={world} T={T}", flush=True)
        for mode, out in results.items():
            print(f"[KCP-MOD] {mode:8} rel {rel(out):.3e}", flush=True)
        print(f"[KCP-MOD] control  rel {rel(control):.3e} (isolated shards)",
              flush=True)
        ok = rel(results["kcp"]) < 5e-2 and rel(control) > 10 * rel(results["kcp"])
        print("[KCP-MOD] PASS" if ok else "[KCP-MOD] FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
