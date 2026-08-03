"""Whole-model CP parity: KDA on KCP and MLA on Ulysses, in the same forward.

This composition is the official design, not a half-finished one. Report sec 5.4
cites DeepSpeed Ulysses [50] as K3's context parallelism, and sec 5.1.2 (KCP) is
specifically the KDA extension -- softmax attention has to exchange KV blocks
that grow with sequence length, while linear attention carries a fixed-size
recurrent state, so the two attention types genuinely want different schemes.
Making MLA use KCP would DIVERGE from the release.

What that leaves worth checking is the seam. Under KCP the residual stream stays
sequence-sharded; an MLA layer then has to all-to-all into head-sharding, attend
over the full sequence for its local heads, and all-to-all back. Every KDA/MLA
boundary crosses that seam, and with K3's alternating pattern there are many of
them.

So: the full model, sequence-sharded across ranks, against the same model on one
rank with the whole sequence -- compared PER LAYER, not on the logits.

Why per layer. An untrained model with random weights amplifies any perturbation
by roughly 1.6x per layer, so over 21 layers a last-bit difference reaches the
same magnitude as a completely wrong shard. Measured: in bf16 the correct CP path
lands at rel 6.8e-01 on the logits while isolated shards -- no halo, no state
scan, no head all-to-all -- land at 7.5e-01. Those are indistinguishable, so an
end-to-end logits comparison is not a gate at this depth; it is a vacuous A/B
whose control happens to look almost identical to a pass.

The first layer is where the signal is. In fp32 layer 0 comes out at 1.06e-05,
i.e. exact, and the growth from there is the model's own amplification rather
than accumulating error:

    fp32:  L0 1.1e-05   L1 1.3e-03   L4 4.8e-02   L20 3.2e-01
    bf16:  L0 5.6e-03 (bf16 epsilon is ~8e-03)    L20 6.8e-01

This probe therefore gates on the FIRST layer and reports the whole profile, and
runs in fp32 so the gate is not sitting on the bf16 noise floor.

Launch: PYTHONPATH=<titan> torchrun --nproc_per_node=<cp> mixed_cp_parity_probe.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from torchtitan.experiments.kimi_k3.model import (
    KimiDeltaAttention,
    KimiK3Model,
    KimiMLAAttention,
)
from torchtitan.experiments.kimi_k3.model_configs import build_kimi_linear_config

DTYPE = torch.float32  # see the docstring: bf16 puts the gate on the noise floor


def build(cp_mode: str, wire_cp: bool):
    cfg = build_kimi_linear_config("k3mini", vocab_size=256)
    cfg.kda_cp_mode = cp_mode
    torch.manual_seed(0)
    model = KimiK3Model(cfg).cuda().to(DTYPE)
    model.init_weights(buffer_device="cuda")
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    if wire_cp:
        for m in model.modules():
            if isinstance(m, (KimiDeltaAttention, KimiMLAAttention)):
                m._cp_group = dist.group.WORLD
    model.eval()
    return model, cfg


def hook_layers(model):
    seen: dict[str, torch.Tensor] = {}
    for name, layer in model.layers.items():
        layer.register_forward_hook(
            lambda mod, args, out, n=name: seen.__setitem__(
                n, (out[0] if isinstance(out, tuple) else out).detach()
            )
        )
    return seen


def profile(cp_mode: str, wire_cp: bool, tokens, sl, world):
    model, cfg = build(cp_mode, wire_cp)
    seen = hook_layers(model)
    with torch.no_grad():
        model(tokens[:, sl].contiguous() if wire_cp or sl is not None else tokens)
    out = {}
    for name, local in seen.items():
        if sl is None:
            out[name] = local.float()
            continue
        buf = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(buf, local.contiguous())
        out[name] = torch.cat(buf, dim=1).float()
    del model
    torch.cuda.empty_cache()
    return out, cfg


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    assert T % world == 0

    tokens = torch.randint(0, 256, (1, T), device="cuda")
    dist.broadcast(tokens, src=0)
    part = T // world
    sl = slice(rank * part, (rank + 1) * part)

    ref, cfg = profile("ulysses", wire_cp=False, tokens=tokens, sl=None, world=world)
    arms = {
        mode: profile(mode, True, tokens, sl, world)[0]
        for mode in ("kcp", "ulysses")
    }
    # CONTROL: sharded with NO cp wiring -- isolated shards.
    control, _ = profile("ulysses", False, tokens, sl, world)

    if rank == 0:
        full = set(cfg.full_attn_layers)
        names = sorted(ref, key=lambda s: int(s))

        def rel(a, n):
            r = ref[n]
            return ((a[n] - r).norm() / r.norm()).item()

        first = names[0]
        print(f"[MIXED-CP] cp={world} T={T} dtype={DTYPE}", flush=True)
        print("[MIXED-CP] layer kind      kcp      ulysses     control", flush=True)
        for n in names:
            kind = "MLA" if (int(n) + 1) in full else "KDA"
            print(
                f"  {int(n):3} {kind}   {rel(arms['kcp'], n):.3e}  "
                f"{rel(arms['ulysses'], n):.3e}  {rel(control, n):.3e}",
                flush=True,
            )
        # Gate at the first layer, where the metric still discriminates.
        kcp0, ctrl0 = rel(arms["kcp"], first), rel(control, first)
        ok = kcp0 < 1e-3 and ctrl0 > 100 * max(kcp0, 1e-12)
        print(
            f"[MIXED-CP] layer {first}: kcp {kcp0:.3e} vs control {ctrl0:.3e}",
            flush=True,
        )
        print("[MIXED-CP] PASS" if ok else "[MIXED-CP] FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
