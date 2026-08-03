"""Are TP gradients wrong, or is grad_norm just mis-measured under TP?

The overnight seed-checkpoint run showed every TP configuration reporting
grad_norm ~2.7 while every non-TP configuration reported ~8.5, from an IDENTICAL
initialization -- so the usual explanation (different init, different trainable
set) is gone. A 3x systematic gap is either a real gradient bug on the TP axis or
a reporting artifact, and loss curves cannot tell those apart: AdamW is
scale-invariant, so a uniformly mis-scaled gradient trains almost the same.

So measure param.grad directly, the same way the FSDP gradient-division bug was
caught. Both arms run on the same ranks with the same weights and the same batch:

  reference: every rank builds the full model and runs the full batch alone
  tp arm:    the same model with apply_tp over the ranks, same batch

If the materialized gradients agree, the 3x is a grad_norm metric artifact. If
they disagree, TP gradients are wrong and every TP result to date is suspect.

Launch: PYTHONPATH=<titan> torchrun --nproc_per_node=2 tp_grad_probe.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from torchtitan.experiments.kimi_k3.model import KimiK3Model
from torchtitan.experiments.kimi_k3.model_configs import build_kimi_linear_config


def build(device="cuda"):
    cfg = build_kimi_linear_config("k3mini", vocab_size=256)
    # MLA-only. Two confounds have to go at once: bf16 noise (a ~0.5% forward
    # difference between the arms, amplified through 21 layers) and KDA's
    # tilelang kernel, which has no fp32 path. Dropping KDA lets the whole model
    # run in fp32, so a surviving per-parameter difference is a TP bug rather
    # than precision. TP's correctness on the KDA layers is covered separately
    # by the CP/KCP probes, which compare against a single-rank reference.
    cfg.kda_layers = []
    cfg.full_attn_layers = list(range(1, cfg.num_hidden_layers + 1))
    torch.manual_seed(0)
    # fp32: in bf16 the forward already differs by ~0.5% between the arms, and
    # a random-init deep model amplifies that, so per-parameter gradient
    # comparisons drown in precision noise rather than showing a TP bug.
    m = KimiK3Model(cfg).to(device).float()
    m.init_weights(buffer_device=device)
    for p in m.parameters():
        dist.broadcast(p.data, src=0)
    return m, cfg


def grads_of(model) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        # A TP-sharded grad is a DTensor; materialize the global tensor so the
        # two arms are comparable. full_tensor() also reduces Partial grads,
        # which is exactly the step a naive norm would skip.
        out[name] = (g.full_tensor() if hasattr(g, "full_tensor") else g).float()
    return out


def global_norm(grads: dict[str, torch.Tensor]) -> float:
    return torch.sqrt(sum(g.pow(2).sum() for g in grads.values())).item()


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Seed the batch: an unseeded randint makes every invocation a different
    # experiment, which is useless for a comparison this tight.
    gen = torch.Generator(device="cuda").manual_seed(4242)
    tokens = torch.randint(0, 256, (1, 256), device="cuda", generator=gen)
    dist.broadcast(tokens, src=0)

    def step(model):
        model.train()
        logits = model(tokens).float()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tokens.view(-1)
        )
        loss.backward()
        return loss.item()

    # ---- reference: no parallelism, every rank does the whole thing ----
    ref_model, cfg = build()
    ref_loss = step(ref_model)
    ref = grads_of(ref_model)
    _ref_keep = ref_model

    # ---- TP arm ----
    from torchtitan.experiments.kimi_k3.parallelize import apply_tp_kimi_k3

    tp_model, _ = build()
    tp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world,), mesh_dim_names=("tp",)
    )
    apply_tp_kimi_k3(tp_model, tp_mesh)
    tp_loss = step(tp_model)
    tp = grads_of(tp_model)

    # What the TRAINER would report, on the same gradients we just materialized.
    from torchtitan.distributed.utils import clip_grad_norm_ as titan_clip

    reported_tp = titan_clip(
        [p for p in tp_model.parameters()], max_norm=1e9, foreach=False
    )
    reported_ref = titan_clip(
        [p for p in _ref_keep.parameters()], max_norm=1e9, foreach=False
    )
    if rank == 0:
        print(f"[TPGRAD] trainer-reported norm: ref={float(reported_ref):.4f}  "
              f"tp={float(reported_tp):.4f}  ratio="
              f"{float(reported_ref)/max(float(reported_tp),1e-12):.4f}", flush=True)
    if rank == 0:
        print(f"[TPGRAD] world={world}  ref_loss={ref_loss:.6f} tp_loss={tp_loss:.6f}",
              flush=True)
        print(f"[TPGRAD] materialized global grad norm: "
              f"ref={global_norm(ref):.4f}  tp={global_norm(tp):.4f}  "
              f"ratio={global_norm(ref)/max(global_norm(tp),1e-12):.4f}", flush=True)
        common = sorted(set(ref) & set(tp))
        print(f"[TPGRAD] params compared: {len(common)} "
              f"(ref-only {len(set(ref)-set(tp))}, tp-only {len(set(tp)-set(ref))})",
              flush=True)
        worst = []
        for n in common:
            a, b = ref[n], tp[n]
            if a.shape != b.shape:
                worst.append((float("inf"), n, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
                continue
            d = (a - b).norm() / a.norm().clamp(min=1e-12)
            worst.append((d.item(), n, f"ratio {(a.norm()/b.norm().clamp(min=1e-12)).item():.4f}"))
        worst.sort(reverse=True)
        print("[TPGRAD] worst 8 per-parameter relative differences:", flush=True)
        for d, n, extra in worst[:8]:
            print(f"    {d:.4e}  {n}   ({extra})", flush=True)
        agree = sum(1 for d, _, _ in worst if d < 5e-2)
        print(f"[TPGRAD] params agreeing within 5e-2: {agree}/{len(worst)}", flush=True)
        verdict = ("METRIC ARTIFACT (gradients agree)" if agree == len(worst)
                   else "REAL TP GRADIENT DIFFERENCE")
        print(f"[TPGRAD] VERDICT: {verdict}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
