"""Step-0 graft-identity verification on real Kimi-Linear-48B weights.

PLAN section 1 / CLAUDE.md anchor claim: "Kimi-Linear-48B + AttnRes
(zero-init) is numerically identical to the original checkpoint at
step 0". Three phases (one torchrun invocation each; native DCP is
produced once and reused):

  export  -- CPU-offloaded HF load (proven path) + save native DCP.
  baseline -- GPU-resident baseline, native load, forward fixed batch,
              save logits.
  graft   -- GPU-resident block_attn_res build (110 AttnRes params
             zero-init), key-filtered native dcp.load, forward the
             same batch, compare logits.

Usage:
    torchrun --nproc_per_node=8 verify_48b_graft_step0.py <phase> <hf_dir>
"""

import sys

import torch
import torch.distributed as dist

NATIVE = "/workspace/smoke_runs/kimi48b_native/checkpoint"
LOGITS = "/workspace/smoke_runs/graft0_baseline_logits.pt"
TOKENS_SEED = 0


def make_config(flavor_fn, *, offload, model_dir=None, native_load=False):
    from torchtitan.experiments.kimi_k3 import config_registry

    config = getattr(config_registry, flavor_fn)()
    config.training.steps = 1
    config.training.local_batch_size = 1
    config.training.seq_len = 256
    config.training.enable_cpu_offload = offload
    config.checkpoint.enable = True
    config.checkpoint.last_save_model_only = True
    config.checkpoint.folder = NATIVE
    if model_dir is not None:
        config.checkpoint.initial_load_path = model_dir
        config.checkpoint.initial_load_in_hf = True
        config.checkpoint.initial_load_model_only = True
    elif native_load:
        config.checkpoint.initial_load_path = f"{NATIVE}/step-0"
        config.checkpoint.initial_load_model_only = True
    config.dump_folder = f"/workspace/smoke_runs/graft0_{flavor_fn}_{'x' if offload else 'g'}"
    config.metrics.enable_tensorboard = False
    return config


def fixed_tokens():
    g = torch.Generator().manual_seed(TOKENS_SEED)
    return torch.randint(0, 163840, (1, 256), generator=g).cuda()


def forward_logits(model):
    # CPU-offloaded FSDP moves PARAMETERS per-forward but leaves buffers
    # where they were materialized (CPU) -- the MoE router's
    # expert_bias_E then meets GPU activations. Move buffers up front.
    for m in model.modules():
        for name, b in list(m.named_buffers(recurse=False)):
            if b is not None and b.device.type != "cuda":
                setattr(m, name, b.cuda())
    model.eval()
    with torch.no_grad():
        return model(fixed_tokens()).float().cpu()


def main(phase, model_dir):
    rank_print = lambda *a: (dist.get_rank() == 0) and print(*a, flush=True)

    if phase == "export":
        config = make_config(
            "kimi_linear_48b_baseline", offload=True, model_dir=model_dir
        )
        trainer = config.build()
        trainer.checkpointer.load(step=-1)
        trainer.checkpointer.save(curr_step=0, last_step=True)
        rank_print("[GRAFT0] native DCP exported to", NATIVE)

    elif phase == "baseline":
        config = make_config(
            "kimi_linear_48b_baseline", offload=True, native_load=True
        )
        trainer = config.build()
        trainer.checkpointer.load(step=-1)
        logits = forward_logits(trainer.model_parts[0])
        if dist.get_rank() == 0:
            torch.save(logits, LOGITS)
            print(f"[GRAFT0] baseline logits saved: shape "
                  f"{tuple(logits.shape)} mean {logits.mean():.6f}",
                  flush=True)

    elif phase == "graft":
        import torch.distributed.checkpoint as dcp

        config = make_config("kimi_linear_48b_block_attn_res_gated", offload=True)
        config.checkpoint.enable = False
        trainer = config.build()
        model = trainer.model_parts[0]
        sd = model.state_dict()
        def is_graft_key(k):
            return "attn_res" in k or "mlp_res" in k
        graft_only = [k for k in sd if is_graft_key(k)]
        load_sd = {k: v for k, v in sd.items() if not is_graft_key(k)}
        rank_print(f"[GRAFT0] loading {len(load_sd)} keys; keeping "
                   f"{len(graft_only)} zero-init AttnRes params")
        dcp.load(load_sd, checkpoint_id=f"{NATIVE}/step-0")
        logits = forward_logits(model)
        if dist.get_rank() == 0:
            base = torch.load(LOGITS)
            d = (base - logits).abs()
            agree = (base.argmax(-1) == logits.argmax(-1)).float().mean()
            print(f"[GRAFT0] max|dlogit| = {d.max():.6e}", flush=True)
            print(f"[GRAFT0] mean|dlogit| = {d.mean():.6e}", flush=True)
            print(f"[GRAFT0] top-1 agreement = {agree*100:.2f}%", flush=True)
            verdict = ("IDENTITY" if d.max() == 0 else
                       "NEAR-IDENTITY" if d.max() < 1e-3 else "DIVERGENT")
            print(f"[GRAFT0] verdict: {verdict}", flush=True)
    else:
        raise ValueError(phase)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
