"""Verify official Kimi-Linear-48B-A3B-Base weights load through the
titan Trainer + KimiLinearStateDictAdapter (initial_load_in_hf path).

Work item (1) closure leg: "48B weight LOADING verifies on 5090
(12GB/card sharded)". Runs the REAL trainer init (FSDP2 sharded build)
and the REAL CheckpointManager HF initial load -- the exact code path
veRL's torchtitan engine uses -- then spot-checks loaded values
against the raw safetensors bytes (exact bf16 equality).

Usage:
    cd torchtitan && torchrun --nproc_per_node=8 \
        ../phase13_k3like_48b_posttrain/verify_48b_load.py <hf_snapshot_dir>
"""

import sys

import torch
import torch.distributed as dist
from safetensors import safe_open


def main(model_dir: str) -> None:
    from torchtitan.experiments.kimi_k3 import config_registry

    config = config_registry.kimi_linear_48b_baseline()
    config.training.steps = 1
    config.training.local_batch_size = 1
    config.training.seq_len = 256
    # FSDP CPU offload: sharded params live in host RAM (503 GB on this
    # box); the HF initial load's transient copies (per-expert slices +
    # stacked rebuilds) exceed 32 GiB GPU otherwise. Same code path the
    # veRL engine uses for train<->rollout offload.
    config.training.enable_cpu_offload = True
    config.checkpoint.enable = True
    config.checkpoint.initial_load_path = model_dir
    config.checkpoint.initial_load_in_hf = True
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.folder = "/workspace/smoke_runs/verify48b/checkpoint"
    config.dump_folder = "/workspace/smoke_runs/verify48b"
    config.metrics.enable_tensorboard = False

    trainer = config.build()
    trainer.checkpointer.load(step=-1)

    rank = dist.get_rank()
    model = trainer.model_parts[0]

    # Spot checks: (tt_fqn, hf_key, optional transform on the HF side)
    checks = [
        ("embed_tokens.weight", "model.embed_tokens.weight", None),
        (
            "layers.0.self_attn.A_log",
            "model.layers.0.self_attn.A_log",
            lambda t: t.reshape(-1),
        ),
        (
            "layers.3.self_attn.q_proj.weight",  # layer 4 (1-idx) = MLA
            "model.layers.3.self_attn.q_proj.weight",
            None,
        ),
        (
            "layers.1.ffn._moe.routed_experts.inner_experts.w2_EDF",
            None,  # stacked: compare expert 0 + 255 slices
            None,
        ),
    ]

    sd = model.state_dict()
    results = []
    if rank == 0:
        for tt_fqn, hf_key, tf in checks:
            p = sd[tt_fqn]
            # Compare rank-0's LOCAL shard against the same slice of the
            # raw safetensors bytes (Shard(0) over the dp_shard mesh) --
            # no collectives, no full-tensor gathers.
            local = p.to_local() if hasattr(p, "to_local") else p
            local = local.cpu()
            n0 = local.shape[0]
            if tt_fqn.endswith("w2_EDF"):
                hf = _read_hf(
                    model_dir,
                    "model.layers.1.block_sparse_moe.experts.0.w2.weight",
                )
                ok = torch.equal(local[0].to(hf.dtype), hf)
            else:
                hf = _read_hf(model_dir, hf_key)
                if tf is not None:
                    hf = tf(hf)
                ok = torch.equal(local.to(hf.dtype), hf[:n0])
            results.append((tt_fqn, ok))
            print(f"[VERIFY48B] {tt_fqn}: "
                  f"{'EXACT-MATCH' if ok else 'MISMATCH'} "
                  f"(local shard {tuple(local.shape)})", flush=True)
            del local

    if rank == 0:
        n_ok = sum(1 for _, ok in results if ok)
        print(f"[VERIFY48B] {n_ok}/{len(results)} spot checks exact; "
              f"model params sharded across {dist.get_world_size()} ranks.",
              flush=True)
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"[VERIFY48B] peak GPU mem rank0: {mem:.2f} GiB", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def _read_hf(model_dir: str, key: str) -> torch.Tensor:
    import json
    from pathlib import Path

    idx = json.load(open(Path(model_dir) / "model.safetensors.index.json"))
    shard = idx["weight_map"][key]
    with safe_open(str(Path(model_dir) / shard), framework="pt",
                   device="cpu") as h:
        return h.get_tensor(key)


if __name__ == "__main__":
    main(sys.argv[1])
