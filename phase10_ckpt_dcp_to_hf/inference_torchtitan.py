"""Phase 10 — torchtitan-based forward-only inference for fabric trace.

Bypasses both the SGLang serving runtime (sgl_kernel incompatible with
our env) and torchtitan's Trainer (training-only). Builds the
kimi_linear AttnRes model directly, applies torchtitan's parallelize_fn
for TP+EP+FSDP, dcp.loads the phase4 step-8000 ckpt, then loops
forward-only batches with NCCL trace capture.

Mesh: FSDP=2 x TP=2 x EP=2 = 8 (PP dropped to keep the forward path
single-call — _ScheduleForwardOnly is available but adds 200+ LOC of
schedule plumbing for marginal fabric coverage; v11 training trace
already covers PP fwd+bwd send/recv pattern, so the *delta* between
training and inference fabric on the PP axis is documentable from
existing data).

Captured fabric:
* FSDP AllGather (per layer, weight reconstitution; appears in both
  training and inference)
* FSDP ReduceScatter SHOULD NOT appear (no backward in inference)
* EP all-to-all dispatch (per MoE-layer forward)
* EP all-to-all combine (per MoE-layer forward)
* TP AllReduce (intra-node, NVLink — typically does not show on
  inter-host fabric, but logged by NCCL trace)

Run via phase10_ckpt_dcp_to_hf/run_inference_torchtitan.sh.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))
sys.path.insert(0, str(_WS / "torchtitan"))


def _init_dist():
    if not dist.is_initialized():
        dist.init_process_group("nccl")


def _build_parallel_dims(world_size: int):
    """Build a ParallelDims for FSDP=2 x TP=2 x EP=2 = 8 mesh.

    PP=1, CP=1, dp_replicate=1. dp_shard=FSDP=2 (the world / others).
    """
    from torchtitan.distributed.parallel_dims import ParallelDims

    # EP overlays on dp_shard (does NOT multiply into world_size).
    # Target inference fabric: FSDP=4 x TP=2 + EP=2-overlay = 8 ranks.
    fsdp = 4
    tp = 2
    ep = 2
    pp = 1
    cp = 1
    dp_replicate = 1
    if fsdp * tp * pp * cp * dp_replicate != world_size:
        raise RuntimeError(
            f"Mesh product (excl ep) != world_size: "
            f"FSDP={fsdp} x TP={tp} x PP={pp} x CP={cp} x dp_rep={dp_replicate} "
            f"= {fsdp*tp*pp*cp*dp_replicate}, world={world_size}, EP={ep} overlays"
        )
    pd = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=fsdp,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=tp,  # expert TP shares the tp mesh by default in our config
        world_size=world_size,
    )
    return pd


def _build_model_and_parallelize(parallel_dims, device):
    """Build kimi_linear AttnRes and apply torchtitan parallelism."""
    from torchtitan.experiments.kimi_linear.config_registry import (
        kimi_linear_436m_block_attn_res_n4,
    )
    from torchtitan.experiments.kimi_linear.attn_res_model import (
        KimiLinearAttnResModel,
    )
    from torchtitan.experiments.kimi_linear.parallelize import parallelize_kimi_linear
    from torchtitan.config import (
        ParallelismConfig,
        TrainingConfig,
        ActivationCheckpointConfig,
        CompileConfig,
    )

    cfg = kimi_linear_436m_block_attn_res_n4()
    spec = cfg.model_spec.model
    model = KimiLinearAttnResModel(spec.kimi_config, num_blocks=spec.num_blocks)
    model = model.to(dtype=torch.bfloat16)

    # Construct minimal config objects for parallelize_fn.
    parallelism = ParallelismConfig()
    training = TrainingConfig()
    ac = ActivationCheckpointConfig(mode="none")
    compile_cfg = CompileConfig()
    compile_cfg.enable = False

    model = parallelize_kimi_linear(
        model,
        parallel_dims=parallel_dims,
        training=training,
        model_converters=[],
        parallelism=parallelism,
        compile_config=compile_cfg,
        ac_config=ac,
        dump_folder=None,
    )

    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_dcp(model, ckpt_dir: Path):
    sd = model.state_dict()
    print(f"[infer] loading DCP from {ckpt_dir} (state_dict keys: {len(sd)})")
    dcp.load(sd, checkpoint_id=str(ckpt_dir))
    # state_dict view is a live reference for sharded params; no extra
    # load_state_dict needed.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bs", type=int, default=4)
    args = p.parse_args()

    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if rank == 0:
        print(f"[infer] world={world} rank={rank} device={device}")

    pd = _build_parallel_dims(world)
    if rank == 0:
        print(
            f"[infer] mesh: FSDP={pd.dp_shard} TP={pd.tp} EP={pd.ep} "
            f"(PP={pd.pp}, CP={pd.cp})"
        )

    model = _build_model_and_parallelize(pd, device)
    _load_dcp(model, args.ckpt)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[infer] model params: {n:,}")

    # Forward-only loop with synthetic batches.
    vocab = 163840
    if rank == 0:
        print(f"[infer] starting {args.n_steps} forward batches "
              f"(micro_bs={args.micro_bs}, seq_len={args.seq_len})")

    t0 = time.time()
    # NOTE: torch.no_grad() (not inference_mode) — EP all_to_all_single
    # requires the autograd-aware variant which has no CUDA registration
    # under inference_mode in torch 2.11.
    with torch.no_grad():
        for step in range(1, args.n_steps + 1):
            ids = torch.randint(
                0, vocab, (args.micro_bs, args.seq_len),
                device=device, dtype=torch.long,
            )
            try:
                out = model(ids)
            except Exception as e:
                print(f"[rank{rank}] FAILED at step {step}: {type(e).__name__}: {e}")
                raise
            if rank == 0 and step % 5 == 0:
                logits_max = out.float().abs().max().item() if isinstance(out, torch.Tensor) else 0.0
                elapsed = time.time() - t0
                print(
                    f"[infer] step={step:3d}/{args.n_steps} "
                    f"logits_max={logits_max:.2f} t={elapsed:.1f}s"
                )
            del out
    if rank == 0:
        print(f"[infer] DONE {args.n_steps} steps in {time.time()-t0:.1f}s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
