"""Phase 10 Stage F — Real PPO smoke with kimi_linear Block AttnRes.

Captures the production-shape RLHF fabric pattern: actor + frozen ref
co-located on the same 4D mesh, with per-step forward-fwd-backward-opt
loop. Distinct from Phase 9-B's toy MLP smoke in that the actor and
ref are real 1.4B-param Block AttnRes models loaded from real ckpts
under real 4D parallelism.

Mesh: same as v11 production training: FSDP=2 x TP=2 x EP=2 with
PP=1 (PP would require pp_schedule integration for both actor and ref
forwards which is out of scope for the smoke). Memory: 2x 1.4B at
bf16 ~= 5.6 GB params, plus actor's optimizer + grads ~= +12 GB on
top of the actor-only FSDP shard, fits 32 GB GPUs at FSDP=4.

Actor's forward + ref's forward + actor's backward + opt step per
step gives the unique RLHF fabric:
  * 2x FSDP AllGather per step (actor + ref forward)
  * 1x FSDP ReduceScatter per step (actor backward only)
  * 2x EP all-to-all dispatch + combine per MoE layer per step
  * 1x EP all-to-all on backward
  * 1x optimizer-step AllReduce

"Cross-mesh" PPO (separate actor / ref / reward / critic meshes) is
missed because vLLM/SGLang's serving runtime is unbuildable on this
env (sgl_kernel cu130/py314/sm120 wheel doesn't exist). The toy MLP
PPO smoke from Phase 9-B captures the cross-mesh signature.

Run via phase10_ckpt_dcp_to_hf/run_ppo_real.sh.
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
    from torchtitan.distributed.parallel_dims import ParallelDims
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
    return ParallelDims(
        dp_replicate=dp_replicate, dp_shard=fsdp, cp=cp, tp=tp, pp=pp, ep=ep,
        etp=tp, world_size=world_size,
    )


def _build_model(parallel_dims, device, *, frozen: bool = False):
    """Build a kimi_linear AttnRes instance with parallelism applied.

    If ``frozen`` is True, all params have requires_grad=False
    (ref-model use case).
    """
    from torchtitan.experiments.kimi_linear.config_registry import (
        kimi_linear_436m_block_attn_res_n4,
    )
    from torchtitan.experiments.kimi_linear.attn_res_model import (
        KimiLinearAttnResModel,
    )
    from torchtitan.experiments.kimi_linear.parallelize import (
        parallelize_kimi_linear,
    )
    from torchtitan.config import (
        ParallelismConfig, TrainingConfig,
        ActivationCheckpointConfig, CompileConfig,
    )

    cfg = kimi_linear_436m_block_attn_res_n4()
    spec = cfg.model_spec.model
    model = KimiLinearAttnResModel(spec.kimi_config, num_blocks=spec.num_blocks)
    model = model.to(dtype=torch.bfloat16)

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

    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model.train()
    return model


def _load_dcp(model, ckpt_dir: Path, label: str = "model"):
    sd = model.state_dict()
    if dist.get_rank() == 0:
        print(f"[ppo] loading DCP into {label} from {ckpt_dir}")
    dcp.load(sd, checkpoint_id=str(ckpt_dir))


def _logprob_at_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather log-prob of label tokens from logits.

    logits: (B, T, V), labels: (B, T) -> (B, T) log-probs.
    """
    log_probs = logits.float().log_softmax(dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="DCP ckpt for both actor and ref initialization")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--micro-bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-coef", type=float, default=0.05)
    args = p.parse_args()

    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if rank == 0:
        print(f"[ppo] world={world} rank={rank} device={device}")

    pd = _build_parallel_dims(world)
    if rank == 0:
        print(
            f"[ppo] mesh: FSDP={pd.dp_shard} TP={pd.tp} EP={pd.ep} "
            f"(PP={pd.pp}, CP={pd.cp})"
        )

    if rank == 0:
        print("[ppo] building actor model...")
    actor = _build_model(pd, device, frozen=False)
    _load_dcp(actor, args.ckpt, "actor")

    if rank == 0:
        print("[ppo] building ref model...")
    ref = _build_model(pd, device, frozen=True)
    _load_dcp(ref, args.ckpt, "ref")

    # Optimizer: only actor params (ref is frozen).
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(actor_params, lr=args.lr, fused=False)

    if rank == 0:
        n_actor = sum(p.numel() for p in actor_params)
        n_ref = sum(p.numel() for p in ref.parameters())
        print(f"[ppo] actor trainable params: {n_actor:,}")
        print(f"[ppo] ref params (frozen): {n_ref:,}")
        print(f"[ppo] starting {args.n_steps} PPO steps "
              f"(micro_bs={args.micro_bs}, seq_len={args.seq_len}, lr={args.lr}, kl_coef={args.kl_coef})")

    vocab = 163840
    rng = torch.Generator(device="cuda").manual_seed(42 + rank)
    t0 = time.time()
    log_lines: list[str] = []

    for step in range(1, args.n_steps + 1):
        # Mock prompt batch + mock completion (the "rollout" in PPO; we
        # skip actual generation since we're profiling fabric, not RLHF
        # quality. Real generation would need KV cache integration which
        # is out-of-scope per user constraint).
        ids = torch.randint(
            0, vocab, (args.micro_bs, args.seq_len),
            device=device, dtype=torch.long, generator=rng,
        )
        # Use shifted ids as "labels" to compute logprob ratios.
        labels = torch.randint(
            0, vocab, (args.micro_bs, args.seq_len),
            device=device, dtype=torch.long, generator=rng,
        )

        # Ref forward (no_grad — frozen anchor).
        with torch.no_grad():
            ref_logits = ref(ids)
            ref_lp = _logprob_at_labels(ref_logits, labels)
            del ref_logits

        # Actor forward (autograd on).
        actor_logits = actor(ids)
        actor_lp = _logprob_at_labels(actor_logits, labels)
        del actor_logits

        # KL divergence proxy (per-token logprob delta).
        kl_term = (actor_lp - ref_lp).abs().mean()

        # Mock advantage — typical PPO has a value-fn or reward-model
        # producing per-sequence scalars; for fabric we just need a
        # gradient signal.
        advantage = torch.randn(actor_lp.shape, device=device, dtype=torch.float32, generator=rng)

        # PPO ratio loss: -E[advantage * log_ratio] + kl_coef * KL.
        # log_ratio = actor_lp - ref_lp.
        log_ratio = actor_lp - ref_lp
        ppo_loss = -(advantage * log_ratio).mean() + args.kl_coef * kl_term

        optimizer.zero_grad(set_to_none=True)
        ppo_loss.backward()
        optimizer.step()

        if rank == 0 and (step % 5 == 0 or step == 1):
            elapsed = time.time() - t0
            line = (
                f"[ppo] step={step:3d}/{args.n_steps} "
                f"loss={ppo_loss.item():.4f} kl={kl_term.item():.4f} "
                f"actor_lp={actor_lp.float().mean().item():.4f} "
                f"ref_lp={ref_lp.float().mean().item():.4f} "
                f"t={elapsed:.1f}s"
            )
            print(line, flush=True)
            log_lines.append(line)

    if rank == 0:
        print(f"[ppo] DONE {args.n_steps} steps in {time.time()-t0:.1f}s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
