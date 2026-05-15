"""Phase 9-B PPO smoke (vLLM-free) — fabric trace only.

Captures the unique RLHF fabric pattern: actor + ref on disjoint
sub-meshes, with cross-mesh logprob/KL exchange. No real reward,
no real rollout (uses a fixed-length sequence) — just enough work
to fire the distinctive collectives.

Layout (8 GPUs):
    ranks 0-3 = actor mesh (FSDP=2 x PP=2 / for trace simplicity: dp4)
    ranks 4-7 = ref   mesh (FSDP=2 x PP=2 / for trace simplicity: dp4)
    world PG  = all 8 ranks, used for cross-mesh KL allreduce

Per step:
    1. Both meshes forward a fake batch through their own copy of the
       model (random init llama3_debugmodel).
    2. Each mesh allreduces its logits along its sub-PG (FSDP grad sync
       analog).
    3. Cross-mesh: ranks 0-3 send their (sample-mean) logprob to
       ranks 4-7 via broadcast on world_pg, ref returns its logprob via
       reduce on world_pg. KL = actor_lp - ref_lp.
    4. Actor backwards & sgd step. Ref does nothing (frozen).

Captures:
    * Sub-mesh AllReduce / AllGather (FSDP-style)
    * Cross-mesh Broadcast / Reduce (the unique RLHF signature)

Run: torchrun --nproc_per_node=8 ppo_smoke_no_vllm.py
"""
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn


def make_actor_or_ref():
    # Tiny fake "language model": embed -> 2x mlp -> head.
    # Random init; we don't need real ckpts for fabric trace.
    vocab = 4096
    hidden = 1024
    return nn.Sequential(
        nn.Embedding(vocab, hidden),
        nn.Linear(hidden, hidden * 4),
        nn.GELU(),
        nn.Linear(hidden * 4, hidden),
        nn.GELU(),
        nn.Linear(hidden, vocab, bias=False),
    )


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    assert world == 8, f"This smoke requires world=8 (got {world})"
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    world_pg = dist.group.WORLD

    # Sub-meshes
    actor_ranks = [0, 1, 2, 3]
    ref_ranks = [4, 5, 6, 7]
    actor_pg = dist.new_group(ranks=actor_ranks)
    ref_pg = dist.new_group(ranks=ref_ranks)

    am_actor = rank in actor_ranks
    sub_pg = actor_pg if am_actor else ref_pg
    role = "ACTOR" if am_actor else "REF"

    if rank == 0:
        print(f"[smoke] launched: world={world}, actor={actor_ranks}, ref={ref_ranks}")

    # Both roles instantiate their model. Frozen for ref, trainable for actor.
    torch.manual_seed(42 if am_actor else 123)
    model = make_actor_or_ref().cuda()
    if am_actor:
        opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    seq_len = 256
    bs = 4
    vocab = 4096

    STEPS = 50
    t0 = time.time()
    for step in range(1, STEPS + 1):
        # Fake "rollout": each role gets the same fake batch (no actual
        # generation). Real PPO would have actor.generate here.
        torch.manual_seed(step)  # same on all ranks for determinism in mock
        ids = torch.randint(0, vocab, (bs, seq_len), device="cuda")

        # Forward.
        logits = model(ids)
        log_softmax = torch.log_softmax(logits, dim=-1)
        # Use last-token logprob of token 0 as a scalar summary.
        my_lp = log_softmax[:, -1, 0].mean()

        # Sub-mesh AR: simulate FSDP-style grad sync analog by
        # AllReducing the scalar summary across the role's 4 ranks.
        sub_lp = my_lp.detach().clone()
        dist.all_reduce(sub_lp, op=dist.ReduceOp.SUM, group=sub_pg)
        sub_lp /= len(actor_ranks)

        # Cross-mesh KL exchange.
        # 1) actor mesh broadcasts its sub_lp to all 8 ranks (src=0).
        actor_lp = sub_lp.detach().clone() if am_actor else torch.zeros_like(sub_lp)
        dist.broadcast(actor_lp, src=0, group=world_pg)
        # 2) ref mesh broadcasts its sub_lp to all 8 ranks (src=4).
        ref_lp = sub_lp.detach().clone() if not am_actor else torch.zeros_like(sub_lp)
        dist.broadcast(ref_lp, src=4, group=world_pg)
        # 3) Both compute KL.
        kl = (actor_lp - ref_lp).abs()

        # Actor: PPO-like loss = -(advantage * lp) + kl_coef * kl.
        if am_actor:
            advantage = torch.randn((), device="cuda")  # mock
            ppo_loss = -(advantage * my_lp) + 0.01 * kl
            opt.zero_grad()
            ppo_loss.backward()
            opt.step()

            # Sub-mesh AR for gradient sync (mock FSDP).
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=actor_pg)

        if rank == 0 and step % 5 == 0:
            elapsed = time.time() - t0
            print(f"[smoke] step={step:3d} kl={kl.item():.4f} actor_lp={actor_lp.item():.4f} ref_lp={ref_lp.item():.4f} t={elapsed:.1f}s", flush=True)

    if rank == 0:
        print(f"[smoke] DONE 50 steps in {time.time()-t0:.1f}s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
