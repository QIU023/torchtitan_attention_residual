"""Phase 10 Stage J — autoregressive-style inference fabric smoke.

Without a real KV cache (porting KDA + MLA cache is out-of-scope per
the user's "don't deep-dive inference framework" constraint), captures
the **two limiting fabric patterns** that autoregressive inference
exhibits:

* ``growing`` mode: the naive no-cache path. Each generation step
  re-forwards the FULL prefix (length grows P, P+1, P+2, ...). Fabric
  volume grows per step; total fabric over N tokens is O(N²) seq
  work + O(N) collectives at growing message size.
* ``single_token`` mode: the ideal-cache path. Each generation step
  forwards a seq=1 batch (the new token). Fabric volume per step is
  constant-tiny; total over N tokens is O(N) collectives at a
  fixed small message size.

Real production decode sits between the two, with a one-time prefill
(growing-style) followed by per-token decodes (single-token-style).
The two modes here bracket the design space for IXIA modeling.

Run via phase10/run_autoregressive.sh.
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

# Reuse Stage D's mesh + model build helpers.
from phase10.inference_torchtitan import (  # noqa: E402
    _build_parallel_dims,
    _build_model_and_parallelize,
    _load_dcp,
    _init_dist,
)


def _next_token(logits: torch.Tensor) -> torch.Tensor:
    """Greedy: take argmax over vocab on the last position."""
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--mode", choices=("growing", "single_token"), required=True)
    p.add_argument("--n-generations", type=int, default=20,
                   help="Number of independent prompt batches.")
    p.add_argument("--n-tokens", type=int, default=20,
                   help="New tokens per generation.")
    p.add_argument("--prompt-len", type=int, default=64,
                   help="Initial prompt length (used as P0 for growing mode).")
    p.add_argument("--micro-bs", type=int, default=2)
    args = p.parse_args()

    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if rank == 0:
        print(f"[autoregress] world={world} mode={args.mode}")

    pd = _build_parallel_dims(world)
    model = _build_model_and_parallelize(pd, device)
    _load_dcp(model, args.ckpt)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[autoregress] model params: {n:,}")
        print(
            f"[autoregress] {args.n_generations} generations x "
            f"{args.n_tokens} tokens (mode={args.mode}, prompt={args.prompt_len})"
        )

    vocab = 163840
    rng = torch.Generator(device="cuda").manual_seed(42 + rank)
    t0 = time.time()
    total_calls = 0

    with torch.no_grad():
        for gen_idx in range(1, args.n_generations + 1):
            # Initial prompt batch for this generation.
            prefix = torch.randint(
                0, vocab, (args.micro_bs, args.prompt_len),
                device=device, dtype=torch.long, generator=rng,
            )

            for tok in range(args.n_tokens):
                if args.mode == "growing":
                    # Re-forward full prefix (P + i tokens at step i).
                    logits = model(prefix)
                    next_tok = _next_token(logits.float())
                    prefix = torch.cat([prefix, next_tok], dim=-1)
                elif args.mode == "single_token":
                    # Forward only the new (or single) token. The model
                    # has no real KV cache so this fabric profile is
                    # what *would* happen with a cache; output is
                    # discarded.
                    if tok == 0:
                        # First call uses the prompt to "prime" — fabric
                        # for single_token mode counts only the single-
                        # token decode steps below, so we skip this.
                        pass
                    one_tok = torch.randint(
                        0, vocab, (args.micro_bs, 1),
                        device=device, dtype=torch.long, generator=rng,
                    )
                    logits = model(one_tok)
                    next_tok = _next_token(logits.float())
                    # No prefix update needed — single_token fabric is
                    # constant per step.
                total_calls += 1
                del logits

            if rank == 0 and gen_idx % 5 == 0:
                elapsed = time.time() - t0
                print(
                    f"[autoregress] gen={gen_idx:3d}/{args.n_generations} "
                    f"calls={total_calls} t={elapsed:.1f}s"
                )

    if rank == 0:
        print(
            f"[autoregress] DONE mode={args.mode} "
            f"{total_calls} forward calls in {time.time()-t0:.1f}s"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
