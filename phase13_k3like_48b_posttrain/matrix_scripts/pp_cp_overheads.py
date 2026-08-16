"""Two overheads the review flagged as "profile it first", measured.

Both are cheap to reason about wrongly, which is why they were ranked last rather than
guessed at.

C5 -- ``_keepalive_touch``. The last pipeline stage keeps the received delta tensor on
its autograd graph with ``0.0 * prev_recv_tensor.sum()``. Mathematically it is a no-op
that cannot perturb the loss, but it is an O(T*D) reduction plus a broadcast add on
every micro-batch of every stage. The question is whether that is measurable next to a
stage's own forward.

C6 -- ``_build_cp_subgroups``. Every sub-CP group layout is created up front, since
``new_group`` must be called by all ranks in the same order and a per-batch call would
hang. For cp=8 that is the divisors 1, 2, 4, 8, so four rounds of group creation. The
question is what that costs at startup, and how it grows.

    torchrun --nproc_per_node=8 pp_cp_overheads.py

C6 needs the world size it is measuring; C5 does not care and runs on rank 0.
"""

import statistics
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)


def median_ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


# ----- C5: the keepalive touch against a comparable matmul ------------------- #
if rank == 0:
    print("C5  keepalive touch vs one projection, per micro-batch", flush=True)
    print("  T x D        touch(ms)  linear(ms)  share", flush=True)
    # Shapes the adapter actually sees: (batch*seq) x hidden, at the gate's sequence
    # lengths and both the debug and the K3-mini hidden sizes.
    for tokens, dim in ((256, 512), (2048, 512), (4096, 2048), (16384, 2048)):
        x = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
        w = torch.randn(dim, dim, device=device, dtype=torch.bfloat16)

        def touch():
            return 0.0 * x.sum()

        def linear():
            return x @ w

        t_touch = median_ms(touch)
        t_lin = median_ms(linear)
        print(
            f"  {tokens:5d} x {dim:<5d} {t_touch:9.4f}  {t_lin:10.4f}  "
            f"{t_touch / max(t_lin, 1e-9) * 100:5.1f}%",
            flush=True,
        )
    print(
        "  A stage runs many projections per micro-batch, so its forward is a large\n"
        "  multiple of the linear column; the touch is charged once per stage per\n"
        "  micro-batch. Read the share as an upper bound on the per-stage cost.",
        flush=True,
    )

dist.barrier()

# ----- C6: pre-creating every sub-CP group layout --------------------------- #
# Timed on the real helper so the number describes the code that ships, not a sketch of
# it. It all-gathers the CP rank lists first, so every rank iterates the same global
# list in the same order -- that ordering is the reason this cannot be done per batch.
from torchtitan.models.kimi_k3.parallelize import _build_cp_subgroups  # noqa: E402

torch.cuda.synchronize()
t0 = time.perf_counter()
groups = _build_cp_subgroups(dist.group.WORLD)
elapsed = (time.perf_counter() - t0) * 1e3

divisors = [d for d in range(1, world + 1) if world % d == 0]
if rank == 0:
    print(
        f"\nC6  _build_cp_subgroups at cp={world}: {elapsed:.1f} ms for "
        f"{len(divisors)} layout(s) {divisors}, {len(groups)} kept on this rank",
        flush=True,
    )
    print(
        "  new_group count grows with the DIVISORS of cp, not with cp itself, so the\n"
        "  cp=16 and cp=32 cases add one or two rounds rather than doubling. NCCL\n"
        "  creates the communicator lazily on first use, so an unused layout costs\n"
        "  only this call.",
        flush=True,
    )
dist.destroy_process_group()
