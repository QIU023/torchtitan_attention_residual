"""#4380 collective micro-benchmark, logbook kit only: forward + backward time of the band gather as the PR does it
(torch.distributed.nn.functional.all_gather, list API) against spmd_types' redistribute S(0) -> R (all_gather_into_tensor,
reduce_scatter_tensor backward in bf16 or in fp32 as main's CP attention gathers K/V), bf16, ranks synchronised before every iteration.
env: ROWS (default 2592, a 1008 px band at CP2), ITERS (50). run: torchrun --nproc_per_node=<n> gather_bench.py"""

import os
import statistics
import time

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    group = dist.new_group(list(range(world)))
    rows, iters = int(os.environ.get("ROWS", "2592")), int(os.environ.get("ITERS", "50"))
    x = torch.randn(rows, 12, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(rows * world, 12, 128, device="cuda", dtype=torch.bfloat16)

    def pr_list():
        with spmd.no_typecheck():
            out = torch.cat(list(dist_nn.all_gather(x.contiguous(), group=group)))
        (out * upstream).sum().backward()

    def tensor_api(reduce_dtype):
        def run():
            with spmd.no_typecheck():
                out = spmd.redistribute(
                    x, group, src=spmd.S(0), dst=spmd.R, backward_options={"op_dtype": reduce_dtype}
                )
            (out * upstream).sum().backward()

        return run

    results = {}
    variants = (
        ("dist_nn list", pr_list),
        ("spmd bf16 reduce", tensor_api(torch.bfloat16)),
        ("spmd fp32 reduce", tensor_api(torch.float32)),
    )
    for name, fn in variants:
        for _ in range(5):
            fn()
        times = []
        for _ in range(iters):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        results[name] = 1e6 * statistics.median(times)
    x.grad = None
    with spmd.no_typecheck():
        a = torch.cat(list(dist_nn.all_gather(x.contiguous(), group=group)))
        b = spmd.redistribute(x, group, src=spmd.S(0), dst=spmd.R, backward_options={"op_dtype": torch.float32})
    if rank == 0:
        mb = x.numel() * 2 / 2**20
        print(
            f"GATHER_BENCH world {world} rows {rows} ({mb:.1f} MiB per rank) "
            + " ".join(f"{k}: {v:.0f} us" for k, v in results.items())
            + f" forward equal {torch.equal(a, b)}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
