"""Does a receive posted ahead of its send hold back later GPU work on the posting rank?

Two ranks on NCCL. Rank 1 posts an irecv from rank 0, then runs a small kernel and reads it
back (the DEP prologue: _post_receives, then the encode's grid_thw.tolist()). Rank 0 sends only
after rank 1 reports that its read-back finished (the features come only after the peer's
encode). MODE picks where rank 1 runs its kernel:
  legacy  - the current stream (the legacy default stream, as the trainer runs)
  side    - a torch-created side stream
  late    - the same as legacy, but the irecv is posted after the read-back (control)
"""

import datetime
import os
import sys
import time

import torch
import torch.distributed as dist

mode = sys.argv[1]
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=45))
rank = dist.get_rank()
torch.cuda.set_device(rank)
store = dist.distributed_c10d._get_default_store()
group = dist.new_group([0, 1])
payload = torch.arange(1024, dtype=torch.float32, device="cuda")
dist.barrier()
if os.environ.get("WARM", "1") == "1":
    # The first P2P between two ranks blocks until both reach it (vision_dep's _connect);
    # warm with the same size and the same irecv/send pattern the test uses.
    for _ in range(2):
        if rank == 1:
            w = dist.irecv(torch.empty(1024, device="cuda"), src=0, group=group)
            w.wait()
        else:
            dist.send(payload, dst=1, group=group)
        torch.cuda.synchronize()
torch.cuda.synchronize()
dist.barrier()

if rank == 1 and os.environ.get("PRELOAD") == "1":
    # The same ops once before the receive is posted, so their kernels are loaded.
    (torch.ones(8, device="cuda") * 3).sum().tolist()
    torch.cuda.synchronize()

if rank == 1:
    buf = torch.empty(1024, device="cuda")
    work = None
    if mode != "late":
        t_post = time.time()
        work = dist.irecv(buf, src=0, group=group)
        print(f"[{mode}] rank 1 irecv returned after {time.time() - t_post:.3f} s", flush=True)
    stream = torch.cuda.Stream() if mode.startswith("side") else torch.cuda.current_stream()
    t0 = time.time()
    if mode.endswith("_event"):
        with torch.cuda.stream(stream):
            out = torch.ones(8, device="cuda") * 3
            done = torch.cuda.Event()
            done.record(stream)
        while not done.query() and time.time() - t0 < 5:
            time.sleep(0.01)
        print(f"[{mode}] rank 1 kernel event {'completed' if done.query() else 'NOT completed'} "
              f"after {time.time() - t0:.3f} s with the receive pending", flush=True)
    else:
        with torch.cuda.stream(stream):
            value = (torch.ones(8, device="cuda") * 3).sum().tolist()
        print(f"[{mode}] rank 1 read back {value} after {time.time() - t0:.3f} s "
              f"with the receive {'pending' if work is not None else 'not posted'}", flush=True)
    store.set("read_back", "1")
    if work is None:
        work = dist.irecv(buf, src=0, group=group)
    work.wait()
    torch.cuda.synchronize()
    print(f"[{mode}] rank 1 received, equal={torch.equal(buf, payload)}", flush=True)
else:
    store.wait(["read_back"], datetime.timedelta(seconds=40))
    dist.send(payload, dst=1, group=group)
    torch.cuda.synchronize()
    print(f"[{mode}] rank 0 sent", flush=True)

dist.destroy_process_group()
