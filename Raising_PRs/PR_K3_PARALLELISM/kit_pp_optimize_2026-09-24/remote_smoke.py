import os, sys, traceback
sys.path.insert(0, "/tmp/wt_ppbal")
import torch, torch.distributed as dist
from torchtitan.distributed.activation_storage import RemoteBackend
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r); dev = torch.device("cuda", r)
try:
    rb = RemoteBackend(dist.group.WORLD, dests={3: 7, 1: 6, 2: 5}, pool_bytes=2 << 30, staging_bytes=256 << 20, device=dev)
    ok = True
    if rb.parks:
        stream = torch.cuda.Stream(dev)
        x = torch.randn(4096, 2048, device=dev, dtype=torch.bfloat16)
        stream.wait_stream(torch.cuda.current_stream())
        p = rb.put(x, stream)
        out = torch.empty_like(x)
        rb.get(p, out, stream)
        stream.synchronize()
        ok = torch.equal(out, x)
        rb.free(p)
    print(f"rank {r}: parks={rb.parks} roundtrip_equal={ok}", flush=True)
except Exception:
    print(f"rank {r} FAILED:\n{traceback.format_exc()}", flush=True)
dist.barrier(); dist.destroy_process_group()
