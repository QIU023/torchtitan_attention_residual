import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
dev = torch.device("cuda", r)
warm = torch.zeros(1, device=dev)
dist.all_reduce(warm); torch.cuda.synchronize()
if r == 0:
    base = torch.randn(64, 1024, 1024, device=dev)  # 256 MiB
    view = base[:, :16].transpose(0, 1)  # non-contiguous view
    before = torch.cuda.memory_allocated()
    works = dist.batch_isend_irecv([dist.P2POp(dist.isend, base[8:], 1)])
    del base, view
    torch.cuda.synchronize(); time.sleep(3)
    print(f"rank0 after send, before wait: {torch.cuda.memory_allocated()/2**20:.0f} MiB (had {before/2**20:.0f})", flush=True)
    for w in works: w.wait()
    torch.cuda.synchronize()
    print(f"rank0 after wait: {torch.cuda.memory_allocated()/2**20:.0f} MiB", flush=True)
    print("AVOID_RECORD_STREAMS env:", os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS"), flush=True)
else:
    buf = torch.empty(56, 1024, 1024, device=dev)
    for w in dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf, 0)]): w.wait()
    torch.cuda.synchronize()
dist.barrier(); dist.destroy_process_group()
