# Can mooncake's TCP transport land a write in a peer's registered GPU memory?
# Two ranks on two GPUs: rank 1 registers a device pool, rank 0 writes into it from a
# pinned host staging buffer (sync and stream-ordered), then reads it back.
import ctypes, glob, os, socket, time
import torch
import torch.distributed as dist

def load_engine():
    try:
        from mooncake.engine import TransferEngine
    except ImportError:
        import nvidia
        for root in nvidia.__path__:
            for lib in glob.glob(os.path.join(root, "cuda_runtime", "lib", "libcudart.so.12*")):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                break
        from mooncake.engine import TransferEngine
    return TransferEngine

def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    eng = load_engine()()
    host = socket.gethostname()
    with socket.socket() as s:
        s.bind(("", 0)); port = s.getsockname()[1]
    assert eng.initialize(f"{host}:{port}", "P2PHANDSHAKE", "tcp", "") == 0
    session = f"{host}:{eng.get_rpc_port()}"
    n = 64 << 20
    pool_ptr = 0
    if rank == 1:
        pool = torch.zeros(n, dtype=torch.uint8, device=dev)
        pool_ptr = pool.data_ptr()
        assert eng.register_memory(pool_ptr, n) == 0
    book = [None, None]
    dist.all_gather_object(book, (session, pool_ptr))
    dist.barrier()
    res = {}
    if rank == 0:
        stage = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        assert eng.register_memory(stage.data_ptr(), n) == 0
        src = torch.randint(0, 255, (n,), dtype=torch.uint8, device=dev)
        stage.copy_(src.cpu())
        dsess, dptr = book[1]
        for _ in range(40):
            try:
                if eng.transfer_sync_write(dsess, stage.data_ptr(), dptr, 512) == 0:
                    break
            except RuntimeError:
                pass
            time.sleep(0.25)
        t = time.perf_counter(); rc = eng.transfer_sync_write(dsess, stage.data_ptr(), dptr, n); dt = time.perf_counter() - t
        res["sync_write_rc"] = rc; res["sync_write_GBps"] = n / dt / 1e9
        dist.barrier()                      # rank 1 checks the sync write
        stream = torch.cuda.Stream(dev)
        stage2 = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        assert eng.register_memory(stage2.data_ptr(), n) == 0
        src2 = torch.randint(0, 255, (n,), dtype=torch.uint8, device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream):
            stage2.copy_(src2, non_blocking=True)
        t = time.perf_counter()
        eng.transfer_write_on_cuda(dsess, stage2.data_ptr(), dptr, n, stream.cuda_stream)
        stream.synchronize(); dt = time.perf_counter() - t
        res["cuda_write_GBps"] = n / dt / 1e9
        dist.barrier()                      # rank 1 checks the stream-ordered write
        back = torch.zeros(n, dtype=torch.uint8, pin_memory=True)
        assert eng.register_memory(back.data_ptr(), n) == 0
        rc = eng.transfer_sync_read(dsess, back.data_ptr(), dptr, n)
        res["read_rc"] = rc; res["read_ok"] = bool(torch.equal(back, src2.cpu()))
        back2 = torch.zeros(n, dtype=torch.uint8, pin_memory=True)
        assert eng.register_memory(back2.data_ptr(), n) == 0
        out = torch.zeros(n, dtype=torch.uint8, device=dev)
        t = time.perf_counter()
        eng.transfer_read_on_cuda(dsess, back2.data_ptr(), dptr, n, stream.cuda_stream)
        with torch.cuda.stream(stream):
            out.copy_(back2, non_blocking=True)
        stream.synchronize(); dt = time.perf_counter() - t
        res["cuda_read_GBps"] = n / dt / 1e9
        res["cuda_read_ok"] = bool(torch.equal(out.cpu(), src2.cpu()))
        torch.save(src.cpu(), "/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/probe_src1.pt")
        torch.save(src2.cpu(), "/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/probe_src2.pt")
        dist.barrier()
    else:
        dist.barrier()
        torch.cuda.synchronize()
        pool_snapshot1 = pool.cpu()
        dist.barrier()
        torch.cuda.synchronize()
        pool_snapshot2 = pool.cpu()
        dist.barrier()
        s1 = torch.load("/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/probe_src1.pt")
        s2 = torch.load("/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/probe_src2.pt")
        res["pool_after_sync_write_ok"] = bool(torch.equal(pool_snapshot1, s1))
        res["pool_after_cuda_write_ok"] = bool(torch.equal(pool_snapshot2, s2))
        res["pool_nonzero_frac"] = float((pool_snapshot2 != 0).float().mean())
    print(f"rank {rank}: {res}", flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
