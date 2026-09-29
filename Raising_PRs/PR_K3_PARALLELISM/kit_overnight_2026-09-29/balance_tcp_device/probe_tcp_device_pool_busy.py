# Does a mooncake tcp write into a peer's device pool wait for work queued on the peer's GPU?
# Rank 1 holds the pool and queues a spin kernel of about SPIN_S seconds, on the default (legacy)
# stream or on a side stream; rank 0 then writes 64 MiB and times the write.
# Cases: device pool + spin on default stream, device pool + spin on a side stream,
# pinned host pool + spin on default stream (the PR's tcp layout).
# PROTO=tcp (default) or nvlink_intra / rdma on a box that has them (H100: check before the balance cells).
import ctypes, glob, os, socket, time
import torch
import torch.distributed as dist

SPIN_S = 4.0

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
    proto = os.environ.get("PROTO", "tcp")
    assert eng.initialize(f"{host}:{port}", "P2PHANDSHAKE", proto, os.environ.get("DEVICE_NAMES", "")) == 0
    session = f"{host}:{eng.get_rpc_port()}"
    n = 64 << 20
    # calibrate the spin kernel on every rank
    t = time.perf_counter(); torch.cuda._sleep(int(1e8)); torch.cuda.synchronize(); per = (time.perf_counter() - t) / 1e8
    cycles = int(SPIN_S / per)
    pools = {}
    if rank == 1:
        pools["device"] = torch.zeros(n, dtype=torch.uint8, device=dev)
        pools["host"] = torch.zeros(n, dtype=torch.uint8, pin_memory=True)
        for p in pools.values():
            assert eng.register_memory(p.data_ptr(), n) == 0
    book = [None, None]
    dist.all_gather_object(book, (session, {k: v.data_ptr() for k, v in pools.items()}))
    stage = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    if rank == 0:
        assert eng.register_memory(stage.data_ptr(), n) == 0
        stage.random_(0, 255)
        dsess, ptrs = book[1]
        for _ in range(40):  # open the segment while the peer is idle
            try:
                if eng.transfer_sync_write(dsess, stage.data_ptr(), ptrs["device"], 512) == 0:
                    break
            except RuntimeError:
                pass
            time.sleep(0.25)
    cases = [("device", "default"), ("device", "side"), ("host", "default"), ("device", "idle")]
    side = torch.cuda.Stream(dev)
    for pool, where in cases:
        dist.barrier()
        if rank == 1:
            t0 = time.perf_counter()
            if where == "default":
                torch.cuda._sleep(cycles)
            elif where == "side":
                with torch.cuda.stream(side):
                    torch.cuda._sleep(cycles)
        dist.barrier()                       # the spin is queued on rank 1 before rank 0 writes
        if rank == 0:
            t = time.perf_counter()
            rc = eng.transfer_sync_write(dsess, stage.data_ptr(), ptrs[pool], n)
            dt = time.perf_counter() - t
            print(f"{proto} pool={pool:6s} spin on {where:7s}: write rc={rc} took {dt:6.2f} s (spin {SPIN_S} s)", flush=True)
        if rank == 1:
            torch.cuda.synchronize()
            print(f"   rank 1: spin done after {time.perf_counter() - t0:6.2f} s", flush=True)
        dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
