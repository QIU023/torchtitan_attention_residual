import ctypes, glob, json, os, socket
import torch, torch.distributed as dist
import nvidia
for root in nvidia.__path__:
    for lib in glob.glob(os.path.join(root, "cuda_runtime", "lib", "libcudart.so.12*")):
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL); break
from mooncake.engine import TransferEngine
dist.init_process_group("gloo")
r = dist.get_rank(); torch.cuda.set_device(r)
host = socket.gethostname()
with socket.socket() as s:
    s.bind(("", 0)); port = s.getsockname()[1]
eng = TransferEngine()
assert eng.initialize(f"{host}:{port}", "P2PHANDSHAKE", "tcp", "") == 0
session = f"{host}:{eng.get_rpc_port()}"
topo = json.loads(eng.get_local_topology())
nbytes = 8 << 20
buf = torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True)
assert eng.register_memory(buf.data_ptr(), nbytes) == 0
book = [None, None]
dist.all_gather_object(book, (session, buf.data_ptr()))
stream = torch.cuda.Stream()
if r == 0:
    src = torch.arange(nbytes, dtype=torch.int64).to(torch.uint8)
    buf.copy_(src)
    import time
    t = time.time()
    rc = eng.transfer_write_on_cuda(book[1][0], buf.data_ptr(), book[1][1], nbytes, stream.cuda_stream)
    print("write_on_cuda returned", rc, flush=True)
    stream.synchronize()
    print(f"write done in {time.time()-t:.4f}s", flush=True)
    dist.barrier()
    back = torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True)
    assert eng.register_memory(back.data_ptr(), nbytes) == 0
    rc = eng.transfer_read_on_cuda(book[1][0], back.data_ptr(), book[1][1], nbytes, stream.cuda_stream)
    stream.synchronize()
    print("read back equal:", torch.equal(back, src), flush=True)
    rc2 = eng.transfer_submit_write(book[1][0], buf.data_ptr(), book[1][1], nbytes)
    print("submit_write batch id", rc2, "status", eng.transfer_check_status(rc2), flush=True)
    dist.barrier()
else:
    dist.barrier()
    print("peer received equal:", torch.equal(buf, torch.arange(nbytes, dtype=torch.int64).to(torch.uint8)), flush=True)
    dist.barrier()
dist.destroy_process_group()
