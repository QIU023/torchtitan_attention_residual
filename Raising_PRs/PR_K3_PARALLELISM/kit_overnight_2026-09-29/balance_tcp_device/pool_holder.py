# 5060 test only: holds one destination rank's balance pool in its own process on that rank's GPU, so
# mooncake's tcp copies into it run on this process's legacy default stream, which no training work uses.
# Usage: pool_holder.py <cuda index> <bytes> <protocol> [device_names]; prints "READY <session> <base>",
# then blocks until its stdin closes (the trainer exits).
import ctypes
import glob
import os
import socket
import sys

import torch


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
    index, nbytes, protocol = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = sys.argv[4] if len(sys.argv) > 4 else ""
    torch.cuda.set_device(index)
    pool = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{index}")
    engine = load_engine()()
    host = os.environ.get("MC_LOCAL_HOSTNAME") or socket.gethostname()
    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    assert engine.initialize(f"{host}:{port}", "P2PHANDSHAKE", protocol, names) == 0
    assert engine.register_memory(pool.data_ptr(), nbytes) == 0
    print(f"READY {host}:{engine.get_rpc_port()} {pool.data_ptr()}", flush=True)
    sys.stdin.read()


if __name__ == "__main__":
    main()
