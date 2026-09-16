"""Does this box support the NVLink multicast MoonEP's buffers assert? Run before renting hours.

    python moonep_multicast_probe.py

Prints, per visible GPU: name, compute capability, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED (driver attribute 132),
and torch's _SymmetricMemory.has_multicast_support. MoonEP needs every value True (Hopper or newer behind
an NVSwitch: an HGX board; four H100 SXM on direct NVLinks read False on 2026-09-15).
"""
import torch


def main() -> None:
    import ctypes

    n = torch.cuda.device_count()
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    try:
        from torch._C._distributed_c10d import _SymmetricMemory

        def sym(i: int) -> bool:
            return _SymmetricMemory.has_multicast_support(torch._C._autograd.DeviceType.CUDA, i)

    except Exception as exc:  # noqa: BLE001
        print("torch symmetric memory probe unavailable:", exc)
        sym = None
    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        dev, val = ctypes.c_int(), ctypes.c_int()
        cu.cuDeviceGet(ctypes.byref(dev), i)
        rc = cu.cuDeviceGetAttribute(ctypes.byref(val), 132, dev)  # CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
        print(
            f"cuda:{i} {torch.cuda.get_device_name(i)} sm{cap[0]}{cap[1]} "
            f"multicast_attr={val.value if rc == 0 else f'rc={rc}'} "
            f"torch_symm_mem_multicast={sym(i) if sym else None}"
        )


if __name__ == "__main__":
    main()
