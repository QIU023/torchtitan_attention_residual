# Local shim for the fake MoonEP package on GPUs without NVSwitch multicast; never committed.
import os

if os.environ.get("FAKE_MULTICAST") == "1":
    try:
        from torch._C._distributed_c10d import _SymmetricMemory

        _SymmetricMemory.has_multicast_support = staticmethod(lambda *args: True)
    except Exception:
        pass
