#!/bin/bash
# Run before installing anything: MoonEP's buffers assert NVSwitch multicast on every GPU.
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m | head -6
python3 - <<'PY'
import ctypes
cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit(0)
count = ctypes.c_int()
cuda.cuDeviceGetCount(ctypes.byref(count))
values = []
for i in range(count.value):
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), i)
    v = ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(v), 132, dev)  # CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
    values.append(v.value)
print("multicast per GPU:", values)
print("MOONEP OK" if count.value >= 4 and all(values) else "NOT USABLE FOR MOONEP: return this box")
PY
