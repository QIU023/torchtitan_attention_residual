import torch

torch.zeros(1, device="cuda")
maps = open("/proc/self/maps").read().splitlines()
libs = sorted({l.split()[-1] for l in maps if "libcuda" in l or "libnvidia-ptxjitcompiler" in l})
print("torch", torch.__version__, "libs", libs, flush=True)
