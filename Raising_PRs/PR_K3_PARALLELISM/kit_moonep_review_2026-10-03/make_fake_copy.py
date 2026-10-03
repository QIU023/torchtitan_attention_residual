"""Write the fake-backend copy of the PR's GPU test: identical except that the multicast check returns True."""

import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
old = """    return all(
        _SymmetricMemory.has_multicast_support(torch._C._autograd.DeviceType.CUDA, i)
        for i in range(torch.cuda.device_count())
    )
"""
assert src.count(old) == 1
open(dst_path, "w").write(src.replace(old, "    return True\n"))
