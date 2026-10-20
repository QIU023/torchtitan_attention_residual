import os

import torch
import triton
import triton.language as tl


@triton.jit
def _add(x, y, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)


x = torch.randn(4096, device="cuda")
y = torch.randn(4096, device="cuda")
out = torch.empty_like(x)
_add[(4,)](x, y, out, 4096, BLOCK=1024)
print("triton raw", bool(torch.equal(out, x + y)), "ptxas", os.environ.get("TRITON_PTXAS_PATH", "bundled"), flush=True)
f = torch.compile(lambda a: (a.sin() * 2).sum(dim=-1))
print("inductor", float((f(torch.randn(64, 256, device="cuda")) ).abs().sum()) > 0, flush=True)
