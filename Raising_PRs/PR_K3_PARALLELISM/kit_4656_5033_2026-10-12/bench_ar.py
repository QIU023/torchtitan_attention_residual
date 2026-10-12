"""fwd+bwd time and incremental peak of the aggregation region at a K3-like shape: main (stack + partial, symbolic N, bmm),
list (stack inside, static N, bmm), list_sum (stack inside, weighted sum as a reduction)."""
import sys, time, torch
from arfuncs import main_version, list_version
from bi_attrib import list_version_sum
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm
T, N, D = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
projection = Linear.Config(in_features=D, out_features=1).build().cuda().bfloat16()
norm = RMSNorm.Config(normalized_shape=D).build().cuda().bfloat16()
torch.manual_seed(0)
src = [torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(N)]
g = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
for name, fn, args in (("main", main_version, lambda: (src[-1], torch.stack(src[:-1], 1).detach().requires_grad_(), projection, norm)),
                       ("list", list_version, lambda: (src, projection, norm)),
                       ("list_sum", list_version_sum, lambda: (src, projection, norm))):
    f = torch.compile(fn, fullgraph=True, dynamic=True)
    a = args()
    for _ in range(3):
        out = f(*a); out.backward(g)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    out = f(*a); out.backward(g); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    t0 = time.perf_counter()
    for _ in range(20):
        out = f(*a); out.backward(g)
    torch.cuda.synchronize()
    print(f"{name}: {1000 * (time.perf_counter() - t0) / 20:.3f} ms fwd+bwd, incremental peak {peak:.3f} GiB")
