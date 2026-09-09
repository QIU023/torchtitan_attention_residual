"""E2: sensitivity of attn-gym's KDA kernel (fused Triton vs reference) to a tiny input perturbation, fp32 inputs."""
import torch, sys
sys.path.insert(0, "/tmp/attn_gym_up")
from attn_gym.linear.kda.api import chunk_kda
torch.manual_seed(0); dev = "cuda"
B, T, H, K = 1, 256, 16, 128
def inputs(dtype):
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).to(dtype)
    v = torch.randn(B, T, H, K, device=dev).to(dtype)
    gate = (-torch.rand(B, T, H, K, device=dev) * 2.0).to(dtype)   # nonpositive log decay in [-2, 0]
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev)).to(dtype)
    return q, k, v, gate, beta
def run(impl, q, k, v, gate, beta):
    out = chunk_kda(q, k, v, gate, beta, impl=impl, autotune=False)
    return (out[0] if isinstance(out, tuple) else out).float()
def rel(a, b): return ((a - b).norm() / a.norm()).item()
for dtype in (torch.float32, torch.bfloat16):
    q, k, v, gate, beta = inputs(dtype)
    g = torch.Generator(device=dev).manual_seed(1)
    eps = 3e-7 if dtype == torch.float32 else 4e-3
    pert = lambda t: (t.float() * (1 + eps * torch.randn(t.shape, device=dev, generator=g))).to(dtype)
    qp, kp, vp = pert(q), pert(k), pert(v)
    res = {}
    for impl in ("fused", "reference"):
        try:
            o = run(impl, q, k, v, gate, beta); o2 = run(impl, qp, kp, vp, gate, beta); o3 = run(impl, q, k, v, gate, beta)
            res[impl] = (o, rel(o, o2), rel(o, o3))
        except Exception as e:
            print(f"{dtype} {impl}: failed {type(e).__name__}: {str(e)[:160]}")
    for impl, (o, d, d3) in res.items():
        print(f"{str(dtype):15s} {impl:10s}: output change for a {eps:.0e} relative input perturbation = {d:.2e}; same input twice = {d3:.1e}")
    if "fused" in res and "reference" in res:
        print(f"{str(dtype):15s} fused vs reference on the same input: {rel(res['reference'][0], res['fused'][0]):.2e}")
