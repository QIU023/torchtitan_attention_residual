"""Error against a float64 golden of main's compiled region, the list region (bmm) and the list region (reduction)."""
import torch
from arfuncs import main_version, list_version
from bi_attrib import list_version_sum
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm

def golden(sources, pw, nw, eps):
    v = torch.stack(sources, 1).double()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    s = (k * (nw.double() * pw.squeeze(0).double())).sum(-1)
    p = torch.softmax(s, -1)
    return (p.unsqueeze(-1) * v).sum(1)

D = 256
projection = Linear.Config(in_features=D, out_features=1).build().cuda().bfloat16()
norm = RMSNorm.Config(normalized_shape=D).build().cuda().bfloat16()
with torch.no_grad():
    projection.weight.normal_(0, 0.05)
fm = torch.compile(main_version, fullgraph=True, dynamic=True)
fl = torch.compile(list_version, fullgraph=True, dynamic=True)
fs = torch.compile(list_version_sum, fullgraph=True, dynamic=True)
errs = {"main": [], "list": [], "sum": []}
gerrs = {"main": [], "list": [], "sum": []}
for seed in range(10):
    torch.manual_seed(seed)
    src = [torch.randn(512, D, device="cuda", dtype=torch.bfloat16) for _ in range(5)]
    g = torch.randn(512, D, device="cuda", dtype=torch.float64)
    gs = [x.double().requires_grad_() for x in src]
    gold = golden(gs, projection.weight, norm.weight, norm.eps)
    ggrad = torch.autograd.grad(gold, gs, g)
    for name, f, mk in (("main", fm, lambda s: (s[-1], torch.stack(s[:-1], 1), projection, norm)),
                        ("list", fl, lambda s: (s, projection, norm)), ("sum", fs, lambda s: (s, projection, norm))):
        a = [x.detach().requires_grad_() for x in src]
        if name == "main":
            br = torch.stack(a[:-1], 1).detach().requires_grad_(); pa = a[-1]
            out = f(pa, br, projection, norm); gr = torch.autograd.grad(out, (br, pa), g.bfloat16()); grads = [*gr[0].unbind(1), gr[1]]
        else:
            out = f(a, projection, norm); grads = torch.autograd.grad(out, a, g.bfloat16())
        errs[name].append((out.double() - gold).abs().max().item())
        gerrs[name].append(max((x.double() - y).abs().max().item() for x, y in zip(grads, ggrad)))
for name in errs:
    print(f"{name}: output max err vs fp64 {max(errs[name]):.3e}, grad max err {max(gerrs[name]):.3e}")
