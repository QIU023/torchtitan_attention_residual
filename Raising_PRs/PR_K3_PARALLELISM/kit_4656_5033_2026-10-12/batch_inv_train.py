import sys, torch
from arfuncs import main_version, list_version
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm
D = 128
projection = Linear.Config(in_features=D, out_features=1).build().cuda().bfloat16()
norm = RMSNorm.Config(normalized_shape=D).build().cuda().bfloat16()
fm = torch.compile(main_version, fullgraph=True, dynamic=True)
fl = torch.compile(list_version, fullgraph=True, dynamic=True)
bad = {"main": 0, "list": 0, "neq": 0}
for seed in range(20):
    torch.manual_seed(seed)
    src = [torch.randn(64, D, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    outs = {}
    for name, f, args in (("main", fm, lambda s: (s[3], torch.stack(s[:3], 1), projection, norm)), ("list", fl, lambda s: (s, projection, norm))):
        a = [x.detach().requires_grad_() for x in args(src) if isinstance(x, torch.Tensor)]
        full = f(*args([x.detach().requires_grad_() for x in src]))
        parts = [f(*args([x.detach().chunk(2)[p].requires_grad_() for x in src])) for p in range(2)]
        outs[name] = full
        bad[name] += not torch.equal(full, torch.cat(parts))
    bad["neq"] += not torch.equal(outs["main"], outs["list"])
print("over 20 seeds, training graphs: batch-variant main", bad["main"], "list", bad["list"], "; main != list", bad["neq"])
# eager bmm invariance
torch.manual_seed(0)
p = torch.softmax(torch.randn(64, 1, 4, device="cuda"), -1); v = torch.randn(64, 4, D, device="cuda")
print("eager bmm batch-invariant", torch.equal(torch.bmm(p, v), torch.cat([torch.bmm(p[:32], v[:32]), torch.bmm(p[32:], v[32:])])))
