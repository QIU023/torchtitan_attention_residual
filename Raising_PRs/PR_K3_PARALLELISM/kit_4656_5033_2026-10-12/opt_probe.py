import sys, torch
from arfuncs import main_version, list_version
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm
D = 128
projection = Linear.Config(in_features=D, out_features=1).build().cuda().bfloat16()
norm = RMSNorm.Config(normalized_shape=D).build().cuda().bfloat16()
fm = torch.compile(main_version, fullgraph=True, dynamic=True)
variants = {
    "default": {},
    "no_unroll": {"unroll_reductions_threshold": 1},
    "no_unroll_no_persist": {"unroll_reductions_threshold": 1, "triton.persistent_reductions": False},
}
for vname, opts in variants.items():
    torch._dynamo.reset()
    fm = torch.compile(main_version, fullgraph=True, dynamic=True)
    fl = torch.compile(list_version, fullgraph=True, dynamic=True, options=opts or None)
    neq_out = neq_grad = 0
    for seed in range(20):
        torch.manual_seed(seed)
        src = [torch.randn(64, D, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        g = torch.randn(64, D, device="cuda", dtype=torch.bfloat16)
        a = [x.detach().requires_grad_() for x in src]
        bm = torch.stack(a[:3], 1)
        om = fm(a[3], bm, projection, norm)
        gm = torch.autograd.grad(om, a, g)
        b = [x.detach().requires_grad_() for x in src]
        ol = fl(b, projection, norm)
        gl = torch.autograd.grad(ol, b, g)
        neq_out += not torch.equal(om, ol)
        neq_grad += not all(torch.equal(x, y) for x, y in zip(gm, gl))
    print(vname, "outputs differ", neq_out, "/20, grads differ", neq_grad, "/20")
