import torch
from torch import Tensor

seen = {}

@torch.library.custom_op("probe::two_out", mutates_args=())
def two_out(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return x * 2, w * 3, torch.tensor([7])

def setup_context(ctx, inputs, output):
    ctx.n = int(output[2])

def backward(ctx, g_a, g_b, *_):
    seen["g_b"] = g_b
    return g_a * 2, g_b.float() * 3

two_out.register_autograd(backward, setup_context=setup_context)
two_out.register_effect(torch.library.EffectType.ORDERED)

x = torch.randn(4, requires_grad=True)
w = torch.randn(4, requires_grad=True)
a, b, pid = two_out(x, w)
a.sum().backward()  # b unused
print("unused output grad:", type(seen["g_b"]).__name__, None if seen["g_b"] is None else seen["g_b"].abs().sum().item())

def block(x, w):
    a, b, pid = two_out(x, w)
    key = int(pid)
    return a.sum() + b.sum() + key

for fg in (False, True):
    torch._dynamo.reset()
    try:
        out = torch.compile(block, fullgraph=fg, backend="eager")(x, w)
        print(f"compile fullgraph={fg}: ran, out {out.item():.3f}")
    except Exception as e:
        print(f"compile fullgraph={fg}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
