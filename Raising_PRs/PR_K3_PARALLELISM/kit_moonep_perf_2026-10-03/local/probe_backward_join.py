"""Shared experts on the side stream with their backward delayed: are the gradients still exact without the
x_TD hook? Run from a worktree root: python probe_backward_join.py <repeats>."""
import sys

import torch

sys.path.insert(0, "tests/unit_tests/gpu")
import test_moe_shared_experts_stream as t  # noqa: E402


class _SleepInBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        torch.cuda._sleep(200_000_000)
        return grad


class _DelayedBoth(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x_TD):
        torch.cuda._sleep(100_000_000)
        return _SleepInBackward.apply(self.inner(x_TD))


t._Delayed = _DelayedBoth
repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
_, out_ref, gx_ref, g_ref = t._run(shared_experts_stream=False)
bad = 0
for i in range(repeats):
    moe, out, gx, g = t._run(shared_experts_stream=True)
    ok_out = torch.equal(out, out_ref)
    ok_params = all(torch.equal(g[n], g_ref[n]) for n in g_ref)
    gx_err = (gx - gx_ref).abs().max().item()
    ok_gx = torch.allclose(gx, gx_ref, rtol=1e-5, atol=1e-4)
    bad += not (ok_out and ok_params and ok_gx)
    print(f"run {i}: output exact {ok_out}, parameter grads exact {ok_params}, input grad max abs diff {gx_err:.3e} within tol {ok_gx}")
print("ALL OK" if bad == 0 else f"{bad} BAD")
