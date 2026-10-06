"""Scratch check (not committed): a single-GPU test that needs the MoE input hook.

Two micro-batches accumulate into the shared experts' .grad, so the second one adds in place on the
side stream; a tensor hook on each shared-expert parameter sleeps on that stream right before the add.
A reader wrapped around the MoE input clones the shared experts' gradients on the main stream in its
backward, which runs after the MoE input's gradient exists, as FSDP2's post-backward does."""
import sys
import torch

sys.path.insert(0, "tests/unit_tests/gpu")
import test_moe_shared_experts_stream as t  # noqa: E402
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config, make_routed_experts_config, make_router_config, make_shared_expert_ffn_config)


class _ReadSharedGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, moe, out):
        ctx.moe, ctx.out = moe, out
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.out.update({n: p.grad.clone() for n, p in ctx.moe.shared_experts.named_parameters()})
        return grad, None, None


def run(stream: bool, sleep: bool):
    config = make_moe_config(
        num_experts=t.E,
        router=make_router_config(dim=t.D, num_experts=t.E, gate_param_init={}, score_func=Sigmoid.Config(), top_k=t.K),
        routed_experts=make_routed_experts_config(dim=t.D, hidden_dim=t.H, num_experts=t.E, top_k=t.K, param_init={}),
        shared_experts=make_shared_expert_ffn_config(dim=t.D, hidden_dim=t.H, w1_param_init={}, w2w3_param_init={}),
    )
    config.shared_experts_stream = stream
    torch.manual_seed(0)
    moe = config.build().cuda()
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.05)
    if sleep:
        for p in moe.shared_experts.parameters():
            p.register_hook(lambda g: (torch.cuda._sleep(300_000_000), g)[1])
    torch.manual_seed(1)
    xs = [torch.randn(t.T, t.D, device="cuda", requires_grad=True) for _ in range(2)]
    seen = {}
    for x in xs:
        moe(_ReadSharedGrads.apply(x, moe, seen)).square().sum().backward()
    torch.cuda.synchronize()
    return seen


ref = run(stream=False, sleep=False)
for trial in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    got = run(stream=True, sleep=True)
    bad = [n for n in ref if not torch.equal(got[n], ref[n])]
    print(f"trial {trial}: {len(bad)} of {len(ref)} shared-expert gradients differ from the stream-off run {bad}")
