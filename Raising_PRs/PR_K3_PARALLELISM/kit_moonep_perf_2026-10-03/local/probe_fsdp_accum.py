"""Does FSDP2's mid-backward reduce of the shared experts' gradients race with their accumulation on the side
stream when the x_TD hook is gone? One GPU, world size 1. Micro-batch 1 runs without gradient sync, micro-batch 2
with it, so micro-batch 2 adds into the unsharded gradients on the side stream; a sleep on the side stream right
before that add (a tensor hook on the unsharded parameters) widens the window. Run from a worktree root."""
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, "tests/unit_tests/gpu")
import test_moe_shared_experts_stream as t  # noqa: E402

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
dist.init_process_group("nccl", rank=0, world_size=1)
torch.cuda.set_device(0)
mesh = init_device_mesh("cuda", (1,))


class Block(torch.nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.pre = torch.nn.Linear(t.D, t.D, device="cuda")
        self.moe = moe

    def forward(self, x):
        return self.moe(self.pre(x))


def build(stream: bool):
    torch.manual_seed(0)
    moe, *_ = (None,)
    config_moe = t._run.__globals__  # reuse the test's config builder below
    from torchtitan.models.common.activation import Sigmoid
    from torchtitan.models.common.config_utils import (
        make_moe_config, make_routed_experts_config, make_router_config, make_shared_expert_ffn_config)
    config = make_moe_config(
        num_experts=t.E,
        router=make_router_config(dim=t.D, num_experts=t.E, gate_param_init={}, score_func=Sigmoid.Config(), top_k=t.K),
        routed_experts=make_routed_experts_config(dim=t.D, hidden_dim=t.H, num_experts=t.E, top_k=t.K, param_init={}),
        shared_experts=make_shared_expert_ffn_config(dim=t.D, hidden_dim=t.H, w1_param_init={}, w2w3_param_init={}),
    )
    config.shared_experts_stream = stream
    moe = config.build().cuda()
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.05)
    block = Block(moe)
    torch.manual_seed(1)
    with torch.no_grad():
        block.pre.weight.normal_(0, 0.05)
        block.pre.bias.zero_()
    fully_shard(block.moe, mesh=mesh)
    fully_shard(block, mesh=mesh)
    return block


def run(stream: bool, sleep: bool):
    block = build(stream)
    torch.manual_seed(2)
    xs = [torch.randn(t.T, t.D, device="cuda") for _ in range(2)]
    hooked = []
    for mb, x in enumerate(xs):
        block.moe.set_requires_gradient_sync(mb == 1)
        block.set_requires_gradient_sync(mb == 1)
        out = block(x)
        if sleep and mb == 1 and not hooked:
            for fp in block.moe._get_fsdp_state()._fsdp_param_group.fsdp_params:
                if "shared_experts" in fp._param_fqn:
                    fp._unsharded_param.register_hook(lambda g: (torch.cuda._sleep(300_000_000), g)[1])
                    hooked.append(fp._param_fqn)
        out.square().sum().backward()
    torch.cuda.synchronize()
    return {n: p.grad.full_tensor().clone() for n, p in block.named_parameters() if "shared_experts" in n}, hooked


ref, _ = run(stream=False, sleep=False)
for trial in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    got, hooked = run(stream=True, sleep=True)
    worst = max((got[n] - ref[n]).abs().max().item() / (ref[n].abs().max().item() + 1e-30) for n in ref)
    print(f"trial {trial}: delayed {len(hooked)} shared-expert params; worst relative gradient diff {worst:.3e}")
dist.destroy_process_group()
