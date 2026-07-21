"""Phase 0 decision gate: does a frozen MXTensor base survive FSDP2?

Mirrors the KimiLoRALinear QLoRA shape: a Linear whose frozen weight is
MXFP4 (torchao MXTensor, block 32) + a trainable low-rank adapter, wrapped
in fully_shard on 2 GPUs. Tests the exact NF4 pain point -- can FSDP2
shard/all-gather the tensor-subclass param and can we dequantize it after.

Tries the DIRECT approach (MXTensor as the param). If FSDP2 chokes, we
fall back to *_mxfp4 packed-uint8 storage + reconstruct (the NF4 pattern).
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torchao.prototype.mx_formats.mx_tensor import MXTensor

BLK = 32


def _full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


class MXFP4LoRALinear(nn.Module):
    """Frozen MXFP4 base + trainable rank-r adapter (QLoRA shape).

    SPLIT-STORAGE: MXTensor's packed qdata is half-width so the logical
    view is non-contiguous and FSDP2 rejects it. Store qdata (uint8) and
    scale (e8m0) as plain contiguous frozen params (both dim-0 = out_f, so
    FSDP shards them by row consistently) + the flatten ctx as metadata;
    reconstruct the MXTensor after all-gather via __tensor_unflatten__.
    """

    def __init__(self, in_f, out_f, rank=8):
        super().__init__()
        w = torch.randn(out_f, in_f, dtype=torch.bfloat16).cuda() * 0.02
        mx = MXTensor.to_mx(
            w, elem_dtype=torch.float4_e2m1fn_x2, block_size=BLK
        )
        inner, ctx = mx.__tensor_flatten__()  # ['qdata','scale'], ctx
        self.base_qdata = nn.Parameter(
            mx.qdata.contiguous(), requires_grad=False
        )
        # FSDP2 all-gather (foreach_copy) has no Float8_e8m0fnu kernel; store
        # the 1-byte scale as uint8 bytes and view back on reconstruction.
        self._scale_dtype = mx.scale.dtype
        self.base_scale = nn.Parameter(
            mx.scale.view(torch.uint8).contiguous(), requires_grad=False
        )
        self._mx_ctx = ctx
        self._mx_inner = inner
        self.lora_a = nn.Parameter(torch.zeros(rank, in_f, dtype=torch.bfloat16))
        self.lora_b = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.bfloat16))
        nn.init.normal_(self.lora_a, std=0.02)
        nn.init.normal_(self.lora_b, std=0.02)

    def _dequant_base(self):
        qdata = _full(self.base_qdata)
        scale = _full(self.base_scale).view(self._scale_dtype)
        mx = MXTensor.__tensor_unflatten__(
            {"qdata": qdata, "scale": scale}, self._mx_ctx, None, None
        )
        return mx.dequantize()

    def forward(self, x):
        w = self._dequant_base()
        out = F.linear(x, w.to(x.dtype))
        return out + F.linear(F.linear(x, self.lora_a), self.lora_b)


class Net(nn.Module):
    def __init__(self, d=512, n=3):
        super().__init__()
        self.layers = nn.ModuleList([MXFP4LoRALinear(d, d) for _ in range(n)])
        self.head = nn.Linear(d, d, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        for lyr in self.layers:
            x = torch.relu(lyr(x))
        return self.head(x)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def log(*a):
        if rank == 0:
            print("[MXFP4-FSDP2]", *a, flush=True)

    torch.manual_seed(0)
    net = Net().cuda()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    try:
        for lyr in net.layers:
            fully_shard(lyr, mp_policy=mp)
        fully_shard(net, mp_policy=mp)
        log("fully_shard OK (MXTensor param sharded)")
    except Exception as e:
        log("fully_shard FAILED:", type(e).__name__, str(e)[:160])
        log("RESULT: DIRECT MXTensor-as-param does NOT shard -> need *_mxfp4 storage")
        dist.destroy_process_group()
        return

    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    try:
        for step in range(3):
            x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
            out = net(x)
            loss = out.float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            log(f"step {step}: loss {loss.item():.5f} gnorm {gnorm.item():.4f}")
        # confirm grads only on adapters, base stayed frozen MXFP4
        grad_params = sum(1 for p in trainable if p.grad is not None)
        log(f"trainable tensors with grad: {grad_params}/{len(trainable)}")
        log("RESULT: DIRECT MXTensor-as-param SHARDS + trains -> Option A = mirror NF4")
    except Exception as e:
        import traceback
        if rank == 0:
            traceback.print_exc()
        log("fwd/bwd FAILED:", type(e).__name__, str(e)[:160])
        log("RESULT: shards but fwd/bwd breaks -> need dequant/storage workaround")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
