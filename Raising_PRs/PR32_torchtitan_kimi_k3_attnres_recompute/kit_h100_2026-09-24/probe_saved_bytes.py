"""Bytes one attention-residual call keeps alive for backward, per implementation.

Run from a torchtitan checkout of attnres_review1; fla 0.6.0 (git main) on the
path adds its fused op, called the way an override of the node would call it:

    PYTHONPATH=<fla main>:. python <kit>/probe_saved_bytes.py [--time]

For each of eager, torch.compile(eager) and, when importable, fla fused, with and
without the torch_remat checkpoint the model puts around the call, it reports:

  held      memory allocated after forward minus before, minus the output:
            what the autograd graph of this one call keeps until backward
  peak      max allocated during forward + backward, over the inputs
  new       bytes of saved tensors whose storage is not one of the inputs'
  rel err   max over output and input grads of max|diff| / max|eager|

--fla-res-from-saved rebuilds fla's source pointer table from the saved
tensors in backward instead of keeping it on ctx.res, the fla-side change
that would let a checkpoint release the contiguous source copies.

Bytes are shape facts and hold on any CUDA device. --time adds a median
forward + backward time; quote it only from the H100.
"""

import argparse
import statistics
import types

import torch
import torch_remat as remat

from torchtitan.models.kimi_k3.model import AttentionResidual

try:  # a broken fla install can raise more than ImportError (tilelang, tvm_ffi)
    from fla.ops.attnres import fused_attnres as fla_fused_attnres
except Exception:
    fla_fused_attnres = None

EPS = 1e-6


def make_inputs(tokens, entries, dim, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    stack = torch.randn(tokens, entries, dim, generator=g).to(device, torch.bfloat16)
    prefix = torch.randn(tokens, dim, generator=g).to(device, torch.bfloat16)
    proj = torch.randn(1, dim, generator=g).mul(0.02).to(device, torch.bfloat16)
    norm = torch.rand(dim, generator=g).add(0.5).to(device, torch.bfloat16)
    return stack, prefix, proj, norm


def call(fn, stack, prefix, proj, norm):
    projection = types.SimpleNamespace(weight=proj)
    rms = types.SimpleNamespace(weight=norm, eps=EPS)
    return fn(prefix, stack, projection, rms)


def fla_fused(prefix_sum_TD, block_residual_TND, projection, norm):
    """fla's fused op on the node's arguments: the stack's block views plus the running sum."""
    return fla_fused_attnres(
        projection.weight.squeeze(0),
        [*block_residual_TND.unbind(dim=1), prefix_sum_TD],
        norm.weight,
        rms_eps=norm.eps,
    )


def patch_fla_res_from_saved():
    """Keep fla's pointer table off ctx; rebuild it from ctx.saved_tensors in backward."""
    from fla.ops.attnres import fused as fla_fused
    from fla.utils import autocast_custom_bwd, input_guard

    function = fla_fused.FusedAttnresFunction
    forward = function.forward

    def forward_without_res(ctx, *args):
        outputs = forward(ctx, *args)
        ctx.res = None
        return outputs

    @input_guard
    @autocast_custom_bwd
    def backward_with_res(ctx, do, dp=None):
        # fla's backward body, with the table built from the unpacked tensors.
        del dp
        query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse, *residuals = (
            ctx.saved_tensors
        )
        dvs, dq, dw, dow = fla_fused.fused_attnres_bwd(
            do=do,
            q=query,
            residuals=residuals,
            res=fla_fused._build_ptr_table(residuals),
            w=rms_weight,
            ow=output_rms_weight,
            o_pre=o_pre,
            rstd=rstd,
            logit=logit,
            lse=lse,
            eps=ctx.eps,
            scale=ctx.scale,
            checkpoint_level=ctx.checkpoint_level,
        )
        return (dq, dw, dow, None, None, None, None, *dvs)

    function.forward = staticmethod(forward_without_res)
    function.backward = staticmethod(backward_with_res)


def measure(fn, inputs, checkpoint, grad_out):
    stack, prefix, proj, norm = (t.detach().clone().requires_grad_(True) for t in inputs)
    input_storages = {t.untyped_storage().data_ptr() for t in (stack, prefix, proj, norm)}
    saved = {}

    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in input_storages:
            saved[ptr] = t.untyped_storage().nbytes()
        return t

    run = remat.checkpoint(region_name="attention_res")(fn) if checkpoint else fn
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = call(run, stack, prefix, proj, norm)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base - out.untyped_storage().nbytes()
    out.backward(grad_out)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    grads = (stack.grad, prefix.grad, proj.grad, norm.grad)
    return out.detach(), grads, held, peak, sum(saved.values())


def timed(fn, inputs, checkpoint, grad_out, iters=20, warmup=5):
    stack, prefix, proj, norm = (t.detach().clone().requires_grad_(True) for t in inputs)
    run = remat.checkpoint(region_name="attention_res")(fn) if checkpoint else fn
    times = []
    for i in range(warmup + iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call(run, stack, prefix, proj, norm).backward(grad_out)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--fla-res-from-saved", action="store_true")
    parser.add_argument(
        "--shapes", default="2048x8x7168,8192x4x1024", help="tokens x entries x dim"
    )
    args = parser.parse_args()
    if args.fla_res_from_saved:
        assert fla_fused_attnres is not None, "--fla-res-from-saved needs fla 0.6.0"
        patch_fla_res_from_saved()
    device = torch.device("cuda")
    eager = AttentionResidual.Config().build()
    compiled = torch.compile(eager.__call__)
    impls = {"eager": eager, "compile(eager)": compiled}
    if fla_fused_attnres is not None:
        impls["fla fused"] = fla_fused
    mib = 1 << 20
    print(
        f"device {torch.cuda.get_device_name()}  torch {torch.__version__}"
        f"  fla ctx.res {'rebuilt in backward' if args.fla_res_from_saved else 'as shipped'}"
    )
    for shape in args.shapes.split(","):
        tokens, entries, dim = (int(x) for x in shape.split("x"))
        inputs = make_inputs(tokens, entries, dim, device)
        grad_out = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
        stack_mib = tokens * entries * dim * 2 / mib
        print(
            f"\nT={tokens} stack entries={entries} D={dim}: bf16 stack {stack_mib:.1f} MiB, "
            f"one FP32 copy of stack + prefix {tokens * (entries + 1) * dim * 4 / mib:.1f} MiB"
        )
        for fn in impls.values():
            for checkpoint in (False, True):
                measure(fn, inputs, checkpoint, grad_out)
        ref_out, ref_grads, *_ = measure(eager, inputs, False, grad_out)
        header = f"{'impl':16} {'remat':6} {'held MiB':>9} {'peak MiB':>9} {'new saved MiB':>13} {'rel err':>9}"
        print(header + ("  median fwd+bwd ms" if args.time else ""))
        for name, fn in impls.items():
            for checkpoint in (False, True):
                out, grads, held, peak, new = measure(fn, inputs, checkpoint, grad_out)
                err = max(
                    ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
                    for a, b in zip((out, *grads), (ref_out, *ref_grads))
                )
                line = (
                    f"{name:16} {str(checkpoint):6} {held / mib:9.1f} {peak / mib:9.1f} "
                    f"{new / mib:13.1f} {err:9.2e}"
                )
                if args.time:
                    line += f"  {timed(fn, inputs, checkpoint, grad_out):8.3f}"
                print(line)


if __name__ == "__main__":
    main()
