"""CPU+gloo 2-proc parity probe for the packed-MXFP4 TP path.

Reference: unsharded packed module forward. Test: TP-sharded packed
module (colwise and rowwise) via apply_packed_mxfp4_tp + _forward_packed_tp.
Gate: forward parity to fp32-accumulation tolerance, and backward grads
on adapters match the reference (incl. the Partial all-reduces).
"""
import copy, os, sys
import torch, torch.distributed as dist
import torch.nn as nn
sys.path.insert(0, "/workspace/torchtitan_attention_residual/torchtitan")
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torchtitan.experiments.kimi_k3.lora import KimiLoRALinear

def main():
    dist.init_process_group("gloo")
    rank, tp = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    mesh = init_device_mesh("cpu", (tp,), mesh_dim_names=("tp",))

    out_f, in_f, r = 16, 128, 4
    for colwise in (True, False):
        torch.manual_seed(7)
        base = nn.Linear(in_f, out_f, bias=False)
        ref = KimiLoRALinear(copy.deepcopy(base), rank=r, alpha=8.0, quantize_base="mxfp4")
        tpm = KimiLoRALinear(copy.deepcopy(base), rank=r, alpha=8.0, quantize_base="mxfp4")
        # identical adapters (kaiming init is per-instance; copy over)
        with torch.no_grad():
            tpm.lora_a.copy_(ref.lora_a); tpm.lora_b.copy_(ref.lora_b)
        tpm.apply_packed_mxfp4_tp(mesh, colwise=colwise)
        # adapters distributed as parallelize does
        from torch.distributed.tensor import distribute_tensor, Shard
        a_pl = [Replicate()] if colwise else [Shard(1)]
        b_pl = [Shard(0)] if colwise else [Replicate()]
        tpm.lora_a = nn.Parameter(distribute_tensor(tpm.lora_a.data, mesh, a_pl))
        tpm.lora_b = nn.Parameter(distribute_tensor(tpm.lora_b.data, mesh, b_pl))

        x = torch.randn(2, 6, in_f)
        y_ref = ref(x)
        if colwise:
            x_in = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
            y_tp_dt = tpm(x_in)   # DTensor Shard(-1)
            y_tp = y_tp_dt.full_tensor()
        else:
            # rowwise consumes the local in/tp shard (plain), returns plain replicated
            x_loc = x.chunk(tp, dim=-1)[rank].contiguous()
            y_tp = tpm(x_loc)
        fwd_err = (y_tp - y_ref).abs().max().item()

        # backward: grads on adapters
        g = torch.randn_like(y_ref)
        ref.zero_grad(); y_ref2 = ref(x); y_ref2.backward(g)
        tpm.zero_grad()
        if colwise:
            y2 = tpm(DTensor.from_local(x, mesh, [Replicate()], run_check=False))
            y2.backward(DTensor.from_local(g.chunk(tp, dim=-1)[rank].contiguous(), mesh, [Shard(2)], run_check=False))
        else:
            y2 = tpm(x.chunk(tp, dim=-1)[rank].contiguous())
            y2.backward(g)
        def full(p):
            gr = p.grad
            return gr.full_tensor() if isinstance(gr, DTensor) else gr
        ga_err = (full(tpm.lora_a) - ref.lora_a.grad).abs().max().item()
        gb_err = (full(tpm.lora_b) - ref.lora_b.grad).abs().max().item()
        if rank == 0:
            mode = "colwise" if colwise else "rowwise"
            ok = fwd_err < 1e-4 and ga_err < 1e-4 and gb_err < 1e-4
            print(f"[{mode}] fwd={fwd_err:.2e} grad_a={ga_err:.2e} grad_b={gb_err:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
