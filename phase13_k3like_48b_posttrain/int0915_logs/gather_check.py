# Two gloo ranks each hold half of a [T, V] logits tensor and compute the SAME full loss on the gathered
# tensor, as the engine does under tp2. Compare each rank's input gradient with the single-process one.
import os, torch, torch.distributed as dist
from verl.utils.ulysses import gather_outputs_and_unpad
dist.init_process_group("gloo")
r = dist.get_rank()
torch.manual_seed(0)
full = torch.randn(1, 6, 8, dtype=torch.float64)
w = torch.randn(1, 6, 8, dtype=torch.float64)
ref = full.clone().requires_grad_(True); (ref * w).sum().backward()
for scaler in (True, False):
    local = full[..., r * 4:(r + 1) * 4].clone().requires_grad_(True)
    g = gather_outputs_and_unpad(local, gather_dim=2, grad_scaler=scaler, group=dist.group.WORLD)
    (g * w).sum().backward()
    ratio = (local.grad / ref.grad[..., r * 4:(r + 1) * 4]).mean().item()
    if r == 0: print(f"grad_scaler={scaler}: local grad / exact grad = {ratio:.3f}")
dist.destroy_process_group()
