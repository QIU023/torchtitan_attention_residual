"""What one forward of the 8-layer Kimi K3 debug model keeps for its backward, no PP.

Usage: python saved_probe.py <none|selective|full> [T]

Prints, per layer scope, every tensor autograd saved (shape, dtype, bytes) with
duplicates by storage marked, then the memory the forward left allocated.
"""

import collections
import sys

import torch

from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.kimi_k3 import model_registry

mode = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 256
torch.manual_seed(0)

spec = model_registry("debugmodel")
model = spec.model.build().cuda().to(torch.bfloat16)
model.train()
if mode == "selective":
    SelectiveAC.Config().build(dump_folder="/tmp/saved_probe_dump").apply(model)
elif mode == "full":
    FullAC.Config().build(dump_folder="/tmp/saved_probe_dump").apply(model)
elif mode != "none":
    raise SystemExit(f"unknown mode {mode}")

tokens = torch.randint(0, spec.model.vocab_size, (T,), device="cuda")
positions = torch.arange(T, dtype=torch.int32, device="cuda")
masks = model.get_attention_masks(positions)

scope = {"name": "embedding"}
for name, layer in model.layers.items():
    layer.register_forward_pre_hook(
        lambda m, a, n=name: scope.__setitem__("name", f"layer {n}")
    )
    layer.register_forward_hook(
        lambda m, a, o, n=name: scope.__setitem__("name", "head")
    )

saved = collections.OrderedDict()
seen = {}


def pack(t):
    if not isinstance(t, torch.Tensor):
        return t
    st = t.untyped_storage()
    key = (st.data_ptr(), st.nbytes())
    first = key not in seen
    if first:
        seen[key] = scope["name"]
    saved.setdefault(scope["name"], []).append(
        (tuple(t.shape), str(t.dtype).replace("torch.", ""), t.numel() * t.element_size(), first, seen[key])
    )
    return t


def unpack(t):
    return t


torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_allocated()
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    out = model(tokens, attention_masks=masks, positions=positions)
    loss = out.float().logsumexp(-1).mean()
torch.cuda.synchronize()
after = torch.cuda.memory_allocated()
peak = torch.cuda.max_memory_allocated()

D = spec.model.dim if hasattr(spec.model, "dim") else None
print(f"mode={mode} T={T} params bf16; forward left {(after - before) / 2**20:.1f} MiB allocated, peak {(peak - before) / 2**20:.1f} MiB above the parameters")
total_new = 0
for sc, items in saved.items():
    new = [i for i in items if i[3]]
    dup = [i for i in items if not i[3]]
    nbytes = sum(i[2] for i in new)
    total_new += nbytes
    print(f"--- {sc}: {len(items)} saved refs, {len(new)} new storages = {nbytes / 2**20:.2f} MiB, {len(dup)} refs to storages saved earlier")
    # group new storages by (shape, dtype)
    groups = collections.Counter((i[0], i[1]) for i in new)
    for (shape, dt), n in sorted(groups.items(), key=lambda kv: -kv[0][0][0] if kv[0][0] else 0):
        if len(shape) >= 2 and shape[0] == T:
            print(f"      {n} x {shape} {dt}")
    stacks = [i for i in items if len(i[0]) == 3 and i[0][0] == T and i[0][2] and i[1] == "bfloat16"]
    if stacks:
        print(f"      bf16 [T, N, D] refs: {[(i[0][1], 'new' if i[3] else 'from ' + i[4]) for i in stacks]}")
print(f"new storages saved in total: {total_new / 2**20:.1f} MiB")
loss.backward()
torch.cuda.synchronize()
print(f"after backward: {(torch.cuda.memory_allocated() - before) / 2**20:.1f} MiB above the parameters (gradients)")
