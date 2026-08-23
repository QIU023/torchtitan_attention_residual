"""Does the whole model under CP reproduce the whole model without it?

The per-layer checks pinned KDA and the all-to-all. This pins the model: same
weights, same tokens, one rank running the full sequence against every rank
running its shard. Nothing about the trainer, the dataloader or the loss is
involved, so a difference here is the model's.
"""
import copy, sys, torch, torch.distributed as dist
from torchtitan.models.kimi_k3 import model_registry

MODE = sys.argv[1] if len(sys.argv) > 1 else "kcp"
dist.init_process_group("nccl")
r, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)

torch.manual_seed(0)
spec = model_registry("debugmodel_text")
for layer in spec.model.layers:
    if layer.delta_attention is not None:
        layer.delta_attention.cp_mode = MODE
model = spec.model.build().to("cuda")
model.init_states()
model.eval()

T = 512
tokens = torch.randint(0, 1000, (T,), device="cuda")
positions = torch.arange(T, device="cuda")
masks = model.get_attention_masks(positions)

cp = copy.deepcopy(model)          # copy BEFORE hooks: deepcopy copies them too,
                                   # and inherited ref-hooks would overwrite the
                                   # reference captures during the CP forward.
caught = {}
def _hook(name):
    def f(mod, args, out):
        caught[name] = out[0] if isinstance(out, tuple) else out
    return f
names = list(model.layers.keys())
for nm in names:
    model.layers[nm].register_forward_hook(_hook(("ref", nm)))
from torchtitan.models.common.attention import (
    create_attention_mask,
    get_causal_mask_mod,
)

plain_causal = create_attention_mask(
    get_causal_mask_mod(), None, None, T, T, device=tokens.device
)
with torch.no_grad():
    ref = model(tokens, positions=positions, attention_masks=masks)
    ref_plain = model(tokens, positions=positions, attention_masks=plain_causal)
if r == 0:
    d0 = (ref.float() - ref_plain.float()).abs().max().item()
    print(f"  [mask check] document-mask vs plain-causal on the SAME run: {d0:.3e}", flush=True)

cp._cp_group = dist.group.WORLD
for m in cp.modules():
    if hasattr(m, "_cp_group") and m is not cp:
        m._cp_group = dist.group.WORLD
for nm in names:
    cp.layers[nm].register_forward_hook(_hook(("cp", nm)))
t_loc = T // world
lo = r * t_loc
with torch.no_grad():
    got = cp(
        tokens[lo : lo + t_loc],
        positions=positions[lo : lo + t_loc],
        attention_masks=masks,
    )

if r == 0:
    for nm in names:
        a = caught[("ref", nm)][lo : lo + t_loc].float()
        b = caught[("cp", nm)].float()
        kind = "KDA" if model.layers[nm].delta_attention is not None else "MLA"
        print(f"  layer {nm:>2s} {kind}  max_abs={(a-b).abs().max().item():.3e}", flush=True)
want = ref[lo : lo + t_loc]
d = (got.float() - want.float()).abs()
rel = (d.max() / want.float().abs().max()).item()
ok = rel < 3e-2
print(f"rank{r} mode={MODE} max_abs={d.max().item():.3e} rel={rel:.3e} ok={ok}", flush=True)
t = torch.tensor([int(ok)], device="cuda", dtype=torch.int32)
dist.all_reduce(t)
if r == 0:
    print(f"{MODE}: {'PASS' if t.item()==world else f'FAIL ({t.item()}/{world})'}", flush=True)
dist.destroy_process_group()
