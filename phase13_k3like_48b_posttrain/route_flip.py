"""Do CP and non-CP pick the same experts?"""
import copy, torch, torch.distributed as dist
from torchtitan.models.kimi_k3 import model_registry

dist.init_process_group("nccl"); r, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
torch.manual_seed(0)
spec = model_registry("debugmodel_text")
model = spec.model.build().to("cuda"); model.init_states(); model.eval()
cp = copy.deepcopy(model)

grab = {}
def hook(tag, li):
    def f(mod, args, out):
        grab.setdefault((tag, li), []).append(out[1].detach())   # expert_ids_TK
    return f
for li, l in model.layers.items():
    if l.moe is not None: l.moe.router.register_forward_hook(hook("ref", li))
for li, l in cp.layers.items():
    if l.moe is not None: l.moe.router.register_forward_hook(hook("cp", li))

T = 512
tokens = torch.randint(0, 1000, (T,), device="cuda")
positions = torch.arange(T, device="cuda")
masks = model.get_attention_masks(positions)
with torch.no_grad():
    model(tokens, positions=positions, attention_masks=masks)
cp._cp_group = dist.group.WORLD
for m in cp.modules():
    if hasattr(m, "_cp_group") and m is not cp: m._cp_group = dist.group.WORLD
t_loc = T // world; lo = r * t_loc
with torch.no_grad():
    cp(tokens[lo:lo+t_loc], positions=positions[lo:lo+t_loc], attention_masks=masks)

if r == 0:
    for li in sorted({k[1] for k in grab}, key=int)[:6]:
        a = grab[("ref", li)][0][lo:lo+t_loc]
        b = grab[("cp", li)][0]
        flipped = (a != b).any(dim=-1).sum().item()
        print(f"  layer {li:>2s}  tokens whose expert set changed: {flipped}/{t_loc}", flush=True)
dist.destroy_process_group()
