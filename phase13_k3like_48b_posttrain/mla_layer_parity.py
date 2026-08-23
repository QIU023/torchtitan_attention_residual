"""One MLA layer, identical input, CP vs no CP."""
import copy, torch, torch.distributed as dist
from torchtitan.models.kimi_k3 import model_registry

dist.init_process_group("nccl"); r, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
torch.manual_seed(0)
spec = model_registry("debugmodel_text")
model = spec.model.build().to("cuda"); model.init_states(); model.eval()
mla = [l.attention for l in model.layers.values() if l.attention is not None][0]

T, D = 512, model.config.dim
torch.manual_seed(1)
x = torch.randn(T, D, device="cuda")
positions = torch.arange(T, device="cuda")
masks = model.get_attention_masks(positions)

mla._cp_group = None
with torch.no_grad():
    ref = mla(x, attention_masks=masks)

cp = copy.deepcopy(mla); cp._cp_group = dist.group.WORLD
t_loc = T // world; lo = r * t_loc
with torch.no_grad():
    got = cp(x[lo:lo + t_loc], attention_masks=None)

want = ref[lo:lo + t_loc]
d = (got.float() - want.float()).abs()
print(f"rank{r} MLA-only max_abs={d.max().item():.3e} "
      f"rel={(d.max()/want.float().abs().max()).item():.3e}", flush=True)
dist.destroy_process_group()
