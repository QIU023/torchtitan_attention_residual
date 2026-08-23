"""Does a two-stage forward reproduce the whole-model forward?

Splits the built model the way core's PP does -- keep a slice of layers, set the
modules this stage does not own to None -- and chains the two stages by hand.
No schedule, no loss, no microbatching: this isolates the model's stage
interface from everything around it.
"""
import copy, torch
from torchtitan.models.kimi_k3 import model_registry

torch.manual_seed(0)
spec = model_registry("debugmodel_text")
model = spec.model.build().to("cuda").to(torch.bfloat16)
model.init_states()
model.eval()

T = 256
tokens = torch.randint(0, 1000, (T,), device="cuda")
positions = torch.arange(T, device="cuda")
masks = model.get_attention_masks(positions)

with torch.no_grad():
    ref = model(tokens, positions=positions, attention_masks=masks)

names = list(model.layers.keys())
half = len(names) // 2
s0, s1 = copy.deepcopy(model), copy.deepcopy(model)
for n in names[half:]:
    del s0.layers[n]
for n in names[:half]:
    del s1.layers[n]
s0.norm = None; s0.lm_head = None; s0.output_res_proj = None; s0.output_res_norm = None
s1.tok_embeddings = None

with torch.no_grad():
    out0 = s0(tokens, positions=positions, attention_masks=masks)
    assert isinstance(out0, tuple), f"stage 0 returned {type(out0).__name__}, expected a tuple"
    h, res = out0
    got = s1(h, res, positions=positions, attention_masks=masks)

d = (got.float() - ref.float()).abs()
print(f"stage-split vs whole model: max_abs={d.max().item():.3e} "
      f"rel={(d.max()/ref.float().abs().max()).item():.3e}")
print("PASS" if d.max().item() < 1e-2 else "FAIL")
