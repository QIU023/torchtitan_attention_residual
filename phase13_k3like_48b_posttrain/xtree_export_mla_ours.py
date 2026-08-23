"""Our tree: build one KDA and one MLA, save weights, input and output."""
import torch
sp_out = "/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/xtree_mla.pt"
from torchtitan.models.kimi_k3.model import KimiK3Config, KimiMLAAttention
from torchtitan.models.kimi_k3.kda import KimiDeltaAttention

torch.manual_seed(0)
cfg = KimiK3Config(vocab_size=256, hidden_size=256, num_hidden_layers=2,
                   num_attention_heads=4, kda_num_heads=4, kda_head_dim=32,
                   kda_short_conv_kernel_size=4, q_lora_rank=64, kv_lora_rank=64,
                   qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                   intermediate_size=512, kda_use_full_rank_gate=True, mla_gated=True)
mla = KimiMLAAttention.make_config(cfg, layer_idx=1).build().to("cuda").to(torch.float32)
for p in mla.parameters():
    torch.nn.init.normal_(p, std=0.02)
mla.eval()

T = 256
torch.manual_seed(1)
x = torch.randn(1, T, 256, device="cuda")
with torch.no_grad():
    out = mla(x)
out = out[0] if isinstance(out, tuple) else out
torch.save({"sd": {k: v.cpu() for k, v in mla.state_dict().items()},
            "x": x.squeeze(0).cpu(), "out": out.squeeze(0).cpu()},
           sp_out)
print("ours MLA out", tuple(out.shape))
