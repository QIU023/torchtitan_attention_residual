import torch
from arfuncs import main_version, list_version
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm
D = 128
projection = Linear.Config(in_features=D, out_features=1).build().cuda().bfloat16()
norm = RMSNorm.Config(normalized_shape=D).build().cuda().bfloat16()

def list_version_sum(sources, projection, norm):
    values_TND = torch.stack(sources, dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_TN = torch.softmax(scores_TN, dim=-1)
    output_TD = (probs_TN.unsqueeze(-1) * values_float).sum(dim=1)
    return output_TD.to(values_TND.dtype)

if __name__ == "__main__":
    for name, fn, mk in (("main", main_version, lambda s: (s[3], torch.stack(s[:3], 1), projection, norm)),
                         ("list", list_version, lambda s: (s, projection, norm)),
                         ("list_sum", list_version_sum, lambda s: (s, projection, norm))):
        torch._dynamo.reset()
        f = torch.compile(fn, fullgraph=True, dynamic=True)
        bad = 0
        for seed in range(20):
            torch.manual_seed(seed)
            src = [torch.randn(64, D, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
            full = f(*mk([x.detach().requires_grad_() for x in src]))
            parts = [f(*mk([x.detach().chunk(2)[p].requires_grad_() for x in src])) for p in range(2)]
            bad += not torch.equal(full, torch.cat(parts))
        print(name, "batch-variant forward outputs over 20 seeds:", bad)
