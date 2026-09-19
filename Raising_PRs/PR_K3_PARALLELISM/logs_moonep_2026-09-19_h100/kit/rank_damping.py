"""How the standard path's rank-level imbalance depends on experts per rank.

Pure arithmetic on the same Zipf routing the sweep uses: no transport involved.
It answers why four ranks understate what MoonEP removes, since with E / R
experts on a rank the hot ones average out inside the rank.
"""
import torch

E, K = 256, 8
TOKENS = 4096 * 4


def counts(alpha):
    g = torch.Generator().manual_seed(7)
    order = torch.randperm(E, generator=g)
    p = torch.zeros(E)
    p[order] = torch.arange(1, E + 1, dtype=torch.float) ** (-alpha)
    p = p / p.sum()
    ids = torch.multinomial(p.expand(TOKENS, E), K, replacement=False, generator=g)
    return torch.bincount(ids.reshape(-1), minlength=E).float()


print(f"E={E} top_k={K} tokens={TOKENS}; standard path max/mean over ranks")
print(f"{'ranks':>6} {'E/rank':>7} " + " ".join(f"a={a:<5.1f}" for a in (0.0, 1.0, 2.0, 3.0)))
for R in (2, 4, 8, 16, 32, 64):
    if E % R:
        continue
    row = []
    for a in (0.0, 1.0, 2.0, 3.0):
        c = counts(a)
        per_rank = c.view(R, E // R).sum(dim=1)
        row.append(float(per_rank.max() / per_rank.mean()))
    print(f"{R:>6} {E // R:>7} " + " ".join(f"{v:>7.2f}" for v in row))
