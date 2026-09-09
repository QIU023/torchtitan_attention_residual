"""LOCAL PROBE HACK (not committed): with KDA_PERTURB=<eps> the q/k/v projections of every KDA layer are
multiplied by (1 + eps * N(0,1)) on dp1, the size of the colwise matmul's fp32 roundoff under TP, so the
step-1 gradient distribution of this run against plain dp1 is the floor the tp2 dumps are read against."""
import sys
p = sys.argv[1] + "/torchtitan/models/kimi_k3/kda.py"; s = open(p).read()
old = """        out_THV = self.inner_kda(
            self.q_proj(x_TD),
            self.k_proj(x_TD),
            self.v_proj(x_TD),
"""
assert s.count(old) == 1, s.count(old)
new = """        q_TC, k_TC, v_TC = self.q_proj(x_TD), self.k_proj(x_TD), self.v_proj(x_TD)
        _eps = float(__import__("os").environ.get("KDA_PERTURB", "0") or 0)  # LOCAL PROBE HACK (not committed)
        if _eps:
            _g = torch.Generator(device=x_TD.device).manual_seed(1234 + (id(self) % 100000))
            q_TC, k_TC, v_TC = (t * (1 + _eps * torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=_g)) for t in (q_TC, k_TC, v_TC))
        out_THV = self.inner_kda(
            q_TC,
            k_TC,
            v_TC,
"""
open(p, "w").write(s.replace(old, new)); print("perturb hack applied")
