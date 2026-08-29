Before this change KDAKernel hardcodes impl="fused" (PR-4351) and raises on
any CUDA capability outside SM100/SM103, so Kimi K3, which entered as an
eager reference model (PR-4025), runs only on datacenter Blackwell and its
KDA tests skip everywhere else. After it, KDAKernel.Config has impl with
values auto, fused and reference, default auto: auto resolves to fused on
SM100/SM103, so numerics and behavior there are unchanged, and to Attention
Gym's reference implementation elsewhere (same chunk_kda/bound_gate API),
with an info log. Explicit fused on unsupported hardware still raises. The
two KDA test gates drop the capability check, so the recurrent-reference and
varlen parity oracles run on any CUDA device; both pass on SM120, and the
debug flavor trains there (one seed, warmed compile cache):

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.57037 | 7.56620 | 3.94650 |
| dp2 | 12.51296 | 7.56612 | 3.27074 |

Changed files: torchtitan/models/kimi_k3/kda.py,
tests/unit_tests/test_kda_attention.py, tests/unit_tests/gpu/test_kimi_k3.py.
