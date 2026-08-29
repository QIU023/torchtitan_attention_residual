Kimi K3 entered torchtitan as an eager reference model that ran on any
hardware (PR-4025); the KDA kernel change (PR-4351) hardcoded impl="fused"
and raises on any CUDA capability outside SM100/SM103, so the model now
runs only on datacenter Blackwell and the KDA tests skip everywhere else.
Attention Gym ships a reference implementation behind the same
chunk_kda/bound_gate API. Before this change KDAKernel raises off
SM100/SM103; after it, the kernel routes there to the reference
implementation and K3 trains on everything else again, with fla staying
out of the dependency set.

KDAKernel.Config gains impl with values auto, fused and reference,
default auto: on SM100/SM103 auto resolves to fused, so numerics and
behavior there are unchanged; elsewhere it resolves to reference with an
info log. Explicit fused on unsupported hardware still raises, with the
message now pointing at the knob. The two KDA test gates drop the
capability check, so the recurrent-reference and varlen parity oracles
run on any CUDA device; both pass on SM120 through the reference path,
and the debug flavor trains (loss below, consumer Blackwell, one seed,
warmed compile cache):

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.57037 | 7.56620 | 3.94650 |
| dp2 | 12.51296 | 7.56612 | 3.27074 |

Changed files: torchtitan/models/kimi_k3/kda.py,
tests/unit_tests/test_kda_attention.py, tests/unit_tests/gpu/test_kimi_k3.py.
