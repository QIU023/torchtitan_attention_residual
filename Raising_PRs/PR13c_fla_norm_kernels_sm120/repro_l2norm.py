# Minimal repro for the fla l2norm Blackwell sm_120 autotuner crash (PR13c).
#
# NOT RUN (GPUs busy with a live training run). Static repro only — written to
# document the trigger, mirroring fla #796 / our PR13 fused_norm_gate repro.
#
# Env that triggers it: RTX 5090 / B200 (Blackwell sm_120), Triton 3.6.0,
# fla 0.5.0, bf16.
#
# Root cause #1 (the ONLY l2norm bug): the tiled l2norm_fwd_kernel /
# l2norm_bwd_kernel list a phantom "NB" in @triton.autotune(key=["D", "NB"]).
# NB = cdiv(T, 2048*32) is passed as a constexpr but never used in the kernel
# body. It only mutates the autotune cache key, so every time T crosses a
# 2048*32-token boundary Triton re-autotunes. On sm_120 the autotuner crashes
# (device-side assert) while benchmarking variants at large grids.
#
# l2norm does NOT have grid-overlap bug #2 (unlike fused_norm_gate/layernorm):
# its launcher grid is cdiv(T, meta["BT"]) with NO get_multiprocessor_count NS
# cap, so adjacent programs' make_block_ptr blocks (offset i_t*BT) never overlap.
#
# l2norm is reached in our stack via fla.ops.kda.chunk -> l2norm_fwd/l2norm_bwd
# (q/k normalization, use_qk_l2norm_in_kernel=True). Kimi-Linear KDA training.
#
# Expected on patched l2norm: runs clean. Expected on stock fla 0.5.0:
# device-side assert / illegal memory access once T sweeps several NB ranges.

import torch

from fla.modules.l2norm import l2norm  # D<=512 -> tiled kernel (the buggy path)

D = 128  # KDA head_dim regime; D<=512 selects l2norm_fwd_kernel (tiled, NB key)

# Sweep T across multiple NB = cdiv(T, 2048*32) ranges to force re-autotuning,
# exactly the condition that crashes the sm_120 autotuner.
for T in [1024, 70000, 140000, 210000, 280000]:
    x = torch.randn(T, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = l2norm(x)
    y.sum().backward()  # bwd also re-autotunes (separate NB key)
    torch.cuda.synchronize()
    print(f"T={T} ok")

print("PASS (patched) — stock fla 0.5.0 expected to assert during the sweep")
