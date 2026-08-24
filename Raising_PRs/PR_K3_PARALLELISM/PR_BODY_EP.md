# PR body: expert parallel for Kimi K3 (no MoonEP)

按 maintainer 要求:terse,无小标题,op 级事实在前,数字内联。

---

Adds expert parallelism to kimi_k3 via upstream's AllToAllTokenDispatcher. The
routed experts shard over an expert-data-parallel mesh that excludes the expert
axis -- the shape deepseek_v3 already resolves -- and the token dispatch is the
standard all-to-all. ep2_fsdp2 step-1 loss is bit-identical to dp2 (12.43537);
ep8_fsdp8 is 12.44615 against dp8's 12.44491, a 1e-4 relative gap from 8-way
expert sharding, well inside bf16 epsilon.

The latent MoE routes on the full-width token and runs the experts on the
down-projected latent; the router's per-expert counts and the grouped GEMM
offsets come from the shared dispatcher, so nothing here reindexes tokens by
hand. enable_sp is derived as enable_ep and enable_tp rather than a constant.

MoonEP is deliberately not here. The report's perfectly-balanced EP with online
redundant-expert planning and migration needs a separate dispatcher and backend;
this PR is the plain all-to-all path only. comm_backend is pinned to "standard"
rather than left a config choice so no run silently believes it is on MoonEP.

Files: moe.py (the latent MoE forward and the SiTU experts), model.py
(_set_sharding_config), parallelize.py (the ep mesh and the ep_degree wiring),
__init__.py (the dispatcher config).

Tested: ep2_fsdp2 vs dp2 bit-identical, ep8_fsdp8 vs dp8 within 1e-4, 10 steps.
