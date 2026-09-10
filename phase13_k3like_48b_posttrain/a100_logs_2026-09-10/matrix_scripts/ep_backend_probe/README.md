# EP backend gradient probe (2026-08-28)

Tooling behind `EP_BACKEND_EVIDENCE_2026-08-28.md` §八.1.

* `grad_probe.py` — build the Trainer from the usual CLI, load the seed checkpoint, run ONE
  microbatch forward+backward, dump every parameter's gradient norm (DTensor shards reduced
  across the world) to `$GRAD_PROBE_OUT` as JSON. Launch with torchrun like `torchtitan.train`.
* `grad_diff.py A.json B.json` — per-parameter relative differences, grouped (router / experts /
  latent proj / attention / vision / ...), top-12 list.
* `run_grad_probe.sh` — K3: `kimi_k3_debugmodel` vs `kimi_k3_debugmodel_minimal_async_ep`, ep2 x fsdp2,
  full AC on both, same seed checkpoint.
* `run_grad_probe_dsv3.sh` + `dsv3_probe/` — the same on upstream deepseek_v3_debugmodel with the
  PLAIN GroupedExperts (the upstream minimal_async_ep flavor forces the fused_swiglu override).
* `maep_discriminator.sh` — the 2-step loss/grad_norm comparison that preceded the probe.
* `grads_*.json` — the four dumps the tables were made from.

Paths marked `<scratch>` were this session's scratch directory; point them anywhere writable.
