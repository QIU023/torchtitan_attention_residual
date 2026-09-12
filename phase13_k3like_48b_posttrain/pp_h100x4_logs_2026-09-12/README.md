# PR 4312 (pp_review4) on 4 x H100 PCIe, #4500's protocol (2026-09-12)

Tree: `pp_review4` = `dbc425403` (on upstream main `d9ca9e55a`) plus `matrix_scripts/tp_h100_v2/pp4h_probe.patch` (KDA guard widened, `kimi_k3_debugmodel_pp_naive`, `MB_REVERSE`, `NOSYNC_GA`; none of it in the PR). Box as `tp_h100x4_logs_2026-09-12/README.md`. Kit: `run_pp.sh`. Protocol: `seed=42`, `--debug.deterministic`, bf16, one seed checkpoint per batch shape, one inductor cache and a per-rank Triton cache per cell, 100 steps, `cc12m-test`; pp8 x vp4 left out. #4500's 256 tokens per step cannot hold a pipeline (the collator needs 256 per micro-batch, the pipeline one micro-batch per stage), so pp2 runs at 1024 (4 x 256) and 512 (2 x 256), pp2 x vp2 at 1024.

- The 1024-token cells reproduce the 2026-09-11 2 x H100 PCIe run (`h100_logs_2026-09-11/`, pp_review4 `8aea9ef03`) value for value: dp1 `12.605700` / `3.114620`, pp2 `3.227050`, pp2 x vp2 cached `3.150940` / naive `3.514970`, reversed accumulation `3.247610`.
- Step 10, 1024 tokens, against plain dp1: dp1 with the pipeline's accumulation (`NOSYNC_GA`, no pipeline) +3.85 %, reversed accumulation +4.27 %, pp2 +3.61 %, vp2 cached +1.17 %, vp2 naive +12.9 %. Against the matched-accumulation dp1: pp2 -0.24 %, vp2 cached -2.59 %, vp2 naive +8.67 %.
- 512 tokens (a two-term accumulation, exact either way): pp2 +4.71 % at step 10.
- Step 100 at 1024 tokens is memorised (dp1 0.19) and compares nothing; `pp_cc12m_run.log` repeats the 1024 cells on the streamed cc12m.
- The matched-accumulation row is a probe. Upstream #4597 lets `training.mixed_precision_reduce` be bfloat16, which makes FSDP accumulate in the parameter dtype in both paths; pp_review4's base predates it.
