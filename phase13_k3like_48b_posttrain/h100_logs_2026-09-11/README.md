# H100 box archive (2026-09-11)

Everything small from `/workspace` on the rented 2 x NVIDIA H100 PCIe box (capability 9.0, 64 cores, 251 GB RAM): the driver scripts, the uncommitted probe patches as patch files, and every cell's `results.txt` and `*_measure.log`. Checkpoints, the inductor and Triton caches, the venv and the source trees are not here.

Environment, identical to the 8 x RTX 5060 Ti box except where noted: torch `2.15.0.dev20260906+cu130` (CUDA 13.0), triton `3.8.0+gitc01b6774`, spmd-types 0.2.5, torch-remat 0.2.0, nvidia-cutlass-dsl 4.6.0; Attention Gym at upstream main `b16d6d3` (the other box runs the fork's `b19162e`). Branch `pp_review4` = `8aea9ef03` on upstream main `d9ca9e55a`, except the CP replication, which is PR 4500's head `2884d82a9` fetched from `pull/4500/head`.

Probe patches, all uncommitted on the box and never committed to any branch:

- `probe_patches_pp_review4.patch` -- the KDA capability guard lift, the `kimi_k3_debugmodel_ppnaive` alias (whole-stack transport), the `kimi_k3_debugmodel_deep` / `_deep_ppnaive` aliases (33-layer flavour), `MB_REVERSE` (reverses the order the gradient accumulation groups are consumed in), and `LOSS_FULL` (logs the step loss and grad norm at full precision).
- `probe_patch_4500_cudagraph.patch` -- on the 4500 head only: `torch.cuda._annotate_cuda_graph_trace` does not exist in this torch nightly, so the import moves inside the profiling post-processor that is its only user.

Run directories, one line each:

- `mx3_pp100_a_0911_070707` -- dp1, pp2, pp2 x vp2 cached, 24-layer shared debug model, 1024 tokens per step as 4 x 256, 100 steps. The delivered PP table.
- `mx3_pp100_b_0911_071828` -- the same at whole-stack transport (`pp2vp2n`), same table.
- `mx3_pp100_c_0911_072148` -- `dp1rev`, the noise floor: no pipeline, only the order the four accumulation groups are consumed. Same table.
- `mx3_deep_a_0911_074918`, `mx3_deep_b_0911_080251`, `mx3_deep_c_0911_080651` -- the same five cells on the 33-layer flavour the pipeline stress cell uses. Feeds the PR body's stress-cell section, not the numerics reply.
- `out_4500` -- PR 4500's own cells on its own head: `cp1` (CP=1, spmd_types, the reference), `cp2ag` (all-gather), `cp2u` (Ulysses), 256 tokens per step, 100 steps, no seed checkpoint, her protocol.
- `fl_dp1.log`, `fl_pp2.log` -- step-1 only at 33 layers with `LOSS_FULL=1`: the full-precision step-1 comparison (`12.336345672607422` against `12.336344718933105`, one float32 ulp).

Drivers: `setup.sh` (environment), `pp100_controls.sh` + `mx3.sh` (the five controls), `pp100_deep.sh` (the 33-layer set), `fullloss.sh` (the full-precision step-1 pair), `run_4500_cp.sh` (as archived from the A100 box) and `run_4500_h100.sh` (the same with this box's paths and GPU ids).
