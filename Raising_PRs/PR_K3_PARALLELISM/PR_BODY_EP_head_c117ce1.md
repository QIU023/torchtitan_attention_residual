### Summary

Enables expert parallelism for Kimi K3 on the existing all-to-all token dispatcher. Expert parallel comes off the unsupported list; MoonEP (the report's balanced EP with online redundant-expert planning) is not here, it needs its own dispatcher and backend, and `comm_backend` is pinned to "standard" so no run can silently believe it is on MoonEP.

### EP Design

#### Design points

- `set_moe_sharding_config` declares the routed-expert layout, `w1_EFD S(1), w2_EDF S(2), w3_EFD S(1)`, on the expert axis (`sharding.py:26-46`, `set_expert_parallel_sharding_config`).
- `set_decoder_sharding_config` sits above it so the activations reaching the MoE boundary are DTensors that can be redistributed onto the expert mesh.
- parallelize builds the expert-data-parallel mesh, `efsdp`, that excludes the expert axis and hands it to `apply_fsdp_to_decoder` with `ep_degree` (`parallelize.py:67-69`, `parallelize.py:104`): the shape `deepseek_v3` already resolves, so expert parameters shard on `(dp_shard x cp x tp) / ep` and everything else on the full data axis.
- Without `expert_parallel_degree > 1` none of this executes.

### K3 EP runs:

To reproduce, from the torchtitan checkout root on this branch, 8 GPUs. Every cell loads the same seed checkpoint; run each cell twice and read the second run (a cold compile cache moves step 1). The runner we used, with the seed-load assertion and a disk gate, is https://github.com/QIU023/torchtitan_attention_residual/blob/611385d4e123d4d0527c6d08b06f8d701bb63e21/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh.

```sh
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
cell dp1 1 $D 1;  cell dp2 2 $D 2;  cell ep2_fsdp2 2 $D 2 $E 2
cell dp4 4 $D 4;  cell ep4_fsdp4 4 $D 4 $E 4;  cell dp8 8 $D 8;  cell ep8_fsdp8 8 $D 8 $E 8
```

EP shards experts inside the data axis, so each cell is compared against the pure-data-parallel run at the same world size. `kimi_k3_debugmodel`, seed 42, `--debug.deterministic`, one seed checkpoint loaded by every cell; at step 2 the EP cells are 3.3e-3, 9.0e-3 and 1.6e-5 from their own-degree baseline, against 1.7e-2, 2.5e-2 and 5.2e-2 for the dp2, dp4 and dp8 rows measured the same way against dp1 -- sharding the experts moves the loss less than changing the data-parallel degree does, at every degree:

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.59245 | 7.46936 | 3.18111 |
| dp2 | 2 | 12.58904 | 7.61204 | 3.33502 |
| ep2 x fsdp2 | 2 | 12.59108 | 7.59337 | 3.29070 |
| dp4 | 4 | 12.58372 | 7.65474 | 3.24291 |
| ep4 x fsdp4 | 4 | 12.58128 | 7.59719 | 3.17962 |
| dp8 | 8 | 12.60943 | 8.26416 | 3.35955 |
| ep8 x fsdp8 | 8 | 12.61045 | 8.38947 | 3.20613 |

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py              +46  set_expert_parallel_sharding_config: the
                                   routed-expert layout and the decoder-level
                                   distribution above it (new file, following
                                   qwen3_5/sharding.py)
      model.py                  +4  the one call under the ep>1 gate
      __init__.py            +6/-2 core's dispatcher factory, pinned to the
                                   standard all-to-all
      parallelize.py         +14/-2 the efsdp mesh, ep_degree through to
                                   apply_fsdp_to_decoder, and expert parallel off
                                   the unsupported list
    tests/
      unit_tests/test_kimi_k3.py  +45  the declaration checks, folded into the
                                   file the original K3 PR created
      integration_tests/models.py       the model test gains ep2
    torchtitan_recipes/tests/models.py  its configuration

### CI/CD Coverage

Two CPU checks in test_kimi_k3.py (plain DP declares nothing; EP shards the routed experts on the expert axis); the existing kimi_k3 model integration test becomes fsdp2 x ep2 on the same 2 GPUs.

### Numerical Correction run with unmerged upstream grad-norm precision forced to FP32

The same matrix with the grad-norm reduction carried in float32 (https://github.com/pytorch/torchtitan/pull/4135, a separate change not on this branch): dp1, dp4 and ep4 move (largest 2.0e-2 at step 3, on the dp1 baseline itself) and the other four cells are bitwise unchanged. The total norm is reduced over an expert group and a non-expert group separately, and which cells cross a bf16 rounding boundary shifts with what is in the groups -- the vision tower's parameters are part of the non-expert group here.

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.59245 | 7.62176 | 3.19563 |
| dp2 | 2 | 12.58904 | 7.61204 | 3.32351 |
| ep2 x fsdp2 | 2 | 12.59108 | 7.59337 | 3.28553 |
| dp4 | 4 | 12.58372 | 7.61710 | 3.23680 |
| ep4 x fsdp4 | 4 | 12.58128 | 7.61100 | 3.14353 |
| dp8 | 8 | 12.60943 | 8.26416 | 3.33049 |
| ep8 x fsdp8 | 8 | 12.61045 | 8.38947 | 3.20156 |

### EP backend verify result

Review ask: try the other dispatcher backends. This branch keeps `comm_backend` pinned to `standard`; the other three were tried on `ep_review1` (https://github.com/QIU023/torchtitan/tree/ep_review1, this branch plus `moe_comm_backend` as a spec parameter, `deepep` / `minimal_async_ep` debug flavors, and one core line so the dispatcher buffers are sized by the experts' input width: K3's routed experts consume the 512-wide latent stream, not the 1024-wide model dim, and MinimalAsyncEP's receive buffer otherwise fails the expert GEMM on the contraction dim). None of the three is usable yet, so that branch stays off this PR. 2 x H100 NVL, no NVLink between the pair (`nvidia-smi topo -p2p n`: NS), torch 2.15.0.dev20260827+cu130, DeepEP v2 at the commit CI pins (`01dc3aa`), `kimi_k3_debugmodel`, seed 42, `--debug.deterministic`, one seed checkpoint, ep2 x fsdp2 on 2 GPUs against dp2:

| backend | step 1 | step 2 | step 3 | step 10 | grad_norm at step 1 | result |
|---|---|---|---|---|---|---|
| dp2 (reference) | 12.59885 | 9.55945 | 7.58868 | 3.30412 | 13.6250 | |
| standard (this PR) | 12.59885 | 9.55519 | 7.55794 | 3.32943 | 13.6250 | bitwise dp2 at step 1; full activation checkpointing is bitwise neutral |
| minimal_async_ep | 12.59281 | 9.93810 | 7.56969 | 3.22479 | 11.3750 | runs, but the routed experts' `w1_EFD` / `w3_EFD` gradients are lost (1/500 of `standard` in every MoE layer; exactly zero on upstream `deepseek_v3_debugmodel` with the plain `GroupedExperts`); cloning the dispatched rows before the expert GEMM restores them to 3e-4. The expert weight-gradient GEMM saves a view of MinimalAsyncEP's two-slot receive buffer, which the combine backward rewrites first. deepseek_v3's flavor is shielded by the `fused_swiglu` override, which K3's SiTU-GLU cannot use. A MinimalAsyncEP fix, not a K3 one |
| deepep | - | - | - | - | - | dies in the first dispatch: `DeepEP NVLink barrier timeout`, then `CUDA_ERROR_LAUNCH_FAILED`; DeepEP's own `test_barrier.py` / `test_ep.py` fail the same way at 2 ranks. DeepEP requires NVLink intranode; to be rerun on an NVLink pair |
| hybridep | - | - | - | - | - | not run: DeepEP's separate `hybrid-ep` branch, GB200 / NVL72 only |

Logs, the per-parameter gradient probe and the environment recipe: https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase13_k3like_48b_posttrain/EP_BACKEND_EVIDENCE_2026-08-28.md
