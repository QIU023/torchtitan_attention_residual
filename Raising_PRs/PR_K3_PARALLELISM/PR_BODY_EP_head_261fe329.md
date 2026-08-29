### Summary

Enables expert parallelism for Kimi K3 on the existing all-to-all token dispatcher. Expert parallel comes off the unsupported list; The dispatcher backend is a spec parameter, `moe_comm_backend`, threaded from `model_registry` to `make_token_dispatcher_config` the way deepseek_v3 and gpt_oss do it: `standard` (PyTorch all-to-all) by default, `deepep`, `hybridep` and `minimal_async_ep` selectable as on those models. The tables below are measured on `standard`. MoonEP (the report's balanced EP with online redundant-expert planning) is not here; it needs its own dispatcher and backend.

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

    torchtitan/models/common/
      token_dispatcher.py    +5/-1  the DeepEP / HybridEP / MinimalAsyncEP
                                   buffers are sized by
                                   routed_experts.inner_experts.dim (the
                                   latent width on K3) instead of the model dim
      config_utils.py           +1  the minimal_async_ep factory branch
                                   forwards hidden_dim
    torchtitan/models/kimi_k3/
      sharding.py              +47  set_expert_parallel_sharding_config: the
                                   routed-expert layout (new file, following
                                   qwen3_5/sharding.py)
      model.py                  +3  the one call under the ep>1 gate
      __init__.py           +23/-6  moe_comm_backend threads from model_registry
                                   to core's dispatcher factory: standard by
                                   default, deepep / hybridep / minimal_async_ep
                                   selectable as on deepseek_v3
      parallelize.py         +17/-2 the efsdp mesh, ep_degree through to
                                   apply_fsdp_to_decoder, expert parallel off
                                   the unsupported list, and a comment naming
                                   the backends this model runs
    tests/integration_tests/models.py       the model test gains ep2
    torchtitan_recipes/tests/models.py    its configuration

### CI/CD Coverage

The existing kimi_k3 model integration test becomes fsdp2 x ep2 on the same 2 GPUs.

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

Review ask: try the other dispatcher backends. `moe_comm_backend` is a spec parameter now (`standard` by default; `deepep`, `hybridep`, `minimal_async_ep` selectable as on deepseek_v3); the debug flavor stays on `standard`, and the non-standard rows below were run by pointing the registry parameter at each backend (MinimalAsyncEP additionally needs full activation checkpointing). One thing had to change before any of them ran on K3: the DeepEP / HybridEP / MinimalAsyncEP buffers were sized by the model dim, but K3's routed experts consume the latent stream (`routed_down` runs before the dispatch: 512 against a model dim of 1024 on the debug flavor, 3584 against 7168 on the released shape), so the receive buffer came back 1024 wide and the expert grouped GEMM failed on the contraction dim. Per review, the width is configuration: the registry passes `hidden_dim=latent_dim` through `make_token_dispatcher_config`, and core defaults to the model dim only when the config leaves it unset -- deepseek_v3 and qwen3 are unchanged.

2 x H100 NVL, PCIe only (no NVLink between the pair), torch 2.15.0.dev20260827+cu130, DeepEP v2 at the commit CI pins (`01dc3aa`, NCCL 2.30.7). `kimi_k3_debugmodel`, seed 42, `--debug.deterministic`, one seed checkpoint loaded by every cell, ep2 x fsdp2 on 2 GPUs against the dp2 row at the same world size. MinimalAsyncEP requires full activation checkpointing, so a standard + full-AC row isolates that; it is bitwise the selective-AC row.

| cell | backend | AC | step 1 | step 2 | step 3 | step 10 | grad_norm at step 1 |
|---|---|---|---|---|---|---|---|
| dp2 | - | selective | 12.59885 | 9.55945 | 7.58868 | 3.30412 | 13.6250 |
| ep2 x fsdp2 | standard | selective | 12.59885 | 9.55519 | 7.55794 | 3.32943 | 13.6250 |
| ep2 x fsdp2 | standard | full | 12.59885 | 9.55519 | 7.55794 | 3.32943 | 13.6250 |
| ep2 x fsdp2 | minimal_async_ep | full | 12.59281 | 9.93810 | 7.56969 | 3.22479 | 11.3750 |
| ep2 x fsdp2 | deepep | selective | fails in the first dispatch (below) | | | | |

`standard` is bitwise dp2 at step 1 and 4.3e-3 from it at step 2, the same shape as the 8-GPU table above. `deepep` loads the seed checkpoint and dies in the first dispatch with `DeepEP NVLink barrier timeout` then `CUDA_ERROR_LAUNCH_FAILED`; DeepEP's own `tests/elastic/test_barrier.py` and `test_ep.py` fail the same way at 2 ranks on this box, and its README lists NVLink as an intranode requirement, so that backend needs an NVLink machine, which this one is not. `hybridep` is not run: it lives on DeepEP's separate `hybrid-ep` branch and targets GB200 / NVL72, the same split CI makes.

`minimal_async_ep` runs to step 10 but is not training the same model: with identical weights the forward is 6.0e-3 off at step 1 while the step-1 gradient norm is 17% lower, and step 2 is 3.8e-1 off. A per-parameter gradient dump on one microbatch from the seed checkpoint (https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase13_k3like_48b_posttrain/matrix_scripts/ep_backend_probe) has every other parameter group within 1e-2 of `standard` and the routed experts' `w1_EFD` / `w3_EFD` gradients at 1/500 of theirs in every MoE layer; `w2_EDF` matches. This is not K3-specific: the same probe on upstream `deepseek_v3_debugmodel` with the plain `GroupedExperts` (the `deepseek_v3_debugmodel_minimal_async_ep` flavor forces the `fused_swiglu` override, which K3's SiTU-GLU cannot use) gives exactly zero `w1_EFD` / `w3_EFD` gradients, and cloning the dispatched rows before the expert GEMM (`x_RD = x_RD.bfloat16().clone()`) brings them to within 3e-4 of `standard`. The expert weight-gradient GEMM reads the dispatch output it saved for backward, and that tensor is a view of MinimalAsyncEP's two-slot receive buffer, which the combine backward has rewritten by then. That fix belongs in MinimalAsyncEP rather than in this PR; the backend stays selectable on K3 and the row above is what it produces today. With the MinimalAsyncEP fix applied on top (a separate PR: the dispatch output is cloned under autograd), the same K3 cell reads 12.59281 / 9.50324 / 7.45474 / 3.22755 with a step-1 gradient norm of 13.4375 against 13.6250, and no parameter group differs from `standard` beyond bf16 reduction order.

#### Second machine: 2 x H100 with NVLink (NV18)

The same five cells on a pair with real NVLink, environment identical otherwise. DeepEP v2 runs here (its NVLink-barrier failure on the first box was topology, not code):

| cell | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp2 | 12.59951 | 7.45599 | 3.26481 |
| ep2 standard | 12.59951 | 7.43228 | 3.30036 |
| ep2 standard, full AC | 12.59951 | 7.43228 | 3.30036 |
| ep2 minimal_async_ep | 12.59768 | 7.56392 | 3.29721 |
| ep2 deepep | 12.59438 | 7.46660 | 3.24408 |

standard prints dp2's step-1 loss to every digit, and its full-AC twin matches it through step 10. minimal_async_ep runs but its step-3 deviation is the visible face of an upstream issue we are filing separately: the dispatcher hands the expert GEMM a view of a receive buffer that the combine backward overwrites, so the routed experts' w1/w3 gradients are ~lost (per-parameter probe: 0.998 relative difference on exactly those parameters and no others; upstream's own fused path reproduces; a one-line owned-dispatch copy restores every parameter group to the noise band). deepep's differences are ordinary backend arithmetic.

Review round two is in this head: the buffer width comes from configuration as asked; `enable_sp` is a real parameter (default False, the only value shipped here) rather than a call-site literal; and the decoder-level sharding call is deleted outright -- the ablation shows removing it changes nothing to every printed digit, the MoE boundary lifts its own input. Sequence parallel itself is deliberately not in this PR; it composes with tensor parallel and follows that PR.
