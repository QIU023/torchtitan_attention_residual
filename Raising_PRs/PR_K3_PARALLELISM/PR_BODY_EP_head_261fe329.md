### Summary

Enables expert parallelism for Kimi K3 on the existing all-to-all token dispatcher. Expert parallel comes off the unsupported list; the dispatcher backend is a spec parameter, `moe_comm_backend`, threaded from `model_registry` to `make_token_dispatcher_config` the way deepseek_v3 and gpt_oss do it: `standard` (PyTorch all-to-all) by default, `deepep`, `hybridep` and `minimal_async_ep` selectable as on those models. The tables below are measured on `standard`. MoonEP (the report's balanced EP with online redundant-expert planning) is not here; it needs its own dispatcher and backend.

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
      token_dispatcher.py    +4/-1  a hidden_dim set on the dispatcher
                                   config wins; model dim is only the default
                                   (K3 dispatches the latent stream)
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

2 x H100 SXM (NVLink), torch 2.15.0.dev20260827+cu130, DeepEP v2 at the commit CI pins (`01dc3aa`). `kimi_k3_debugmodel`, seed 42, `--debug.deterministic`, one seed checkpoint loaded by every cell; the backend is chosen through `model_registry(..., moe_comm_backend=...)`, the debug flavor itself stays on `standard`. MinimalAsyncEP requires full activation checkpointing, so a `standard` full-AC row isolates that. `hybridep` is not run: it lives on DeepEP's `hybrid-ep` branch and targets GB200 / NVL72.

| cell | backend | AC | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp2 | - | selective | 12.59951 | 7.45599 | 3.26481 |
| ep2 x fsdp2 | standard | selective | 12.59951 | 7.43228 | 3.30036 |
| ep2 x fsdp2 | standard | full | 12.59951 | 7.43228 | 3.30036 |
| ep2 x fsdp2 | minimal_async_ep | full | 12.59768 | 7.56392 | 3.29721 |
| ep2 x fsdp2 | deepep | selective | 12.59438 | 7.46660 | 3.24408 |

`standard` prints dp2's step-1 loss to every digit and its full-AC twin matches it through step 10; `deepep` differs by ordinary backend arithmetic. `minimal_async_ep`'s step-3 gap is an upstream MinimalAsyncEP bug, not K3's: the expert GEMM saves a view of the recycled receive buffer that the combine backward overwrites, so the routed `w1_EFD` / `w3_EFD` gradients are lost; the one-line owned-dispatch fix goes in a separate PR, and with it the same cell is back in the noise band.
