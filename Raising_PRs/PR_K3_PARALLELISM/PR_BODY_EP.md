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

EP shards experts inside the data axis, so each cell is compared against the pure-data-parallel run at the same world size. `kimi_k3_debugmodel`, seed 42, `--debug.deterministic`, one seed checkpoint loaded by every cell; at step 2 the EP cells are 2.9e-4, 1.2e-3 and 1.1e-3 from their own-degree baseline, against 3.7e-3, 3.5e-3 and 5.2e-3 for the dp2, dp4 and dp8 rows measured the same way against dp1:

<!-- NUMBERS BELOW ARE FROM THE RETIRED TEXT FLAVOR: re-measure on kimi_k3_debugmodel in flight (mx3_ep_mm / mx3_ep_mm_gn); swap both tables before filing -->
| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.58955 | 7.55365 | 3.46009 |
| dp2 | 2 | 12.57725 | 7.57439 | 3.19128 |
| ep2 x fsdp2 | 2 | 12.57718 | 7.64509 | 3.20056 |
| dp4 | 4 | 12.59366 | 7.20509 | 3.30981 |
| ep4 x fsdp4 | 4 | 12.59386 | 7.25484 | 3.31417 |
| dp8 | 8 | 12.58336 | 7.32668 | 3.36729 |
| ep8 x fsdp8 | 8 | 12.58286 | 7.40832 | 3.44799 |

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py              +46  set_expert_parallel_sharding_config: the
                                   routed-expert layout and the decoder-level
                                   distribution above it (new file, following
                                   qwen3_5/sharding.py)
      model.py               +7/-36 the one call under the ep>1 gate; the
                                   declaration body moves to sharding.py
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

The same matrix with the grad-norm reduction carried in float32 (https://github.com/pytorch/torchtitan/pull/4135, a separate change not on this branch): only ep8 moves (7.40832 to 7.42252 at step 3), because the reduction is taken over an expert group and a non-expert group separately and the split only rounds differently in bf16 once there are enough experts per group.

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.58955 | 7.54919 | 3.50680 |
| dp2 | 2 | 12.57725 | 7.57439 | 3.19060 |
| ep2 x fsdp2 | 2 | 12.57718 | 7.64509 | 3.20221 |
| dp4 | 4 | 12.59366 | 7.20509 | 3.31005 |
| ep4 x fsdp4 | 4 | 12.59386 | 7.25484 | 3.30867 |
| dp8 | 8 | 12.58336 | 7.32668 | 3.36718 |
| ep8 x fsdp8 | 8 | 12.58286 | 7.42252 | 3.43101 |
