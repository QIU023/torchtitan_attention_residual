# PR title: [kimi_k3] add a KDA impl knob so the model runs off SM100/SM103 again

Branch `k3_kda_impl` (`3b34e306`, on upstream/main `13da2d77`). Evidence: `phase13_k3like_48b_posttrain/KDA_IMPL_KNOB_2026-08-29.md`. Paste between the markers into the PR body.

--- PASTE BEGIN ---

### Summary

Before this change `KDAKernel` hardcodes `impl="fused"` (PR-4351) and raises on any CUDA capability outside SM100/SM103, so Kimi K3, which entered as an eager reference model (PR-4025), runs only on datacenter Blackwell and its KDA tests skip everywhere else.

After it, `KDAKernel.Config` has `impl` with values `auto`, `fused` and `reference`, default `auto`: `auto` resolves to `fused` on SM100/SM103, so numerics and behavior there are unchanged, and to Attention Gym's reference implementation elsewhere (same `chunk_kda` / `bound_gate` API), with an info log. Explicit `fused` on unsupported hardware still raises. The two KDA test gates drop the capability check, so the recurrent-reference and varlen parity oracles run on any CUDA device.

### Results

Everything below is on an RTX 5060 Ti (SM120, CUDA capability 12.0), where `auto` resolves to `reference` (the log prints `KDA: CUDA capability (12, 0) has no fused kernel; using Attention Gym's reference implementation`); on SM100/SM103 the resolved path is `fused` and nothing changes. `pre-commit run --all-files` is clean. The two upstream oracles that skipped off SM100 now run and pass: `tests/unit_tests/test_kda_attention.py::TestKDA::test_varlen_matches_independent_documents` (packed sequences against the documents run one at a time, forward and gradient) and `tests/unit_tests/gpu/test_kimi_k3.py::TestKimiK3::test_attention_gym_kda_kernel_matches_recurrent_reference` (kernel against the FP32 sequential recurrence).

To reproduce the training rows, from the torchtitan checkout root on this branch. Every cell loads the same seed checkpoint; run each cell twice and read the second run (a cold compile cache moves step 1):

```sh
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 512 --training.max-context-length 512 --checkpoint.enable"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
D="--parallelism.data_parallel_shard_degree"
cell dp1 1 $D 1;  cell dp2 2 $D 2
```

`kimi_k3_debugmodel` through the reference path, seed 42, `--debug.deterministic`, one seed checkpoint loaded by both cells; the dp2 row is the usual dp-degree shift of the debug flavor (the two cells see different tokens per step), not a kernel difference:

| cell | world | step 1 | step 2 | step 3 | step 10 | grad_norm at step 1 |
|---|---|---|---|---|---|---|
| dp1 | 1 | 12.57037 | 9.74440 | 7.56620 | 3.94650 | 15.8125 |
| dp2 | 2 | 12.51296 | 10.10287 | 7.56612 | 3.27074 | 16.6250 |

### Changed files

    torchtitan/models/kimi_k3/kda.py         the impl knob and its resolution
    tests/unit_tests/test_kda_attention.py   capability gate dropped
    tests/unit_tests/gpu/test_kimi_k3.py     capability gate dropped

--- PASTE END ---
