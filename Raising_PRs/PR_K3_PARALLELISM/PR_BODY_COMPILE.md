# PR title: [Kimi K3] Compile the transformer blocks and the vision tower

Fork branch `k3_compile_blocks` = [`a4e6f4e3b`](https://github.com/QIU023/torchtitan/commit/a4e6f4e3b), one commit on main `b21f7d43e`, Kimi K3 only (`parallelize.py` +10/-2), independent of the parallelism PRs. Design choice (user, 2026-09-14): main's `apply_compile` unchanged, no KDA carve-out, no core argument. Body in the #4577 format (2026-09-14).

Not ready to file, per the numerics rules:
- The table has no noise-floor row (eager twice on the shared cache).
- `aot_eager` matches eager at step 1 only (step 2 `9.88943` against `9.89441`); unexplained.
- The commit message says inductor's code for `KDAKernel.forward` and for the tower moves the step-1 loss and the other blocks read eager bitwise when those two stay eager. No logbook record (probe, tree, logs) backs that yet, so the body does not cite it.
- The numbers are from `1cae3fd62` (decoder only). Text-only runs never execute the tower, so the tower compile added in `a4e6f4e3b` is not exercised by any run; rerun on `a4e6f4e3b` with a multimodal cell. Raw logs are not in the logbook.
- `PASS-COUNT`: fill from a run on `a4e6f4e3b` before filing.

--- PASTE BEGIN ---

## Summary

Enable `torch.compile` for Kimi K3, which `parallelize_kimi_k3` refused with `NotImplementedError` for `--compile.enable --compile.components model`.

- Call the shared `apply_compile` on the decoder after activation checkpointing and before FSDP, and on the vision tower when present, as Qwen3.5 and Kimi K2.5 do.
- No core change: Dynamo traces each block, the KDA kernel wrapper included, with `fullgraph=True`.

## Design

`apply_compile` compiles every block of `model.layers` with `fullgraph=True`, and the KDA kernel wrapper traces as it is, so Kimi K3 needs no carve-out and no new argument on the core function. The vision tower goes through the same call, in the order the other vision models use: activation checkpointing, compile, FSDP.

## Results

Kimi K3 debug model (24 layers), one data-parallel rank, 8192 tokens per step in 256-token micro-batches, seed 42, deterministic, three steps, one inductor cache warmed by a one-step compiled pass; text-only data, so the vision tower does not run.

```bash
NGPU=1 CONFIG=kimi_k3_debugmodel MODULE=kimi_k3 ./run_train.sh --parallelism.data_parallel_shard_degree 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --debug.seed 42 --debug.deterministic --training.steps 3 --compile.enable --compile.components model
```

| run | loss 1 / 2 / 3 | grad norm 1 / 2 / 3 | peak memory |
| --- | --- | --- | ---: |
| eager | `12.54770` / `9.89441` / `7.69925` | `15.1250` / `14.4375` / `8.0000` | 12.64 GiB |
| compiled (inductor) | `12.54321` / `9.82954` / `7.87903` | `15.0000` / `14.6875` / `9.5000` | 12.63 GiB |
| compiled, `--compile.backend aot_eager` | `12.54770` / `9.88943` / `7.75367` | `15.1250` / `14.3750` / `8.6875` | 12.62 GiB |

The same command on main stops at `NotImplementedError: Kimi K3 does not support model compilation yet.` Compile is exercised at one data-parallel rank; TP, PP and CP under compile are not.

## Test plan

- `pytest tests/unit_tests/cpu/test_compile_config.py tests/unit_tests/cpu/test_integration_test_definitions.py tests/unit_tests/cpu/test_no_new_cli_options.py tests/unit_tests/cpu/test_skip_dp.py tests/unit_tests/cpu/test_context_parallel_validation.py tests/unit_tests/cpu/test_loss.py -q` (`PASS-COUNT`)
- `ufmt check torchtitan/models/kimi_k3/parallelize.py` and `pyrefly check`: no new error against main (`PASS-COUNT`)
- The GPU command above.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
