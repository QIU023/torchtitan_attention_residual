# PR title: [Kimi K3] Compile the transformer blocks and the vision tower

Fork branch `k3_compile_blocks` = [`5247ca305`](https://github.com/QIU023/torchtitan/commit/5247ca3057096a52e1da33da4fbb31c900b41db7), one commit on main `b21f7d43e`. Touches core once: `raise_dynamo_recompile_limit` moves from a private gpt_oss helper into `distributed/compile.py` (gpt_oss keeps its value). Body in the #4577 format (2026-09-14).

Notes for filing:
- The GPU numbers were measured on `a4e6f4e3b` and `640b101e1`; `5247ca305` changes only comments, docstrings and the test's style against `640b101e1`, and the 24-layer rows were identical with and without the helper.
- The debug flavor trains on the multimodal `cc12m-test` set, so the vision tower runs in every row.
- One RTX 5060 Ti, KDA capability guard lifted locally for the runs (not part of the branch). Raw logs: `Raising_PRs/PR_K3_PARALLELISM/logs_compile_2026-09-14/`.
- Located as far as the probes go: under `aot_eager` every activation gradient of layer 23 is bitwise and only the weight gradients of `wq_a`, `wq_b`, `wkv_a`, `wkv_b` and `attention_res_proj` differ (rel 6.4e-7 to 1.6e-4; 2 of 32 micro-batches tapped), so the gap is in those weight-gradient matmuls, not in flex attention or the residual path; the kernel or layout is not located. Under inductor the step-1 shift comes from compiling `KDAKernel.forward` and the tower (with both left eager an earlier revision read eager bitwise over three steps); not located below module level.
- Not run: the gpt_oss path on GPU (unit-tested only) and the 93-layer layout.

--- PASTE BEGIN ---

## Summary

Enable `torch.compile` for Kimi K3, which `parallelize_kimi_k3` refused with `NotImplementedError` for `--compile.enable --compile.components model`.

- Call the shared `apply_compile` on the decoder after activation checkpointing and before FSDP, and on the vision tower when present, as Qwen3.5 and Kimi K2.5 do.
- `raise_dynamo_recompile_limit` (`distributed/compile.py`): raise Dynamo's per-code-object recompile limit to at least a model's bound, never lower it; gpt_oss's private helper becomes this call with the same value.
- Kimi K3 passes a bound computed from its config, so layouts deeper than the debug flavor compile.
- Add a CPU test of the helper, gpt_oss's value and the Kimi K3 bound.

## Design

Every decoder and tower block reaches Dynamo through one code object, the checkpoint wrapper's `forward`, and compiles it once per distinct block shape: MLA or KDA, whether the layer opens an attention-residual block, and the width of the residual stack it reads. The 24-layer debug model needs 8 variants, exactly Dynamo's default limit, and a 33-layer model needs 9, which stops before the first step under `fullgraph=True`. The bound counts those combinations from the config and adds 4 for the tower's image shapes and the FSDP wrapper types: 10 for the debug model, 12 for 33 layers, 28 for the released 93-layer layout.

The helper lives in `distributed/compile.py` next to `apply_compile` because two models now need it; gpt_oss computes its own bound as before.

Compiled numerics differ from eager at the rounding level. With `--compile.backend aot_eager` the forward is exact (step-1 loss and grad norm match eager) and the difference enters in a few projection weight-gradient matmuls; with inductor it comes from the generated code for the KDA kernel wrapper and the tower.

## Results

Kimi K3 debug model (24 layers and the vision tower, multimodal debug data), one data-parallel rank, 8192 tokens per step in 256-token micro-batches, seed 42, deterministic, three steps, one inductor cache warmed by a one-step pass of each run.

```bash
NGPU=1 CONFIG=kimi_k3_debugmodel MODULE=kimi_k3 ./run_train.sh --parallelism.data_parallel_shard_degree 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --debug.seed 42 --debug.deterministic --training.steps 3 --compile.enable --compile.components model
```

| run | loss 1 / 2 / 3 | grad norm 1 / 2 / 3 | peak memory | step-1 gradients bitwise with eager |
| --- | --- | --- | ---: | ---: |
| eager | `12.54770` / `9.89441` / `7.69925` | `15.1250` / `14.4375` / `8.0000` | 12.64 GiB | reference |
| eager, fresh cache A | same | same | 12.64 GiB | 726 / 726 |
| eager, fresh cache B | same | same | 12.64 GiB | 726 / 726 |
| compiled (inductor) | `12.55324` / `9.86780` / `7.58297` | `15.0625` / `14.6875` / `10.6875` | 12.62 GiB | 0 / 726 |
| compiled, `--compile.backend aot_eager` | `12.54770` / `9.88943` / `7.75367` | `15.1250` / `14.3750` / `8.6875` | 12.61 GiB | 25 / 726 |

The inductor row read the same three steps twice. With 33 layers the compiled run trains (`12.54158` / `10.23301` / `7.98758`, 9 wrapper variants under a bound of 12); without the helper it stops before step 1 with `recompile limit exceeded`. The same command on main stops at `NotImplementedError: Kimi K3 does not support model compilation yet.` Compile is exercised at one data-parallel rank; TP, PP and CP under compile are not.

## Test plan

- `pytest tests/unit_tests/cpu/test_recompile_limit.py tests/unit_tests/cpu/test_compile_config.py tests/unit_tests/cpu/test_integration_test_definitions.py tests/unit_tests/cpu/test_no_new_cli_options.py tests/unit_tests/cpu/test_skip_dp.py tests/unit_tests/cpu/test_context_parallel_validation.py tests/unit_tests/cpu/test_loss.py -q` (72 passed)
- `ufmt check` on the four changed files; `pyrefly check`: 45 errors, the same per-file set as main.
- The GPU command above.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
