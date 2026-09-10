# PR title: [Kimi K3] Compile each transformer block, with the KDA kernel wrapper carved out

Fork branch `k3_compile_blocks` = `5ee84f0c4`, one commit on main `ac10ca48f`, independent of the parallelism PRs (#4312, #4499, #4500). Measured on one RTX 5060 Ti (SM120) with the KDA capability guard lifted locally for the run; the branch leaves the guard as main has it.

--- PASTE BEGIN ---

## Summary

`parallelize_kimi_k3` raised `Kimi K3 does not support model compilation yet` for `--compile.enable --compile.components model`. Now each transformer block compiles through the shared `apply_compile` helper, and the KDA kernel wrapper is kept out of the graph:

- `KDAKernel.forward` (the Attention Gym Triton kernels with their gate and normalisation preprocessing, and under context parallelism the per-fragment state exchange) is marked `torch.compiler.disable(recursive=True)` once, at class level, so it runs eagerly and the graph breaks around it; dynamo does not trace through those kernels.
- The shared helper gains a `fullgraph` argument (default `True`, unchanged for every other model); Kimi K3 passes `fullgraph=False`, since the graph breaks around the kernel wrapper make a full graph impossible. A block alternates between KDA and MLA attention, so the compiled blocks specialise per attention class.

## Implementation

`torchtitan/distributed/compile.py`: `apply_compile(..., fullgraph: bool = True)`, threaded into `transformer_block.compile(backend=backend, fullgraph=fullgraph)`. `torchtitan/models/kimi_k3/parallelize.py`: the refusal is gone; `_apply_compile_kimi_k3` runs after activation checkpointing is applied and before FSDP, as for the other models; `_carve_kernels_out_of_dynamo` applies the class-level disable once, not per model part.

## Limitations

- Numerics and peak memory only; throughput is not measured (the debug model's steps are seconds either way).
- Compile is exercised at one data-parallel rank; TP, PP and CP under compile are not (they are not on main's Kimi K3).
- The kernel wrapper runs eagerly, so the compiled region is everything around it: projections, norms, the MoE and the attention residual.

## Tests

```text
pytest tests/unit_tests/cpu/test_integration_test_definitions.py   # 13 passed
pre-commit run --files torchtitan/distributed/compile.py torchtitan/models/kimi_k3/parallelize.py
```

Every hook passes on the two files; `pyrefly check` yields the same error set as main `ac10ca48f` (the environment's missing-import entries, none in the touched files), with the pinned 0.45.1 and with 1.2.0.

## Results

The Kimi K3 debug model (24 layers), one data-parallel rank, spmd_types, 4096 tokens per step in 256-token micro-batches, seed 42, deterministic, the same step-0 checkpoint for both runs, three steps:

| run | loss 1 | loss 2 | loss 3 | grad norm 1 / 2 / 3 | peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | `12.53584` | `9.49238` | `6.60885` | `14.6250` / `16.2500` / `9.7500` | 12.64 GiB |
| compiled (`--compile.enable --compile.components model`) | `12.53584` | `9.49238` | `6.60885` | `14.6250` / `16.2500` / `9.7500` | 12.64 GiB |

Bitwise over the three steps at the same peak memory. The same compiled cell on main `ac10ca48f` without this commit stops at

```text
NotImplementedError: Kimi K3 does not support model compilation yet.
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
