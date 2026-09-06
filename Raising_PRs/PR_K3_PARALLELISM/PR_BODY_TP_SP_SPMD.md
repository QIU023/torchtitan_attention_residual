# PR title: [Kimi K3] Tensor parallelism with sequence parallel, on the spmd_types declarations

PR branch `k3_tp_sp` on the fork (`c0ab32d07`, same commit as `tpsp_review3`: three commits on the declarations PR `k3_spmd_decl` `dbc60701d`, on upstream/main `390e2985b` with 4446 merged). Stacked: filed as a draft on top of the declarations PR and rebases when that one lands. CPU: 19 tests pass on the branch's files, pyrefly count equal to main's. The TP types were checked at tp2 with float32 per-parameter gradient dumps (see Results). Paste between the markers; the first line of the GitHub body says which PR it stacks on.

--- PASTE BEGIN ---

### Summary

Enables tensor parallelism for Kimi K3, with sequence parallel on the same mesh, on the previous PR's declarations. Before this change `parallelize_kimi_k3` rejects `tensor_parallel_degree > 1` and the declarations are issued at tp = 1; after it TP comes off the unsupported list, `model.parallelize` runs under it, and `parallelism.enable_sequence_parallel` (core default: on) decides whether the stream between modules is Replicate or the TP-axis Shard(0). Both attention kinds are head-parallel: MLA on its head projections, KDA on its per-head state with Attention Gym's kernel on the local heads behind a `local_map` on `inner_kda`.

### Design

- Sequence parallel (`model.py`): norms compute on the sequence shard, the attention boundaries gather it on the way in and `wo` / `output_proj` reduce-scatter back to `S(0)` -- the GQA pattern. The stream weights follow `norm_config`'s rule: invariant without SP, replicated under it.
- The multimodal splice: *vision_positions* index the global token axis and a placeholder run can cross a shard boundary, so `_splice_under_sequence_parallel` gathers the stream (`S(0) -> R`, reduce-scatter backward), splices on the whole sequence, and `forward` cuts the shard back out past the DP-local vision region. Found by the tp4 cell, whose 64-token shards cut the debug image.
- Two type-checking seams at tp > 1: the shared vision encoder mutated its position tables to `R` on every axis while the tower declares them invariant on tp, and `local_head_split` asserted heads sharded on tp. Both keep the declared TP type now.
- `clip_grad_norm_` (`distributed/utils.py`): parameters are grouped by mesh before the norm. Undeclared, hence replicated, modules under TP hold gradients on `(fsdp, tp)` and `(fsdp,)`, and `get_total_norm`'s foreach stack refuses to mix them; disjoint groups combine exactly ($(\sum \lVert g Vert^p)^{1/p}$, max for inf), the algebra the EP path already uses. With one mesh it is the single call it always was.

### Results

Correctness is claimed on the gradient: at tp2, with and without sequence parallel, float32 end-to-end step-1 gradient dumps (a per-expert float32 loop for the experts) against dp1 show every replicated parameter bitwise identical across the two TP ranks and within noise of dp1 -- 744 and 742 of 750 parameters, the rest `A_log` entries whose gradients are 1e-4-scale noise in bf16.

`kimi_k3_debugmodel` (multimodal), `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read; on an RTX 5060 Ti with Attention Gym at upstream/main `b19162e` (2026-09-04) and its SM100/SM103 guard in `kda.py` lifted locally, which routes KDA through Attention Gym's portable kernels. The first row names `partial_dtensor` (since 4446 the default backend is `spmd_types`), every other row runs under `spmd_types`. The rows are grouped by DATA STREAM, because the loader shards documents by dp rank (`components/data/sources.py`): tensor parallel does not shard the data, so every tp cell in the first block reads exactly dp1's samples and is comparable to it; the second block's cells all read dp2's, and the two blocks are not comparable to each other. The last three rows run the flavor the way 4446's B200 cell does (type checking on, activation checkpointing off): dp1 and tp2 are bitwise their checked-off rows, the dp2 x ep2 x tp2 row moves from step 3 on with activation checkpointing off, as the declarations PR's dp2 rows do.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
# sequence parallel off: add --parallelism.no-enable-sequence-parallel; expert parallel: --parallelism.expert_parallel_degree 2
```

Data stream A (dp degree 1: dp1's samples):

| cell | world | backend | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | partial_dtensor | 12.52977 | 7.36833 | 2.91045 |
| dp1 | 1 | spmd_types | 12.52977 | 7.36833 | 2.91045 |
| dp1 | 1 | spmd_types, another compile cache | 12.52977 | 7.27107 | 2.98077 |
| tp2 (SP on) | 2 | spmd_types | 12.54164 | 7.39513 | 3.16153 |
| tp2 (SP off) | 2 | spmd_types | 12.55332 | 7.31955 | 3.08457 |
| tp4 (SP on) | 4 | spmd_types | 12.52816 | 7.11398 | 2.99496 |
| dp1 | 1 | spmd_types, type checking, AC off | 12.52977 | 7.36833 | 2.91045 |
| tp2 (SP on) | 2 | spmd_types, type checking, AC off | 12.54164 | 7.39513 | 3.16153 |

The third row is this flavor's noise floor: the same cell, same seed and samples, step-1 gradients bitwise, on another compile cache -- 1.3% at step 3 and 2.4% at step 10 from the row above it. The tp cells' step-1 loss sits 0.1 to 0.2 percent from dp1 (the head-sharded matmuls and the boundary collectives round differently, and the random-init MoE router is sensitive to it) and their later steps move within that floor.

Data stream B (dp degree 2: a different set of samples from step 1 on, so these rows compare only with each other):

| cell | world | backend | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp2 | 2 | spmd_types | 12.53137 | 7.25082 | 3.15411 |
| dp2 x ep2 | 2 | spmd_types | 12.53146 | 7.13441 | 3.09174 |
| dp2 x tp2 (SP on) | 4 | spmd_types | 12.53383 | 7.22083 | 3.27481 |
| dp2 x ep2 x tp2 (SP on) | 4 | spmd_types | 12.53826 | 7.34735 | 3.19884 |
| dp2 x ep2 x tp2 (SP on) | 4 | spmd_types, type checking, AC off | 12.53826 | 7.28868 | 3.22917 |

The SP benefit case is long-sequence; at this scale it shows neither a memory win nor a speed cost worth reporting.

### Changed files

    torchtitan/models/kimi_k3/
      model.py              +82/-7   enable_sp from the parallelism config; the splice under sequence parallel; the stream's layout after the vision region; the latent MoE norm on the sequence shard
      parallelize.py        +14/-2   tensor parallel off the unsupported list; model.parallelize under TP; the tp group for the splice
    torchtitan/distributed/
      utils.py              +42/-4   clip_grad_norm_ grouped by parameter mesh
    torchtitan/models/common/
      attention.py          +7/-2    local_head_split keeps an invariant TP type
    torchtitan/models/kimi_k2_7/
      vision_encoder.py     +3/-6    the position tables keep their declared TP type under type checking
    tests/unit_tests/cpu/
      test_kimi_k3_sp_splice.py  +76/-0  the sequence-parallel splice on a one-rank group (new)

### CI/CD Coverage

One CPU test (the sequence-parallel splice through the real collectives on a one-rank group). A tp2 cell next to 4446's `kimi_k3_debugmodel_mm_fsdp2` on B200 is the natural addition; the type-checked tp2 row above is that cell run locally. Type checking passes at tp2 with and without sequence parallel and at dp2 x ep2 x tp2 (the last three rows).

--- PASTE END ---
