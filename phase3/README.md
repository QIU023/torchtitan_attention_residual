# Phase 3 — Pipeline Parallel Integration

Goal: get Block AttnRes running end-to-end under torchtitan's pipeline
parallelism (`PP`) on 8× GPUs, first verifying numeric correctness against
the single-GPU reference, then adding the **cross-stage caching adapter**
that makes per-stage bandwidth constant instead of `O(stage_id × d)`.

The adapter is what the PR #2 headline hangs on
("AttnRes PP overhead < 5 % even over PCIe"); it is NOT required for
correctness — Phase 2's `AttnResLlama3Model.forward` already returns the
right tuple at middle stages and PyTorch's `_PipelineSchedule` unpacks
tuples automatically.

## Staging (direct 8-GPU path)

1. **8-GPU prep** (rental, ~15 min): env + tokenizer + C4 shard prefetch
   (`prefetch_c4.py`) so long runs don't depend on HF streaming.
2. **8-GPU naive PP smoke** (~20 min): `PP=8, VP=2, FSDP inner,
   Llama3-150M AttnRes, 500 steps, adapter OFF`. Proves the PP path
   boots end-to-end on real NCCL; baselines the per-stage send size
   (expected to grow linearly in stage id).
3. **8-GPU adapter smoke** (~20 min): same config, `TORCHTITAN_ATTNRES_CACHE=1`,
   adapter ON. Loss must match naive within bf16 tolerance; per-stage
   send size becomes constant. A/B the comm trace (`nsys profile` or
   `torch.profiler`).
4. **Scale run** (full PR #2 headline): 1.5–2 B dense AttnRes, `PP=8,
   VP=2`, 20B tokens. Produces the money plot.

The `go_8gpu.sh` orchestrator runs steps 1-3 end-to-end so on a fresh
rental box you do:

```bash
bash phase3/go_8gpu.sh
```

## Files in this folder

| File | Role |
| --- | --- |
| [`go_8gpu.sh`](./go_8gpu.sh) | **Orchestrator**. Env check → install → tokenizer → C4 prefetch → unit tests → naive PP → adapter PP → compare. Run this first on a fresh rental box. |
| [`prefetch_c4.py`](./prefetch_c4.py) | Parallel C4 shard download into HF cache. Default 150 shards (~45 GB, ~22B tokens). Addresses the streaming httpx crash we hit on Phase 2 N=12. |
| [`fake_pg_test.py`](./fake_pg_test.py) | **Optional** single-GPU `PP=4` fake-process-group smoke. Useful when debugging numerics locally; not required for the 8-GPU path. |
| [`launch_8gpu_naive.sh`](./launch_8gpu_naive.sh) | `PP=8, VP=2` launcher for the naive path. Uses existing `--module attn_res` configs. Sets `LOG_RANK=0` so only rank 0 tees to `train.log`. |
| [`launch_8gpu_adapter.sh`](./launch_8gpu_adapter.sh) | Same as above but exports `TORCHTITAN_ATTNRES_CACHE=1` to switch on the adapter (wiring in `torchtitan/experiments/attn_res/pipeline_adapter.py`). |
| [`adapter_design.md`](./adapter_design.md) | State machine + invariants for the adapter. Read before debugging: it enumerates the five open unknowns the design rests on (microbatch keying, VP chunk order, backward hook reliability, activation-checkpoint interaction, FSDP reshard composition). The adapter itself now lives in-experiment at [`torchtitan/experiments/attn_res/pipeline_adapter.py`](../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py). |
| [`compare_pp_vs_single.py`](./compare_pp_vs_single.py) | After 2+3 above finish, extract first-N-step loss arrays from each run's TB events and print max-abs diff; sanity check that PP didn't silently break numerics. |

## Not in this folder (must still be done)

- A 1.5 B `attn_res` Trainer config — add to
  `torchtitan/experiments/attn_res/config_registry.py` once we pick the
  exact shape (first draft: dim=2048, n_layers=16, n_blocks=8 so
  `layers_per_block = 2` and `PP_stage_layers = 2` line up).
- Wiring so `parallelism.pipeline_parallel_degree > 1` does not silently
  disable the AttnRes path. Need to check `Llama3Model.update_from_config`
  (it currently only complains about weight-tying + PP).

## Tomorrow's 8-GPU playbook

```bash
# 0. From laptop, rent 8× RTX 5090 PCIe on vast.ai. Copy:
scp -r phase3/ experiments-attn_res/ rental:~/work/

# 1. On the rental box, set up env (same recipe as phase2)
bash phase2/setup_env.sh

# 2. Single-GPU fake-PG smoke first (free, 3 min)
python phase3/fake_pg_test.py

# 3. 8-GPU naive PP, 500 steps
NGPU=8 bash phase3/launch_8gpu_naive.sh

# 4. 8-GPU adapter, 500 steps (expect same loss, lower comm time)
NGPU=8 TORCHTITAN_ATTNRES_CACHE=1 bash phase3/launch_8gpu_adapter.sh

# 5. Compare
python phase3/compare_pp_vs_single.py \
    --single phase3/runs/single_reference/tb \
    --pp phase3/runs/pp8_naive/tb \
    --pp_cached phase3/runs/pp8_adapter/tb
```
