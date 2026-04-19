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

## Staging (cheap → expensive, do in order)

1. **Fake-PG smoke** (single GPU, free): build the AttnRes 150M model under
   a 4-stage fake process group, run 10 steps, compare loss to a reference
   single-GPU run. Pure correctness.
2. **Real 8-GPU naive PP** (rental, ~2 h): `PP=8, VP=2, FSDP inner`,
   Llama3-150M AttnRes, 500 steps. No caching adapter yet — every stage
   sends the full growing `stacked_blocks`. Measure step time + NCCL comm
   volume to get the "before" number.
3. **Enable adapter** (same 8 GPUs): flip the adapter on via flag, repeat
   the 500-step run, measure step time + NCCL comm volume. The adapter
   target: per-stage forward send/recv size becomes constant in stage id.
4. **Scale run** (full PR #2 headline): 1.5–2 B dense AttnRes, `PP=8,
   VP=2`, 20B tokens, full config. Produce the money plot.

## Files in this folder

| File | Role |
| --- | --- |
| [`fake_pg_test.py`](./fake_pg_test.py) | Single-GPU `PP=4` fake-process-group smoke. Verifies the tuple-return pattern round-trips and AttnRes numerics match single-GPU to within rtol=1e-4. |
| [`launch_8gpu_naive.sh`](./launch_8gpu_naive.sh) | `PP=8, VP=2` launcher for the naive path. Uses existing `--module attn_res` configs. Sets `LOG_RANK=0` so only rank 0 tees to `train.log`. |
| [`launch_8gpu_adapter.sh`](./launch_8gpu_adapter.sh) | Same as above but exports `TORCHTITAN_ATTNRES_CACHE=1` to switch on the adapter (see `adapter.py`). |
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
