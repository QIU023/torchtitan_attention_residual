# PR #4 — `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer`

**Target repo**: `pytorch/torchtitan`
**Target file**: `torchtitan/experiments/rl/actors/trainer.py` (function `PolicyTrainer._build_model`)
**Fork reference**: torchtitan `attention_residual_dev` branch; pattern lives in fork's `experiments/rl/actors/trainer.py`.
**Effort**: ~1 day (patch + per-flavor smoke + PR description).
**Risk**: low — additive kwargs with sensible defaults already in the existing trainer config.

---

## Suggested PR title

> [experiments/rl] `PolicyTrainer._build_model` — pass full `parallelize_fn` kwarg surface so non-Qwen3 model_specs work out of the box

---

## Suggested PR body

### Summary

`PolicyTrainer._build_model` currently invokes
`model_spec.parallelize_fn(model, parallel_dims=, parallelism=,
compile_config=)`. This works for Qwen3 (whose `parallelize_qwen3` accepts
only those three) but fails for every other torchtitan `parallelize_*`
function (`parallelize_kimi_linear`, `parallelize_deepseek_v3`,
`parallelize_llama`, `parallelize_attn_res`), which all require four
additional kwargs: `training`, `model_converters`, `ac_config`,
`dump_folder`.

The result is `TypeError: parallelize_kimi_linear() missing 4 required
positional arguments: 'training', 'model_converters', 'ac_config',
'dump_folder'` the moment a user points `PolicyTrainer` at a non-Qwen3
model_spec. There is no per-flavor escape hatch — every RL entry-point
currently has to ship a hand-rolled adapter wrapper.

### Patch

```python
# torchtitan/experiments/rl/actors/trainer.py
# inside PolicyTrainer._build_model, replacing the existing
# parallelize_fn call:

parallelize_kwargs = dict(
    parallel_dims=self.parallel_dims,
    parallelism=config.parallelism,
    compile_config=config.compile,
    # NEW kwargs below — additive, sensibly defaulted from config.
    training=config.training,
    model_converters=getattr(config, "model_converters", None)
                     or _DefaultModelConverters(),
    ac_config=config.activation_checkpoint,
    dump_folder=config.dump_folder,
)
model = model_spec.parallelize_fn(model, **parallelize_kwargs)
```

`_DefaultModelConverters` is a no-op fallback so model_specs that don't
need converters keep working without code changes on the caller side.

### Why all four kwargs are needed

- `training`: `parallelize_kimi_linear` reads `training.async_tensor_parallel`
  to pick between sync and async TP collectives. Without it, falls back
  to the wrong default and silently degrades throughput.
- `model_converters`: passed to FP8 / MXFP4 converters for quantization-
  aware parallelism (the converter sees TP / FSDP plans before applying).
  RL trainers that don't enable quant get a no-op converter.
- `ac_config`: activation-checkpoint policy is per-parallelism-plan
  (e.g. AC interacts with PP microbatch scheduling). Without it,
  PP-enabled RL trainers silently disable AC.
- `dump_folder`: parallelize_fn writes plan diagnostics here. Without
  it, plan inspection / debugging across the trainer's own dump folder
  breaks.

### Test plan

1. Existing Qwen3 RL smoke (`experiments/rl/tests/test_qwen3_smoke.py` or
   equivalent) stays green — Qwen3's `parallelize_fn` ignores the new
   kwargs.
2. Add a smoke test for `PolicyTrainer` instantiation against
   `kimi_linear_447m_aligned_block_attn_res_n4` (or any non-Qwen3
   model_spec) — currently fails with `TypeError`, after patch boots.

### Why upstream / why land it

torchtitan's `experiments/rl` is the path toward an in-tree RL trainer.
Today it's Qwen3-only by accident of the kwarg surface; this PR removes
the accidental coupling so the trainer becomes model-agnostic. Removes
the need for per-flavor adapter shims that downstream forks
(ours included) currently duplicate.

### Discovered via

While porting our Kimi-Linear AttnRes (Block Attention Residuals,
[arXiv:2603.15031](https://arxiv.org/abs/2603.15031)) through
torchtitan's `PolicyTrainer` for GRPO multimodal training. Hit the
signature mismatch on day-1; patched in our fork
(`QIU023/torchtitan@attention_residual_dev`) and confirmed every
non-Qwen3 model_spec hits the same failure.

### Out of scope (separate follow-up PRs)

- **Engine-agnostic `Generator` abstraction + SGLang reference impl**
  (`experiments/rl/actors/sglang_generator.py` + RFC).
  Tracked as a separate RFC since it requires upstream design
  discussion on the Generator interface; depends on this PR landing
  first.
- **DCP-native checkpoint load path in `PolicyTrainer`** —
  currently each RL launcher rolls its own. Worth standardising in
  a follow-up PR.

---

## Filing checklist

- [ ] Fork branch up to date with torchtitan `main`.
- [ ] Single-commit PR titled per above.
- [ ] Test: existing Qwen3 RL smoke green.
- [ ] Test: new `kimi_linear` smoke for `PolicyTrainer._build_model`.
- [ ] PR body links the Generator RFC follow-up so reviewers see the
      multi-PR shape.
- [ ] CC `@fegin` / `@wconstab` if they want to weigh in on the
      `_DefaultModelConverters` default (or accept any reviewer-suggested
      shape).
