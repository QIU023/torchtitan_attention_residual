# The MoE rollout was never blocked on SITU — 2026-09-02

## Retraction first

The standing note that "GRPO with MoE is blocked on vLLM SITU kernel coverage
(no CUDA kernel on SM120)" is **wrong and is withdrawn**. The K3 debug model
with its routed experts restored now serves on a 5060 Ti (SM120) through the
current vLLM build, and nothing on the path rejects the SITU activation.

Reproduce: `matrix_scripts/vllm_moe_rollout_probe.py` under `/venv/vllm_k3`.

```
GENERATED: ' licensors move孟 Deputy公开市场但并且可以坦荡'
```

The text is noise because the weights are a seeded random init; the point is
that the engine builds the MoE, selects a kernel and decodes.

## What the blocker actually was: three unrelated faults in a row

Each one produced a failure that looked like the previous diagnosis, which is
how the wrong conclusion survived.

**1. The measurement was taken on the wrong vLLM.** `venv_verl` carries
vLLM `d6938b740` (2026-08-03), whose unquantized triton MoE path predates SITU:

    # venv_verl, triton_moe.py
    _supports_activation -> [SILU, GELU, GELU_TANH, SWIGLUOAI, ...]   # no SITU

    # vllm_k3 6dc76a9ad, same function
    _supports_activation -> [SILU, GELU, GELU_TANH, SITU, SWIGLUOAI, ...]

A property of one build was written down as a property of vLLM.

**2. `num_experts: 896` in the exported config.** vLLM reads `num_experts`
(`models/kimi_k3/nvidia/model.py:395`), not `n_routed_experts` (32), so it
allocated the released expert count and died with `torch.OutOfMemoryError:
Tried to allocate 4.59 GiB`. That reads as "16 GB is too small for MoE", which
is the same shape of conclusion as the first fault.

**3. `routed_expert_hidden_size: 3584` in the same config.** The released
latent width, against a debug latent of 512, so the latent down projection
failed its load-time shape assert: `Tried to load weights of size
torch.Size([512, 1024]) to a parameter of size torch.Size([3584, 1024])`.

## Why nothing caught faults 2 and 3 earlier

The debug HF directory descends from a copy of the reference config, so every
field the downscaled model does not exercise still holds the released value.
Dense and text-only exports have no routed experts and no latent MoE, so
neither field was ever read -- they served correctly for weeks with both wrong
values sitting in their config.json. The faults appeared exactly when the RL
flavor regained its MoE stack.

## What now guards it

`matrix_scripts/hf_export_shape_check.py` reads an export's `config.json` and
its safetensors and compares every shape-bearing field against the tensors:
vocab, hidden, layer count, latent width, MLA low ranks, dense and expert
intermediates, and the expert count. Verified non-vacuous by re-running it
against the pre-fix config, which it fails on both fields.

`matrix_scripts/export_rl_moe.py` now writes those fields from the titan
config instead of inheriting them, and calls the checker as its last step, so
the export cannot succeed while disagreeing with its own weights. All six K3
exports on this box pass.

## Open

- veRL still runs against the old vLLM in `venv_verl`. The newer wheel is built
  against torch 2.13 while that venv is on 2.14.dev, so the swap needs its own
  venv and an ABI check first -- not an in-place `pip install`.
- Quantized MoE remains untried. On CUDA the auto priority list is TRTLLM
  MXFP8, DeepGEMM FP8, then Marlin; K3's deployment format is weight-only
  MXFP4 (W4A16), which is the Marlin entry, so `--moe-backend marlin` is the
  first thing to test rather than assuming it is unsupported.
