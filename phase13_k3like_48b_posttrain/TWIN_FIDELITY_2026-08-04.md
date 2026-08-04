# The #4025 twin was architectural only, and that hid three legs

Refs: pytorch/torchtitan#3029, pytorch/torchtitan#4025

Corrects the conclusion in `CP_MULTIMODAL_HANG_RESOLVED_2026-08-04.md` that
three legs of the twin matrix were blocked by a hardware ceiling.

## What the twin actually reproduced

`kimi_k3_debugmodel_pr_4025` copied every architectural extent from #4025's
`_debugmodel`: 13 layers at dim 256, 4 heads, q_lora 128 / kv_lora 64,
qk_nope 32 / qk_rope 16 / v 32, full attention on {4, 8, 12}, KDA elsewhere,
AttnRes block size 12, vocab 163840, and the 4-layer MoonViT.

It did not copy that PR's `Trainer.Config`. Those fields came from
`kimi_k3_mini_vl`'s chain instead:

| field | #4025 | twin, before | effect |
|---|---|---|---|
| `training.dtype` | `bfloat16` | `float32` | changes which legs run at all |
| `max_patches` | 256 | 1024 | one image nearly fills the sequence |
| `max_patches_per_side` | 16 | 64 | as above |
| `max_pixels` | 224x224 | 1048576 | as above |
| `seq_len` | 256 | 512 (CLI) | different shard sizes under CP |
| `local_batch_size` | 1 | 4 (CLI) | whether accumulation runs at all |

So the comparison was "our parallelism on their model" rather than "our
parallelism on their configuration", which is what it claimed to be.

## Why dtype is the one with teeth

`training.dtype` is applied to the model itself. `mixed_precision_param` is
consumed only by FSDP. torchtitan's FSDP mesh is `dp_shard x cp`, so a layout
with `dp_shard 1` and no CP has an FSDP mesh of size 1, FSDP is not applied,
and nothing casts the parameters. The twin therefore ran KDA on fp32 operands
in exactly three legs -- dp1, pp2, tp2 -- and the fla kernel at these head
dimensions then asks for 108160 bytes of dynamic shared memory:

    tvm.error.InternalError: Failed to set the allowed dynamic shared memory
    size to 108160

This GPU (RTX 5060 Ti, consumer Blackwell cc 12.0) allows 101376.

The pattern fits the dtype explanation exactly, including the case that first
looked like a counterexample: `cp2` also has `dp_shard 1` yet passes, because
turning CP on makes the FSDP mesh size 2, FSDP applies, and the parameters
become bf16.

Confirmed by prediction rather than by story. If dtype is the variable, then
forcing `--training.mixed_precision_param float32` onto the *passing* `fsdp2`
leg should reproduce the failure. It does, with the identical 108160. And
batch size is not involved: `local_batch_size` 1, 2 and 4 all request 108160.

## The correction

Recorded earlier as "a hardware limit of the box, not a defect, and not ours".
Half right and the wrong half mattered. 108160 > 101376 is a real hardware
limit, but the twin only asked for 108160 because it was running fp32, and it
was running fp32 because the flavor omitted a field #4025 sets. On #4025's
actual configuration the question does not arise.

The general lesson is narrower than "check dtype": a config that copies a
reference model's *architecture* and inherits everything else is not a twin,
and the fields it silently inherits are exactly the ones nobody re-derives.

## What changed

The flavor now carries #4025's dtype, image budget, sequence length, batch
size, learning rate and schedule. Two consequences beyond the three legs:

* `local_batch_size 1` against global batch 8 makes gradient accumulation the
  default, which is the configuration that exposed the zero-sentinel CP defect
  (`CP_MULTIMODAL_HANG_RESOLVED_2026-08-04.md`). The matrix now runs it by
  default instead of avoiding it.
* Matrix runs no longer override `seq_len` or `local_batch_size` on the command
  line, so what runs is the flavor as registered.
