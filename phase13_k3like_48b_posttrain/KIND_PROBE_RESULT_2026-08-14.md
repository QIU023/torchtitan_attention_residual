# What the KIND probe found, and what it takes off the critical path

## The run

`dtensor_kind_probe.py` copied to `sitecustomize.py` on `PYTHONPATH`, so every rank patches
itself at interpreter startup, inside the real trainer. One step of
`ep2_fsdp2_tp2_cp2` on `tp_declarative_refactor`, the cell that has been failing.

This is the third attempt at localising this. The first two were 8-GPU bisections, and an
earlier standalone probe died with "Unknown c10d backend type FAKE" because it rebuilt
`ParallelDims` outside the trainer. Running inside the trainer removes that entirely.

## The answer

```
RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor

[kind] trainer.py:866          all plain
[kind] trainer.py:756          all plain
[kind] multimodal_model.py:950 input_ids plain, pixel_values plain, features plain, embeds plain
[kind] attn_res_model.py:524   tokens plain, h plain, partial_block plain
[kind] attn_res_model.py:205   partial_block plain, h plain, attn_out plain, ffn_out DTensor(R)
```

One tensor in the residual stream is a DTensor and the other three are not. The add fails.

**The vision tower is not involved.** Its frame is entirely plain, in forward, on the rank
that raised.

## What this changes

The 08-13 handoff carried this as "the vision tower's dynamic-CP backward fails
(`_encode_images_dynamic_cp`, `from_local` receiving a DTensor gradient)", placed there by
`--debug.detect-anomaly`, with "write a KIND-printing probe" as the next action.

The probe was written and it says something different: this cell fails FIRST, in forward, in
the AttnRes residual add, with a different error. Whatever happens in the vision tower is
downstream of a failure that happens before it. So the vision tower comes off the critical
path -- not because it was cleared, but because it is not what is failing here.

It is also not a new problem. `DECLARATIVE_MIGRATION_2026-08-13.md` predicted exactly this:

> Measured: the dense FFN alone dies with `aten.add.Tensor got mixed torch.Tensor and
> DTensor`, while the layer norms migrate byte-identically -- their output feeds attention,
> not the residual add.

So the blocker is the one already understood: **the migration unit is a residual stream,
not a module.** `ffn_out` went declarative; `partial_block`, `h` and `attn_out` did not.

## Next action

The stream has to flip together, which is what the step-B plan already said. Three
plain sources feed it -- the two AttnRes accumulators and the attention output -- and each
needs either a declaration or an `in_src` lift. This is the same wall the migration hit on
08-13, now with the failing operand named instead of inferred.

The upstream K3 tree makes this tractable in a way ours does not: it threads
`block_residual_TND` and `prefix_sum_BLD` through the block's forward SIGNATURE, where a
`sharding_config` can reach them, instead of holding them in Python locals. That is
migration step 3, and it is the reason to do the migration rather than keep patching this
tree.

## Method note

The probe cost twenty minutes to write and one run. The five 8-GPU runs before it advanced
one layer each and produced an attribution that this contradicts. The general rule was
already in `HOW_I_GET_THIS_WRONG_2026-08-13.md` -- instrument instead of guessing -- and
this is the second time in two days it paid for itself, after
`probe_declaration_conflicts.py` listed all 79 conflicts in a single run.
