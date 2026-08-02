# Converting to upstream LoRAConverter is blocked on a prerequisite refactor

## Why converting is the right direction

`torchtitan/components/lora.py` builds LoRA by SUBCLASSING the base linear:

    class LoRALinear(parent_cls):
        self.lora_a = Linear.Config(..., sharding_config=lora_a_sharding).build()
        self.lora_b = Linear.Config(..., sharding_config=lora_b_sharding).build()

    def forward(self, input):
        base_out = super().forward(input)
        lora_out = self.lora_b(self.lora_a(input))
        return base_out + self._lora_scaling * lora_out

Both adapters are real `Linear` modules carrying their own `sharding_config`, and
the forward contains no `to_local`, no `from_local` and no hand-written grad
placement. `_lora_adapter_sharding` derives the adapter placements from the base
linear's -- the same rule our code implements by hand.

Our `KimiLoRALinear` wraps instead of subclassing: it branches on `x_is_dt`,
unwraps both adapters, and re-wraps at the end. That is what produced the defect
in LORA_TP_DEFECT -- o_proj receives a plain x, both adapters get unwrapped to
local shards, and their product is a partial sum nothing reduces. A subclassing
implementation cannot reach that state because the adapters never leave DTensor.
So this would remove a class of bug, not relocate one.

## The blocker

`LoRAConverter` starts from `_lora_adapter_sharding(config.sharding_config)`.
K3's linears do not have one: `sharding_config` appears twice in model.py, both
for the MoE via `set_moe_sharding_config`. Attention and FFN linears are built as
bare `Linear.Config(...)` and get their TP from the imperative plan in
parallelize.py. So the converter would derive `None, None` for every adapter and
place none of them.

Converting therefore requires first moving K3's TP from the imperative plan
(~1000 lines in parallelize.py) to declarative per-linear `sharding_config`. That
is a substantial refactor of the file whose correctness this whole investigation
has been establishing, and every parallelism result would need re-taking against
it.

## What this changes about the plan

Not attempted tonight, and not because it is hard -- because doing it in the same
overnight run as the smoke matrix would invalidate the matrix. The order that
makes sense:

1. Measure LoRA across the parallelism combinations as it stands. That is worth
   doing even though the implementation is slated for replacement: it establishes
   which combinations are affected, and the replacement has to reproduce those
   numbers or better.
2. Do the sharding_config refactor as its own change, gated on the full-parameter
   parallelism numbers staying put.
3. Then convert LoRA, and require it to fix the o_proj defect rather than merely
   move it.
