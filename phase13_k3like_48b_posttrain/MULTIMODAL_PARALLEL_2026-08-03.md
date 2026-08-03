# The multimodal flavor across the parallelism axes

`kimi_linear_k3mini_vl`, 3 steps, seed 42, deterministic, global batch 8.

    fsdp2          7.71632 7.63655 7.43903
    cp2            7.71686 7.65315 7.44388
    ep2 x fsdp2    7.71632 7.64275 7.44239
    fsdp2 x tp2    7.72128 7.63490 7.45163
    pp2            blocked, see below

## Two fixes the axes exposed

**TP addressed the wrong module.** `apply_tp_kimi_linear` applies its top-level
plan by name -- `embed_tokens`, `norm`, `lm_head` -- but the multimodal wrapper
keeps the text model at `.language_model`, so none of those names resolved and
the embedding stayed un-sharded. It surfaced as `aten.embedding.default got
mixed torch.Tensor and DTensor`. The plan now descends to the text model; the
vision tower stays replicated, which is what MoonViT wants at this size (no head
axis to shard).

**`init_weights` assumed a whole model.** Under PP the splitter sets the pieces
a stage does not own to None, so `self.vision_tower.init_weights(...)` hit None
on every stage but the first. Both children are guarded now.

## PP is blocked, and not by something a guard fixes

With those two done, PP reaches `Optimizer param_groups pattern '.*' matched no
parameters` and, on the 4-GPU leg, a stage with no `language_model` at all. The
splitter divides `model.layers`, which for this model is a property forwarding
into `.language_model` -- so it hands some stages a wrapper whose text model is
gone while the wrapper's own forward still expects it.

Making PP work here means teaching the split about the wrapper: which stage owns
the tower, how the vision features cross a stage boundary, and what the
non-owning stages present as. That is design work on the PP adapter, which is
the one component with a clean per-parameter bill of health (0.00000 over 548
parameters), so it is not something to improvise against at speed.

Recorded as a real gap rather than worked around. Text-only PP is unaffected.
