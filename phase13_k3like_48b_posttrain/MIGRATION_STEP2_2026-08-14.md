# Migration step 2: porting our flavors onto the upstream config tree

Step 1 (vendor their model unmodified, run it) was done on 2026-08-13. This is step 2, and
the first thing it turned up made half of it unnecessary.

## Their debugmodel IS our report_arch

Read side by side, `kimi_k3_up`'s `_debugmodel` and our `kimi_k3_debugmodel_report_arch`
are the same architecture in every free parameter:

| | theirs | ours |
|---|---|---|
| layers / dim / heads | 13 / 256 / 4 | 13 / 256 / 4 |
| full attention layers | `{4, 8, 12, 13}` | `[4, 8, 12, 13]` |
| q_lora / kv_lora | 128 / 64 | 128 / 64 |
| qk_nope / qk_rope / v | 32 / 16 / 32 | 32 / 16 / 32 |
| KDA head / conv | 32 / 4 | 32 / 4 |
| dense hidden | 1024 | 1024 |
| latent / expert hidden | 128 / 128 | 128 / 128 |
| experts / top-k / shared | 8 / 2 / 2 | 8 / 2 / 2 |
| vocab | 163840 | 163840 |
| vision | 4L, 256, qkv 384, hidden 1024, 3 heads | same |
| AttnRes block size | 12 | 12 |

Both were read off the same source -- the upstream K3 PR's debug model -- which is why they
agree. The one thing our `report_arch` changed from the twin was adding layer 13 to the
full-attention set, per report sec 2.1's trailing global-attention layer; their current head
has it too.

**Consequence**: the multimodal arm needs no flavor port at all. It needs a controlled
head-to-head of the two implementations on one architecture.

That head-to-head has now run, and the honest reading is that it is **not yet controlled**:

    ours:   12.05090 11.98433 11.82505 11.52380 11.03637 10.56118 10.23745 10.05560 9.99942 9.91659
    theirs: 12.45026 11.93980 10.50697  9.13622  7.38922  8.05780  7.57110  7.18937  6.31638 7.02810

Two things differ besides the implementation, and both can produce a gap this size:

* **Parameter init.** Their tree carries its own `param_init` per module -- `_LINEAR_INIT`,
  `_output_linear_init(dim)`, `_EMBEDDING_INIT` at normal std 1.0. Ours does not match it.
* **The loss wrapper.** Theirs is `ChunkedLossWrapper(CrossEntropyLoss)`; ours is bare
  `CrossEntropyLoss`.

Everything else -- dtype, seq_len, local batch, dataset, tokenizer, image budget, lr
schedule -- is the same, checked field by field.

So the numbers above measure "their config end to end" against "our config end to end", not
"their model" against "our model". A control arm is only a control for what you have
checked it shares. To make it one, either align init and the loss wrapper, or -- cleaner --
load one checkpoint into both and compare a single forward at step 0, which is the same
anchor trick the 48B graft uses.

## The text flavor, ported

`kimi_k3_mini_block_attn_res` is our text instrument -- 21 layers at dim 512, no vision
tower, deep enough that AttnRes commits more than one block. Expressed with their
`_kimi_k3_config` generator it builds on meta at 80.9M parameters with 15 KDA and 6 MLA
layers, matching ours exactly.

Their `KimiK3Model.Config` takes `vision_encoder: ... | None = None`, so text-only needed
no change on their side.

### What their generator cannot express

Dropped explicitly rather than silently defaulted:

| ours | theirs | matters for this flavor? |
|---|---|---|
| `latent_moe_use_norm` | always norms | no -- ours is True here |
| `moe_router_activation_func` | fixed | no -- ours is sigmoid, theirs too |
| `num_expert_group` | none | no -- ours is 1 |
| `attn_res_gated` | none | no -- off |
| `lora_rank` / `lora_quantize_*` / `mxfp4_qat` | none | no -- off; these are step 3 |
| `per_head_muon` | none | no -- off |

Every one of them is off in this flavor, which is why it is the flavor that ports first.
The LoRA and MXFP4 entries are step 3's work and their absence here is the accurate
statement of the gap, not a problem to solve now.

## `router_input_BLD` dissolves at step 3, and this is why

Their `KimiLatentMoE(Module)` does not subclass `common.MoE`. It builds its own router,
calls it on the full-width token, and separately projects to the latent for dispatch --
report Eq. 11's semantics, with **no change to `models/common/moe.py`**.

Ours reaches the same semantics by adding a `router_input_BLD` keyword to core. That is
three backward-compatible lines and no other model can observe it, so it stays for now, but
their approach is the more conservative one and step 3 adopts it. Recorded in
`CORE_CHANGES_AUDIT_2026-08-14.md`.

## Step 2's gate, and what it can and cannot cover

Queued: the text flavor on the upstream model, `dp1` and `fsdp2`.

Only those two cells are expressible. Their `parallelize.py` is 100 lines that call
`apply_fsdp_to_decoder` and `apply_fsdp_to_vision_encoder` and nothing else -- there is no
TP, CP or PP on their side yet. Naming the two runnable cells beats running the matrix and
reading eleven `NotImplementedError`s.

So step 2's gate is necessarily weaker than the three-arm matrix: it proves our topology
builds and trains on their tree, not that any parallelism survives. The parallelism is
step 3, and it re-attaches to their modules one axis per branch.

## Still ahead

3. **Re-attach the increments**, in dependency order: CP (attaches to attention modules, no
   interaction with the AttnRes carrier), then LoRA/MXFP4 (attaches to Linear leaves), then
   DEP, then PP.
4. **PP last**, adapter kept, its cache re-targeted at columns of `block_residual_TND`
   instead of elements of a Python list. The adapter is not redundant: `N` grows with depth,
   so the naive carrier costs ~0.88 GiB per microbatch on the last hop at 2.8T dims.
