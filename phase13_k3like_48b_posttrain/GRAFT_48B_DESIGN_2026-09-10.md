# The Kimi-Linear-48B graft on the new tree: the decision it needs (2026-09-10)

The one piece of the "train from released weights" group (goal item 4) not ported today. The released checkpoint format, the two released-topology downscales, the quantized reader and the vision-preprocessing parity are on `k3_released_format` (`65badf428`); this note is about grafting a K3-shaped post-train stack onto the released **Kimi-Linear-48B-A3B** weights, which the old tree did with the `_gated` / `_gated_lora` flavors.

## What the old tree has (the port source)

Old tree = the submodule `torchtitan/` (branch `gb200_fixes`, `507c34fac`), folder `torchtitan/models/kimi_k3/`:

- one shape builder (`model_configs.py`) with an `is_k3` switch: K3 gets `q_lora_rank`, two shared experts, `routed_scaling_factor 1.0`, `mla_gated=True`, the latent MoE (Eq. 11, `latent_moe_use_norm`); Kimi-Linear gets no q_lora, one shared expert, `routed_scaling_factor 2.446`, `mla_gated=False`, a plain MoE, SiLU (the `activation_situ_*` betas are K3's);
- a second model name `kimi_linear` registered from the same folder, flavors `kimi_linear_<size>_<variant>` (e.g. `kimi_linear_436m_block_attn_res_n4`, `kimi_linear_447m_aligned_block_attn_res_n4`);
- the graft itself: `_GraftSuffix("_gated")` / `("_gated_lora", lora_rank=16)` flavor suffixes, `attn_res_gated=True` (alpha-gated attention-residual reads, so the grafted model starts as the base), `mla_gated` / `attn_gate_param="full_rank"`, and the text-only key spelling in `hf_key_map.py` / `state_dict_adapter.py`;
- footprint of the graft-specific lines: about 80 (attn_res_model 25, config_registry 19, state_dict_adapter 12, model 10, hf_key_map 7, `__init__` 4). The bulk of "Kimi-Linear support" is the `is_k3` generality of the modules, not the graft.

On the new tree (upstream main's `kimi_k3/`) the modules implement K3 only: gated MLA with q_lora, the latent MoE with two shared experts, K3's activation. Nothing there takes the Kimi-Linear shapes, so the graft is not an 80-line port; it is "where do the Kimi-Linear-shaped modules live".

## The two options

**A. The old tree's way: `is_k3` switches inside upstream's K3 folder.** Add the knobs (MLA gate off, no q_lora, plain MoE with one shared expert and a routed scale, SiLU) to `kimi_k3/model.py`, `moe.py`, the config dataclass and `model_configs`, register `kimi_linear` from the folder, port the graft suffixes, the gated reads and the text-only spelling. Exact port of validated code (the "never write a replacement" rule holds). Cost: upstream's K3 modules, reviewed and merged as K3-only, grow five non-K3 switches; every later K3 PR carries them.

**B. A `torchtitan/models/kimi_linear/` folder composed from what upstream already has.** Ungated MLA, the standard MoE with shared experts and a routed scale, SiLU: these are DeepSeek-V3 / Kimi K2.5 pieces in `models/common` and `kimi_k2_7`; the KDA layer and the attention-residual add-on are imported from `kimi_k3`. The graft flavors, the gated reads and the text-only spelling are ported into that folder. Matches upstream's one-folder-per-released-model layout and leaves K3 as K3. Cost: it is a composition, not a port (the "port from the old tree only" rule must be lifted for it), and it needs its own `parallelize`, `state_dict_adapter`, configs and tests (the K3 ones as templates).

Recommendation: **B**, because the reviewers merged K3 as K3-only and Kimi-Linear-48B is a released model of its own; A is the faster path if the graft is fork-internal and never meant for upstream.

## What verification can and cannot do here

- The released 48B weights are not on this box (the HF cache holds `moonshotai/Kimi-K3` only; the 48B is ~97 GB bf16) and the old graft runs were on H200s; a graft train cannot be verified on 8 x 16 GB.
- What can be verified here, either option: the Kimi-Linear-shaped debug flavors against the old tree at the same seed and shapes (bitwise at 3 steps, as the other ports), the key map round trip against a synthetic 48B-layout index (658-key style test), and the gated-read identity (alpha = 0 reproduces the base) as a CPU test.

## Decision needed

Which option, and whether the graft is meant for upstream at all. With B and an upstream target, the folder should be planned with the maintainers before code (same as the transport decision on #4312). Until then this item stays as recorded in `K3_INT_20260910.md`: not ported by design.

## Addendum, same day: ported under option A as a fork-internal branch

`k3_linear_graft` = `3b6f03347` (three commits on `ac10ca48f`, body `Raising_PRs/PR_K3_PARALLELISM/PR_BODY_LINEAR_GRAFT.md`): the Kimi-Linear shapes in the K3 modules (no q compression, ungated MLA, the plain MoE from core's router/experts with route scale 2.446 and one shared expert, the KDA output gate as `g_a/g_b`, the unbounded decay), `kimi_linear_48b` (49.1 B parameters, the released topology) and `kimi_linear_debugmodel`; the alpha-gated reads with the `_gated` suffix (`_gated_lora` stays with the LoRA branch); the text-only spelling with reads and alphas left out of the export so official checkpoints load and the alphas keep their zero init. K3's debug model stays bitwise with main; gated at alpha = 0 equals the plain backbone bitwise. Two limits found on the way: attn-gym's fused chunk kernels take per-token decays in roughly [-5.9, 0], so the unbounded Kimi-Linear decay runs on the eager reference path (exact, not a training kernel; the old tree used fla-core, absent here); and the old tree's flavors need fla-core, so no old-vs-new bitwise pair was run. Moving the code to a `kimi_linear` folder (option B) is a relocation of these three commits, not a rewrite.
