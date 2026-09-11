# PR 4499 (Kimi K3 TP/SP): diff audit of the branch against Shuhua's review and CLAUDE.md (2026-09-11)

Branch `k3_tp_sp` = `tp_sp_on_main` = `bd55160a8`: five commits on main `da2f82670`, 8 files, +432/-53. Checked against Shuhua's 18 inline comments (review CHANGES_REQUESTED 2026-09-11 00:40; no newer comment on the PR) and the CLAUDE.md rules (commit messages, diff audit, flavors, PR text, abstraction). Method: `git diff da2f82670 bd55160a8` read line by line; every docstring of `sharding.py` parsed and read in full.

## Passes

- Commit messages: no cross-repo reference, no URL, no trailer, no logbook path; all five commits authored and committed by QIU023.
- Added source lines: no logbook path, box name or measured number.
- Shuhua's 18, checked in the head's code: the KDA capability change is gone (#1); the `kda.py:266` comment is gone (#2); MLA and KDA split heads with core's `local_head_split`, `_local_head_split` is gone (#3-5); the `unsqueeze(1).expand(...)` suggestion and `flatten(-2)` are in, `h_local` is gone (#6, #7); one entry point `set_kimi_k3_sharding_config(enable_ep, enable_tp, enable_sp)` (#8, #13); `Decoder.Config.update_from_config` runs first (#9); no `_sp_group` (#10); the embedding stays TP-replicated with a tower and layer 0's boundary restores the layout, mirroring K2.5's `_shard_decoder_after_embedding_scatter` (#11); no `CP` alias (#12); the tower takes K2.5's `set_vision_transformer_block_sharding_config` and the invariant-tower helper is gone (#14); the K2.5 table retype stays, with the evidence in the reply (#15); `common/attention.py` and `common/linear.py` are untouched (#16, #17); no `test_kimi_k3_sp_splice.py` (#18).
- Flavors: one test recipe in `torchtitan_recipes/tests/b200.py` and its b200 cell; no model-registry flavor.
- Abstraction: multimodal input layouts from core's `multimodal_input_sharding()`; head splits through core's `local_head_split`; the tower through K2.5's block plan. K3 keeps its own `_shard_decoder_after_embedding_scatter` because its layer 0 also carries `block_residual_TND`, which K2.5's has no notion of.

## Findings to fix

1. **Docstrings carry the design argument (diff-audit rule).** `sharding.py` is 463 lines with 75 docstring and 32 comment lines, 23%; K2.5's `sharding.py`, the closest upstream peer, is 131 lines with 13, about 10%. The long ones explain why a declaration is what it is rather than what the function declares: `set_kimi_k3_sharding_config` (16 lines), `_set_kda_sharding` (10), `_stream_param_config` (8), `_set_vision_encoder_sharding` (8), `_tp_replicate_config` (6). Trim each to what it declares, in one or two lines; the reasons belong in the PR body.
2. **Code comments over two lines (PR-text rule).** `parallelize.py`, the partial_dtensor refusal (5 lines, the design argument; the error message already names the requirement); `model.py`, the `k_rope` region (3) and the scatter note (3); `sharding.py`, the MLA, KDA and MoE boundary comments (3 to 5 lines each).
3. **Diff noise.** `model.py` deletes one blank line after `h_TD = tokens` with no other change there.
4. **Reply draft (`REPLY_4499_2026-09-11.md`).** The top-level paragraph says four commits; there are five (`bd55160a8`, the dispatcher condition). The `model.py:162` reply says "Applied." while the code also wraps the expand in a `spmd.local()` region with a type assertion; say so in the reply.
5. **PR body (`PR_BODY_TP_SP_v2_DRAFT.md`) against the PR-text rule.** `##` headings instead of `###`; sections Stack / Implementation / Limitations / Tests instead of Summary / Design / Results / Changed files / CI/CD Coverage; Results at steps 1 / 10 / 20 instead of 1 / 3 / 10; no `torchrun` reproduction block before the main table; no Changed files block; one hard-wrapped paragraph.

Findings 1-3 are docstring, comment and whitespace edits, numerically neutral. None is applied yet; `k3_tp_sp` is unchanged by this audit.
