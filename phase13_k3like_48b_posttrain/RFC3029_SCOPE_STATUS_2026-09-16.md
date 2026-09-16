# RFC 3029 scope against the integration tree and veRL (2026-09-16)

Scope as the RFC states it (body updated 2026-08-26): Kimi K3 support "for every component needed in the torchtitan stack with 5D parallelism and QLoRA and veRL post-training integration"; slots A (TP), B (EP), C (PP with the Block AttnRes cross-stage cache), D (CP: Ulysses for MLA, merged KCP for KDA, dynamic CP for the vision encoder); LoRA, Muon, quantile balancing and MXFP4 QAT carried but "not proposed in this pass". Maintainer steer on the thread (shuhuayu, 08-23): land one by one, test composability gradually; TP non-critical; focus on dp, ep, pp, cp (dynamic CP for the vision encoder, linear-attention CP for KDA, CP for MLA).

Trees: integration `k3_on_4025` = `45ed0e3c1` (tag `k3_int_20260916c`, main `810e62786` + 72 commits; the morning's `8e0ae8381` = `k3_int_20260916b` was main `1949c297f` + 73, the extra commit the stacked #4577 copy that merged upstream); veRL fork branch `kimi_k3_integration_rebased` = `a3661ae3` (upstream verl `00cd5b44` + 46 commits). No verl upstream PR exists yet.

| scope item | upstream | integration tree | veRL branch | gap |
| --- | --- | --- | --- | --- |
| eager model: KDA, Gated MLA, latent MoE, Block AttnRes, MoonViT, 2.8T config | merged (#4025 and follow-ups; main carries `debugmodel` only) | main's model; flavors `Kimi-K3` (2.8T), `report_arch`, `k3mini`, `debugmodel_33_layers` ours | runs the `rl` flavor (released MLA dims, kv rank 512) | the 2.8T flavor and `report_arch` are fork-only |
| FSDP2 / HSDP | main | main | fsdp2 cell passes (09-16) | none |
| B: EP + grouped GEMM | merged earlier (core) | main's EP; MoonEP backend ours (`k3_moonep_seam`, 5 commits) | ep2 cell passes (09-15) | MoonEP unfiled: needs an NVSwitch Hopper box |
| A: TP / SP | merged 09-16 (#4499) | main's | tp2 cell passes after the gather-scaling fix (`a3661ae3`) | none |
| C: PP + AttnRes VPP cache | PR 4312 open, Tianyu's round of 09-11 unanswered by him since; 49 behind main, 5-file conflict | `de6f29514` as one change, plus fork-only: neighbor transport, cache offload, pp_balance, `has_backward` eval fix, empty-optimizer parts | pp2 cell passes (token budget, offload off) | the held fixes (`pp_review4_consume6840`) wait for his next round; rebase only when asked |
| D: CP text side (MLA Ulysses / all-gather KV, KDA KCP) | maintainers' PR 4639 (replaced their 4500; our 4313 still open, superseded) | 4639 `e06dcbee3` + our three adaptations (tower on the cp axis, block stack on cp, head splits) + the transform's tower exclusion | CP written for the pre-4639 module-internal Ulysses; refuses packed micro-batches; not ported to `preprocess_inputs` / `kda_cp_routing` | engine port to 4639's API; close 4313 or leave, user's call |
| D: dynamic CP for the vision encoder (report 5.2.3) | PR 4380 draft = `cpmm_review1` `774e0b9b5` on 4639 | on the tree since `68609b2e4` (evening of 09-16) | not applicable (engine is text-only) | none on the tree |
| DEP: tower on its own pipeline stage, bubble / prefetch | PR 4381 draft = `dep_review1` `384d576dc` on 4312 | yes; `pp8vp4_vit_dep` passes | PP under the engine keeps `vit_dep_stages=1` flavors (old rule) | none on the tree |
| composability (the RFC's 18-cell 3-of-4 matrix, seeded) | the b200 suite runs single-axis cells | the seeded `mx4.sh` matrix on `8e0ae8381` (`K3_INT_20260916.md`): every 3-of-4 cell incl. the three with CP, single-axis cells, cp x ep, cp x fsdp, ep8 x fsdp8; PP bitwise at step 1 in five pairings; CP with TP/PP needed 4639's refusal lifted and three fixes (`8e0ae8381`) | fsdp2, ep2, tp2, pp2, LoRA, QAT cells pass separately; no combined cells | compiled variants and the max-degree tp4 / cp4 cells not run |
| AttnRes companion optimisation (report 5.2.2): residual math checkpointed | PR 4656 repurposed = `ac_review3` `ea1316606` on main tip | yes (`a490441f8` + `checkpoint_residual`) | n/a | H100 table on the new head before review; the block stack is still re-`cat` per block (sentence 1 of the paragraph), a design decision |
| compile (blocks + tower, recompile bound) | not filed (`k3_compile_blocks`, body ready) | yes; `dp1_compile` passes | not used by the engine | file after 4312 or alongside |
| LoRA / QLoRA (NF4, packed MXFP4 bases, packed experts, adapter export, merge) | not filed (`lora_review1/2` stale); main moved to `LoRATransform` + handlers (#4684) | yes, still on `LoRAConverter`, coexisting with main's `LoRATransform` | LoRA merged sync passes (09-16); QLoRA packed-base sync not exercised in the engine | port to the handler seam before filing; a QLoRA GRPO cell |
| MXFP4-weight / MXFP8-activation QAT | not filed (`k3_mx_qat`) | yes (`mx_qat` passes) | fake-quant sync (`23477df8`), QAT cell passes | file |
| Muon (DistMuon per-head layouts) | JZ's #4596 competes; ours stays ours | yes (`muon_dp2` passes) | n/a | decide against #4596 |
| quantile balancing | maintainers' #4577 open (content unchanged since our stacking) | stacked as one change | n/a | none |
| MTP | not filed (`k3_mtp_layers`) | yes (`mtp` passes) | n/a (released config ships MTP off) | file |
| released checkpoint format (A_log layout, packed MXFP4 import, to_hf) | 4025's adapter upstream; released layout ours (`k3_released_format`) | yes | `to_hf` used by the sync; the debug export needed the `-rel` copy | file the released-layout fix |
| veRL engine: Kimi K3 end to end (folded stream, masks, packed offsets) | no verl PR | n/a | yes; text only ("multimodal not yet supported" in the engine) | multimodal GRPO; the CP port; then a verl upstream PR |
| 48B graft (Kimi-Linear-48B + AttnRes at alpha 0) | outside the RFC's text; our validation anchor | `k3_linear_graft` (3 commits), design decision open | n/a | decide |

Order proposed for the tree, from the table: (1) dynamic CP into the tree and the cut tower under cp x pp / cp x tp: done 09-16; (2) the seeded 3-of-4 composability matrix on this tree: done 09-16; (3) the engine's CP on 4639's API; (4) the LoRA port to the handler seam; (5) the AttnRes block-stack representation.

## Audit of `RFC_BODY_2026-09-16.md` against the evening's state (2026-09-16, night)

Rows checked one by one against GitHub (the user's open PRs as of 09:20 UTC: 4135, 4281, 4312, 4313, 4380, 4381, 4576, 4656, 4751) and the fork. Two rows were stale and are edited in the body file: MoonEP reads `#4751` instead of "not filed"; the 5D row names `k3_on_4025` = `k3_int_20260916c` (main `810e62786` + 72) as the current full tree instead of "pending". Everything else in the status tables matches. Rows that change as soon as the user acts, left as they are:

- "PP rank store offload, PP activation balance: drafts stacked on #4312" and deliverable 1's "filed once #4312's transport is decided": the review branches are ready (`pp_offload_review1` `d49bb388b`, `pp_balance_review1` `080f44208`, bodies in `Raising_PRs/PR_K3_PARALLELISM/`); once the user files `k3_pp_offload` / `k3_pp_balance` the row takes their numbers and the deliverable line reads "filed as drafts on #4312".
- "#4281: the pre-rebase tree": whether to refresh its branch (`k3_pr_classified_v2`) from `k3_int_20260916c` or close it in favour of the tree link is the user's call; a refresh is a force push to a published PR branch.
- Deliverable 2, "released layout round trip on the debug downscale and on the released index": the debug half is done (`report_arch`, 0 missing / 0 extra keys, bitwise export round trip); the released-index half needs the released checkpoint on a box with the disk for it. The line can be split into two boxes with the first checked when the body is next pasted.

What to push next, in the order that unblocks the most:

1. File the two PP drafts (user). Then paste the RFC body with the two rows updated.
2. LoRA on main's `LoRATransform` handler seam (#4576's deliverable): survey tonight, port next.
3. README deliverable (RFC section 5): the parallelism list of `models/kimi_k3/README.md`; patch and body prepared tonight, a one-file PR against main once the user names the branch.
4. The tree's numbers on `45ed0e3c1`: the seeded matrix rerun tonight (`mx4_int0916c_0916_094907`); the Elfie plan takes them.
5. H100 tables before any draft leaves DO NOT review: #4656, #4380, the two PP drafts (and MoonEP on an NVSwitch box, per `MOONEP_TEST_PLAN_2026-09-16.md`).
6. Compile: locate the compiled-vs-eager gap below module level (deliverable 4); a probe campaign on this box, one worktree, after the matrix.
7. veRL engine CP on 4639's API once 4639 stops moving; #4313 close or leave, user's call.
8. 4312: rebase only when the maintainer asks; the consume6840 fixes wait for his next round.

