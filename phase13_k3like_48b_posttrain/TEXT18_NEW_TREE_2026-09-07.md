# The 18-cell text-arm matrix on the integration tree (2026-09-07)

Tree: `k3_int_20260906` = `6dbc6805b` (fork) plus three uncommitted run-worktree aliases in `/tmp/wt_text18`
(`local_hacks/text18_aliases_newtree.patch`): the cp2 aliases, `kimi_k3_debugmodel_text` and
`kimi_k3_debugmodel_text_cp2`. The flavor rule keeps text-only flavors off the branch, so the text arm is an
alias: `kimi_k3_debugmodel` with the text loader (`c4_test`, concat-then-split packing) and **no vision
tower** (`vision_encoder = None`), which is what the old tree's text flavor was. Protocol = the mm18 one
(`matrix_scripts/text18_matrix.sh`, derived from `mm18_matrix.sh`): 33 layers, spmd_types everywhere, 4096
tokens per step in 256-token micro-batches, 8 pipeline micro-batches, SP on under TP except with CP, seed
42, one seed checkpoint (`.mx3_seeds_text18b`), Attention Gym `b19162e`, 10 measured steps after a warm run.
The text stream is not the multimodal stream: these rows are not comparable to `MM18_NEW_TREE_2026-09-05.md`.

| cell | world | step 1 | step 3 | step 10 |
| --- | --- | --- | --- | --- |
| dp1 | 1 | 12.40875 | 7.85552 | 3.27523 |
| fsdp2 | 2 | 12.39817 | 7.33636 | 3.23535 |
| pp2 | 2 | 12.40875 | 7.82651 | 3.32024 |
| tp2 (SP on) | 2 | 12.40917 | 7.68093 | 3.29654 |
| ep2 x fsdp2 | 2 | 12.39701 | 7.53269 | 3.20408 |
| cp2 | 2 | 12.41009 | 7.75217 | 3.34138 |
| pp4 | 4 | 12.40875 | 7.82651 | 3.32201 |
| tp4 (SP on) | 4 | 12.40976 | 7.57858 | 3.24684 |
| cp4 | 4 | 12.41597 | 7.67653 | 3.31860 |
| pp8 | 8 | 12.40875 | 7.82651 | 3.32075 |
| ep8 x fsdp8 | 8 | 12.42136 | 7.50619 | 3.38366 |
| fsdp2 x tp2 x pp2 | 8 | 12.40796 | 7.72779 | 3.33930 |
| ep2 x fsdp2 x tp2 x pp2 | 8 | 12.41173 | 8.11731 | 3.33334 |
| fsdp2 x pp2 x cp2 | 8 | 12.40740 | 7.56151 | 3.36438 |
| ep2 x fsdp2 x pp2 x cp2 | 8 | 12.41034 | 7.67304 | 3.35541 |
| fsdp2 x tp2 x cp2 (SP off) | 8 | 12.42177 | 7.61877 | 3.30650 |
| tp2 x pp2 x cp2 (SP off) | 8 | 12.41960 | 8.11042 | 3.29420 |
| ep2 x fsdp2 x tp2 x cp2 (SP off) | 8 | 12.40549 | 7.51178 | 3.34342 |

18/18. Step 1 of pp2 / pp4 / pp8 is bitwise dp1 (12.40875) and their step 3 is bitwise each other (7.82651):
the pipeline changes nothing at one micro-batch per stage of this batch, as on the mm arm. dp2-class cells
read other samples (the loader shards documents by dp rank), TP and CP re-order reductions; all step-1
offsets are within the flavor's floor (`PP_STEP10_SPREAD_2026-09-04.md`). Runs: `/workspace/mx3_text18_*`
(`fsdp2_tp2_pp2` from `mx3_text18_c8fix_0907_085726`: its first attempt lost torchrun's random master port,
`EADDRINUSE`, and was rerun alone).

## The first pass, and why a text arm has no tower

The first pass (`/workspace/text18_pass1/`) used the debug model with its vision tower present but never run:
every PP cell died at the first backward with `AttributeError: 'FSDPParam' object has no attribute
'_unsharded_param'`, while the non-PP cells passed. Located with a `sitecustomize` guard that names the
parameter (`mx3_text18_pp2diag_0907_072916`): all 64 are `vision_encoder.*`. FSDP2's root final callback
(`_fsdp_state.py: _root_post_backward_final_callback`) runs `post_backward` for every parameter group not
already in POST_BACKWARD, including groups that never ran forward; under a pipeline schedule the non-last
micro-batches run with gradient reduction off, and that branch of `post_backward` upcasts every parameter's
gradient through `_unsharded_param` without the `hasattr` guard the reduce branch has. A tower that never
forwards has no `_unsharded_param`. The mm arm runs the tower every step; fsdp2 without PP takes the guarded
branch; only text x PP meets the pair. No parallelism code changed: the text alias drops the tower, as the
old tree's text flavor had none, and the same pp2 cell then passes (with the guard in place it also ran:
12.40875 / 7.82651, i.e. PP itself was never at fault). The missing guard is a torch defect to report
upstream; verl's torchtitan engine carries the same guard for its text GRPO cells under PP.

With this arm the new tree carries the three-arm 54/54: `MM18_NEW_TREE_2026-09-05.md` (18),
`MM18_LORA_2026-09-06.md` (18), this file (18) -- all on `k3_int_20260906`, none on the old tree.
