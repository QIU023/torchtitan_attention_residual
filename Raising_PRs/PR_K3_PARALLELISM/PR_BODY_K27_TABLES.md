# PR title: [Kimi K2.5] The vision tables keep the tower's tensor-parallel declaration under type checking

Branch `k27_vision_tables_tp` = `c0e1584df` (one commit on main `ac10ca48f`). The bottom commit of the K3 TP/SP stack (PR 4499); file first, 4499 states the dependency.

--- PASTE BEGIN ---

### Summary

`_compute_learned_pos_embeds` and `_compute_2d_rope_cache` in `kimi_k2_7/vision_encoder.py` gather the per-image position tables into the packed stream and, under `spmd_types` type checking, retype the result. The retype named a tp destination (`{"dp": V, "tp": I}`) as well as the dp one; with tensor parallelism on, the checker reads the table's source type from the declaration the tower's parameters carry on the tp axis and refuses the mismatch (`mutate_type: expected current type R on axis mesh_tp, got I`). The retype now names the dp axis only: per rank on dp, as the grids are, while on tp the table's own declaration holds. Both helpers are shared with Kimi K3, whose tensor-parallel PR needs this; K2.5 itself hits the same check at tp > 1 with type checking on.

The change runs only under `spmd_types` with type checking, so it moves no number.

### Evidence

Kimi K2.5 debug model, dp2 x tp2, `spmd_types` with `--debug.spmd-typechecking`, AdamW (the DistMuon container refuses TP), `activation-checkpoint:none` (type checking rejects selective AC with flex attention), 3 steps, 8 x RTX 5060 Ti:

- main `ac10ca48f`: every rank fails at the tables with `SpmdTypeError: mutate_type: expected current type PerMeshAxisLocalSpmdType.R on axis mesh_tp, got PerMeshAxisLocalSpmdType.I`.
- with this commit: past the tables; the run then fails in `kimi_k2_7/qk_clip.py` with `ValueError: QK clip scales do not match the MLA weight shape` (the qk-clip hook against a tp-sharded MLA weight), a tensor-parallel problem of K2.5's own that no CI cell exercises and this PR does not touch.

Pinned pyrefly (0.45.1) clean on the file.

### Changed files

    torchtitan/models/kimi_k2_7/vision_encoder.py   +3/-6

--- PASTE END ---
