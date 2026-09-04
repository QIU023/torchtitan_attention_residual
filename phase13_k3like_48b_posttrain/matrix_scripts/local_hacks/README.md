# Local hacks for the run worktrees (never committed to a torchtitan branch)

Kept here so a host restart (the worktrees live under /tmp) does not lose them.

- `sm120_hack_main.patch`: lifts Attention Gym's SM100/SM103 guard in `kimi_k3/kda.py` on main-based trees (this box is an RTX 5060 Ti, SM120); `sm120_hack_cprev.patch` is the same for the attn-gym-base trees.
- `grad_dump_hack.py` / `grad_dump_hack_full.py` / `grad_hash_hack.py`: patch `trainer.py` to dump per-parameter gradient norms (local, or cp-reduced with `full_tensor()`) and sha1 before `clip_grad_norm_` when `GRAD_DUMP` is set; compare with `../cmp_grad_dumps.py`.
- `registry_aliases_cp5.py` / `registry_alias_pp_naive.py`: appended to `kimi_k3/config_registry.py` in a run worktree so recipe flavors (and the upstream generic kernels, and the whole-stack transport) are reachable through `--config`.
- `vision_cp_axis_fix.py`, `vision_norm_cp_axis_fix.py`, `resolve_sp_cp_splice.py`: the edit and conflict-resolution scripts of the 2026-09-04 rebase, for the record.

Recreate a run worktree: `git worktree add --detach /tmp/wt_X <branch>`, `git apply sm120_hack_main.patch`, append the alias file to `config_registry.py`.
