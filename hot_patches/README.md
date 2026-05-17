# hot_patches/

Live patches we apply to system-installed packages while waiting for upstream
fixes to land. Each subdir is a self-contained patch kit with:

- `original_*.py` — exact bit-identical copy of the file BEFORE patching.
  md5 should match the installed file at the moment of backup.
- `patched_*.py` — the patched version actually deployed.
- `apply.sh` — copies `patched_*.py` over the live file (idempotent).
- `rollback.sh` — restores `original_*.py` over the live file (idempotent).
- `README.md` — what, why, when, related upstream PRs.

## Rules

1. **Backup before any edit.** `original_*.py` md5 must match the live file
   at backup time, or the patch is not applied against the expected baseline.
2. **Document upstream provenance.** Each patch links to the open upstream PR
   or issue it is anticipating. When upstream lands, the live file gets bumped
   via pip and this patch becomes a no-op; delete or mark stale.
3. **Idempotent.** `apply.sh` should be safe to run twice; `rollback.sh` should
   restore exact bytes from the backup.
4. **Mark every changed block.** Patches embed a `PATCHED-YYYY-MM-DD (PR##
   hot_patches/)` comment so a reader of the live file can find what we changed
   and where to read the rationale.
5. **Rollback on failure.** If a patch causes new crashes 1-2 retries in a row,
   rollback first, then investigate.

## Current patches

| Dir | Target | Reason | Upstream | Status |
|---|---|---|---|---|
| [`fla_fused_norm_gate_sm120_kda/`](fla_fused_norm_gate_sm120_kda/) | fla 0.5.0 / `fla/modules/fused_norm_gate.py` | KDA `o_norm` device-side assert on Blackwell sm_120 + Triton 3.6.0 — every ~2500 SFT steps | [PR #796 mirror; our PR draft → Raising_PRs/PR13](../Raising_PRs/PR13_fla_fused_norm_gate_sm120_kda_crash/) | 🟢 **Applied 2026-05-17 20:02** |
