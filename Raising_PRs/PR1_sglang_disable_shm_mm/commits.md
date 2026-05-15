# Backing commits — PR #1 `SGLANG_DISABLE_SHM_MM`

## Discovered in

**Phase 11** — VLM SGLang Engine boot under Monarch actor mesh
(`phase11/rlhf/run_grpo_*.py`). First reproduced when the GRPO actor
mesh provisioner unlinked `/psm_*` before the SGLang scheduler subprocess
opened it; `FileNotFoundError: '/psm_xxx'` hard-killed engine startup.

## Fork source

| Field | Value |
|---|---|
| Repo | `git@github.com:QIU023/sglang.git` |
| Branch | `main` (also on `attention_residual_inference`) |
| Commit | `74083ffae5520b12579bea48678758ca0afdffb2` |
| Author / date | QIU023 — 2026-05-10 |
| Title | `[VLM] SGLANG_DISABLE_SHM_MM env var to force CPU tensor transport` |
| Files touched | `python/sglang/srt/managers/tokenizer_manager.py` (single file) |
| Total diff | +9 / -1 lines |

Status: **clean, ready to cherry-pick as-is.**

## Cherry-pick recipe

```bash
# 1. Clone the upstream sglang and add our fork as a remote.
git clone https://github.com/sgl-project/sglang.git
cd sglang
git remote add qiu023 https://github.com/QIU023/sglang.git
git fetch qiu023

# 2. Branch off latest upstream main.
git checkout -b sglang-disable-shm-mm upstream/main

# 3. Cherry-pick the fork commit.
git cherry-pick 74083ffae5520b12579bea48678758ca0afdffb2

# 4. Resolve any conflicts (none expected; single-function change).
# 5. Push to your sglang fork on GitHub.
git push origin sglang-disable-shm-mm

# 6. Open the PR on github.com/sgl-project/sglang using PR.md as the body.
```

## Conflict surface

Low risk. The function `_determine_tensor_transport_mode` in
`tokenizer_manager.py` is small and rarely touched upstream. The diff
adds a single env check at the top of the function; merging is trivial
unless upstream has refactored this helper in the interim.

If upstream has moved the function to a different module, port the
9-line change by hand (the env-var name and semantics stay the same).

## Notes for the PR opener

- The commit message on our fork is fine to reuse verbatim, but feel
  free to expand the "why" paragraph using the PR.md body (which has
  more context about Monarch / Ray / SLURM lifecycle scenarios).
- Author email on the commit is the personal `yiqiaoqiu@hotmail.com`;
  re-sign with the address you want associated with the upstream PR
  if different.
