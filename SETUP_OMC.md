# Oh-My-ClaudeCode (OMC) integration

This repo opts into the [oh-my-claudecode](https://github.com/Yeachan-Heo/oh-my-claudecode)
plugin to make the multi-phase, multi-GPU experiment workflow easier to drive
from inside Claude Code. OMC is **not required** to use the repo — every
phase has plain shell scripts under `phaseN/` — but the long-horizon
orchestration tasks (overnight pretraining, alignment matrices, NCCL trace
tiers, RFC iteration) benefit from OMC's `team` / `autopilot` / `ralph`
modes and from project-local skills that capture the recurring recipes.

## Why OMC for this repo

- **Long horizons** — phase 6 v9/v10 runs span thousands of steps; OMC's `ralph`
  mode keeps verifying-then-fixing until the goal is met instead of stopping at
  the first failure (matches the `run_v8_crash_resilient_pretrain.sh` pattern
  we already use).
- **Many parallel knobs** — the 8-GPU alignment matrix runs FSDP × PP × TP × EP
  combos sequentially today; OMC's `ultrawork` / `team N:executor` parallelizes
  the variant launches and reconciles the alignment reports.
- **Recipe reuse** — workflows like "bump torchtitan submodule pointer", "kick
  off a tier_a NCCL trace", "regenerate the alignment CSV" are encoded as
  project-local skills in `.omc/skills/` and version-controlled with the repo.
- **No infra cost** — OMC is a Claude Code plugin; nothing new to deploy. It
  modifies `~/.claude/` (already gitignored here) and ships project skills
  under `.omc/skills/` (committed).

Risks worth naming: third-party plugin (32k stars but external code), 32 bundled
agents that can overlap with the built-in Claude Code subagents, and `/omc-setup`
sets `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in the user-global
`~/.claude/settings.json`. None of that touches the training code or the
torchtitan submodule.

## Prereqs

Already present on the dev box used for this repo:

- Claude Code CLI ≥ 2.1.126 (`claude --version`)
- Node ≥ v20 (we have v24.14.1 at `/opt/nvm/.../node`)
- `tmux` (for `omc wait` and `/omc-teams` workers — optional)

You must have a Claude Max/Pro subscription or `ANTHROPIC_API_KEY` set; the
plugin does not change auth.

## Install (run inside a Claude Code session)

These three commands run in the Claude Code REPL, **not** in bash. They modify
`~/.claude/` globally, so they're a per-developer one-time setup, not part of
`phase2_attnres_baseline_loss/setup_env.sh`:

```text
/plugin marketplace add https://github.com/Yeachan-Heo/oh-my-claudecode
/plugin install oh-my-claudecode
/omc-setup
```

After install, restart Claude Code once so the new agents/skills register.
Verify with:

```text
/omc-help
/skill list
```

You should see the project-local skills from `.omc/skills/` (see below) listed
alongside the framework skills.

Optional cross-model workers (only if you want `/omc-teams` or `/ccg`):

```bash
npm install -g @google/gemini-cli
npm install -g @openai/codex
```

## Project layout

| Path | What it is | Tracked? |
| --- | --- | --- |
| `.omc/skills/` | Repo-specific recipes (alignment matrix, submodule bump, NCCL trace) | ✅ committed |
| `.omc/RELEASE_RULE.md` | Cached release/checklist rule (optional, populated by `/release`) | ✅ committed if produced |
| `.omc/plans/` | Per-session intermediate plans handed between skills | ❌ gitignored (runtime) |
| `.omc/state/`, `.omc/cache/` | Any other runtime scratch OMC writes | ❌ gitignored |
| `~/.claude/` | OMC plugin install, settings, skill DB, sessions | ❌ already gitignored |

The split mirrors how OMC documents project- vs user-scope: project skills win
on conflict and travel with the repo; user-scope skills (`~/.omc/skills/`) and
plugin state live outside the worktree.

## Recommended modes per workflow

| Workflow (what it maps to in this repo) | Mode | Why |
| --- | --- | --- |
| Single bug fix in `torchtitan/experiments/attn_res/` | default Claude Code (no mode) | OMC overhead not worth it for one-shot edits |
| Iterate on RFC drafts (`RFC_DRAFT_v3.md`) | `team 2:executor` | Plan → draft → critic loop, two agents |
| Launch + babysit phase 6 v10 pretraining | `ralph: ...` | Persistent verify-then-fix until step target hit; tolerates NCCL hangs |
| Run the 8-GPU alignment matrix end-to-end | `autopilot:` + `/oh-my-claudecode:run-8gpu-alignment` | 4–5 variant launches, sequential by GPU contention |
| Collect tier_a/b/c NCCL traces and rebuild `pattern_catalog.md` | `/oh-my-claudecode:nccl-trace-tier` | Wraps `phase7_nccl_traffic_catalog/run_tier_b_a_traces.sh` + `extract_collectives.py` |
| Bump `torchtitan/` submodule pointer after a fork merge | `/oh-my-claudecode:bump-torchtitan-submodule` | Encodes the "submodule is a gitlink, not a worktree" rule from `.gitignore` |
| Long ambiguous research request ("port AttnRes to Kimi K2") | `deep-interview` then `ralplan` | Forces the spec out before any code lands |

The defaults (Sonnet for routine, Opus for hard reasoning) are fine; do not
override the OMC model router unless a phase specifically needs it.

## Project-local skills shipped here

Three seed skills live under `.omc/skills/` and are invokable as soon as OMC is
installed. They are intentionally thin wrappers around the existing
`phase{6,7}/*.sh` scripts so the shell entry points stay the source of truth.

- `run-8gpu-alignment` — drive `phase6_upstream_pr_prep/run_8gpu_alignment_matrix.sh`, parse the
  per-variant `alignment_8gpu_*.txt` reports, summarize PASS/FAIL per cell.
- `bump-torchtitan-submodule` — fetch fork, choose target SHA, update gitlink,
  smoke `pytest tests/unit_tests/test_attn_res.py`, write the commit message in
  the style of `753026c phase 6: bump submodule pointer to 84d42c9`.
- `nccl-trace-tier` — run `phase7_nccl_traffic_catalog/run_tier_b_a_traces.sh` for a chosen tier,
  call `extract_collectives.py`, regenerate `phase7_nccl_traffic_catalog/pattern_catalog.md`.

Add new skills with `/skill add <name>` (interactive) or by dropping a
`.omc/skills/<name>/SKILL.md` following the frontmatter format documented in
the OMC repo (`skills/AGENTS.md`).

## Maintenance

- **Update plugin:** `/plugin update oh-my-claudecode` then restart Claude Code.
- **Disable temporarily:** `/plugin disable oh-my-claudecode` (per-session); the
  `.omc/skills/` files stay in the repo and have no effect when disabled.
- **Uninstall fully:** `/plugin uninstall oh-my-claudecode`. Leaving
  `.omc/skills/` in the repo is harmless for collaborators who never install
  the plugin — they're just markdown files.
- **Diagnostics:** `/oh-my-claudecode:omc-doctor` reports plugin/version
  mismatches and missing optional CLIs.

## Not changed by this integration

- `phase2_attnres_baseline_loss/setup_env.sh` and the conda env (`attnres`) — OMC has nothing to do
  with the training stack.
- `torchtitan/` submodule pointer — OMC does not touch git state on its own;
  the `bump-torchtitan-submodule` skill is opt-in and human-driven.
- `.claude/settings.local.json` — kept as-is (only the `nvidia-smi` allow rule).
  Any global tweaks `/omc-setup` makes go to `~/.claude/settings.json`, which
  is per-developer and outside the repo.
