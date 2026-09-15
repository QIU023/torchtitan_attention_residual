# TP/SP #4499 on 4 x H100: the run kit

Round 3 (2026-09-15). Branch `QIU023:tpsp_review4` = `1dec3ee17`: the four PR commits rebased onto upstream main `d34a13fdf`, plus the four round-3 commits (unified b200 cell, K2.5 comment, names and docstrings, `routed_down` on the token shard under EP). The reference for the dp1 and K2.5 tables is main `d34a13fdf`, checked out as a second worktree.

The rebase moved that reference: #4535 made the fused `w13` the default for K3's dense FFN, with its own init. On one RTX 5060 Ti, tp=1 step-1 loss went from `12.50616` (old base `56a721b64`) to `12.60343` (`d34a13fdf`), and the PR head matched the new main on 3 steps (`Raising_PRs/PR_K3_PARALLELISM/logs_tpsp_r3_2026-09-15/`). Every table in the current PR body is therefore stale and is replaced by this run. The 5060 numbers are smoke only and never go into the body.

`run_v3.sh` runs everything on four GPUs. The `run_v2*.sh` scripts belong to the pre-rebase head `22eeec412` and are kept for the record.

## Setup

```bash
export KIT=/path/to/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2
git clone -b tpsp_review4 https://github.com/QIU023/torchtitan.git tt && cd tt
git log --oneline -1                      # 1dec3ee17
git worktree add ../tt_parent d34a13fdf   # upstream main

python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install --force-reinstall --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch
pip install -e .                          # spmd_types==0.2.5 and torch_remat come pinned from pyproject
pip install "git+https://github.com/meta-pytorch/attention-gym@main"
python -c "import torch, spmd_types, attn_gym; print(torch.__version__, torch.cuda.get_device_capability())"
```

The previous body ran torch `2.15.0.dev20260906+cu130` and Attention Gym main `499404b`; use those, or record what was used.

One local patch, never committed: main's KDA guard admits only SM 10.0 / 10.3, so widen it in BOTH trees.

```bash
python $KIT/hacks/kda_capability_hack.py "$PWD"
python $KIT/hacks/kda_capability_hack.py "$PWD/../tt_parent"
```

(The round-2 lazy-import patch for `torch.cuda._annotate_cuda_graph_trace` is no longer needed: main does not import it any more.)

## Tests first (a few minutes)

```bash
pytest tests/unit_tests/cpu/test_integration_test_definitions.py tests/unit_tests/cpu/test_no_new_cli_options.py
pytest tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py
```

## The run

```bash
export TT=$PWD TT_PARENT=$PWD/../tt_parent OUT=$PWD/tp_h100_out && mkdir -p $OUT
bash $KIT/run_v3.sh 2>&1 | tee $OUT/run.log
```

Seventeen cells: five dp1 cells and five dp2 cells at 100 steps, two K2.5 cells at 100 steps, three type-checking smokes at 3 steps (plus two seed builds). Budget three to four hours. The tables and the smoke summary are the last lines of the output; each cell's log is `$OUT/<cell>.log`.

## What the tables have to show

1. **`tp1` bitwise with `tp1_parent` at every step.** The PR's acceptance bar: at tp=1 the branch computes exactly what main computes. If it does not, report that first; nothing else matters.
2. **`tp1_again` is the noise floor**: the same cell as `tp1` on a fresh inductor cache. The tp=2 rows at steps 10 and 20 are read against it, not against zero.
3. **`tp2_sp` and `tp2_nosp` against `tp1_parent`**, steps 1 / 10 / 20. Step 1 carries the weight.
4. **The dp2 rows against `dp2`**, never against tp1 (a second dp rank reads other samples). `dp2_ep2_tp2_nosp` is new: it is the path the `routed_down` change touches (EP without SP).
5. **`k27_dp2` bitwise with `k27_dp2_parent`**: the PR touches `kimi_k2_7`, and K2.5 refuses tp > 1 on main.
6. **Smokes**: `tc_mm` is the b200 cell exactly as CI runs it (fsdp 2 x tp 2 x ep 2, type checking on, AC off); `tc_mm_nosp` the same without SP; `tc_tp2` dp1 x tp2. Each must show 3 steps and no traceback.

Only steps 1 / 10 / 20 go into the body: past that the debug set is memorised and a percentage divides two collapsing curves. The logs keep every step.

## Notes for the write-up

- Report the torch build, the Attention Gym commit, spmd-types and CUDA versions with the tables, and name the KDA guard patch.
- If a cell fails, keep its log and report the failure rather than the cell's absence.

## run_pp_c4.sh (2026-09-12): the PP matrix on c4_test text-only rows

`pp4h_probe_c4.patch` on dbc425403 adds `kimi_k3_debugmodel_c4` (+ `_pp_naive`): c4_test through the multimodal
loader, one text-only row per doc, its first 256 tokens (a row must fit one 256-token micro-batch). 2000 rows,
~0.5M tokens; 100 steps read 21% (1024/step) or 43% (2048/step) once. Plumbing smoke on the 5060 (bfx9 venv):
dp1, pp2, pp2 x vp2 cached all run. Memorisation check, dp1 1024/step, 100 steps, 5060 (data behaviour only,
not numerics): loss 1:12.566 10:3.507 20:3.029 30:2.999 50:2.684 70:2.760 90:2.476 100:2.544, minimum 2.385 --
no collapse toward zero, unlike cc12m-test (0.19 at step 100). JIT/temp dirs must not sit under a top-level
`/tmp/triton_*` / `torchinductor_*` / `torchelastic_*` name while the disk watchdog is active.
