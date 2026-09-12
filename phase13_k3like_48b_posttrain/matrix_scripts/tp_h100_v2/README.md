# TP/SP #4499 on 2 x H100: the run kit

Branch: `QIU023:tp_sp_on_main` (five commits on upstream main `da2f82670`). The reference for
every table is tp=1 on that same main commit, checked out as a second worktree.

Two GPUs, so every cell runs to completion before the next one. `dp2 x tp2`,
`dp2 x ep2 x tp2` and any `tp=4` need four GPUs and are not in this kit; the tp=4 cells also
need a vision tower whose head count 4 divides (the debug tower has 6, the released one 12).

## Setup

```bash
git clone -b tp_sp_on_main https://github.com/QIU023/torchtitan.git tt && cd tt
git worktree add ../tt_parent da2f82670

python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install --force-reinstall --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch
pip install "spmd_types==0.2.5"
pip install "git+https://github.com/meta-pytorch/attention-gym@main"
pip install "git+https://github.com/meta-pytorch/remat@$(grep -o 'remat[^"]*' pyproject.toml | head -1 | sed 's/.*@//')"
python -c "import torch, spmd_types, attn_gym; print(torch.__version__, torch.cuda.get_device_capability())"
```

Two local patches are needed on a Hopper box. Neither is committed:

```bash
# 1. The KDA guard is main's and refuses SM 9.0. Widen it in BOTH trees.
python phase13_k3like_48b_posttrain/matrix_scripts/tp_h100/hacks/kda_capability_hack.py "$PWD"
python phase13_k3like_48b_posttrain/matrix_scripts/tp_h100/hacks/kda_capability_hack.py "$PWD/../tt_parent"

# 2. If the chosen nightly has no torch.cuda._annotate_cuda_graph_trace, make that import lazy
#    in torchtitan/distributed/cudagraph.py (it is only used by the profiling post-processor).
python - <<'PY'
import pathlib
for t in (".", "../tt_parent"):
    p = pathlib.Path(t) / "torchtitan/distributed/cudagraph.py"
    s = p.read_text()
    if "_annotate_cuda_graph_trace" in s:
        print(t, "check this import against your nightly")
PY
```

## Tests first (a few minutes)

```bash
pytest tests/unit_tests/cpu/test_integration_test_definitions.py
pytest tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py
```

## The matrix

```bash
cd /path/to/tt
export TT=$PWD TT_PARENT=$PWD/../tt_parent OUT=$PWD/tp_h100_out
bash phase13_k3like_48b_posttrain/matrix_scripts/tp_h100/run_bf16_100.sh 2>&1 | tee $OUT/run.log
```

Nine cells, 100 steps each, serialized on two GPUs: budget two to three hours. The three tables
are the last lines of the output; each cell's log is `$OUT/<cell>.log`.

## What the tables have to show

1. **`tp1` bitwise with `tp1_parent`.** This is the PR's acceptance bar: with tp=1 the branch must
   compute exactly what main computes, at every one of the 100 steps. The table prints `(bitwise)`
   when it does. If it does not, nothing else in the run matters -- report that first.
2. **`tp1_pd` bitwise with `tp1_parent_pd`.** Same, on the backend the branch does not touch.
3. **`tp1_again` is the noise floor**: the same cell as `tp1`, on a fresh inductor cache. Whatever
   it moves by at steps 10 and 20 is what this flavour does by itself; the tp=2 rows are read
   against that, not against zero.
4. **`tp2_sp` and `tp2_nosp` against `tp1_parent`**, steps 1 / 10 / 20. Step 1 is the one that
   carries weight; steps 10 and 20 are shown next to the floor row.
5. **`dp2_ep2` against `dp2`**, not against tp1 -- a second dp rank reads other samples, so those
   two are only comparable with each other.

Steps 50 and 100 are deliberately not in the table: by then this flavour's loss has fallen far
enough that a percentage divides two collapsing curves. The logs keep every step if they are
wanted.

## Notes for the write-up

- Report the torch build, the Attention Gym commit, spmd-types and CUDA versions with the tables;
  the two local patches above must be named as well, since a reader cannot reproduce the run
  without them.
- If a cell fails, keep its log and report the failure rather than the cell's absence.
