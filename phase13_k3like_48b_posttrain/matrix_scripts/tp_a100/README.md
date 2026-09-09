# TP/SP on the 4527 base: the A100 x 8 run kit

Branch: fork `tp_sp_on_4527` (4 commits on `shuhuayu:k3` = `d1e3979c7`, PR 4527, itself on main `53326e559`). The reference for every table is tp=1 on the parent commit (`d1e3979c7`, checked out as a second worktree).

## Environment

```
git clone -b tp_sp_on_4527 git@github.com:QIU023/torchtitan.git tt && cd tt
git worktree add ../tt_parent d1e3979c7
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install --force-reinstall --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch   # we ran 2.15.0.dev20260906 (cu130 on SM120); any nightly >= 20260906
pip install "spmd_types==0.2.5"
pip install "git+https://github.com/meta-pytorch/attention-gym@main"                                    # we ran b19162e; the KDA Triton path needs capability >= 8.0, which the branch's last commit accepts
pip install "git+https://github.com/meta-pytorch/remat@$(grep -o 'remat[^"]*' pyproject.toml | head -1 | sed 's/.*@//')"  # or whatever pyproject pins
python -c "import torch, spmd_types, attn_gym; print(torch.__version__)"
```

If `torchtitan/distributed/cudagraph.py` fails to import `torch.cuda._annotate_cuda_graph_trace` on the chosen nightly, main removed that import in 4493; rebase or make the import lazy locally.

## Tests first

```
pytest tests/unit_tests/cpu/test_kimi_k3_sp_splice.py tests/unit_tests/gpu/test_kda_attention.py tests/unit_tests/gpu/test_kimi_k3.py
```

(the splice test passes on the 5060 Ti box; the two GPU files need free GPUs there and are the ones 4527's own CI runs)

## What to run, in order

1. `bash run_bf16_100.sh` -- the table in 4500's format: seed 42, deterministic, 256 tokens per step, 100 steps, one seed checkpoint; parent tp=1, this branch tp=1 (both backends), tp=2 SP on/off (both backends), tp=4 SP on/off, then the dp2 stream: dp2, dp2 x tp2 (both backends), dp2 x ep2, dp2 x ep2 x tp2 (their reference is dp2: a second dp rank reads other samples). Prints both tables. ~1.5 h on 8 GPUs.
2. `bash run_fp32m_100.sh` -- the same table with float32 masters (`--training.dtype float32`, bf16 compute), the DSV3 table's regime, on the full 24-layer model (dp1 needs ~36 GB).
3. `bash run_fp32_loss.sh` -- true float32 (masters and compute, fp32 experts loop), 2 steps, loss and grad norm printed for dp1 and every tp2 cell: the forward-equivalence check on the full model.
4. `bash run_fp32_grads.sh` -- true float32 step-1 gradient dumps for dp1, the tp2 cells and the dp1 perturbation control (`KDA_PERTURB=3e-7`); prints the per-class relative-difference distribution. The tp2 cells should reproduce the control's distribution class by class (on the 5060 Ti box: median 1.2e-4, p90 2.0e-3, max 1.8e-2 vs control 1.6e-4 / 2.0e-3 / 1.5e-2).

Scripts 3 and 4 patch a throwaway copy of the tree with the hacks in `hacks/` (fp32 experts loop, dump-and-exit, the perturbation); never commit those.

Every script writes under `$OUT` (default `./tp_a100_out`); the tables are the last lines of each script's output.
