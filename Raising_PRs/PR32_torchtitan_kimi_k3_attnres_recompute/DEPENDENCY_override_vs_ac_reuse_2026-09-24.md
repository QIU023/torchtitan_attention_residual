# PR 4780: the override (the compile part) and AC reuse (#4656), re-evaluated

2026-09-24, for tianyu-l's review r4090059193 on #4780. The user's instruction: leave the compile part untouched and re-evaluate how it depends on AC reuse. Everything below was measured or read from source on this box; timing is left to the H100 kit.

## Verdict

- The torch_remat checkpoint (#4656) and a fused or compiled override are complementary, not alternatives. The checkpoint removes what the eager aggregation keeps from forward to backward when no AC policy covers the block. An override lowers the transient peak inside each call, which is what still matters under the flavor's default selective AC, but only at real widths: at the released width it is 3052 against 728 MiB per call, while at the debug width (dim 1024, three stack entries at most) the FP32 copies are a few MiB and the training peak does not move (5060 smoke below).
- In code, both depend only on the Configurable commit and not on each other. The review branch orders them Configurable, torch_remat, override. The override commit is byte-identical to the one pushed before this round and can be dropped on its own (tianyu: "can leave to @acisseJZhong").
- At run time the call-site checkpoint wraps whichever implementation the Config node builds. For a torch.compile'd aggregation that is the same trade as for eager. For fla's fused op as shipped it is pure overhead: fla keeps its contiguous source copies on `ctx.res`, outside `save_for_backward`, so no saved-tensor hook can release them, and the checkpoint only adds a fused forward in backward.
- So the recompute decision belongs to the model, at the call site, where the AC coverage is known. An override owns only its math. The fla wrinkle is fla's to fix (rebuild the pointer table from the saved tensors in backward); with that change, fused under the checkpoint keeps nothing.
- #4656 is superseded by the review branch's torch_remat commit: same mechanism and same tests, ported onto the Configurable and into `KimiK3Model.parallelize` after #4810. Whether to close #4656 or keep it as the carrier is the user's call (options at the end).

## What each form keeps for backward (one call, measured)

`kit_h100_2026-09-24/probe_saved_bytes.py` on one RTX 5060 Ti, `attnres_review1` (torch 2.14 nightly, torch_remat 0.2.0, fla 0.6.0 from git main), bf16 inputs. "Held" is memory allocated after forward minus before, minus the output: what the call's autograd graph keeps until backward. "Peak" is the max allocated over forward and backward, above the inputs. Bytes are a property of the shapes, not of the GPU; timing is not measured here.

T=2048 tokens, 8 stack entries, D=7168 (bf16 stack 224 MiB; one FP32 copy of stack plus prefix 504 MiB):

| implementation | torch_remat checkpoint | held MiB | peak MiB | max rel err vs eager |
|---|---|---:|---:|---:|
| eager (main's reference form) | no | 1008.2 | 3052.0 | 0 |
| eager | yes | 0.0 | 3052.0 | 0 (bitwise) |
| fla fused, as shipped | no | 224.1 | 728.0 | 1.7e-3 |
| fla fused, as shipped | yes | 224.0 | 784.2 | 1.7e-3 |
| fla fused, `ctx.res` rebuilt in backward | no | 224.1 | 560.2 | 1.6e-3 |
| fla fused, `ctx.res` rebuilt in backward | yes | 0.0 | 560.2 | 1.6e-3 |
| torch.compile(eager) | no | 504.1 | 1296.1 | 1.7e-3 |
| torch.compile(eager) | yes | 0.0 | 1296.1 | 1.7e-3 |

T=8192, 4 entries, D=1024 gives the same pattern (eager 320.3 held / 976.0 peak, 0 with the checkpoint; fla 64.3 held with or without it; compile 160.3 held, 0 with it). The relative error is the largest `max|diff| / max|eager|` over the output and the four input gradients, one bf16 rounding.

Reading the table:

- Eager keeps two FP32 copies of the stack (1008 MiB here). The checkpoint drops both and rebuilds them in backward, so the retained memory goes to zero and the in-call peak is unchanged.
- fla keeps one bf16 copy of the stack. The override passes `block_residual_TND.unbind(1)`, strided views, and fla's wrapper makes each source contiguous (`r.reshape(-1, D).contiguous()`), so the copies are new tensors. `_build_ptr_table` returns those tensors themselves and `FusedAttnresFunction.forward` stores them as `ctx.res`, a plain attribute, besides `save_for_backward`. The checkpoint drops the saved-tensor references but `ctx.res` keeps the storage, so held stays 224 MiB. Rebuilding the table from `ctx.saved_tensors` in backward (the probe's `--fla-res-from-saved`, about ten lines on fla's side) is enough for the checkpoint to release them.
- torch.compile's partitioner keeps one FP32 stack-sized tensor instead of two; the checkpoint releases it like eager's.
- The in-call peak is what the fused and compiled forms improve: 3052 MiB eager against 728 fused and 1296 compiled. Under selective, full or region AC this peak is what a block's recompute reaches, and the checkpoint does nothing there (the block's own checkpoint already covers it).

## Whole-model smoke and same-cache identity (5060, not for the PR)

`kimi_k3_debugmodel` (stock: DistMuon, multimodal cc12m-test), dp1 on one RTX 5060 Ti, `/workspace/venv_bfx9` (torch 2.15.0.dev20260906), KDA guard lifted for SM120 in both worktrees (uncommitted), 10 steps, 2048 tokens per step in 512-token micro-batches, `--debug.seed 42 --debug.deterministic`. Every cell ran on its own copy of one warm cache (inductor and Triton), filled by main with AC off (10 steps) and one fused step. Trees: main `9e159aed7`; head `attnres_review1` `4e4baa4f3`. Scripts: `scratchpad/smoke4780/{cell,wave}.sh`, table by `compare.py` (now the kit's `tables_attnres.py`). Memory is the trainer's max reserved; tps is not quoted from this box.

| cell | AC | overrides | step 1 loss / grad norm | step 10 loss / grad norm | steps equal to main (loss and grad norm) | peak GiB |
|---|---|---:|---|---|---:|---:|
| main | none | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | reference | 13.22 |
| main again (noise floor) | none | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | 10 / 10 | 13.22 |
| head | none | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | 10 / 10 | 12.79 |
| head, fla override | none | 49 | `12.62280` / `20.1250` | `7.30048` / `19.3750` | 0 / 10 | 12.87 |
| main | selective | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | reference | 11.32 |
| head | selective | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | 10 / 10 | 11.32 |
| head, fla override | selective | 49 | `12.62280` / `20.1250` | `7.30048` / `19.3750` | 0 / 10 | 11.28 |
| main | full | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | reference | 11.18 |
| head | full | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | 10 / 10 | 11.18 |
| head, fla override | full | 49 | `12.62280` / `20.1250` | `7.30048` / `19.3750` | 0 / 10 | 11.28 |
| main | region (`save_regions` `feed_forward.w13`) | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | reference | 11.18 |
| head | region | 0 | `12.62200` / `20.3750` | `7.43741` / `22.7500` | 10 / 10 | 11.18 |
| head, fla override | region | 49 | `12.62280` / `20.1250` | `7.30048` / `19.3750` | 0 / 10 | 11.28 |

- The Configurable and the checkpoint are bitwise neutral in every AC mode, 10 of 10 steps, loss and grad norm.
- The checkpoint path executed: only the AC-off row moves memory (13.22 to 12.79 GiB); under selective, full and region AC the peak is main's to the hundredth.
- Region AC ran on the head, so `KimiK3Model.parallelize` cleared the flag before RegionAC wrapped the blocks (a nested checkpoint would have raised at step 1).
- The fla override rows agree with each other across all four AC modes and differ from eager from step 1 (loss by 8e-4, grad norm by two bf16 steps). That gap is not located yet (router near-tie or the op's own rounding); it is held back from every draft until it is, per the locate-and-rerun rule. The compile part was left untouched tonight, so this stays open with it.

## Facts from source

- torch_remat 0.2.0 rejects a nested `remat.checkpoint` in eager (`_api.py`, `"nested torch_remat.checkpoint regions are not supported"`) and under compile (`_compile.py`, `compiled_checkpoint`). RegionAC wraps each block's forward in `remat.checkpoint`, so the residual's own checkpoint must be off under RegionAC, whatever implementation runs. The review branch's CPU test `test_region_ac_runs_the_residual_as_a_plain_call` pins that case (the nested form raises; the plain call is bitwise equal to no AC).
- Under full and selective AC (PyTorch checkpoint wrappers) the nested form runs, bitwise, but costs one more pass. Measured on the CPU two-block harness from the tests (three aggregations per forward): 9 aggregation calls with the residual's own checkpoint left on, 6 with it off, bitwise equal to no AC in all four runs; RegionAC with it on raises, with it off gives 6 calls, bitwise.
- Under `torch.compile`, `remat.checkpoint` runs the region under non-reentrant `torch.utils.checkpoint` and the partitioner recomputes it wholesale (`_compile.py` docstring). Main's `KimiK3Model.parallelize` refuses model compile ("Kimi K3 does not support model compilation yet."), so today a torch.compile'd aggregation can only enter as an override of this node, and it composes with the call-site checkpoint exactly like eager (table above).
- `torchtitan/overrides/README.md` ("Custom kernels and torch.compile"): a kernel override must be a registered custom op (`torch.library.triton_op` or `custom_op`, `register_fake`, `register_autograd`) to compose with compile. The fla override calls the autograd Function directly and covers SPMD type checking with `spmd.register_decomposition`. That is enough while K3 refuses compile; it is the compile work the override still owes once K3 compiles, and it is independent of #4656.
- MemoryBudgetAC needs model compile, so it cannot run on K3 today; the residual checkpoint stays on under it by construction (only full, selective and region AC mark the blocks).
- Environment note, not a code finding: `/venv/main`'s fla-core 0.5.1 fails to import (`AttributeError` from tilelang 0.1.9 against apache-tvm-ffi 0.1.14.post0, installed 09-19). The override's optional-import sentinel catches only `ImportError` (helion_rope's pattern), so in that environment the override module itself fails to import. Tests and smokes here use fla 0.6.0 from git main on `PYTHONPATH` (`scratchpad/pr4780/fla_main`).

## Dependency, commit by commit (review branch `attnres_review1`)

1. `kimi_k3: the attention residual aggregation is a Configurable`: no dependency. Pure move of main's eager form into `AttentionResidual.__call__`; one config node per residual site.
2. `kimi_k3: recompute the attention residual in backward with torch_remat`: needs 1 (it passes the built Function to `remat.checkpoint`). The block keeps `checkpoint_residual`; `KimiK3Model.parallelize` clears it for selective, full and region AC before building the policy. The model's output aggregation always takes its own checkpoint (no block wraps it). Six CPU tests, four ported from #4656 plus the RegionAC composition and the parallelize marking; both controls fail as expected (without the port five tests fail, without the marking the parallelize test fails).
3. `overrides: Kimi's attention residual fused with flash-linear-attention`: needs 1 only; untouched this round (patch byte-identical to the pre-rebase commit). With 2 below it, AC-off runs with the override pay one fused forward per call in backward and save nothing until fla stops holding `ctx.res`.

Open items for whoever owns the override (acisseJZhong, per tianyu), none done tonight:

- fla: rebuild the pointer table from the saved tensors in backward. Then the override under the model's checkpoint keeps nothing, and the combination is strictly better than either alone.
- Until then, if the AC-off plus override combination matters, a small opt-out (a class attribute on `AttentionResidual` that the fused subclass clears, read at the call site). This touches both files, so it waits for the override's owner.
- A stride-aware fla kernel would also remove the per-call source copies (224 MiB here) without any checkpoint.
- `torch.library.triton_op` registration before K3 turns on model compile.

## Options for the PR structure (the user decides)

- A. 4780 carries commits 1 and 2 (and 3 if kept); #4656 is closed as superseded. This answers tianyu's second bullet in the thread where he asked it, and the three call sites and `parallelize` change once. The H100 kit has to re-measure the AC-off memory and tps rows that #4656 measured on H200 at `a3a819c67`, a pre-#4810 tree.
- B. 4780 carries commit 1 (and 3); #4656 is rebased onto it and carries commit 2 unchanged. Smaller PRs, but #4656 cannot land before 4780, and the answer to tianyu's bullet becomes "see #4656".

Either way commit 2 is the same code. Recommendation: A.
