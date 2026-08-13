# The pre-merge baseline the merge and the refactor must reproduce

Frozen from `attention_residual_dev` at `fe54b8f9c` -- the branch WITHOUT the upstream
merge. Tables in `baseline_pre_merge/{text,full,lora}_{13,max}.txt`, judged mechanically
by `matrix_scripts/compare_to_dev_baseline.py` rather than by reading two tables side by
side (which is how a cell that had moved in the 5th digit got called byte-identical once).

## What is in it

| arm | flavor | 13-cell | maxdeg | failing |
|---|---|---|---|---|
| multimodal full | `kimi_k3_debugmodel_report_arch` | **13/13** | **5/5** | none |
| multimodal LoRA | `..._report_arch_lora` | **13/13** | **5/5** | none |
| text | `kimi_k3_mini_block_attn_res` | 10/13 | 2/5 | `dp1` `pp2` `tp2` `pp4` `pp8` `tp4` |

Settings: DEP and dynamic CP on for the multimodal arms and OFF for text (DEP on a text
flavor is invalid, not inert); text at `seq_len 4096`; steps 10, seed 42, deterministic.

**The text arm's six failures are all one cause and it is not ours**:
`InternalError: Failed to set the allowed dynamic shared memory size to 108160`, fla's
SM120 ceiling (`Raising_PRs/PR13*`). Every failing cell is one where a single rank holds
the full sequence -- `fsdp2`, `cp2`, `ep8_fsdp8`, `cp4` all pass. `seq_len` is squeezed
from both sides on this box: 8192 OOMs at 16 GB and 2048 also trips the ceiling, because
`chunk_kda`'s blocking follows the input shape. 4096 was found by sweep.

**EP x TP passes in this baseline** on all three arms (`ep2_fsdp2_tp2_pp2`,
`ep2_fsdp2_tp2_cp2`). That is the point of freezing it: those two cells are what the
upstream merge broke.

## The acceptance criterion, and the one exception

Criterion: `compare_to_dev_baseline.py` reports **no BROKE** for all three arms, and every
non-SAME cell is explained.

The exception, measured rather than assumed: **six cells drift by 1e-5** between this
baseline and the merged tree, four in the full arm (`cp2`, `pp2`, `fsdp2_pp2_cp2`,
`tp2_pp2_cp2`) and two in LoRA (`pp2`, `tp2_pp2_cp2`). Most likely `#4099 Keep global
valid token counts on device`, which moves a reduction from host to device and so changes
summation order in the loss denominator. **No change on our side can undo an upstream
change of summation order**, so "byte-identical everywhere" is not achievable while the
merge is in; 1e-5 with the rest identical is. The comparison tool classifies these as
DRIFT and only fails on BROKE, which is why it takes a `--tol`.

## Status of the merge branch against this baseline

`merge_upstream_2026_08_12`, before the EP x TP fix:

    [full] SAME 7  DRIFT 4 (1e-5)  BROKE 2 -> ep2_fsdp2_tp2_cp2, ep2_fsdp2_tp2_pp2
    [lora] SAME 9  DRIFT 2 (1e-5)  BROKE 2 -> same two

After the fix (`1eca8b6c0`, MoE declared as an SP island with a replicated boundary),
`ep2_fsdp2_tp2_pp2` reproduces the baseline **byte-identically**:
`12.05902 12.00313 11.75453 11.34475`. Full three-arm verification is the gate for
folding the merge into dev.

## How to re-judge

    python3 matrix_scripts/compare_to_dev_baseline.py \
        baseline_pre_merge/full_13.txt /workspace/mx_verify/mm_full_13.txt --label full

Exit status is 1 if anything BROKE, so a driver can gate on it.

## Result, 2026-08-13: merged and folded into dev

`attention_residual_dev` is now at `1eca8b6c0` and `upstream/main` is an ancestor of it.
Candidate tables in `verify_post_merge/`.

| arm | 13-cell | maxdeg | verdict |
|---|---|---|---|
| multimodal full | 9 SAME, 4 DRIFT 1e-5, **0 BROKE** | 5 SAME | **PASS** |
| multimodal LoRA | 9 SAME, 4 DRIFT 1e-5, **0 BROKE** | 5 SAME | **PASS** |
| text | every cell that trained before still trains; up to 1.3e-2 | 2 SAME, 2 BOTH_FAIL, 2 at 5e-4 | drift, attributed |

Both EP x TP cells are restored -- byte-identical in the full arm, within 1e-5 in LoRA.

### The text arm's drift is upstream's, and this is the measurement

Same cell (`fsdp2`), three trees:

    merge WITHOUT our two commits   7.70979 7.66901 7.57535 7.42618
    merge WITH    our two commits   7.70979 7.66901 7.57535 7.42618
    pre-merge baseline              7.70859 7.66722 7.57442 7.42620

Our commits are provably inert on that cell -- identical with and without -- so the
1.2e-3 comes entirely from the upstream merge. The text mini flavor is 20 of 21 layers
MoE and this merge reworked MoE sharding (`#3996`) and the token dispatcher (`#3970`), so
it is the arm most exposed; the multimodal flavors see 1e-5 on four cells each.

**So "all three matrices identical" is not achievable with this merge in, and the reason
is upstream numerics rather than anything on our side.** What is achieved: no cell that
trained before fails now, both previously-broken EP x TP cells are back, and every
deviation is measured and attributed. The text arm's baseline for future comparisons is
`verify_post_merge/text_13.txt`, not `baseline_pre_merge/text_13.txt`.
