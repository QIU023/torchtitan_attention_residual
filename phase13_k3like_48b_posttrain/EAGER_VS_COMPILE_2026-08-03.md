# eager vs torch.compile, text and multimodal

## The baseline was already eager

Both matrix flavors ship `compile.enable=False`. So the entire 13-leg text matrix
and the 7-leg multimodal matrix were already eager runs -- the "add an eager
baseline" request was, in effect, already satisfied, and what was actually
missing was the COMPILED comparison. Checked rather than assumed:

    kimi_k3_mini_diag_4l_moe_depth: compile.enable=False
    kimi_k3_mini_vl:                compile.enable=False

## The comparison

fsdp2, 3 steps, seed 42, deterministic, global batch 8:

    text  eager      7.67856 7.26904 6.29599
    text  compiled   7.67872 7.26823 6.30244    max |delta| 0.00645

    vl    eager      7.71632 7.63655 7.43903
    vl    compiled   7.71277 7.63872 7.43590    max |delta| 0.00355

Both within the bf16 spread the parallelism combinations themselves show
(<= 0.010 across the matrix), so compile does not change the numerics beyond
arithmetic reordering, on either the text or the multimodal path.

Worth stating what this does not show: it is one axis (fsdp2) and three steps.
It establishes that compile runs and agrees, not that every combination compiles.
