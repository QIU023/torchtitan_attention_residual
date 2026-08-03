# Multimodal parallelism matrix

`kimi_k3_mini_vl` (87,376,316 params: 80.9M text + 6.44M MoonViT-V2 at
4 layers / hidden 256), 3 steps, seed 42, deterministic, global batch 8, on
torch dev20260802.

## Passing

    fsdp2                7.71632 7.63655 7.43903
    cp2                  7.71686 7.65315 7.44388
    fsdp2 x tp2          7.72128 7.63490 7.45163
    fsdp2 x cp2          7.71162 7.63189 7.43270
    fsdp2 x tp2 x cp2    7.71143 7.62768 7.40642
    ep2 x fsdp2          7.71632 7.64275 7.44239
    ep2 x fsdp2 x tp2 x cp2   7.70906 7.64958 7.44055

Seven legs, including a choose-3 and an EP-bearing choose-3. TP, CP, EP and FSDP
all compose with the vision tower.

## Blocked, two distinct reasons

**dp1 and bare tp2** hit the KDA shared-memory ceiling, the same wall every
flavor on this box hits without FSDP's mixed-precision cast: fla's kernel asks
for 108,160 B against this GPU's 101,376 B. Not multimodal-specific, and the
FSDP-bearing legs above cover the same axes.

**Every PP leg** fails with `Optimizer param_groups pattern '.*' matched no
parameters`. The torch upgrade did not change this, which is what it should do:
the cause is structural, not a version skew. The PP splitter divides
`model.layers`, which on the multimodal model is a property forwarding into
`.language_model`, so some stages receive a wrapper whose text model has been
taken away while the wrapper's own forward still expects it.

Making PP multimodal means teaching the split about the wrapper -- which stage
owns the tower, how vision features cross a boundary, what non-owning stages
present as. That is design work on the PP adapter, the component with the clean
per-parameter record (0.00000 over 548 parameters), so it is not something to
improvise. Text-only PP is unaffected and reproduces its baseline exactly.
