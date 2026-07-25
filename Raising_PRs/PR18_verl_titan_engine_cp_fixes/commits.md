# Source commits (fork verl `kimi_k3_integration`)

| commit | scope |
|---|---|
| `60b185fe` | `torchtitan engine: working CP for kimi_k3 (module-internal Ulysses)` -- 5 defects in 1 file (+68/-7). |

`engine_cp_PR18_full.patch` is the FULL fork commit. It is NOT the PR patch:
defects 1 and 4 (`context_parallel_load_balancer=None`, attention-mask CP
bypass) are gated on `torchtitan_name == "kimi_k3"` and must not be filed as a
name check -- see PR.md "NOT proposed here". File only:

| hunk | what | generic? |
|---|---|---|
| `_get_data_parallel_mesh` -> `"batch"` mesh | sampler axis must exclude cp | yes |
| `extra_kwargs["positions"] = extra_inputs["positions"]` | CP input-prep contract | yes |
| cp logits `all_gather` in the forward | seq-sharded logits vs full-seq loss | yes |
| `context_parallel_load_balancer=None` when kimi | needs a ModelSpec capability flag upstream | NO |
| `attention_masks` popped before CP prep when kimi | same | NO |

Upstream state verified 2026-07-25 against `volcengine/verl` main @ `983cb0f2`:
`_get_data_parallel_mesh` still returns the `fsdp` mesh; no `positions`
bridging anywhere in the file.
