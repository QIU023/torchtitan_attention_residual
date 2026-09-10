# PR title: [Kimi K3] LoRA and QLoRA through core's LoRAConverter: adapter export, NF4 and packed-MXFP4 bases, packed tensor parallel, and the debug flavors

Branch `lora_review2` = `72bbcb639` (nine commits on main `ac10ca48f`; independent of the parallelism PRs, no stack). The top commit is a one-line flake8 fix; the four pyrefly errors the earlier head `f092d37d9` carried in `lora.py` were removed by `3f931cf47` before this. Pinned pyrefly 0.45.1 over the project: the branch's error set equals main's (21 project errors on main, none in the touched files).

--- PASTE BEGIN ---

## Summary

Adapter training for Kimi K3 through core's `LoRAConverter`, with the pieces that make it usable end to end:

- **Adapter export.** `merge_lora_state_dict` folds `W + (alpha / rank) * B @ A` in the modules (a parameter-object swap, restored afterwards) and returns the model's original key set, so every serialization hook produces its own keys and wrapper segments (activation checkpointing, FSDP, compile) resolve instead of being guessed at. `trainable_state_dict` is the adapter-only payload. `to_hf` leaves adapters out instead of raising. `LoRALinearBase` and `MXFP4ExpertsBase` are typed marker bases: they declare the members every built subclass has, so the merge, the quantizer, the DCP script and the tests read them without suppressions.
- **QLoRA.** `quantize_base="nf4"` packs the frozen bases after init through torchao (imports follow torchao 0.18, where `NF4Tensor` moved); library builds only, since FSDP2's lazy init cannot take a post-hoc packed parameter. `quantize_base="mxfp4"` swaps the base for split storage at build time, `qdata` uint8 `[out, in/2]` plus an e8m0 scale viewed as uint8 `[out, in/32]`, so FSDP2 shards packed bytes (pack, then shard) and from-scratch init draws each rank's rows locally, exact because MX block-32 is row-blockwise. `quantize_experts="mxfp4"` does the same for the grouped experts through dequant properties, forward unchanged. Under expert parallelism the experts dequantize their local shard (the dequant view sizes its leading dim from the shard) and the merge gathers explicitly for export.
- **Packed tensor parallel.** A base the declarations keep invariant on tp gets replicated adapters; colwise / rowwise packed bases run the packed-TP forward (local dequant, local matmul, `Partial` output under rowwise, explicit gradient placements so the tp reductions happen). Expert weights tp-sharded on their inner dims refuse: the 2-D flatten cannot express them. The packed experts carry the declared `spmd_types` entry through to the packed pair, so the packing's inner-dim refusal no longer fires on a declaration that names a tp placement the run does not use.
- **A fully frozen model part gets no optimizer.** Adapter training under pipeline parallelism produces stages with nothing trainable (a vision-tower stage carries no LoRA targets; the tower stays frozen); such a part is skipped with a log line, and the pattern-level raise stays for parts that do train, where an unmatched pattern is a config typo.
- **`scripts/quantize_lora_dcp.py`** repacks a bf16 LoRA DCP checkpoint into the packed twin's layout. The key map is derived from the packed flavor built on `meta` and walked, so it is exactly the packed model's state-dict contract, and no rank materializes the full bf16 model.
- **Debug flavors.** `kimi_k3_debugmodel_lora` (the MLA projections, the dense FFN and the latent-MoE projections; every decoder layer carries an adapter) and `kimi_k3_debugmodel_qlora_mxfp4` (the same with packed bases and experts). The MLA output gate is not adapted: it is named `gate`, as is every MoE router's gate, and last-segment matching cannot tell them apart.

## Implementation

Everything lives in `torchtitan/components/lora.py` (the converter, the marker bases, the packing, the merge) plus thirteen lines in `torchtitan/components/optimizer/optimizer.py` (the frozen-part skip), the two flavors in `torchtitan/models/kimi_k3/config_registry.py`, and the script. No model code changes.

## Limitations

- LoRA on a tensor-parallel Kimi K3 is not exercised here: the model refuses tensor parallelism on main (that is PR 4499); the packed-TP forward is covered by the unit tests on small linears.
- NF4 packing is library-build only (see above); MXFP4 is the path that composes with FSDP2.
- Pipeline-parallel adapter training (the frozen-stage case the optimizer change is for) is not measured in this PR; the CPU tests cover the optimizer construction on a frozen part only through the converter tests.

## Tests

```text
pytest -q tests/unit_tests/cpu/test_lora.py tests/unit_tests/cpu/test_kimi_k3_qlora_experts.py
```

Result: 19 passed (16 in `test_lora.py`: build, forward, converter order, class cache, rank validation, freezing of direct parameters on composite and root modules, frozen-config type checks, the adapter-only state dict, the merge's keys / zero-init identity / delta / serialization hooks / wrappers, NF4 pack-forward-merge, packing at init, MXFP4 pack-at-build-and-merge; 3 in `test_kimi_k3_qlora_experts.py`: experts pack at build with the dequant property, the merge restores the original keys, packing under the declared sharding). pre-commit passes on the touched files (the pinned pyrefly's project-wide error set is main's).

## Results

`kimi_k3_debugmodel_lora`, `seed=42`, deterministic, 3 steps, on 8 x RTX 5060 Ti (SM120; the KDA capability guard, which on main admits SM100/103 only, lifted locally for the run). The top commit removes an unused import, so the branch head must reproduce the previous head to every digit:

| cell | step 1 loss / grad norm | step 2 | step 3 | peak memory |
| --- | ---: | ---: | ---: | ---: |
| dp1, previous head `aaa823064` | `12.48753` / `2.1875` | `12.20719` / `4.5625` | `12.23086` / `3.1562` | 4.05 GiB |
| dp1, this head `72bbcb639` | bitwise | bitwise | bitwise | 4.05 GiB |
| fsdp2, previous head `aaa823064` | `12.64212` / `2.0625` | `12.47497` / `3.0156` | `12.19887` / `3.3906` | 3.46 GiB |
| fsdp2, this head `72bbcb639` | bitwise | bitwise | bitwise | 3.46 GiB |

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
