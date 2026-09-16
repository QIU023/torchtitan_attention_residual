# PR title: [Kimi K3] README: list the merged parallelisms; the refusal names what is supported

RFC 3029 deliverable 5 ("`models/kimi_k3/README.md` lists the parallelisms main supports"). Prepared 2026-09-16 night; no branch created (user's rule). Patch `readme_parallelisms.patch` in this folder applies on main `810e62786` (`git apply --check` against the main index passed): 7 changed lines in two files, `README.md` (two table rows and one sentence) and `parallelize.py` (the refusal's message). No test asserts the old message (grepped main's `tests/` and `torchtitan/`). Facts in the rows: TP / SP from #4499 (head-parallel KDA and MLA, core's expert sharding for the latent MoE, the MoonViT plan shared with Kimi K2.5), EP from #4314, the B200 CI cell `kimi_k3_debugmodel_mm` (`torchtitan_recipes/tests/b200.py`: dp_shard 2, tp 2, sequence parallel on, ep 2). PP and CP are the two names `parallelize_kimi_k3` still refuses on main.

To file: a branch from main `810e62786` with the patch applied as one commit ("kimi_k3: README lists TP/SP and EP; the refusal names the supported parallelisms"), PR against `main`, title as above, body from the paste section.

--- PASTE BEGIN ---

## Summary

Update the Kimi K3 README's parallelism table for what main now runs, and make `parallelize_kimi_k3`'s refusal say the same.

- `models/kimi_k3/README.md`: rows for tensor parallelism with sequence parallel (#4499) and expert parallelism (#4314), and one sentence that pipeline and context parallelism are refused for now.
- `models/kimi_k3/parallelize.py`: the `NotImplementedError` raised for PP or CP said "supports FSDP2 data parallelism only"; it now names FSDP2 / HSDP, tensor parallelism and expert parallelism.

## Test plan

- Documentation and a message string; no behaviour change. `pytest tests/unit_tests/cpu -q -k kimi_k3` unchanged.

--- PASTE END ---
