# New PR: zero initialisation of Kimi K3's attention residual projections (split out of #4780)

tianyu-l on #4780 (r4090059193): "fix the init in its own PR". Branch `k3_attnres_zero_init` on the fork, one commit on upstream main `9e159aed7`. Open it from the fork; the number then goes into the #4780 reply and body v3 (`#INIT_PR`).

Title: `[Kimi K3] Zero-initialise the attention residual projections`

--- PASTE BEGIN ---

## Summary

Initialise Kimi K3's attention residual projections to zero, as Section 5 of the Attention Residuals technical report requires, so the depth softmax starts uniform and each residual starts as an equal-weight average of its sources.

- `torchtitan/models/kimi_k3/__init__.py`: `attention_res_proj`, `ffn_res_proj` and `output_res_proj` take `nn.init.zeros_` instead of `trunc_normal_` at std 0.02.
- `tests/unit_tests/cpu/test_kimi_k3_attention_residual_init.py`: the projections are zero initialised; at zero the aggregation returns the mean of its sources exactly; the projection still gets a gradient, and the norm weight does not until the projection leaves zero.

Split out of #4780, as suggested there.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_attention_residual_init.py -q` (`3 passed`; the initialisation test fails on main)
- Scoped pre-commit checks, including formatting, lint, and Pyrefly (`passed`)

--- PASTE END ---
