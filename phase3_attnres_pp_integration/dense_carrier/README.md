# Dense AttnRes test carrier (relocated from the torchtitan fork)

The Llama3-shape dense + DSv3-shape MoE **AttnRes test carrier** and its
test suite, relocated out of `torchtitan/experiments/kimi_k3/` on
2026-07-18 so the fork folder contains only the K3 implementation.

This carrier is the algorithm's A/B baseline and PP-pressure-test bed:

- `dense_model.py` — `AttnResModel` / `AttnResTransformerBlock` (dense GQA
  + DSv3-MoE variants over the shared `Decoder` base)
- `test_attn_res.py` — primitive + dense-model unit tests
- `test_attn_res_dsv3.py` — DSv3-shape MoE + MLA + AttnRes tests
- `test_attn_res_tp_grad.py` — TP gradient tests
- `test_pipeline_adapter.py` — the 1460-line PP cross-stage adapter grid
  (TP + PP + AC combinations)

Also here: `__init__.py` (model flavors `llama3_175m_*` / `dsv3_attn_res_*`
+ `model_registry`) and `config_registry.py` (trainer configs) — extracted
from the fork at `666cf7ad6` and rewired so this package **runs against the
current fork** as a downstream consumer (it imports the AttnRes primitive,
layout tables and PP adapter from `torchtitan.experiments.kimi_k3`).

**How to run** (torchtitan's ConfigManager accepts fully-qualified module
paths):

```bash
export PYTHONPATH="$WS:$PYTHONPATH"   # logbook root, next to the fork pkg
torchrun ... -m torchtitan.train     --module phase3_attnres_pp_integration.dense_carrier     --config llama3_175m_attn_res_L16_n8 ...
# tests
PYTHONPATH="$WS/torchtitan:$WS" pytest phase3_attnres_pp_integration/dense_carrier/ -q
```

The phase2/3 launchers (`launch_4gpu_*`, `launch_8gpu_*`,
`run_pp_pressure_test.sh`, phase2 `launch*.sh`) have been updated to this
module path. The phase2/3 evidence (paper Table 1 reproduction, PP=8×VP=4
pressure tests, see
[`../PRESSURE_TEST_REPORT_2026-05-12.md`](../PRESSURE_TEST_REPORT_2026-05-12.md))
was produced on this carrier.
