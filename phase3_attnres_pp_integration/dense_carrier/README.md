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

**Provenance / how to run**: these files are archived verbatim. The flavor
definitions (`llama3_175m_*`, `dsv3_attn_res_*`) and a fully runnable
state live in the fork's git history — last complete commit
[`666cf7ad6`](https://github.com/QIU023/torchtitan/tree/666cf7ad6/torchtitan/experiments/kimi_k3)
on branch `attention_residual_dev`. The phase2/3 evidence
(paper Table 1 reproduction, PP=8×VP=4 pressure tests, see
[`../PRESSURE_TEST_REPORT_2026-05-12.md`](../PRESSURE_TEST_REPORT_2026-05-12.md))
was produced on this carrier.
