"""LLaVA-style downstream benchmark eval pipeline for the Kimi-AttnRes VLM.

This package builds an offline-only (no sglang, no DCP→HF) generation loop
on top of ``MultimodalTrainer``'s ckpt-loading pathway. Each benchmark is
a separate ``score_*.py`` that subclasses ``EvalRunner`` and provides:
  * record iterator: (record_id, PIL image, prompt str, ground truth)
  * postprocess(model_text) → predicted answer
  * scorer(predictions, gts) → metric dict

The orchestrator ``run_all_evals.sh`` launches each score_*.py under
torchrun --nproc_per_node=8 (1D FSDP), distributes records round-robin
across ranks, gathers per-rank result JSONLs and writes REPORT.md.
"""
