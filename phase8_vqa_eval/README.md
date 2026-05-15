# Phase 8 — VQA Evaluation

Quantitative assessment of v11 / v12 / SFT'd checkpoints on standard
multimodal benchmarks. Validates that loss convergence in pretraining
translates to actual visual question-answering capability.

## Benchmarks

* **VQAv2** (val split, 214K Q-A pairs) — open-ended VQA on COCO
  images; primary multimodal benchmark
* **GQA** (testdev_balanced, 12K pairs) — compositional reasoning on
  Visual Genome scenes
* **ScienceQA** (test, 21K pairs) — multiple-choice science / textbook
  questions, sub-benchmark image+text vs text-only

## Targets

| Checkpoint | Expected (LLaVA-1.5 7B reference) |
|---|---|
| v11 step-5000 (post-stage1 alignment) | ~30-40 % VQAv2 (alignment baseline) |
| v12 step-5000 | similar to v11 (parallelism doesn't change quality) |
| SFT (post 9-A) | **+15-25 %** improvement on VQAv2 |

## Eval framework

`lmms-eval` (LMMs-Lab/lmms-eval) — popular, supports many models +
benchmarks, easy to plug a custom inference adapter.

## Inference path

DCP shards (FSDP+TP+EP) → consolidated `.bin` / `.safetensors` →
HuggingFace-style loadable model. Need a small converter
(`phase8_vqa_eval/dcp_to_hf.py`).

Alternative: vllm with custom model loader — faster but more setup.

For 436M model and 50K-200K eval samples, raw transformers inference
at batch 8-16 is fine (~30 min per benchmark per ckpt).

## Files (to be written)

```
phase8_vqa_eval/run_vqa_eval.sh         # entry: per-ckpt × per-benchmark grid
phase8_vqa_eval/dcp_to_hf.py            # DCP shards → HF-style state_dict
phase8_vqa_eval/lmms_eval_adapter.py    # plugs kimi_linear + AttnRes into lmms-eval
phase8_vqa_eval/eval_results/           # JSON outputs per (ckpt, benchmark)
```

## Cost

* Per ckpt: 30 min (3 benchmarks × 10 min ea)
* 3 ckpts: 1.5h
* Plus 30-60 min framework setup first time
