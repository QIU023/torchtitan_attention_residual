# Official K3 artifacts (fetched 2026-07-27)

`config.json` -- verbatim from
https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json
(fields reordered alphabetically within objects and the HF
generation-defaults boilerplate dropped; every architecture value is
unmodified). This is the reconciliation baseline for
K3_RELEASE_IMPACT_2026-07-16.md sec 4.

Other sources used:
- Official blog: https://www.kimi.com/blog/kimi-k3 -- names Stable LatentMoE
  ("effectively activating 16 out of 896 experts"), Gated MLA, SiTU (Sigmoid
  Tanh Unit), "MXFP4 weights with MXFP8 activations" with "quantization-aware
  training from the SFT stage onward", KDA prefix caching, native vision, 1M
  context. No layer-level numbers in the blog -- those come from config.json.
