# Official K3 infra: what exists, what we integrated (2026-07-27)

## Inventory

| artifact | where | status |
| --- | --- | --- |
| **reference modeling code** | HF `moonshotai/Kimi-K3`: `modeling_kimi_linear.py`, `modeling_kimi_k3.py`, `configuration_kimi_k3.py`, vision processors | **now used** -- was not, and that was the biggest gap |
| **weight index** | `model.safetensors.index.json`, 497,220 keys | **now used** as ground truth |
| **MoonEP** | `github.com/MoonshotAI/MoonEP` | **cannot run on this hardware** (below) |
| GitHub `MoonshotAI/Kimi-K3` | weights + report + LICENSE only | no infra to integrate |
| minitriton, AgentENV | report footnotes | out of scope |
| vLLM / SGLang recipes | external | out of scope for the training stack |

The reference code was the important find. Everything this repo believed about K3
before it came from the config plus the report's prose -- i.e. from inference --
and the report and the shipped code disagree in at least one material place.

## What checking against the reference changed

Parity was measured by instantiating the official module, copying its weights
into ours, and feeding identical inputs (`ref_parity_probe.py`).

| component | result |
| --- | --- |
| Gated MLA | **rel 7.07e-06** (fp32) -- verified correct |
| KDA | **rel 2.67e-03** (bf16, chunked-scan band) -- verified correct |
| `SituAndMul` | formula identical, including the fp32 compute and cast back |
| LatentMoE | structure identical: `down -> experts -> norm -> up`, shared experts on the ORIGINAL input |
| router | identical: fp32 sigmoid, bias added for SELECTION only, weights gathered from unbiased scores, renormalize, then `routed_scaling_factor` |

So the text side was right. It was verified, not assumed, which is the point.

**The vision side was wrong in seven places** and has been rewritten. The report
says attention "is factorized into intra-frame spatial and inter-frame temporal
passes"; the shipped tower has exactly one `wqkv` and one `wo` per block (27 of
each) and runs a single varlen attention over a sample's whole `t*h*w` token set.
The earlier implementation followed the report, and it matched the reported 0.4B
parameter count -- which proves only that a parameter count cannot distinguish
one projection set used twice from one used once. Also corrected: 2-D RoPE was
missing entirely, the time embedding is a fixed sincos buffer rather than a
learned table, `t == 1` takes no time component, `sd2_tpool` means over ALL
frames rather than pairwise, the projector is post-RMSNorm `PatchMergerMLPV2`
rather than pre-LayerNorm, and inputs are NaViT-packed rather than a rectangular
batch.

## Weight loading

All 497,220 keys map, and every non-expert key round-trips
(`hf_key_map.py`, `test_hf_key_map.py`). Naming differences that would each have
silently dropped or swapped tensors:

* everything text-side is under `language_model.model.`
* **Block Attention Residuals are in the shipped weights**:
  `self_attention_res_proj` / `self_attention_res_norm` (93 each), `mlp_res_proj`
  / `mlp_res_norm` (93 each), and `model.output_attn_res_proj` for the final
  aggregation. Direct confirmation of this repo's premise from the checkpoint.
* MoE layers are `block_sparse_moe`, the one dense layer is `mlp`
* routed experts use `w1/w2/w3` while shared experts in the *same block* use
  `gate_proj/up_proj/down_proj`, so a single global rename corrupts one of them
* **both output gates are named `g_proj`**. Ours differ by attention type, so the
  mapping resolves by layer type; getting it backwards swaps two same-shaped
  tensors, which no shape check catches
* only routed-expert weights are quantized (`.weight_packed` + `.weight_scale`),
  confirming `quant_scope.py` off the checkpoint rather than off the config

## MoonEP: blocked on hardware, not on work

MoonEP is an EP communication library whose contract is "one contiguous
symmetric-memory weight tensor per expert projection, plus a planner-produced
`cu_seqlens`", with a `[E+B, H, H']` group-GEMM weight layout and
`dispatch` / `combine` / `prefetch_weight` / `reduce_grad`. Its value is the
`E/R` redundant-expert bound that guarantees a feasible balanced plan, hence
static shapes and no per-layer host sync.

It cannot run on this box, and the blocker is a capability gap rather than
tuning:

```
csrc/nvl_shared_buffer.cuh: cuMulticastBindMem / CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
moonep/buffer.py:221        assert nvl_multicast_supported(), "Multicast not supported on this device"
```

Measured here: `cuMulticast supported: False` on the RTX 5060 Ti (cc 12.0), and
`nvidia-smi topo -m` reports PHB/NODE throughout with `nvidia-smi nvlink -s`
empty -- PCIe only, no NVLink. The assert has no fallback path.

**Deliberately not written yet.** An integration that cannot be executed even
once is the same failure mode this phase spent the day removing: three features
(routed-expert init, Per-Head Muon, the QAT scope) existed, passed their unit
tests, and did nothing. Shipping a fourth would be worse here, because MoonEP's
correctness contract is about memory aliasing and gradient staging across ranks
-- exactly the class of thing that cannot be reasoned into correctness.

What is ready: the contract is understood and our expert parameters already sit
in the `[E, H, H']` stacked layout MoonEP's group GEMM wants (`w1_EFD`,
`w2_EDF`, `w3_EFD`), so the adapter is a symmetric-memory allocation plus a
token-dispatcher swap, not a model change. It becomes a day's work on a machine
with NVLink and multicast (any H100/H200/B200 node).

## Still open

* MXFP8 activations on the QLoRA path (QAT has them; the packing path is
  weights-only, matching the checkpoint's `input_activations: null`)
* migrate `init_weights` to upstream's declarative per-parameter map -- the
  hand-maintained walk is what broke earlier today
* KCP for the MLA layers. KDA is done and bit-exact at cp2; MLA still uses
  Ulysses, so a KCP run is currently mixed-mode
* actually loading the released weights end to end. The key mapping is complete
  and tested, but the MXFP4 dequant-on-load path for `.weight_packed` +
  `.weight_scale` is not wired into the adapter yet, and the full checkpoint is
  ~600 GB, so this needs a machine that can hold it
