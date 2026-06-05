# AttnRes Inference Implementation — Reference Summary

> Reference doc summarizing the inference-stack work done to make Block
> Attention Residuals (Kimi-Linear + Qwen3 carriers, plus the SigLIP-spliced
> VLM) actually run under SGLang on RTX 5090 / 4070Ti hardware. **Not a
> resume bullet** — kept here so the work is inventoried and can be cited
> in interview discussions. 9 fork commits on `attention_residual_inference`
> map to 11 upstream PR candidates in [`Raising_PRs/`](Raising_PRs/).

## 1. Why this stack exists

The training-side AttnRes work (Phases 2–4) proved the algorithm and the PP
cache adapter inside torchtitan. To close the loop for RLHF rollouts, the
same model shapes had to serve under SGLang — which exposed a long list of
upstream sharp edges in the path from `torchtitan` checkpoints → `transformers`
modeling → SGLang radix-cache / CUDA Graph / fused kernels.

Each item below is a real workaround merged on the fork; the upstream
PR-candidate column is the cleaned-up version intended for filing.

## 2. AttnRes model overlay (inference path)

| Fork commit | Subsystem | Upstream PR | What it does |
|---|---|---|---|
| `2f2e917d8` | model overlay | PR #5 | **Block AttnRes generic inference overlay** — two-phase merge (Phase 1: per-block softmax; Phase 2: cross-block aggregation) + optional seq-shard reduction. Generic over LM carrier so Kimi-Linear and Qwen3 both plug in. |
| `4a27b32e1` | model class | PR #5 | `KimiBlockAttnResForCausalLM` SGLang model class — pseudo-query parameter load from DCP-converted HF ckpt; layer index → block index mapping; intermediate-block output cache for the radix path. |
| `61c83cb30` | bench plumbing | (internal) | Env toggles for two-phase fusion / seq-shard / RMS-norm fusion to isolate microbench deltas. |
| `b8bd81a19` | CUDA Graph compat | PR #5 (carve-out) | Flatten `(B, T, D)` → `(B*T, D)` before `sgl_kernel.rmsnorm` because the fused kernel rejects 3-D inputs under the captured graph. |
| `0ddd84617` | correctness fix | (regression) | Shard-mode AttnRes in the eager fallback path — the seq-sharded variant was reusing the un-sharded scratchpad; one-liner divisor fix. |
| `63ea2ab75` | VLM overlay | PR #5/#2 | Host-side image-token splice (image tokens replace `<image>` sentinels after tokenization, before LM forward) so SigLIP path doesn't need radix-cache changes. |
| `a61c5c79f` | VLM carrier | PR #5 | `KimiBlockAttnResForConditionalGeneration` — SigLIP-Base + MLP projector + the AttnRes LM head; weight-loader for the multimodal DCP→HF bundle. |
| `c9bea5516` | merge | — | Reconciled VLM overlay branch into the AttnRes inference primary. |
| `d6fb3bbd7` | VLM correctness | PR #5 | Set `forward_context` + correct MLA layer wiring + RMSNorm `eps`. Without this, VLM forward produced finite-but-wrong logits because the layer index was 0-based in one place and 1-based in another. |

**PR #5 is the umbrella RFC** for the AttnRes overlay; reviewers asked it be
gated on the Kimi K-series release for legitimacy, so it stays research-track.

## 3. Backend / kernel fixes (carrier-independent)

| Fork commit | Subsystem | Upstream PR | What it does |
|---|---|---|---|
| `74083ffae` | SHM transport | **PR #1** ✅ branched | `SGLANG_DISABLE_SHM_MM` env var — forces CPU tensor transport for the multimodal request path. Workaround for shared-memory mmap deadlock on long-lived workers; the smallest possible first contribution. |
| `334990612` + `e8e7134ee` | MLA backend | **PR #3** (issue + flag) | `flashinfer_mla` returns NaN under bf16 on Blackwell for AttnRes-shaped inputs. Fork falls back to fp32 MLA (eager) on extend-only + write-cache. PR #3 is filed as a cross-repo issue (sglang + flashinfer) because the root cause is in flashinfer's MLA kernel; the sglang side is just the knob + fallback path. |
| `c07392916` | debug | (internal) | NaN-trace instrumentation in `KimiBlockAttnResModel.forward` used to localize PR #3 / PR #9. |
| `ac56bcbc0` | backend dispatch | (internal) | `HybridLinearAttnBackend` dispatch fix for the VLM-wrapped Kimi-Linear path — previously the wrapper class name broke the backend's `isinstance` check. |
| `850ebb715` | image loader | **PR #2** | base64 data-URL support in `attn_res_vl` image loader so the eval harness can pass images without writing tempfiles. Blocks on PR #5 because the loader is registered through the overlay. |
| `a6c46168a` (split) | KDA Triton kernel | **PR #7** ✅ branched, GPU-verified 6/7 on SM 8.9 | `causal_conv1d_triton` type-join: fp16 input + fp16 weight crashed in Triton's autotuner because the type promotion table assumed fp32 accumulators. One-line guard. |
| `a6c46168a` (split) | fp8 MoE kernel | **PR #8** ✅ branched, static-verified | Blackwell shmem autotune for the fp8 weight-only MoE fused kernel — previous default tile sizes exceeded SM 12.0 shmem capacity. Need actual SM 12.0 silicon to GPU-verify the new tiles. |
| `a6c46168a` (split) | AttnRes einsum | **PR #9** | The per-block softmax in Phase 1 was an `einsum` that lowered to cuBLAS bf16-after-fp8-dequant — produced silent NaN on Blackwell. Bypass with explicit `broadcast + sum`. Re-scoped 2026-05-17: cuBLAS root-cause issue filed against `pytorch/pytorch` standalone; sglang-side overlay patch waits on PR #5. |
| `dc154e785` | merge | — | Final reconciliation merge: brings DISABLE_SHM_MM + fp32 MLA fallback + base64 data-URL + NaN trace into the inference primary branch. |

## 4. Fused / unmerged investigation tracks

These never became fork commits but informed the PR-list:

- **PR #6 (RS+merge+AG seq-shard fusion as a feature)** — the seq-shard
  path inside the AttnRes overlay collapses ReduceScatter + merge + AllGather
  into one fused pass; if the broader AttnRes overlay (PR #5) gets adopted,
  this fusion is worth documenting as a stand-alone primitive. RFC-track.
- **PR #10 (`Fp8Config.get_quant_method` ignored-layers warning)** —
  during PR #8 debugging it surfaced that fp8 silently falls back to bf16
  on `ignored_layers` without any user-visible warning. Tentative: file only
  if PR #8's downstream ICA stays unresolved long-term.
- **PR #11 (torchstore sync-endpoint dispatch policy)** — async-vs-sync
  endpoint dispatch in `torchstore` blocked weight-broadcast on the RL
  rollout side; current fork carries a monkey-patch in the RL launcher. Issue
  ready; FILING.md drafted.

## 5. Status snapshot (2026-05-18)

| Upstream PR | Branch state | Filing state |
|---|---|---|
| PR #1 SHM disable | 🚀 pushed `pr1-disable-shm-mm` | PR not opened yet |
| PR #2 base64 data-URL | 🟡 ready (`850ebb715`) | blocks on PR #5 |
| PR #3 MLA bf16 NaN | 🟠 issue ready | sglang + flashinfer cross-link FILING drafted |
| PR #5 AttnRes overlay | 🟠 research-track | needs Kimi K-series release |
| PR #6 RS+merge+AG fusion | 🟠 RFC | depends on PR #5 + adopter |
| PR #7 KDA fp16 type-join | 🚀 pushed, GPU 6/7 on SM 8.9 | PR not opened yet |
| PR #8 fp8 MoE Blackwell shmem | 🚀 pushed, static-verified | needs SM 12.0 GPU smoke |
| PR #9 cuBLAS bypass | 🟠 re-scoped: file cuBLAS root-cause issue now | sglang patch blocks on PR #5 |
| PR #10 fp8 ignored-layers warning | 🔵 tentative | conditional on PR #8 outcome |
| PR #11 torchstore dispatch | 🟠 issue ready | FILING drafted |

Filing index lives at [`Raising_PRs/README.md`](Raising_PRs/README.md).

## 6. What this stack proves (for interview talk-track)

- Working knowledge of SGLang's model-overlay extension surface (custom
  model class registration, `forward_context`, weight-loader, radix-cache
  interaction).
- Comfort reading + patching the Triton kernel layer (`sgl_kernel.rmsnorm`,
  `causal_conv1d_triton`, fp8 MoE fused kernel) and isolating regressions
  to specific GPU SM versions (8.9 / 10.0 / 12.0).
- End-to-end debugging across the bf16/fp16/fp32/fp8 numerical interface,
  including cuBLAS-level root-cause analysis (PR #9) and cross-repo issue
  authorship (sglang ↔ flashinfer ↔ pytorch).
- VLM-specific: SigLIP + projector + LM weight-loader plumbing, host-side
  image-token splice strategy compatible with SGLang's radix cache, and the
  `forward_context` / MLA-layer-wiring / eps-config trifecta that gates
  correct VLM inference.

## 7. What this stack does **not** prove

- Throughput parity with stock SGLang at production scale — bench was
  microbench-level on RTX 5090 (single node, ≤ 8 GPU), not multi-node serving.
- Stability under long-running production traffic — no production deployment.
- TP/EP fan-out > 8 — fork ran TP up to 8, EP up to 16; larger meshes deferred.
- SM 12.0 (Blackwell) GPU smoke for PR #8 — static-verified only; needs
  silicon access to land the kernel autotune cleanly.
