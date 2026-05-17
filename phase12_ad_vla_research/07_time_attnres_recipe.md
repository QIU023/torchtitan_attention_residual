# 07 — Time-AttnRes: initial direction & implementation recipe

> **Status**: initial direction doc (2026-05-17). Not a survey or strategy doc — this is the concrete proposal for **how** to extend AttnRes from depth-axis aggregation to time-axis aggregation in a video VLM. Companion to [`01_video_vlm.md`](01_video_vlm.md) Tier C and [`04_world_models.md`](04_world_models.md) Tier A.

## What this doc is for

Not "should we?" — that's [`06_strategy_workplan.md`](06_strategy_workplan.md). This doc answers "**given** we want Time-AttnRes, what does the code/architecture look like, and what's the minimal-viable experiment to validate it works."

## 0. Recap — depth-AttnRes

```python
# Block AttnRes inside a single LM, per layer l ∈ [1..L]:
V        = stack(B_1, B_2, ..., B_{l-1})    # [l-1, B, T, D]   prior block outputs
K        = RMSNorm(V)                        # [l-1, B, T, D]
Q_l      = learned_query[l]                  # [1, D]            per-layer pseudo-query
alpha    = softmax(Q_l @ K.transpose / sqrt(D))  # [1, l-1]      cross-block weights
carrier  = alpha @ V                         # [B, T, D]         weighted aggregation
h_l      = carrier + B_l(h_{l-1})            # block delta on top
```

**Critical observation**: `V` is just `[N, B, T, D]` where `N` is "number of items to aggregate." Paper sets `N = l-1` (depth axis). Nothing about the mechanism requires N to mean "block index". **Swap N from layer-index to frame-index → Time-AttnRes.**

## 1. Four insertion patterns (recap from chat)

| Pattern | Insertion point | New params | LM modified? | Effort | Recommended for |
|---|---|---:|---|---|---|
| **A** | After per-frame projector | ~10K | ❌ no | 1 week | **prototype validation** |
| **B** | Inside each LM block (parallel to depth-AttnRes) | ~per-layer-D | ✅ yes | 4–6 weeks | architecture paper |
| **C** | Frame-cascade (full LM per frame then aggregate) | small | ❌ no | 1 week | baseline only |
| **D** | Concat all frame tokens → KDA handles long seq | 0 | ❌ no | 1 week | uses KDA, ignores AttnRes mechanism |

This doc focuses on **A** (prototype) and **B** (research). C is a cheap baseline; D doesn't actually exercise AttnRes.

## 2. Pattern A — code sketch

### 2.1 New module

```python
# phase5_vlm_multimodal_sft/time_attn_res.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAttnRes(nn.Module):
    """Time-axis softmax aggregation of past K frames' projector outputs.

    Input:  vision_embeds  [B, T, N_patches, D]  with T = num_frames in the clip
    Output: carrier        [B, T, N_patches, D]  where carrier[:, t] aggregates [t-K..t-1]

    Per-frame learned pseudo-query Q[t] decides which of the past K frames to
    weight. Causal mask ensures frame t never sees frames > t (so this works
    for streaming inference too).
    """

    def __init__(self, dim: int, max_k: int = 4, num_query_per_frame: int = 1):
        super().__init__()
        self.dim = dim
        self.max_k = max_k
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        # Single global query (shared across T) — simplest.
        # For per-frame queries, use shape (max_T, dim) with positional indexing.
        self.q = nn.Parameter(torch.randn(num_query_per_frame, dim) * 0.02)

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        # vision_embeds: [B, T, N, D]
        B, T, N, D = vision_embeds.shape
        out = torch.zeros_like(vision_embeds)
        # First frame: no past → carrier == itself (skip aggregation)
        out[:, 0] = vision_embeds[:, 0]
        for t in range(1, T):
            past = vision_embeds[:, max(0, t - self.max_k): t]  # [B, k_eff, N, D]
            k_eff = past.size(1)
            # Aggregate per-patch (we keep spatial structure)
            V = past  # [B, k_eff, N, D]
            K = self.norm(V)
            # Q shape: [1, D] broadcast to [B, 1, D]
            q_t = self.q.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
            # Per-patch attention: average attention weights across N patches
            # of past frame, then weight whole frame.
            K_pooled = K.mean(dim=2)  # [B, k_eff, D]
            scores = (q_t @ K_pooled.transpose(-1, -2)).squeeze(1) / (D ** 0.5)
            alpha = F.softmax(scores, dim=-1)  # [B, k_eff]
            carrier = (alpha.unsqueeze(-1).unsqueeze(-1) * V).sum(dim=1)  # [B, N, D]
            out[:, t] = carrier + vision_embeds[:, t]  # residual on current frame
        return out
```

### 2.2 Integration into `train_mm.py`

Minimal patch — wrap the projector output, before LM input:

```python
# in MultimodalTrainer.__init__ (after projector built)
if self._mm_video_mode:
    self._time_attn_res = TimeAttnRes(dim=lm_dim, max_k=4).to(self.device)
    self._proj_optim.add_param_group({'params': self._time_attn_res.parameters(),
                                       'lr': proj_lr})

# in post_dataloading_process, after projector(vision_tokens):
if self._mm_video_mode:
    # vision_embeds shape needs to be [B, T, N, D] — dataloader must batch T frames
    vision_embeds = self._time_attn_res(vision_embeds)
    # Flatten T into LM token sequence for the existing scatter logic
    vision_embeds = vision_embeds.reshape(B, T * N, D)
```

### 2.3 Dataset side

Need a video dataset that yields:
- `pixel_values`: `[T, 3, 224, 224]` instead of `[3, 224, 224]`
- `input_ids`, `labels`: text token sequence (caption / QA answer / trajectory)
- Each `<image>` sentinel position gets replaced by `T * N_patches` vision-token positions

Sketch: `phase5_vlm_multimodal_sft/multimodal_video_dataset.py`. ~150 lines, mostly mirrors `LlavaInstructSFTDataset` but with frame-stacking. nuScenes / DriveLM v1.1 already has frame indexing.

### 2.4 Training recipe

| | Value |
|---|---|
| Initial ckpt | `runs/stage2_instruct_sft_447m/checkpoint/step-10400` (when stage 2 done) |
| Trainable | TimeAttnRes module (~10K) + projector (~14M, optional) + LM (~447M, optional) |
| Data | nuScenes CAM_FRONT 4-frame clips with DriveLM-QA labels |
| Stages | (a) freeze LM+projector, train TimeAttnRes only — 500 steps proof-of-life; (b) unfreeze projector + LM, full SFT 2-stage style |
| LR | TimeAttnRes 1e-3 (random init), projector 2e-5, LM 2e-5 |
| Batch | gbs=64, lbs=8, T=4 frames/clip → ~1.5× token cost vs single-frame |

## 3. Pattern B — code sketch (deeper)

### 3.1 Where the change lands

In `torchtitan/torchtitan/experiments/kimi_linear/attn_res_model.py`, the per-layer forward currently does:

```python
# simplified
for layer_idx, layer in self.layers.items():
    h = layer(h)            # block forward
    # depth AttnRes carrier kept inside model state
```

Pattern B adds a **parallel time-AttnRes carrier**, computed per frame:

```python
# Pattern B sketch
class KimiAttnResVideoDecoderLayer(KimiAttnResDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Per-layer query for time aggregation (smaller than depth query — same shape, different params)
        self.time_query = nn.Parameter(torch.randn(1, config.hidden_size) * 0.02)

    def forward(self, h_curr, past_block_outputs, past_frame_block_outputs):
        # past_block_outputs:       [l-1, B, T_tok, D]  (depth axis)  — existing
        # past_frame_block_outputs: [K,   B, T_tok, D]  (time axis)   — NEW
        depth_carrier = self._depth_attn_res(past_block_outputs)   # existing
        time_carrier  = self._time_attn_res(past_frame_block_outputs, self.time_query)
        delta         = self._self_attn_and_mlp(h_curr)
        return depth_carrier + time_carrier + delta
```

### 3.2 Where the per-frame block state lives

The hard part: the LM's `forward` currently processes one input at a time. For Pattern B, we need to maintain a **rolling buffer of past frames' per-layer outputs**, so layer `l` of frame `t` can attend to layer `l` of frames `[t-K..t-1]`.

Two implementation paths:
1. **Sequence-level batching**: process all T frames as one big batch dimension, then refactor each layer's forward to mask cross-frame interactions to causal aggregation only. ~400 lines of `attn_res_model.py` rewrite.
2. **Streaming with explicit state**: maintain a `List[Dict[layer_idx, tensor]]` of past frames' block outputs as a side state. Per-frame forward looks up the past state. ~200 lines but state management is fragile under DCP / FSDP.

Recommend path 1 for cleaner gradients (allows compile + AC to work normally).

### 3.3 Training recipe for Pattern B

Differs from Pattern A in:
- Need to retrain from earlier (probably from stage 1 alignment ckpt) since the LM forward graph changes
- ~30% extra step time (per-frame depth aggregation runs T times instead of 1)
- Eval needs to include cross-frame coherence metrics (e.g. consistency of object descriptions across frames)

## 4. Validation experiments (in order)

### Exp 1 — sanity check Pattern A (1 day)
- 50 steps on tiny nuScenes subset (4 clips × 4 frames each)
- Assert: loss decreases (vs frozen TimeAttnRes random init)
- Assert: `alpha` weights are non-uniform (TimeAttnRes is actually learning, not collapsing to mean-pool)

### Exp 2 — Pattern A vs mean-pool baseline (3 days)
- Same data, two configs: (a) TimeAttnRes (b) mean-pool over T frames
- Both fine-tune projector + LM on DriveLM-QA for 1000 steps
- Eval: held-out DriveLM-QA accuracy
- **Expected**: TimeAttnRes ≥ mean-pool by 1–3% (otherwise the mechanism isn't helping)

### Exp 3 — K sweep (2 days)
- Pattern A with K ∈ {1, 2, 4, 8}
- K=1 ≈ no temporal context (single-frame fallback)
- Eval: which K is best? Where do diminishing returns start?

### Exp 4 — Pattern A frozen vs unfrozen LM (3 days)
- Compare: (a) freeze LM, train TimeAttnRes + projector only; (b) unfreeze everything
- Tells us how much of the gain is from temporal context vs from extra training

### Exp 5 — Pattern B vs Pattern A (after Pattern A validates) (2 weeks)
- Pattern B requires 1-2 weeks of implementation work first
- Same DriveLM-QA eval as Exp 2
- **Expected**: Pattern B > Pattern A by 1–2% (otherwise the depth-time fusion isn't worth the complexity)

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TimeAttnRes alpha collapses to argmax (always picks newest frame) | medium | features degenerate to single-frame | add temperature schedule on softmax; init query orthogonal |
| TimeAttnRes alpha collapses to uniform (= mean-pool) | medium | mechanism gives no gain | regularize query magnitude during training; verify in Exp 1 |
| Pattern B's per-frame state breaks FSDP shard semantics | high | OOM or wrong gradients | use Path 1 (batch-axis frames) not Path 2 (streaming state) |
| KDA's chunked state confuses with frame-batched input | medium | wrong sequences fed to KDA | unit test with debug.deterministic; compare per-frame loss to baseline |
| Video dataloader is slow (disk I/O bottleneck) | high | training stalls | pre-extract frames to local SSD; batch-load entire clips |

## 6. Resource estimate (5090×8)

| Experiment | Compute | Wall time |
|---|---|---|
| Exp 1 (sanity) | 50 steps × ~5s/step (4-frame batch) | 5 min |
| Exp 2 (1k step compare) | 1000 × 2 configs × 5s | 3 hours |
| Exp 3 (K sweep) | 1000 × 4 configs × 5s | 6 hours |
| Exp 4 (freeze ablation) | 1000 × 2 configs × 5s | 3 hours |
| Pattern B implementation | engineering, not compute | 1–2 weeks |
| Exp 5 (Pattern B vs A) | 5000 × 2 × ~7s | 20 hours |

**Total to a publishable result**: ~3 days of compute + 2-3 weeks of engineering for Pattern B. Pattern A alone is ~1 week total (engineering + compute).

## 7. What this validates / doesn't validate

**Validates**: AttnRes is a generic "softmax-weighted aggregation" mechanism whose loop axis can be swapped from depth to time without loss of training stability or downstream quality.

**Does NOT validate**:
- Whether AttnRes is the *best* temporal aggregator vs. e.g. Q-Former, Mamba-2 state, mean-pool. (Need separate ablation against those baselines.)
- Whether this scales to 8-cam × 8-frame video (Exp 1-4 only test single-cam 4-frame). Scale-out is a separate concern.
- Whether it transfers to non-driving domains (Ego4D, instruction videos). Cross-domain ablation is future work.

## 8. What goes in the paper / talking points

- "AttnRes mechanism is axis-agnostic" — depth or time, same math
- "Spatio-Temporal AttnRes" framing positions us alongside Spatio-Temporal Transformers (ViViT, TimeSformer) but with **explicit learned aggregation** vs implicit self-attention
- "Causal mask makes it streaming-compatible" — relevant for AD where frames arrive sequentially
- Ablation surface (K sweep, depth-only / time-only / both for Pattern B) is interview-rich

## 9. Open questions for the user

- Should the prototype run on `mm_sft_447m_full` (joint, available now) or wait for stage 2 finish (paper-aligned, ~12-24h away)?
- nuScenes CAM_FRONT alone or surround-cam from day 1? (Surround = 6× tokens but more realistic for AD.)
- DriveLM-QA eval suffices for the prototype, or do we want trajectory MSE too (= already in VLA territory)?
- Pattern B is the paper bet — commit to that timeline or punt to a follow-up project?
