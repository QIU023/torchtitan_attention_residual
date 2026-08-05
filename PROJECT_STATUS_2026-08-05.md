# Project status and blocked items

Written at the point where the eager reference PR looks close to merging and the
four parallelism branches are waiting to be raised against it.

## Repos

| repo | branch | HEAD | state |
|---|---|---|---|
| fork `QIU023/torchtitan` | `attention_residual_dev` | `b99ff970c` | pushed |
| fork, PR bookmarks | `k3_pr_{a,b,c,d}_*` | `a146d1bf2` | **3 commits behind dev** |
| logbook | `main` | pushed | evidence + drafts |
| `QIU023/verl` | `kimi_k3_integration` | `cb9ac9e1` | pushed |

The four bookmarks are all at the same commit and all trail dev by three: the
QAT/multimodal-build fix, the config exporter, and the text-only checkpoint
layout. They are bookmarks on the full fork, not sliced branches -- the slicing
happens at rebase, which is the point of raising them one at a time.

## The four PR slices, and what each is waiting on

| slot | scope | blocked on |
|---|---|---|
| A | TP: KDA NoParallel, Gated MLA, LatentMoE placements | the eager PR landing; nothing of ours |
| B | EP + grouped GEMM | same, plus it ships the per-expert <-> grouped conversion |
| C | PP + Block AttnRes cross-stage adapter | same; its first commit adds the two decoder-loop seams |
| D | CP: Ulysses for MLA, fla's merged KCP for KDA | same |

None of the four is blocked on a defect. All four are blocked on the same
external event, and the work between now and then is rebasing onto
`torchtitan/models/kimi_k3/` and cutting each slice down to its topic.

## Verified

* **Parallelism matrix, 18/18 eager at 100 steps**, every cell from a shared
  step-0 checkpoint. Step-1 spread 0.009, decomposing exactly by reduction
  structure: seven configurations bit-identical to single-GPU (FSDP, PP at 2/4/8,
  EP at 2/8 introduce no numerical difference), the rest clustering by whether TP
  or CP is present. Compiled **15/18**, the three gaps being one `_grouped_mm`
  limitation.
* **Dense control**, 13/13 -- the run-horizon band is not architecture-specific.
* **MXFP4/MXFP8 QAT matrix, 13/13 at 10 steps** on the report architecture with
  `mxfp4_qat` the only field changed. This is the precision K3 is actually
  post-trained in (report sec 4.1.4), which no earlier matrix covered.
* **vLLM weight-sync contract** against the OFFICIAL K3 implementation
  (`vllm/models/kimi_k3`, now on upstream main): 1032 tensors accepted, engine
  built, generation end to end, no shim, on a source build with the branch's real
  kernels.
* **veRL full-parameter SFT** on the torchtitan engine, end to end.
* **2.8T config audited against report Table 1** field for field.
* `kimi_k3` suite: 260 passed.

Two open questions closed by reading the official inference code rather than by
argument: Block AttnRes is **two per layer**, and the final aggregation over
block representations **exists** -- both matching what we already had.

## Blocked, in order of what unblocks the most

1. **Everything PR-facing: the eager reference landing.** Four branches, the RFC
   update, and the rebase all wait on it. No action available.
2. **veRL GRPO** -- stops at `to_hf` emitting `attn_gate_proj` (our name) where
   the release spells it `g_proj`, so that key did not go through
   `titan_to_official`. The adapter also passes `kda_layers=set()`, which makes
   `_mla_layer` true for every layer; establishing whether that is the cause is a
   read of two functions, not an experiment. Everything before it is verified:
   config resolution, flavor resolution, model build, checkpoint load reaching
   per-key lookup.
3. **LoRA TP** -- real defect. 22 of 24 testable Replicate-on-TP gradients
   disagree across ranks at steps 2-3; the only clean pair is layer 0's
   `q_a_proj` adapters, which is the one parameter upstream of every TP
   collective. LoRA-off is clean (40/40), so the wrapper breaks a boundary rather
   than the adapter math being wrong. Next measurement: the placements of
   `base_out` and `lora_out` immediately before they are summed, since
   `parallelize` redirects the Colwise/Rowwise style onto `.base` and the adapter
   branch runs outside that boundary.
4. **`_grouped_mm` zero contraction dim** -- PR kit written with a five-line
   reproduction; the C++ patch is not compile-verified. File it.
5. **MTP loss wiring** -- architecture done, loss not. Needs the trainer's loss
   interface to carry more than one head.
6. **Cross-stack numerical alignment** -- deliberately parked until the rebase.
   Doing it now means fighting two different module trees; after the rebase it is
   mostly a state-dict round-trip.
7. **Hardware-gated**: `PP8xVP4` at seq 8192, real-data long runs. 8x5090 / H200.
8. **ViT real TP and dynamic CP** (report sec 5.2.3) -- sub-CP-group formation is
   per-batch while `DeviceMesh` is static for the run. A core change.

## Two things that will be lost on this instance

`workspace_is_volume` is **false**, so nothing under `/workspace` survives
recycle or destroy, and `/venv/*` is outside it regardless.

* `/workspace/vllm_k3_wheel/vllm-0.1.dev1+g6dc76a9ad-cp312-cp312-linux_x86_64.whl`
  (464M) -- the K3-capable vLLM, 65 minutes of compile. Pinned to torch
  2.13.0+cu130 by the libtorch C++ ABI; sm_120 cubins, so a 5090 is fine on
  architecture. Not uploaded, by decision.
* The seed checkpoints under `/workspace/*/seed*`, which the LoRA TP measurements
  reproduce from.

## Drafts ready to post

* `Raising_PRs/RFC3029_UPDATE_2026-08-04.md` -- RFC body, faithful-only matrix.
* `Raising_PRs/PR4025_MATRIX_COMMENT_2026-08-05.md` -- the matrix comment for the
  eager PR, seeded numbers.
* `Raising_PRs/REVIEW_4025_CONSOLIDATED.md` -- one top-level comment plus a
  bundled review. Worth re-checking against the PR head before posting, since the
  structure changed after those comments were written.
