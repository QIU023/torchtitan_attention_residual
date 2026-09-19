# MoonEP diff audit, 2026-09-19

Branch `k3_moonep_seam`, twelve commits on main `6c2dadbb3`, read line by line. Final diff: 8 files, 1035 insertions, 17 deletions, of which 675 lines are source and 360 are tests.

## The twelve commits

    8a9dbf0bb  +1183 -10   the dispatcher, the expert module, the spec wiring, the first tests
    bad2c456b   +167 -82   prefetch and slot-grad reduce over moonep's public primitives
    106a23185   +111 -11   core: fill the static token capacity of any persistent EP backend
    047402495   +111 -162  efsdp check over CP and TP, one slot count, core's grouped-mm seam
    4b7050ffe    +47 -115  comments and docstrings cut to statements of fact
    641b18f53   +276 -206  moonep's own launch_prefetch and launch_grad_reduce replace the copies
    ac7778ef1    +19 -20   lint: ufmt layout, file end, buffer guard on grad_reduce_handles
    504a36767   +164       the on-device test on two GPUs against a dense reference
    822642a22     +5 -10   the mesh check reads core's efsdp axis instead of recomputing it
    25dae14b4     +4 -452  the CPU tests drop what the on-device test now covers
    676f91826     +3 -7    ufmt
    c02e6240f    +32 -29   the mesh precondition tested against real ParallelDims meshes

Two of these are net deletions of the branch's own earlier work: `047402495` and `25dae14b4` together remove 614 lines that the first commit added. `641b18f53` is a replacement rather than an addition, swapping a copy transport for MoonEP's own kernels at roughly even line count.

One scope leak: `25dae14b4` is titled for the test trim but also deletes one comment line in `moon_ep_dispatcher.py`, which belongs with `4b7050ffe`. The content is right and the message does not mention it.

## What the feature is, and where each piece lives

`model_registry(..., moe_comm_backend="moonep")` selects a different EP transport for the Kimi K3 latent MoE. Four pieces:

- `MoonEPTokenDispatcher` (`kimi_k3/moon_ep_dispatcher.py`, 285 lines): a `BaseEPTokenDispatcher` subclass. `init_buffer` allocates MoonEP's persistent buffer on the EP group, `dispatch` and `combine` are two `torch.autograd.Function`s whose backward is the other kernel on the same plan. Routing weights are applied on the torchtitan side at combine, so the router trains through the same path as with the standard dispatcher. With no EP mesh both calls delegate to `LocalTokenDispatcher`.
- `MoonEPGroupedExperts` (`kimi_k3/moon_ep_experts.py`, 329 lines): a `GroupedExperts` subclass that computes over this rank's `E / R` expert rows followed by its `B` prefetch slots, through core's `_grouped_mm` seam. A third autograd Function recomputes the forward in backward with the tables as leaves, because the tables are NVLink mappings rather than parameters and autograd cannot reach them otherwise.
- Core change (`models/common/token_dispatcher.py`, +25 -17): `update_ep_token_dispatcher_config` selected DeepEP and HybridEP by name. It now fills any `BaseEPTokenDispatcher.Config` that declares `static_token_capacity`, and skips a backend that declares `ep1_local_fallback` when EP is off. Both existing backends declare the first and not the second, so their behaviour is unchanged.
- Model wiring (`kimi_k3/__init__.py` +33, `kimi_k3/moe.py` +20): the spec picks the two classes when the backend is `moonep`, and `KimiLatentMoE.parallelize` calls `check_moonep_mesh` and attaches the table backend after core's `MoE.parallelize` has wired the EP mesh.

Seams used rather than replaced: `BaseEPTokenDispatcher`, `GroupedExperts` and its `_grouped_mm`, `Module.Config`, `MoE.parallelize`, `ParallelDims.get_optional_mesh`, `LocalTokenDispatcher` for the EP=1 path. No forward hook, no monkeypatch, no module global, no dict keyed by `id(module)`.

## The MoonEP interface surface

Eight public entry points, one private method and four private keys:

    moonep.Buffer(S, H, K, E, num_ep_ranks, num_sms, token_padding, B, group)
    Buffer.dispatch(...)                          forward transport and the combine backward
    Buffer.combine(...)                           forward transport and the dispatch backward
    moonep.buffer.pad_dim0_for_alignment          slot count and row padding, 2 call sites
    moonep.buffer.create_nvl_single_owner_tensor  the bf16 [P + B] expert rows
    moonep.buffer.create_nvl_dist_tensor          the fp32 slot-gradient rows
    moonep.prefetch.launch_prefetch               slot weights in
    moonep.grad_reduce.launch_grad_reduce         slot gradients home
    plan.experts_to_copy                          [R, B] int32, read at 3 call sites

    Buffer._require_ctx()                         PRIVATE, for the reduce kernel's barrier handles
      meta_buf, meta_chunk_padded, BARRIER_OFF, grid_sync_bar

The private reach is the one interface debt. `launch_grad_reduce` needs the Buffer's barrier handles and MoonEP exposes no accessor for them, so `grad_reduce_handles()` reads four keys out of the Buffer's context dict. That is an upstream ask on MoonEP, not something this branch can fix, and it is the thing most likely to break on a MoonEP release.

## Size against the comparable backends

    backend    support code in torchtitan
    HybridEP   131 lines, all inside token_dispatcher.py
    MoonEP     614 lines, two files in the model folder
    DeepEP     713 lines, 605 in distributed/deepep/deepep.py plus 108 in token_dispatcher.py

Comment plus docstring is 10.4 percent of the two MoonEP files (56 of 537 non-blank lines), down from the 15.3 percent the 09-17 audit measured, against #4577's 4.3 percent.

## Is it the minimum needed diff

Mostly, and the part that is not is nameable.

Irreducible, because no other titan EP backend does it: MoonEP moves expert *weights* (prefetch into slots) and expert *gradients* (reduce from slots), so the expert module has to own a table per projection, refresh its own rows each forward, and hand the slot gradients back. That is what `moon_ep_experts.py` is, and it cannot shrink without MoonEP shipping an allocator for the `[E + B]` contiguous range and a public entry for the barrier handles. Both are recorded as upstream asks.

Also irreducible: the third autograd Function. MoonEP's dispatch and combine are not `torch.library` ops the way core wraps DeepEP, so the graph breaks at every dispatch and the expert step needs explicit backward. Wrapping them as custom ops is the obvious follow-up and would remove the compile break, not the line count.

Not irreducible, three items:

- `MoonEPTableBackend`, the 20-line `Protocol` at `moon_ep_experts.py:27-46`. It has exactly one implementation, `MoonEPTableBackendNVLink`, which does not even declare it, and no second user anywhere including the tests. It buys two type annotations. Under the #4577 rules this is speculative generality and should go, with `attach` typed on the concrete class.
- `num_sms` and `token_padding` in `MoonEPTokenDispatcher.Config`. Both default to MoonEP's own defaults, no flavor sets either, and neither is read except to pass straight through. Eight lines including their docstrings.
- The table orientation. Rows are stored `[row, in, out]` while core's grouped-mm seam wants `[row, out, in]`, so `_refresh_own_rows` transposes three parameters in every forward, `_compute` transposes three tables in every call, and the backward transposes three gradients back with a `.contiguous()` each. Storing the tables in core's orientation would remove six transposes per step. Whether the prefetch kernel tolerates the swap is a question, not a finding: it tiles 128 by 128 over the trailing dims and both dims are multiples of 128 at the shapes used, but that has not been tested.

## Correctness notes from the read

The gradient scratch buffers are overwritten rather than accumulated across micro-batches, which is correct because each micro-batch's backward returns its own gradient and autograd accumulates into the parameter. The fp32 `[E, in, out]` table is allocated zeroed and only this rank's span is ever written, so the reduce adds peers' slot contributions onto this rank's own rows without touching the rest. Both hold only while a later micro-batch's forward cannot run before an earlier one's backward, which is the interleaved-pipeline limitation the body already states.
