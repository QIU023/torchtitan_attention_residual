"""Rewrite the torchtitan engine's context-parallel path onto torchtitan main's API."""
import sys
p = sys.argv[1]
s = open(p).read()

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)

# R1: the dead import of the pre-4639 API
rep('''try:
    from torchtitan.distributed.context_parallel import prepare_context_parallel_input
except ImportError:  # trees after PR 4639 shard CP batches through prepare_context_parallel_batch
    prepare_context_parallel_input = None

# torchtitan's CP input API changed shape: the current one takes the named inputs as a dict
# and shards each along its declared sequence axis in place; the earlier one took
# (inputs, labels, extra_kwargs, ...) positionally. Both are still met by trees this engine
# runs on, so the call site below picks by signature.
_CP_INPUT_DICT_API = prepare_context_parallel_input is not None and (
    "input_dict" in inspect.signature(prepare_context_parallel_input).parameters
)
''', '''from torchtitan.config.transform import apply_transforms, ContextParallelTransform
''')

# R2: the compat kwargs no longer key on a model name
rep('''def _parallelism_compat_kwargs(spmd_backend: str, torchtitan_name: str) -> dict:
    """ParallelismConfig fields whose shape differs across the torchtitan trees this engine runs on.

    Older trees take ``spmd_backend``; current ones dropped it (spmd_types is the only backend).
    Kimi K3's CP needs contiguous rank-ordered shards: the head-tail balancer permutes the
    sequence before sharding. Older trees spell the balancer as a string (None disables it),
    current ones as ``ContextParallelLoadBalancerConfig`` with ``load_balancer_type=None``.
    """
    fields = ParallelismConfig.__dataclass_fields__
    kwargs = {}
    if "spmd_backend" in fields:
        kwargs["spmd_backend"] = spmd_backend
    if torchtitan_name == "kimi_k3":
        try:
            from torchtitan.config import ContextParallelLoadBalancerConfig
        except ImportError:
            kwargs["context_parallel_load_balancer"] = None
        else:
            kwargs["context_parallel_load_balancer"] = ContextParallelLoadBalancerConfig(load_balancer_type=None)
    return kwargs
''', '''def _parallelism_compat_kwargs(spmd_backend: str, cp_enabled: bool) -> dict:
    """ParallelismConfig fields whose shape differs across the torchtitan trees this engine runs on.

    Older trees take ``spmd_backend``; current ones dropped it (spmd_types is the only backend).
    Under context parallel the shards are contiguous and rank-ordered: the engine gathers the
    logits back in rank order (_finish_pred), which a permuting balancer (head-tail) would scramble.
    """
    fields = ParallelismConfig.__dataclass_fields__
    kwargs = {}
    if "spmd_backend" in fields:
        kwargs["spmd_backend"] = spmd_backend
    if cp_enabled:
        from torchtitan.config import ContextParallelLoadBalancerConfig

        kwargs["context_parallel_load_balancer"] = ContextParallelLoadBalancerConfig(load_balancer_type=None)
    return kwargs


def _context_parallel_transform(model_config) -> ContextParallelTransform:
    """The CP backends for the inner attentions this model config carries.

    Flex attention takes the all-gather-KV backend; a KDA inner attention, where the tree has
    one, takes its CP routing counterpart. Subtrees whose tokens are not sharded on the cp
    axis (a vision tower) keep their local attention.
    """
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention

    mapping: dict = {FlexInnerAttention.Config: KVAllGatherCPFlexInnerAttention}
    try:
        from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA
        from torchtitan.models.kimi_k3.kda import InnerKDA
    except ImportError:
        pass
    else:
        if any(True for _ in model_config.traverse(InnerKDA.Config)):
            mapping[InnerKDA.Config] = ContextParallelInnerKDA
    return ContextParallelTransform(inner_attention=mapping, exclude_fqn_prefixes=("vision_encoder",))
''')
rep('''            **_parallelism_compat_kwargs(self.engine_config.spmd_backend, torchtitan_name),''',
    '''            **_parallelism_compat_kwargs(
                self.engine_config.spmd_backend, self.engine_config.context_parallel_size > 1
            ),''')

# R3: the name flags
rep('''        # kimi_k3 handles CP module-internally (Ulysses) and is causal-only:
        # no attention_masks consumed, no upstream CP mask sharding needed.
        self._model_cp_is_module_internal = torchtitan_name == "kimi_k3"
        # Kimi K3 takes a folded [T] token stream (the model declared it on the earlier
        # tree; on torchtitan main it is a property of the architecture, not an attribute).
        self._folded_token_stream = torchtitan_name == "kimi_k3"
''', '')

# R4: the CP transform at construction, and the folded-stream probe after the trainer exists
rep('''        with _fp32_matmul_emulation_optional():
            self.trainer = Trainer(self.config)
''', '''        if self.engine_config.context_parallel_size > 1:
            self.config = apply_transforms(
                self.config, [_context_parallel_transform(self.config.model_spec.model)]
            )
        with _fp32_matmul_emulation_optional():
            self.trainer = Trainer(self.config)
        # Decoders on torchtitan main take a folded [T] token stream and own their input
        # preprocessing (masks, CP shards, layouts); probed on the model, not on its name.
        from torchtitan.protocols.model import BaseModel

        self._folded_token_stream = (
            type(self.trainer.model_parts[0]).preprocess_inputs is not BaseModel.preprocess_inputs
        )
''')

# R5: one ParallelDims, the trainer's
rep('''        world_size = torch.distributed.get_world_size()
        self.parallel_dims = ParallelDims(
            dp_shard=self.engine_config.data_parallel_shard_size,
            dp_replicate=self.engine_config.data_parallel_replicate_size,
            cp=self.engine_config.context_parallel_size,
            tp=self.engine_config.tensor_parallel_size,
            pp=self.engine_config.pipeline_parallel_size,
            ep=self.engine_config.expert_parallel_size,
            world_size=world_size,
        )
        self.device_mesh = self.parallel_dims.build_mesh()
''', '''        # The trainer's ParallelDims and mesh: the model's SPMD context and the engine's
        # gathers must name the same process groups.
        self.parallel_dims = self.trainer.parallel_dims
        self.device_mesh = self.parallel_dims._world_mesh
''')

# R6: the CP input path
start = s.index('''        pad_multiple = 1
        if self.parallel_dims.cp_enabled:
            # Context parallel wants a sequence it can cut evenly''')
end_marker = '''                extra_kwargs["attention_masks"] = self.module[0].get_attention_masks(positions=positions)
'''
end = s.index(end_marker, start) + len(end_marker)
old_block = s[start:end]
assert "ulysses_pad(" in old_block and "_cu_seqlens_from_positions(" in old_block and old_block.count("if _CP_INPUT_DICT_API:") == 1
pad_start = old_block.index('''        if pad_multiple > 1:''')
pad_end = old_block.index('''        if self.parallel_dims.cp_enabled:
            # prepare_context_parallel_input contract''')
pad_block = old_block[pad_start:pad_end]
new_block = '''        pad_multiple = 1
        if self.parallel_dims.cp_enabled:
            # Context parallel cuts the packed stream into contiguous rank-ordered shards
            # (_parallelism_compat_kwargs) and the flex BlockMask wants whole 128-token
            # blocks per shard; a packed no-padding stream is neither. ulysses_pad does
            # the padding, with the adjustments below.
            pad_multiple = self.parallel_dims.cp * 128
        if self.parallel_dims.tp_enabled:
            # Sequence parallel scatters the token dim across the TP ranks, so the
            # packed stream must divide by the TP degree as well.
            pad_multiple = math.lcm(pad_multiple, self.parallel_dims.tp)
''' + pad_block + '''        if self.parallel_dims.cp_enabled:
            # The model owns its context-parallel preprocessing on this tree: the masks
            # from the positions, the shards, the KDA routing and the layouts. Hand it
            # the folded [T] stream with the global positions and take back this
            # rank's inputs and kwargs. Labels stay FULL length: verl's loss path
            # (nested no-padding log_prob / loss_mask handling) assumes full
            # sequences, so the engine all-gathers the seq-sharded logits after the
            # model call (_finish_pred) instead of sharding the loss side.
            if position_ids.dim() != 2 or position_ids.shape[0] != 1:
                raise NotImplementedError("context parallel takes a [1, T] position stream")
            batch = {
                "input": input_ids.squeeze(0),
                "labels": labels.squeeze(0),
                "positions": position_ids.squeeze(0),
            }
            local_inputs, _labels_local, extra_kwargs = self.module[0].preprocess_inputs(
                batch,
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
            )
            input_ids = local_inputs.unsqueeze(0)
            extra_inputs = {}
        elif self._folded_token_stream:
            # Kimi K3's KDA and short convolution take the packed stream's
            # document offsets explicitly under flex attention; a micro-batch
            # here packs several sequences, and without them the recurrent
            # state runs across sequences. A document starts where the
            # positions restart at 0.
            cu_seqlens = _cu_seqlens_from_positions(extra_inputs.get("positions"))
            if cu_seqlens is not None:
                extra_kwargs["cu_seqlens"] = cu_seqlens
            if extra_kwargs.get("attention_masks") is not None and hasattr(self.module[0], "get_attention_masks"):
                # The model builds its own masks from the [T] positions: current trees key
                # them by consumer (a flex BlockMask for MLA, varlen offsets for KDA).
                positions = extra_inputs["positions"]
                if positions.dim() == 2 and positions.shape[0] == 1:
                    positions = positions.squeeze(0)
                extra_kwargs["attention_masks"] = self.module[0].get_attention_masks(positions=positions)
'''
s = s[:start] + new_block + s[end:]

# R7: the CP gather's backward takes this rank's slice unscaled, as the tp gather does
rep('''        if parallel_dims.cp_enabled:
            # Inputs were seq-sharded across cp; the loss side works on
            # full sequences (see prepare_model_inputs), so gather the
            # logits back (differentiable -> reduce-scatter backward).
            cp_group = parallel_dims.get_mesh("cp").get_group()
            pred = gather_outputs_and_unpad(
                pred.contiguous(), gather_dim=1, group=cp_group
            )
''', '''        if parallel_dims.cp_enabled:
            # Inputs were seq-sharded across cp; the loss side works on full sequences
            # (see prepare_model_inputs), so gather the logits back. Every cp rank
            # computes the same full loss and FSDP reduces over dp_shard x cp, so the
            # backward takes this rank's slice unscaled, as for tp above.
            cp_group = parallel_dims.get_mesh("cp").get_group()
            pred = gather_outputs_and_unpad(pred.contiguous(), gather_dim=1, grad_scaler=False, group=cp_group)
''')

# R8: the non-pipeline forward names the pipeline path instead of promising one
rep('''            raise NotImplementedError(
                "Pipeline parallelism is not yet supported in model_forward_step. "
                "This will be implemented in a follow-up PR."
            )''', '''            raise RuntimeError(
                "model_forward_step is the non-pipeline path; under pipeline parallelism the "
                "worker drives the schedule through _pp_forward_backward_batch"
            )''')

# R10: comments that named the model where the property is general
rep('''        # The kimi_k3 standard token dispatcher synchronizes with the CPU,
        # which CUDA graphs cannot capture; run eager under expert parallel.''',
    '''        # The standard token dispatcher synchronizes with the CPU, which CUDA
        # graphs cannot capture; run eager under expert parallel.''')
rep('''                # Folded-stream models (kimi_k3) take [T] token streams; the''',
    '''                # Folded-stream decoders take [T] token streams; the''')
# the signature probe was the only inspect user
if "inspect." not in s:
    rep("import inspect\n", "")
open(p, "w").write(s)
print("ok", len(s.splitlines()))
