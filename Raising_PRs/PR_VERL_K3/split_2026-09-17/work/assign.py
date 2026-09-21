# ordinal names, in application order; index in splitlib.ORDER
# "01".."06" = upstream patches, "00" = the local env/diagnostics patch, applied LAST.

IMPL = "verl/workers/engine/torchtitan/transformer_impl.py"

# ranges are inclusive item-index ranges over the item list of that file
ASSIGN = {
    "tests/special_e2e/sft/run_sft_engine.sh": [("*", "00")],
    "verl/third_party/vllm/__init__.py": [("*", "00")],
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py": [((54, 54), "05"), ((74, 79), "00"), ((639, 639), "05")],
    "verl/workers/rollout/utils.py": [("*", "05")],
    "verl/experimental/agent_loop/agent_loop.py": [("*", "05")],
    "verl/utils/tokenizer/__init__.py": [("*", "05")],
    "tests/workers/test_torchtitan_engine_cp_gloo.py": [("*", "04")],
    "verl/workers/rollout/vllm_rollout/utils.py": [("*", "00")],
    "verl/trainer/ppo/v1/trainer_base.py": [("*", "00")],
    "verl/utils/tokenizer/continuous_token_wiring.py": [("*", "05")],
    "verl/utils/tokenizer/tokenizer.py": [("*", "05")],
    "verl/utils/dataset/multiturn_sft_dataset.py": [("*", "05")],
    "verl/utils/dataset/rl_dataset.py": [("*", "05")],
    "tests/workers/test_torchtitan_engine_receiver_names.py": [("*", "01")],
    "tests/workers/test_torchtitan_engine_pp_sync.py": [("*", "02")],
    "tests/workers/test_torchtitan_engine_pp_token_budget.py": [("*", "02")],
    # TestContextParallelBackendIsCheckedByTheConfig exercises the
    # context_parallel_backend field, which patch 04 adds, so the class travels
    # with 04 while the rest of the file stays in 01.
    "tests/workers/test_torchtitan_engine_config_fields.py": [
        ((0, 39), "01"),
        ((40, 53), "04"),
        ((54, 57), "01"),
    ],
    "tests/workers/test_torchtitan_engine_checkpoint_step.py": [("*", "01")],
    "tests/workers/test_torchtitan_engine_transform_order.py": [("*", "04")],
    "tests/workers/test_torchtitan_engine_expert_stacks.py": [("*", "03")],
    "tests/workers/test_torchtitan_engine_lora_sync.py": [("*", "03")],
    "tests/workers/test_torchtitan_engine_peft_config.py": [("*", "03")],
    "tests/workers/test_torchtitan_engine_k3_processor.py": [("*", "05")],
    # the K3 FLOPs estimator: model-specific, so it rides with the K3 patch
    "verl/utils/flops_counter.py": [("*", "05")],

    "verl/utils/debug/metrics.py": [
        ((84, 91), "00"),
        ((122, 125), "06"),
        ((131, 132), "06"),
    ],
    "verl/workers/config/engine.py": [
        ((504, 505), "04"),
        ((506, 511), "01"),
        ((547, 547), "04"),
        ((548, 550), "01"),
        ((564, 564), "00"),
        ((565, 567), "04"),
        ((568, 568), "00"),
    ],
    "verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml": [
        ((43, 43), "04"), ((44, 46), "01"),
        ((193, 193), "04"), ((194, 196), "01"),
        ((537, 537), "04"), ((538, 540), "01"),
    ],
    "verl/trainer/config/engine/torchtitan.yaml": [
        ((60, 62), "04"), ((63, 71), "01"),
    ],
    "verl/trainer/config/ref/torchtitan_ref.yaml": [
        ((26, 26), "04"), ((27, 29), "01"),
    ],
    "verl/workers/engine/torchtitan/utils.py": [
        ((27, 28), "01"),
        ((49, 52), "05"),
        ((80, 87), "05"),
        ((91, 98), "01"),
        ((115, 116), "01"),
        ((122, 129), "01"),
        ((138, 139), "01"),
        ((156, 163), "05"),
    ],
    IMPL: [

    ((18,18), "03"),
    ((19,19), "00"),
    ((21,23), "01"),
    ((24,24), "00"),
    ((34,38), "01"),
    ((43,43), "04"),
    ((44,44), "03"),
    ((45,46), "01"),
    ((47,80), "00"),
    ((81,104), "01"),
    ((105,127), "03"),
    ((128,135), "04"),
    ((136,136), "05"),
    ((137,151), "04"),
    ((152,159), "05"),
    ((160,160), "04"),
    ((161,162), "01"),
    ((186,186), "03"),
    ((223,274), "02"),
    ((312,316), "01"),
    ((317,321), "05"),
    ((322,379), "01"),
    ((381,381), "04"),
    ((382,384), "02"),
    ((385,387), "04"),
    ((389,417), "01"),
    ((419,421), "03"),
    ((447,449), "01"),
    ((458,458), "00"),
    ((459,459), "03"),
    ((460,468), "04"),
    ((469,476), "03"),
    ((477,478), "00"),
    ((479,560), "01"),
    ((561,565), "02"),
    ((566,670), "01"),
    ((690,694), "02"),
    ((704,705), "01"),
    ((713,795), "02"),
    ((800,836), "01"),
    ((837,843), "04"),
    ((926,937), "00"),
    ((952,960), "01"),
    ((1034,1134), "03"),
    ((1135,1216), "00"),
    ((1222,1253), "03"),
    ((1254,1264), "00"),
    ((1265,1345), "03"),
    ((1346,1372), "02"),
    ((1373,1402), "03"),
    ((1403,1439), "02"),
    ((1440,1532), "00"),
    ((1533,1534), "01"),
    ((1535,1536), "05"),
    ((1537,1575), "01"),
    ((1576,1597), "02"),
    ((1598,1618), "05"),
    ((1619,1649), "01"),
    ((1650,1877), "03"),
    ((1928,1932), "05"),
    ((1984,1984), "01"),
    ((1985,2007), "02"),
    ((2008,2008), "01"),
    ((2010,2021), "04"),
    ((2022,2029), "01"),
    ((2031,2033), "05"),
    ((2034,2060), "01"),
    ((2061,2064), "00"),
    ((2065,2079), "04"),
    ((2080,2083), "05"),
    ((2084,2090), "04"),
    ((2091,2106), "01"),
    ((2107,2108), "05"),
    ((2110,2126), "01"),
    ],
}

# item -> extra revisions [(ordinal_name, text|None)]
VARIANTS = {
    IMPL: {44: [('03', 'from torchtitan.config.transform import apply_transforms\n'),
      ('04',
       'from torchtitan.config.transform import apply_transforms, ContextParallelTransform\n')],
 317: [('01',
        '        # Two naming spaces in a model package: model_registry parses\n'
        '        # "<size>_<variant>", while some flavors are config_registry FUNCTIONS\n'
        '        # whose names its parser cannot reach.\n'),
       ('05', '        # Two naming spaces in the kimi_k3 package: model_registry parses\n')],
 318: [('01', None)],
 319: [('01', None)],
 320: [('01', None)],
 321: [('01', None)],
 821: [('01', '        # of the vocabulary (the loss-parallel layout under spmd_types) or of\n'),
       ('05',
        '        # of the vocabulary (loss-parallel layout, Kimi K3 under spmd_types) or of\n')],
 1326: [('02',
         '        gen = _gen()\n'
         '        if self.parallel_dims.pp_enabled:\n'
         '            gen = self._iter_pp_gathered(gen, device)\n'
         '        return gen, None\n'),
        ('03', None)],
 1651: [('03', '    """``{fqn: wrapper}`` for every LoRA-wrapped linear in the module.\n'),
        ('05', '    """``{fqn: wrapper}`` for every KimiLoRALinear in the module.\n')],
 1655: [('03', '    qualified suffixes, and it skips subtrees it cannot wrap). A config-derived\n'),
        ('05',
         '    qualified suffixes, and it skips the KDA subtree structurally). A config-derived\n')],
 1788: [('03', '    path is the 4-D ``A_log`` reshape, which ``apply_lora`` never reaches\n'),
        ('05',
         '    path is the 4-D ``A_log`` reshape, and ``apply_lora`` skips the KDA subtree\n')],
 1806: [('03', '        # every layer, and a graft-only target can have no HF\n'),
        ('05',
         '        # every layer, and a graft-only target (the K3 attention gate) can have no '
         'HF\n')],
 2029: [('01',
         '                input_ids, position_ids, sp_size=pad_multiple, pad_value=pad_id\n'
         '            )\n')],
 2030: [('04', None)],
 2067: [('04', '            # from the positions, the shards and the layouts. Hand it\n'),
        ('05',
         '            # from the positions, the shards, the KDA routing and the layouts. Hand '
         'it\n')],
 2091: [('01', '        if self._folded_token_stream:\n'),
        ('04', '        elif self._folded_token_stream:\n')],
 2092: [('01',
         "            # A model whose forward takes the packed stream's document offsets (a\n"),
        ('05',
         "            # A model whose forward takes the packed stream's document offsets "
         "(KDA's\n")],
 2093: [('01', '            # linear-attention recurrence, which would otherwise run across the\n'),
        ('05',
         '            # recurrence and short convolution, which would otherwise run across the\n')],
 2102: [('01',
         '                # them by consumer (a flex BlockMask, varlen offsets, and so on).\n'),
        ('05',
         '                # them by consumer (a flex BlockMask for MLA, varlen offsets for '
         'KDA).\n')]},
    "verl/workers/engine/torchtitan/utils.py": {},
}

# (item, "before"|"after", anchor_item) -- reorder within the item list
MOVES = {
    IMPL: [((459, 476), 'before', 458), ((2022, 2029), 'after', 2030), ((2031, 2033), 'after', 2106)],
    "verl/workers/config/engine.py": [
        ((565, 567), "before", 564),      # the CP assert precedes the spmd assert
    ],
}

# whole-file content per ordinal (path -> {ordinal: local file with the content})
WHOLE = {
    "tests/workers/test_torchtitan_engine_cp_config.py": {
        "01": "cp_config_01.py",
        "04": "cp_config_04.py",
        "05": None,   # None = the HEAD content
    },
}
