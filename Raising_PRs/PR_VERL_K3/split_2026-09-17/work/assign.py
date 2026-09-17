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
    "tests/workers/test_torchtitan_engine_config_fields.py": [("*", "01")],
    "tests/workers/test_torchtitan_engine_expert_stacks.py": [("*", "03")],
    "tests/workers/test_torchtitan_engine_lora_sync.py": [("*", "03")],
    "tests/workers/test_torchtitan_engine_peft_config.py": [("*", "03")],
    "tests/workers/test_torchtitan_engine_k3_processor.py": [("*", "05")],

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
        ((18, 18), "03"),
        ((19, 19), "00"),
        ((21, 22), "01"),
        ((23, 23), "00"),
        ((33, 37), "01"),
        ((42, 42), "04"),
        ((43, 43), "03"),
        ((44, 45), "01"),
        ((46, 79), "00"),
        ((80, 95), "01"),
        ((96, 118), "03"),
        ((119, 126), "04"),
        ((127, 127), "05"),
        ((128, 142), "04"),
        ((143, 150), "05"),
        ((151, 151), "04"),
        ((152, 153), "01"),
        ((177, 177), "03"),
        ((214, 265), "02"),
        ((303, 307), "01"),
        ((308, 312), "05"),
        ((313, 370), "01"),
        ((372, 372), "04"),
        ((373, 375), "02"),
        ((376, 378), "04"),
        ((380, 399), "01"),
        ((403, 405), "03"),
        ((441, 441), "00"),
        ((442, 442), "03"),
        ((443, 451), "04"),
        ((452, 459), "03"),
        ((460, 461), "00"),
        ((462, 472), "01"),
        ((537, 541), "02"),
        ((564, 644), "01"),
        ((664, 768), "02"),
        ((773, 809), "01"),
        ((810, 816), "04"),
        ((899, 910), "00"),
        ((1006, 1106), "03"),
        ((1107, 1188), "00"),
        ((1194, 1225), "03"),
        ((1226, 1236), "00"),
        ((1237, 1317), "03"),
        ((1318, 1344), "02"),
        ((1345, 1374), "03"),
        ((1375, 1411), "02"),
        ((1412, 1472), "00"),
        ((1473, 1474), "01"),
        ((1475, 1476), "05"),
        ((1477, 1489), "01"),
        ((1490, 1511), "02"),
        ((1512, 1532), "05"),
        ((1533, 1563), "01"),
        ((1564, 1791), "03"),
        ((1842, 1846), "05"),
        ((1898, 1898), "01"),
        ((1899, 1921), "02"),
        ((1922, 1922), "01"),
        ((1924, 1935), "04"),
        ((1936, 1943), "01"),
        ((1945, 1947), "05"),
        ((1948, 1974), "01"),
        ((1975, 1976), "00"),
        ((1977, 1991), "04"),
        ((1992, 1995), "05"),
        ((1996, 2002), "04"),
        ((2003, 2018), "01"),
        ((2019, 2020), "05"),
        ((2022, 2038), "01"),
    ],
}

# item -> extra revisions [(ordinal_name, text|None)]
VARIANTS = {
    IMPL: {
        43: [("03", "from torchtitan.config.transform import apply_transforms\n"),
             ("04", "from torchtitan.config.transform import apply_transforms, ContextParallelTransform\n")],
        308: [("01", "        # Two naming spaces in a model package: model_registry parses\n"
                     "        # \"<size>_<variant>\", while some flavors are config_registry FUNCTIONS\n"
                     "        # whose names its parser cannot reach.\n"),
              ("05", "        # Two naming spaces in the kimi_k3 package: model_registry parses\n")],
        309: [("01", None)],
        310: [("01", None)],
        311: [("01", None)],
        312: [("01", None)],
        794: [("01", "        # of the vocabulary (the loss-parallel layout under spmd_types) or of\n"),
              ("05", "        # of the vocabulary (loss-parallel layout, Kimi K3 under spmd_types) or of\n")],
        1298: [("02", "        gen = _gen()\n"
                      "        if self.parallel_dims.pp_enabled:\n"
                      "            gen = self._iter_pp_gathered(gen, device)\n"
                      "        return gen, None\n"),
               ("03", None)],
        1565: [("03", '    """``{fqn: wrapper}`` for every LoRA-wrapped linear in the module.\n'),
               ("05", '    """``{fqn: wrapper}`` for every KimiLoRALinear in the module.\n')],
        1569: [("03", "    qualified suffixes, and it skips subtrees it cannot wrap). A config-derived\n"),
               ("05", "    qualified suffixes, and it skips the KDA subtree structurally). A config-derived\n")],
        1702: [("03", "    path is the 4-D ``A_log`` reshape, which ``apply_lora`` never reaches\n"),
               ("05", "    path is the 4-D ``A_log`` reshape, and ``apply_lora`` skips the KDA subtree\n")],
        1720: [("03", "        # every layer, and a graft-only target can have no HF\n"),
               ("05", "        # every layer, and a graft-only target (the K3 attention gate) can have no HF\n")],
        1943: [("01", "                input_ids, position_ids, sp_size=pad_multiple, pad_value=pad_id\n"
                      "            )\n")],
        1979: [("04", "            # from the positions, the shards and the layouts. Hand it\n"),
               ("05", "            # from the positions, the shards, the KDA routing and the layouts. Hand it\n")],
        2003: [("01", "        if self._folded_token_stream:\n"),
               ("04", "        elif self._folded_token_stream:\n")],
        2004: [("01", "            # A model whose forward takes the packed stream's document offsets (a\n"),
               ("05", "            # A model whose forward takes the packed stream's document offsets (KDA's\n")],
        2005: [("01", "            # linear-attention recurrence, which would otherwise run across the\n"),
               ("05", "            # recurrence and short convolution, which would otherwise run across the\n")],
        1944: [("04", None)],
        2014: [("01", "                # them by consumer (a flex BlockMask, varlen offsets, and so on).\n"),
               ("05", "                # them by consumer (a flex BlockMask for MLA, varlen offsets for KDA).\n")],
    },
    "verl/workers/engine/torchtitan/utils.py": {},
}

# (item, "before"|"after", anchor_item) -- reorder within the item list
MOVES = {
    IMPL: [
        ((442, 459), "before", 441),      # transforms must be built before Trainer()
        ((1936, 1943), "after", 1944),    # TP padding goes after the CP call's closing paren
        ((1945, 1947), "after", 2018),    # old multimodal update stays at the end of the block
    ],
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
