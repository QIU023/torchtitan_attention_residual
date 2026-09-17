# ordinal names, in application order; index in splitlib.ORDER
# "01".."06" = upstream patches, "00" = the local env/diagnostics patch, applied LAST.

IMPL = "verl/workers/engine/torchtitan/transformer_impl.py"

# ranges are inclusive item-index ranges over the item list of that file
ASSIGN = {
    "tests/special_e2e/sft/run_sft_engine.sh": [("*", "00")],
    "verl/third_party/vllm/__init__.py": [("*", "00")],
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py": [("*", "00")],
    "verl/workers/rollout/vllm_rollout/utils.py": [("*", "00")],
    "verl/trainer/ppo/v1/trainer_base.py": [("*", "00")],
    "verl/utils/tokenizer/continuous_token_wiring.py": [("*", "05")],
    "verl/utils/tokenizer/tokenizer.py": [("*", "05")],
    "verl/utils/dataset/multiturn_sft_dataset.py": [("*", "05")],
    "verl/utils/dataset/rl_dataset.py": [("*", "05")],
    "tests/workers/test_torchtitan_engine_receiver_names.py": [("*", "01")],
    "tests/workers/test_torchtitan_engine_pp_sync.py": [("*", "02")],
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
        ((18, 18), "03"),            # import itertools
        ((19, 19), "00"),            # import contextlib
        ((21, 22), "01"),            # import inspect / math
        ((23, 23), "00"),            # import sys
        ((33, 34), "01"), ((36, 37), "01"),
        ((42, 42), "04"),            # drop prepare_context_parallel_input
        ((43, 43), "03"),            # torchtitan.config.transform import (variant at 04)
        ((44, 45), "01"),
        ((46, 79), "00"),            # BFX9 opt-out
        ((80, 95), "01"),            # _parallelism_compat_kwargs
        ((96, 118), "03"),           # _lora_transform
        ((119, 126), "04"),          # _context_parallel_transform
        ((127, 127), "05"),          # its KDA docstring line
        ((128, 142), "04"),
        ((143, 150), "05"),          # the kimi_k3 KDA entry of the CP backend map
        ((151, 151), "04"),
        ((152, 153), "01"),          # verl.utils.ulysses import
        ((177, 177), "03"),          # iter_per_tensor_params_ep import
        ((214, 265), "02"),          # _PipelineLossBridge
        ((303, 307), "01"),
        ((308, 312), "05"),          # kimi-named comment (variant at 01)
        ((313, 338), "01"),
        ((370, 370), "01"),
        ((372, 372), "04"),
        ((373, 375), "02"),
        ((376, 378), "04"),
        ((380, 383), "01"),
        ((385, 399), "01"),
        ((403, 405), "03"),          # disable_cuda_graphs (EP)
        ((441, 441), "00"),
        ((442, 442), "03"),
        ((443, 451), "04"),
        ((452, 459), "03"),
        ((460, 461), "00"),
        ((462, 472), "01"),
        ((537, 541), "02"),
        ((564, 578), "01"),
        ((619, 644), "01"),
        ((664, 668), "02"),
        ((686, 748), "02"),
        ((763, 768), "02"),
        ((773, 790), "01"),
        ((793, 809), "01"),
        ((810, 816), "04"),
        ((899, 910), "00"),
        ((1006, 1019), "03"),
        ((1051, 1106), "03"),
        ((1107, 1188), "00"),        # KIMI_GRPO_* sync diagnostics
        ((1194, 1220), "03"),
        ((1221, 1231), "00"),
        ((1232, 1261), "03"),
        ((1265, 1317), "03"),
        ((1318, 1344), "02"),        # _iter_pp_gathered
        ((1345, 1374), "03"),
        ((1375, 1411), "02"),        # _guard_fsdp_grad_upcast
        ((1412, 1441), "00"),        # _dynamo_probe_once
        ((1442, 1443), "01"),
        ((1444, 1445), "05"),        # _MULTIMODAL_KEY_ALIASES
        ((1446, 1449), "01"),        # _MULTIMODAL_KEYS (needed by _squeeze_folded)
        ((1450, 1470), "05"),        # _model_multimodal_kwargs
        ((1471, 1501), "01"),        # _squeeze_folded, _cu_seqlens_from_positions
        ((1502, 1686), "03"),        # LoRA sync helpers
        ((1687, 1691), "05"),        # measured-numbers docstring paragraph (kimi-named)
        ((1692, 1734), "03"),
        ((1785, 1789), "05"),
        ((1841, 1841), "01"),        # cp_pad_len = 0
        ((1842, 1877), "02"),        # PP token-budget padding
        ((1878, 1878), "01"),        # pad_multiple = 1
        ((1880, 1891), "04"),        # CP: drop prepare_context_parallel_input, pad_multiple
        ((1892, 1899), "01"),        # TP lcm + ulysses_pad  (moved after 1900)
        # item 1900 is a context line; VARIANTS turns it into an ord-04 deletion
        ((1901, 1903), "05"),        # old multimodal TODO  (moved after 1974)
        ((1904, 1930), "01"),        # cp_pad_len renumber / label pad
        ((1931, 1932), "00"),        # dynamo probe call
        ((1933, 1947), "04"),        # CP preprocess_inputs
        ((1948, 1951), "05"),        # multimodal through the CP preprocessing
        ((1952, 1958), "04"),
        ((1959, 1974), "01"),        # folded stream: cu_seqlens + model masks (1959 variant at 04)
        ((1975, 1976), "05"),
        ((1978, 1978), "01"),
        ((1989, 1994), "01"),
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
        1232: [("03", "        if adapter_mode and sd_adapter is None:\n"),
               ("00", "        elif adapter_mode:\n")],
        1298: [("02", "        gen = _gen()\n"
                      "        if self.parallel_dims.pp_enabled:\n"
                      "            gen = self._iter_pp_gathered(gen, device)\n"
                      "        return gen, None\n"),
               ("03", None)],
        1503: [("03", '    """``{fqn: wrapper}`` for every LoRA-wrapped linear in the module.\n'),
               ("05", '    """``{fqn: wrapper}`` for every KimiLoRALinear in the module.\n')],
        1507: [("03", "    qualified suffixes, and it skips subtrees it cannot wrap). A config-derived\n"),
               ("05", "    qualified suffixes, and it skips the KDA subtree structurally). A config-derived\n")],
        1640: [("03", "    path is the 4-D ``A_log`` reshape, which ``apply_lora`` never reaches\n"),
               ("05", "    path is the 4-D ``A_log`` reshape, and ``apply_lora`` skips the KDA subtree\n")],
        1658: [("03", "        # every layer, and a graft-only target can have no HF\n"),
               ("05", "        # every layer, and a graft-only target (the K3 attention gate) can have no HF\n")],
        1899: [("01", "                input_ids, position_ids, sp_size=pad_multiple, pad_value=pad_id\n"
                      "            )\n")],
        1935: [("04", "            # from the positions, the shards and the layouts. Hand it\n"),
               ("05", "            # from the positions, the shards, the KDA routing and the layouts. Hand it\n")],
        1959: [("01", "        if self._folded_token_stream:\n"),
               ("04", "        elif self._folded_token_stream:\n")],
        1960: [("01", "            # A model whose forward takes the packed stream's document offsets (a\n"),
               ("05", "            # A model whose forward takes the packed stream's document offsets (KDA's\n")],
        1961: [("01", "            # linear-attention recurrence, which would otherwise run across the\n"),
               ("05", "            # recurrence and short convolution, which would otherwise run across the\n")],
        1900: [("04", None)],
        1970: [("01", "                # them by consumer (a flex BlockMask, varlen offsets, and so on).\n"),
               ("05", "                # them by consumer (a flex BlockMask for MLA, varlen offsets for KDA).\n")],
    },
    "verl/workers/engine/torchtitan/utils.py": {},
}

# (item, "before"|"after", anchor_item) -- reorder within the item list
MOVES = {
    IMPL: [
        ((442, 459), "before", 441),      # transforms must be built before Trainer()
        ((1892, 1899), "after", 1900),    # TP padding goes after the CP call's closing paren
        ((1901, 1903), "after", 1974),    # old multimodal update stays at the end of the block
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
