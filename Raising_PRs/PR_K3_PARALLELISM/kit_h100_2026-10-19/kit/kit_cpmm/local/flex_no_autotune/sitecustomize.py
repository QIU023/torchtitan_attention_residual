"""5060 test shim, logbook kit only: compile FlexInnerAttention without max_autotune / coordinate descent, which
take tens of minutes per new shape on this box. Both sides of an equivalence test use the same compiled kernel."""

import sys

import torch
from torch.nn.attention.flex_attention import flex_attention

from torchtitan.models.common.attention.attention import FlexInnerAttention

FlexInnerAttention._compiled_flex_attn = torch.compile(
    flex_attention, options={"wrap_inductor_compiled_regions": True, "triton.cudagraphs": False}
)
sys.stderr.write("flex_no_autotune: FlexInnerAttention compiled without max_autotune\n")
