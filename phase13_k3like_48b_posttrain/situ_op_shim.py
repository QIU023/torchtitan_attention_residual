"""Register a Python ``_C::situ_and_mul`` so a precompiled vLLM can build K3.

Purpose is narrow: check whether the official K3 implementation ACCEPTS our
exported weight names and config. It is not a fidelity or performance run --
this is a reference implementation of the op in Python, not Moonshot's kernel.

Why it is needed at all: `/workspace/vllm_k3` was installed with
`VLLM_USE_PRECOMPILED=1`, which reuses a release wheel's binaries and therefore
has none of the K3 branch's new CUDA kernels. `SituAndMul.__init__` does

    if current_platform.is_cuda_alike():
        self.op = torch.ops._C.situ_and_mul

unconditionally, so construction fails on CUDA before the CustomOp dispatch ever
gets to choose `forward_native` -- which exists and would have worked. The real
fix is a source build with kernels; this unblocks the contract check meanwhile.

Import before constructing `LLM`, and run with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` so the engine builds in this process where
the registration is visible.
"""

import torch


def install() -> None:
    if hasattr(torch.ops._C, "situ_and_mul"):
        return

    lib = torch.library.Library("_C", "FRAGMENT")
    lib.define("situ_and_mul(Tensor! out, Tensor input, float beta, float linear_beta) -> ()")

    def _situ_and_mul(out, input, beta, linear_beta):
        d = input.shape[-1] // 2
        gate = input[..., :d].float()
        up = input[..., d:].float()
        gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta > 0:
            up = linear_beta * torch.tanh(up / linear_beta)
        out.copy_((gate * up).to(out.dtype))

    lib.impl("situ_and_mul", _situ_and_mul, "CUDA")
    lib.impl("situ_and_mul", _situ_and_mul, "CPU")
    # Keep the Library alive; a collected Library deregisters its ops.
    globals()["_lib"] = lib
